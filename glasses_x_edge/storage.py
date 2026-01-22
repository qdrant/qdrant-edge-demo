import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import requests
from qdrant_edge import (
    EdgeConfig,
    EdgeShard,
    FieldCondition,
    Filter,
    Mmr,
    Point,
    QueryRequest,
    RangeFloat,
    UpdateOperation,
    VectorDataConfig,
)

from config import (
    API_KEY,
    API_KEY_HEADER,
    BACKEND_URL,
    DISTANCE_METRIC_EDGE,
    IMAGE_PATH_KEY,
    IMMUTABLE_SHARD_DIR,
    MMR_DIVERSITY_FACTOR,
    MMR_MAX_CANDIDATES,
    MUTABLE_SHARD_DIR,
    QUEUE_DB_NAME,
    SEARCH_LIMIT,
    SNAPSHOT_CHUNK_SIZE,
    SYNC_INTERVAL,
    SYNC_TIMESTAMP_KEY,
    VECTOR_DIMENSION,
)

from .queue import create_persistent_queue

HEADERS = {API_KEY_HEADER: API_KEY}
SHARD_CONFIG = EdgeConfig(
    vector_data=VectorDataConfig(size=VECTOR_DIMENSION, distance=DISTANCE_METRIC_EDGE)
)


class VisionStorage:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.mutable_shard = None
        self.immutable_shard = None
        self.upload_queue = None
        self.worker_thread = None
        self.is_running = False

    @property
    def mutable_dir(self) -> Path:
        return self.data_dir / MUTABLE_SHARD_DIR

    @property
    def immutable_dir(self) -> Path:
        return self.data_dir / IMMUTABLE_SHARD_DIR

    def initialize(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.mutable_dir.mkdir(parents=True, exist_ok=True)

        self.mutable_shard = EdgeShard(str(self.mutable_dir), SHARD_CONFIG)

        if self.immutable_dir.exists():
            self.immutable_shard = EdgeShard(str(self.immutable_dir), None)

        self.upload_queue = create_persistent_queue(self.data_dir / QUEUE_DB_NAME)
        self._start_sync_worker()

    def _start_sync_worker(self):
        self.worker_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.is_running = True
        self.worker_thread.start()

    def _upload_batch(self, items: list) -> bool:
        try:
            resp = requests.post(
                f"{BACKEND_URL}/api/upsert",
                json=items,
                headers=HEADERS,
            )
            resp.raise_for_status()
            for item in items:
                self.upload_queue.ack(item)
            return True
        except requests.RequestException:
            for item in items:
                self.upload_queue.nack(item)
            return False

    def _sync_worker(self):
        while self.is_running:
            items = []
            while self.upload_queue.size > 0:
                items.append(self.upload_queue.get(block=False))

            if items:
                self._upload_batch(items)

            time.sleep(SYNC_INTERVAL)

    def force_sync(self):
        items = []
        while self.upload_queue.size > 0:
            items.append(self.upload_queue.get(block=False))
        if items:
            self._upload_batch(items)

    def stop_sync_worker(self):
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join()

    def store_image(self, image_path, embedding) -> str:
        image_id = str(uuid.uuid4())
        payload = {IMAGE_PATH_KEY: str(image_path), SYNC_TIMESTAMP_KEY: time.time()}
        vector = embedding.tolist()

        self.mutable_shard.update(
            UpdateOperation.upsert_points(
                [Point(id=image_id, vector=vector, payload=payload)]
            )
        )
        self.upload_queue.put({"id": image_id, "vector": vector, "payload": payload})
        return image_id

    def search_similar(self, query_embedding, limit: int = SEARCH_LIMIT):
        query = QueryRequest(
            prefetches=[],
            query=Mmr(
                # Using dict comprehension because "lambda" is a reserved keyword in Python
                **{
                    "vector": query_embedding.tolist(),
                    "lambda": MMR_DIVERSITY_FACTOR,
                    "candidates_limit": MMR_MAX_CANDIDATES,
                },
            ),
            limit=limit,
            with_payload=True,
        )

        results = list(self.mutable_shard.query(query))
        if self.immutable_shard:
            results.extend(self.immutable_shard.query(query))

        results.sort(key=lambda x: x.score, reverse=True)

        seen, unique = set(), []
        for r in results:
            if r.id not in seen:
                seen.add(r.id)
                unique.append(
                    {
                        "id": r.id,
                        "score": r.score,
                        IMAGE_PATH_KEY: r.payload[IMAGE_PATH_KEY],
                    }
                )
        return unique[:limit]

    def _download_snapshot(
        self, endpoint: str, target_path: Path, json_data: dict = None
    ):
        resp = requests.post(
            f"{BACKEND_URL}{endpoint}",
            json=json_data,
            headers=HEADERS,
            stream=True,
        )
        resp.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=SNAPSHOT_CHUNK_SIZE):
                f.write(chunk)

    def _cleanup_mutable_shard(self, sync_timestamp: float):
        self.mutable_shard.update(
            UpdateOperation.delete_points_by_filter(
                Filter(
                    must=[
                        FieldCondition(
                            key=SYNC_TIMESTAMP_KEY, range=RangeFloat(lte=sync_timestamp)
                        )
                    ]
                )
            )
        )

    def sync_from_server(self):
        self.stop_sync_worker()
        self.force_sync()

        if not self.immutable_shard:
            raise ValueError(
                "Baseline for partial snapshots is not set. Run a full sync first."
            )

        manifest = self.immutable_shard.snapshot_manifest()
        sync_timestamp = time.time()

        with tempfile.TemporaryDirectory(dir=self.data_dir) as temp_dir:
            snapshot_path = Path(temp_dir) / "partial.snapshot"
            self._download_snapshot(
                "/api/snapshots/partial", snapshot_path, {"manifest": manifest}
            )
            self.immutable_shard.update_from_snapshot(str(snapshot_path))

        self._cleanup_mutable_shard(sync_timestamp)
        self._start_sync_worker()

    def full_sync_from_server(self):
        self.stop_sync_worker()
        self.force_sync()

        sync_timestamp = time.time()

        with tempfile.TemporaryDirectory(dir=self.data_dir) as temp_dir:
            snapshot_path = Path(temp_dir) / "shard.snapshot"
            self._download_snapshot("/api/snapshots/full", snapshot_path)

            self.immutable_shard = None
            if self.immutable_dir.exists():
                shutil.rmtree(self.immutable_dir)
            self.immutable_dir.mkdir(parents=True, exist_ok=True)

            EdgeShard.unpack_snapshot(str(snapshot_path), str(self.immutable_dir))
            self.immutable_shard = EdgeShard(str(self.immutable_dir), None)

        self._cleanup_mutable_shard(sync_timestamp)
        self._start_sync_worker()
