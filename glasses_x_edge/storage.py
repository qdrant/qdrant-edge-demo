import json
import queue
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from qdrant_client import QdrantClient, models
from qdrant_edge import (
    Distance,
    FieldCondition,
    Filter,
    Mmr,
    PayloadStorageType,
    PlainIndexConfig,
    Point,
    QueryRequest,
    RangeFloat,
    SegmentConfig,
    Shard,
    UpdateOperation,
    VectorDataConfig,
    VectorStorageType,
)

from .constants import (
    COLLECTION_NAME,
    IMAGE_PATH_KEY,
    IMMUTABLE_SHARD_DIR,
    MANIFEST_FILE_NAME,
    MMR_DIVERSITY_FACTOR,
    MMR_MAX_CANDIDATES,
    MUTABLE_SHARD_DIR,
    SEARCH_LIMIT,
    SERVER_URL,
    SNAPSHOT_CHUNK_SIZE,
    SYNC_INTERVAL,
    SYNC_TIMESTAMP_KEY,
    VECTOR_DIMENSION,
    VECTOR_NAME,
)


class VisionStorage:
    def __init__(self, data_dir: Path, vector_dim: int = VECTOR_DIMENSION):
        self.data_dir = data_dir
        self.vector_dim = vector_dim
        self.mutable_shard = None
        self.immutable_shard = None
        self.server_client = QdrantClient(url=SERVER_URL)
        self.upload_queue = queue.Queue()
        self.worker_thread = None
        self.is_running = False

    @property
    def mutable_dir(self) -> Path:
        return self.data_dir / MUTABLE_SHARD_DIR

    @property
    def immutable_dir(self) -> Path:
        return self.data_dir / IMMUTABLE_SHARD_DIR

    def _create_shard_config(self) -> SegmentConfig:
        return SegmentConfig(
            vector_data={
                VECTOR_NAME: VectorDataConfig(
                    size=self.vector_dim,
                    distance=Distance.Cosine,
                    storage_type=VectorStorageType.ChunkedMmap,
                    index=PlainIndexConfig(),
                    quantization_config=None,
                    multivector_config=None,
                    datatype=None,
                ),
            },
            sparse_vector_data={},
            payload_storage_type=PayloadStorageType.InRamMmap,
        )

    def initialize(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.mutable_dir.mkdir(parents=True, exist_ok=True)

        config = self._create_shard_config()
        self.mutable_shard = Shard(str(self.mutable_dir), config)

        if self.immutable_dir.exists():
            self.immutable_shard = Shard(str(self.immutable_dir), None)

        self._ensure_server_collection()
        self.worker_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.is_running = True
        self.worker_thread.start()

    def _ensure_server_collection(self):
        if self.server_client.collection_exists(COLLECTION_NAME):
            self.server_client.delete_collection(COLLECTION_NAME)

        self.server_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                VECTOR_NAME: models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.COSINE,
                )
            },
        )

    def _sync_worker(self):
        while self.is_running:
            points_to_upload = []

            while len(points_to_upload) < 10:
                try:
                    points_to_upload.append(self.upload_queue.get_nowait())
                except queue.Empty:
                    break

            if not points_to_upload:
                time.sleep(SYNC_INTERVAL)
                continue

            self.server_client.upsert(
                collection_name=COLLECTION_NAME, points=points_to_upload
            )
            time.sleep(SYNC_INTERVAL)

    def force_sync(self):
        points = []
        while not self.upload_queue.empty():
            try:
                points.append(self.upload_queue.get_nowait())
            except queue.Empty:
                break

        if points:
            self.server_client.upsert(
                collection_name=COLLECTION_NAME, points=points, wait=True
            )

    def stop_server_sync_worker(self):
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join()

    def store_image(self, image_path, embedding) -> str:
        image_id = str(uuid.uuid4())
        sync_timestamp = time.time()
        vector = embedding.tolist()
        payload = {
            IMAGE_PATH_KEY: str(image_path),
            SYNC_TIMESTAMP_KEY: sync_timestamp,
        }

        point = Point(id=image_id, vector={VECTOR_NAME: vector}, payload=payload)
        self.mutable_shard.update(UpdateOperation.upsert_points([point]))

        rest_point = models.PointStruct(
            id=image_id, vector={VECTOR_NAME: vector}, payload=payload
        )
        self.upload_queue.put(rest_point)

        return image_id

    def search_similar(self, query_embedding, limit: int = SEARCH_LIMIT):
        query_request = QueryRequest(
            prefetches=[],
            query=Mmr(
                **{
                    "vector": query_embedding.tolist(),
                    "using": VECTOR_NAME,
                    "lambda": MMR_DIVERSITY_FACTOR,
                    "candidates_limit": MMR_MAX_CANDIDATES,
                }
            ),
            filter=None,
            score_threshold=None,
            limit=limit,
            offset=0,
            params=None,
            with_vector=False,
            with_payload=True,
        )

        mutable_results = self.mutable_shard.query(query_request)

        immutable_results = []
        if self.immutable_shard is not None:
            immutable_results = self.immutable_shard.query(query_request)

        all_results = list(mutable_results) + list(immutable_results)
        all_results.sort(key=lambda x: x.score, reverse=True)

        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)

        return [
            {
                "id": result.id,
                "score": result.score,
                IMAGE_PATH_KEY: result.payload[IMAGE_PATH_KEY],
            }
            for result in unique_results[:limit]
        ]

    def _get_manifest_path(self) -> Path:
        return self.data_dir / MANIFEST_FILE_NAME

    def _load_local_manifest(self) -> Optional[dict]:
        manifest_path = self._get_manifest_path()
        if not manifest_path.exists():
            return None
        with open(manifest_path, "r") as f:
            return json.load(f)

    def _save_local_manifest(self):
        if self.immutable_shard is None:
            return
        manifest = self.immutable_shard.snapshot_manifest()
        manifest_path = self._get_manifest_path()
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _download_partial_snapshot(
        self, manifest: dict, shard_id: int, target_path: Path
    ):
        url = f"{SERVER_URL}/collections/{COLLECTION_NAME}/shards/{shard_id}/snapshot/partial/create"
        response = requests.post(url, json=manifest, stream=True)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=SNAPSHOT_CHUNK_SIZE):
                f.write(chunk)

    def _create_server_snapshot(self, shard_id: int):
        snap_url = (
            f"{SERVER_URL}/collections/{COLLECTION_NAME}/shards/{shard_id}/snapshots"
        )
        resp = requests.post(snap_url)
        resp.raise_for_status()

        result = resp.json().get("result")
        if not result or not result.get("name"):
            raise ValueError("Failed to get snapshot name from response")

        return result["name"], f"{snap_url}/{result['name']}"

    def _download_snapshot(self, url: str, target_path: Path):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(target_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=SNAPSHOT_CHUNK_SIZE):
                    f.write(chunk)

    def _restart_sync_worker(self):
        if not self.is_running:
            self.worker_thread = threading.Thread(target=self._sync_worker, daemon=True)
            self.is_running = True
            self.worker_thread.start()

    def _cleanup_mutable_shard(self, sync_timestamp: float):
        cleanup_filter = Filter(
            must=[
                FieldCondition(
                    key=SYNC_TIMESTAMP_KEY, range=RangeFloat(lte=sync_timestamp)
                )
            ]
        )
        self.mutable_shard.update(
            UpdateOperation.delete_points_by_filter(cleanup_filter)
        )

    def sync_from_server(self, shard_id: int = 0):
        self.stop_server_sync_worker()
        self.force_sync()

        local_manifest = self._load_local_manifest()
        if local_manifest is None:
            raise ValueError("No local manifest found. Run a full sync first.")
        if self.immutable_shard is None:
            raise ValueError("No immutable shard. Run a full sync first.")

        sync_timestamp = time.time()

        with tempfile.TemporaryDirectory(dir=self.data_dir) as temp_dir:
            partial_snapshot_path = Path(temp_dir) / "partial.snapshot"
            self._download_partial_snapshot(
                local_manifest, shard_id, partial_snapshot_path
            )

            self.immutable_shard.update_from_snapshot(str(partial_snapshot_path))
            self._save_local_manifest()

        self._cleanup_mutable_shard(sync_timestamp)
        self._restart_sync_worker()

    def full_sync_from_server(self, shard_id: int = 0):
        self.stop_server_sync_worker()
        self.force_sync()

        sync_timestamp = time.time()

        _, snapshot_url = self._create_server_snapshot(shard_id)

        with tempfile.TemporaryDirectory(dir=self.data_dir) as restore_dir:
            snapshot_path = Path(restore_dir) / "shard.snapshot"
            self._download_snapshot(snapshot_url, snapshot_path)

            self.immutable_shard = None
            if self.immutable_dir.exists():
                shutil.rmtree(self.immutable_dir)
            self.immutable_dir.mkdir(parents=True, exist_ok=True)

            Shard.unpack_snapshot(str(snapshot_path), str(self.immutable_dir))
            self.immutable_shard = Shard(str(self.immutable_dir), None)
            self._save_local_manifest()

        self._cleanup_mutable_shard(sync_timestamp)
        self._restart_sync_worker()
