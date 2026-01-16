import logging
import queue
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from qdrant_client import QdrantClient
from qdrant_client import models as rest_models
from qdrant_edge import (
    Distance,
    EdgeShard,
    Mmr,
    PayloadStorageType,
    PlainIndexConfig,
    Point,
    QueryRequest,
    SegmentConfig,
    UpdateOperation,
    VectorDataConfig,
    VectorStorageType,
)

from .constants import (
    COLLECTION_NAME,
    MMR_DIVERSITY_FACTOR,
    MMR_MAX_CANDIDATES,
    SEARCH_LIMIT,
    SERVER_URL,
    SNAPSHOT_CHUNK_SIZE,
    SYNC_INTERVAL,
    VECTOR_DIMENSION,
    VECTOR_NAME,
)

logger = logging.getLogger(__name__)


class VisionStorage:
    def __init__(self, data_dir: Path, vector_dim: int = VECTOR_DIMENSION):
        self.data_dir = data_dir
        self.vector_dim = vector_dim
        self.shard = None
        self.server_client = QdrantClient(url=SERVER_URL)
        self.upload_queue = queue.Queue()
        self.worker_thread = None
        self.is_running = False

        self._is_restoring = False
        self._restore_buffer = []
        self._restore_lock = threading.Lock()

    def initialize(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)

        config = SegmentConfig(
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
        self.config = config
        self.shard = EdgeShard(str(self.data_dir), self.config)
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
                VECTOR_NAME: rest_models.VectorParams(
                    size=self.vector_dim,
                    distance=rest_models.Distance.COSINE,
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
        timestamp = datetime.now().isoformat()
        vector = embedding.tolist()
        payload = {
            "image_path": str(image_path),
            "timestamp": timestamp,
        }

        point = Point(id=image_id, vector={VECTOR_NAME: vector}, payload=payload)

        with self._restore_lock:
            if self._is_restoring:
                self._restore_buffer.append(point)
            else:
                self.shard.update(UpdateOperation.upsert_points([point]))

        rest_point = rest_models.PointStruct(
            id=image_id, vector={VECTOR_NAME: vector}, payload=payload
        )
        self.upload_queue.put(rest_point)

        return image_id

    def search_similar(self, query_embedding, limit: int = SEARCH_LIMIT):
        results = self.shard.query(
            QueryRequest(
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
        )

        return [
            {
                "id": result.id,
                "score": result.score,
                "image_path": result.payload["image_path"],
                "timestamp": result.payload["timestamp"],
            }
            for result in results
        ]

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

    def restore_snapshot(self, shard_id: int = 0):
        self.stop_server_sync_worker()
        self.force_sync()

        _, snapshot_url = self._create_server_snapshot(shard_id)

        with self._restore_lock:
            self._is_restoring = True

        with tempfile.TemporaryDirectory(dir=self.data_dir.parent) as restore_dir:
            snapshot_path = Path(restore_dir) / "shard.snapshot"

            self._download_snapshot(snapshot_url, snapshot_path)

            self.shard = None
            if self.data_dir.exists():
                shutil.rmtree(self.data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)

            EdgeShard.unpack_snapshot(str(snapshot_path), str(self.data_dir))
            self.shard = EdgeShard(str(self.data_dir), self.config)

            with self._restore_lock:
                points_to_restore = list(self._restore_buffer)
                self._restore_buffer = []
                self._is_restoring = False
            if points_to_restore:
                self.shard.update(UpdateOperation.upsert_points(points_to_restore))

        if not self.is_running:
            self.worker_thread = threading.Thread(target=self._sync_worker, daemon=True)
            self.is_running = True
            self.worker_thread.start()
