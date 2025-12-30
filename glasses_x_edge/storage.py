import logging
import queue
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as rest_models
from qdrant_edge import (
    Distance,
    Mmr,
    PayloadStorageType,
    PlainIndexConfig,
    Point,
    QueryRequest,
    SegmentConfig,
    Shard,
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
        self.is_running = True
        self.worker_thread = None

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

        self.shard = Shard(str(self.data_dir), config)
        self._ensure_server_collection()
        self.worker_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.worker_thread.start()

    def _ensure_server_collection(self):
        if not self.server_client.collection_exists(COLLECTION_NAME):
            logger.info(f"Creating collection {COLLECTION_NAME} on server")
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
        logger.info("Starting sync worker")
        backoff = SYNC_INTERVAL

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

            try:
                self.server_client.upsert(
                    collection_name=COLLECTION_NAME, points=points_to_upload
                )
                backoff = SYNC_INTERVAL
                time.sleep(SYNC_INTERVAL)
            except Exception as e:
                logger.warning(f"Sync failed: {e}. Retrying in {backoff}s")
                for point in points_to_upload:
                    self.upload_queue.put(point)

                time.sleep(backoff)
                backoff = min(backoff * 1.5, 60)

    def stop(self):
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

    def store_image(self, image_path: Path, embedding: np.ndarray) -> str:
        image_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        vector = embedding.tolist()
        payload = {
            "image_path": str(image_path),
            "timestamp": timestamp,
        }

        point = Point(id=image_id, vector={VECTOR_NAME: vector}, payload=payload)
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
                    query_embedding.tolist(),
                    VECTOR_NAME,
                    MMR_DIVERSITY_FACTOR,
                    MMR_MAX_CANDIDATES,
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

    def restore_snapshot(self, snapshot_url: str):
        # TODO (Anush008)
        # 1. Download Snapshot
        # 2. Initialize new shard with snapshot
        # 3. Upsert images added after snapshot download was initiated into new shard
        # 4. Swap old shard with new shard
        pass
