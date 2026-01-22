from pathlib import Path

from qdrant_client.models import Distance
from qdrant_edge import Distance as EdgeDistance

# COMMONS
VECTOR_DIMENSION = 512
API_KEY_HEADER = "X-API-Key"

# SERVER CONFIG
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000
BACKEND_URL = "http://localhost:8000"
QDRANT_URL = "http://localhost:6333"
VALID_API_KEYS = ["demo-api-key"]
COLLECTION_NAME = "smart_glasses"
DISTANCE_METRIC = Distance.COSINE

# EDGE CONFIG
PROJECT_ROOT = Path(__file__).parent.absolute()
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "input.mp4"
DEFAULT_DATA_DIR = PROJECT_ROOT / "demo-data"  # NOTE: Also update Makefile if changed
IMAGES_DIR_NAME = "images"
QDRANT_STORAGE_DIR_NAME = "qdrant_storage"
MUTABLE_SHARD_DIR = "mutable"
IMMUTABLE_SHARD_DIR = "immutable"
DEFAULT_FPS = 1.0
JPEG_QUALITY = 70
DEFAULT_SIMILARITY_THRESHOLD = 0.75
VISION_MODEL_NAME = (
    "Qdrant/clip-ViT-B-32-vision"  # NOTE: Also update Makefile if changed
)
TEXT_MODEL_NAME = "Qdrant/clip-ViT-B-32-text"  # NOTE: Also update Makefile if changed
MODELS_CACHE_DIR = PROJECT_ROOT / "models"
SEARCH_LIMIT = 3
MMR_DIVERSITY_FACTOR = 0.8
MMR_MAX_CANDIDATES = 100
SYNC_INTERVAL = 5  # seconds
SNAPSHOT_CHUNK_SIZE = 8192  # bytes
DISTANCE_METRIC_EDGE = EdgeDistance.Cosine
QUEUE_DB_NAME = "upload_queue"
SYNC_TIMESTAMP_KEY = "sync_timestamp"
IMAGE_PATH_KEY = "image_path"
API_KEY = "demo-api-key"
