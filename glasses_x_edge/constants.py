from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

DEFAULT_VIDEO_PATH = PROJECT_ROOT / "input.mp4"
# NOTE: Also update Makefile if changed
DEFAULT_DATA_DIR = PROJECT_ROOT / "demo-data"
IMAGES_DIR_NAME = "images"

DEFAULT_FPS = 1.0
JPEG_QUALITY = 70
DEFAULT_SIMILARITY_THRESHOLD = 0.75

# NOTE: Also update Makefile if changed
VISION_MODEL_NAME = "Qdrant/clip-ViT-B-32-vision"
TEXT_MODEL_NAME = "Qdrant/clip-ViT-B-32-text"
VECTOR_DIMENSION = 512

VECTOR_NAME = "vision"
SEARCH_LIMIT = 3
MMR_DIVERSITY_FACTOR = 0.8
MMR_MAX_CANDIDATES = 100
SERVER_URL = "http://localhost:6333"
COLLECTION_NAME = "smart_glasses"
SYNC_INTERVAL = 5  # seconds
