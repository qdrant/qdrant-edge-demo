import logging
from pathlib import Path

from fastembed import ImageEmbedding
from PIL import Image

from .constants import MODELS_CACHE_DIR, TEXT_MODEL_NAME, VISION_MODEL_NAME

logger = logging.getLogger(__name__)


class ImageEncoder:
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is None:
            logger.info(f"Loading image embedding model: {VISION_MODEL_NAME}")
            self.model = ImageEmbedding(
                model_name=VISION_MODEL_NAME, cache_dir=str(MODELS_CACHE_DIR)
            )

    def encode_image(self, image):
        if self.model is None:
            self.load_model()

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        embeddings = list(self.model.embed([image]))
        return embeddings[0]


class TextEncoder:
    def __init__(self):
        self.model = None

    def load_model(self):
        if self.model is None:
            from fastembed import TextEmbedding

            logger.info(f"Loading text embedding model: {TEXT_MODEL_NAME}")
            self.model = TextEmbedding(
                model_name=TEXT_MODEL_NAME, cache_dir=str(MODELS_CACHE_DIR)
            )

    def encode_text(self, text: str):
        if self.model is None:
            self.load_model()

        embeddings = list(self.model.embed([text]))
        return embeddings[0]


class CrossModalEncoder:
    def __init__(self):
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

    def encode_image(self, image):
        return self.image_encoder.encode_image(image)

    def encode_text(self, text: str):
        return self.text_encoder.encode_text(text)

    def load_models(self):
        self.image_encoder.load_model()
        self.text_encoder.load_model()
