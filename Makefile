# Also update at glasses-x-edge/constants.py if changed
DATA_DIR = ./demo-data

.PHONY: setup demo clean

setup:
	@command -v uv >/dev/null || (echo "uv not installed." && exit 1)
	@echo "Installing dependencies..."
	@uv sync >/dev/null 2>&1
	@echo "Downloading CLIP models..."
	@uv run python -c "from fastembed import ImageEmbedding; ImageEmbedding(model_name='Qdrant/clip-ViT-B-32-vision')"
	@echo "Vision model ready"
	@# Also update at glasses-x-edge/constants.py if changed
	@uv run python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='Qdrant/clip-ViT-B-32-text')"
	@echo "Text model ready"
	@echo ""
	@echo "Setup complete! Run: make demo"

demo:
	@rm -rf $(DATA_DIR)
	@mkdir -p $(DATA_DIR)
	@uv run streamlit run app.py

clean:
	rm -rf $(DATA_DIR) .venv
