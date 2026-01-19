import sys
import threading
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from glasses_x_edge.capture import VideoCapture
from glasses_x_edge.constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_FPS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_VIDEO_PATH,
    IMAGES_DIR_NAME,
    QDRANT_STORAGE_DIR_NAME,
    SEARCH_LIMIT,
)
from glasses_x_edge.embedding import CrossModalEncoder
from glasses_x_edge.storage import VisionStorage

HIDDEN_CONTROLS_CSS = """
<style>
    video { pointer-events: none; }
</style>
"""


class SystemState:
    def __init__(self):
        self.is_running = True
        self.images_dir = Path(DEFAULT_DATA_DIR) / IMAGES_DIR_NAME
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.storage = VisionStorage(Path(DEFAULT_DATA_DIR) / QDRANT_STORAGE_DIR_NAME)
        self.storage.initialize()

        self.encoder = CrossModalEncoder()
        self.encoder.load_models()

        threading.Thread(target=self.index_video_background, daemon=True).start()

    def index_video_background(self):
        cap = VideoCapture(str(DEFAULT_VIDEO_PATH), fps=DEFAULT_FPS)
        last_frame = None

        with cap:
            for frame in cap.capture_continuous():
                if not self.is_running:
                    break

                # Skip similar frames to save space
                if last_frame is not None:
                    if (
                        VideoCapture.calculate_similarity(frame, last_frame)
                        > DEFAULT_SIMILARITY_THRESHOLD
                    ):
                        continue

                last_frame = frame
                timestamp = int(time.time() * 1000)
                image_path = self.images_dir / f"frame_{timestamp}.jpg"

                cap.save_frame(frame, image_path)
                embedding = self.encoder.encode_image(image_path)
                self.storage.store_image(image_path, embedding)


@st.cache_resource
def get_system_state():
    return SystemState()


@st.fragment
def render_search_interface(system):
    st.header("Smart 🕶️ X Qdrant Edge")

    query = st.text_input("Search for what you saw")

    if query:
        text_embedding = system.encoder.encode_text(query)
        results = system.storage.search_similar(text_embedding, limit=SEARCH_LIMIT)

        if not results:
            st.info("No matching frames found.")
            return

        for result in results:
            image_path = Path(result["image_path"])
            st.image(
                Image.open(image_path),
                width="stretch",
            )


@st.fragment(run_every=2)
def render_sync_status(storage):
    q_size = storage.upload_queue.qsize()
    st.metric("Upload Queue", q_size)


@st.fragment
def render_snapshot_restore(storage):
    if st.button("🔄 Sync From Server"):
        try:
            with st.spinner("Restoring..."):
                storage.restore_snapshot()
            st.success("Restored!")
        except Exception as e:
            st.error(f"Error: {e}")
    st.caption(
        "## What is this?\n"
        "Initially, the glasses store vectors unindexed to save CPU.\n\n"
        "The server builds the HNSW index for fast search.\n\n"
        "Syncing downloads the indexed snapshot from the server.\n\n"
        "**Zero Downtime**: New images captured during this restore are buffered "
        "in memory and added to the swapped storage. "
    )


def main():
    st.set_page_config(page_title="Qdrant Edge Demo", page_icon="👓", layout="wide")
    st.markdown(HIDDEN_CONTROLS_CSS, unsafe_allow_html=True)

    system = get_system_state()

    with st.sidebar:
        st.header("Server Sync Status")
        render_sync_status(system.storage)

        st.header("Sync Index")
        render_snapshot_restore(system.storage)

    col_left, col_right = st.columns([1, 1])

    with col_right:
        st.subheader("You are seeing this")
        if Path(DEFAULT_VIDEO_PATH).exists():
            st.video(str(DEFAULT_VIDEO_PATH), autoplay=True, muted=True)
            st.info("This video is being indexed in real-time.")
        else:
            st.error(f"Video file not found: {DEFAULT_VIDEO_PATH}")

    with col_left:
        render_search_interface(system)


if __name__ == "__main__":
    import sys

    from streamlit.web import cli as stcli

    if st.runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
