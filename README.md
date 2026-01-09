# Smart Glasses x Qdrant Edge

This is a proof-of-concept for smart glasses that remember what they see and help you find your keys even without an internet connection.

Powered by [Qdrant Edge](https://qdrant.tech/edge/).

<img width="1265" height="646" alt="532390022-2a521bc4-642f-497b-952f-772d8ebf1e45" src="https://github.com/user-attachments/assets/e913e142-3a66-4ab7-bed5-e11fc6e68c3a" />


## How?

<img width="1245" height="1264" alt="Architecture" src="https://github.com/user-attachments/assets/3eb92a1b-b3d5-493f-9580-53a3c8f66171" />


The system has two main parts: the glasses with Qdrant Edge and the Qdrant server.

### On the device:

The glasses use a [CLIP model](https://huggingface.co/Qdrant/clip-ViT-B-32-vision) to turn video frames into vectors. We also compare frame similarity to skip redundant frames. We save these directly to a local Qdrant Edge Shard. Unlike the usual Qdrant client, we run the storage engine inside our Python process.

### The Sync & Index:

Indexing (building the HNSW graph) is heavy on the CPU, so we don't do it on the glasses. Instead, we send the vectors to a server too. The server builds the index, creates a snapshot, and the glasses download that snapshot later.

We built a "zero-downtime" restore mechanism for this. When the glasses download a new snapshot, they buffer any new incoming frames in memory. Once the new snapshot is swapped in, those buffered frames are upserted.

## Try it out

You'll need [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and [Docker](https://docs.docker.com/desktop/setup/install/mac-install/) installed.

1. Start the server (handles the vector indexing):

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

2. Install dependencies and download the CLIP models:

```bash
make setup
```

3. Run the demo:

```bash
make demo
```
