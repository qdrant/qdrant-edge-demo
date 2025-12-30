# Smart Glasses And Qdrant Edge

POC visual search with smart glasses and [Qdrant Edge](https://qdrant.tech/edge/).

<img width="1242" height="629" alt="Smart Glasses X Qdrant Demo" src="https://github.com/user-attachments/assets/dcece6b7-7929-4f40-96ea-0aad2cb4f0ef" />

## Usage

```bash
$ docker run -d -p 6333:6333 qdrant/qdrant
$ make setup
$ make demo
```

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Docker](https://docs.docker.com/engine/install/)
