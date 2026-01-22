from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from starlette.middleware.base import BaseHTTPMiddleware

from config import (
    API_KEY_HEADER,
    BACKEND_HOST,
    BACKEND_PORT,
    COLLECTION_NAME,
    DISTANCE_METRIC,
    QDRANT_URL,
    VALID_API_KEYS,
    VECTOR_DIMENSION,
)


class Point(BaseModel):
    id: str
    vector: list[float]
    payload: dict[str, Any]


class SnapshotManifest(BaseModel):
    manifest: dict[str, Any]


qdrant = QdrantClient(url=QDRANT_URL)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get(API_KEY_HEADER)
        if not api_key or api_key not in VALID_API_KEYS:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_DIMENSION, distance=DISTANCE_METRIC
        ),
    )
    yield


app = FastAPI(title="Qdrant Edge Demo Backend", lifespan=lifespan)
app.add_middleware(AuthMiddleware)
api = APIRouter(prefix="/api")


@api.post("/upsert")
async def upsert_points(points: list[Point]):
    qdrant_points = [
        models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=qdrant_points, wait=True)
    return {"status": "ok", "count": len(qdrant_points)}


@api.post("/snapshots/full")
async def create_full_snapshot(shard_id: int = 0):
    snap_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/shards/{shard_id}/snapshots"

    async with httpx.AsyncClient() as client:
        resp = await client.post(snap_url)
        resp.raise_for_status()
        result = resp.json().get("result", {})
        snapshot_name = result.get("name")
        if not snapshot_name:
            raise HTTPException(500, "Failed to create snapshot")

    async def stream():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", f"{snap_url}/{snapshot_name}") as r:
                async for chunk in r.aiter_bytes(8192):
                    yield chunk

    return StreamingResponse(stream(), media_type="application/octet-stream")


@api.post("/snapshots/partial")
async def create_partial_snapshot(request: SnapshotManifest, shard_id: int = 0):
    url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/shards/{shard_id}/snapshot/partial/create"

    async def stream():
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=request.manifest) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes(8192):
                    yield chunk

    return StreamingResponse(stream(), media_type="application/octet-stream")


app.include_router(api)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT)
