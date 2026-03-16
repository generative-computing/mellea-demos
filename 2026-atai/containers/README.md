# atai-containers

Container images for the ATAI platform.

## Images

| Image | Description | Port |
|-------|-------------|------|
| `chromadb` | ChromaDB vector database with pre-computed embeddings | 8000 |
| `langflow-intrinsics` | Langflow with custom intrinsic components for conversational AI | 7860 |
| `langflow-vis` | FastAPI service for Langflow trace visualization | 8080 |

Source code for all images is included in this directory.

## Building

All images target the `quay.io/mellea` registry. Each subdirectory has a `build.sh` that accepts an optional custom image tag (defaults to `quay.io/mellea/<name>:latest`).

### Build all

```bash
./build_all.sh
```

### Build individually

```bash
chromadb/build.sh
langflow-intrinsics/build.sh
langflow-vis/build.sh
```

With a custom tag:

```bash
chromadb/build.sh quay.io/mellea/chromadb:v2
```

## Pushing

Log in to the registry, then push:

```bash
docker login icr.io
docker push quay.io/mellea/chromadb:latest
docker push quay.io/mellea/langflow-intrinsics:latest
docker push quay.io/mellea/langflow-vis:latest
```
