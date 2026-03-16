# chromadb

ChromaDB vector database with pre-computed embeddings baked into the image.

## Building

```bash
./build.sh                              # default tag: quay.io/mellea/chromadb:latest
./build.sh quay.io/mellea/chromadb:v2      # custom tag
```

## Details

- **Base image:** `chromadb/chroma`
- **Port:** 8000
- Adds `curl` for health checks
- Extracts `chromadb_data.tar.gz` into `/data`
