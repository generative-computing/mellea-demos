FROM chromadb/chroma

# Install curl for health checks (minimal Rust-based image has no HTTP tools)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Bake pre-computed embeddings into the image.
# The tarball contains a top-level chromadb_data/ directory;
# ADD auto-extracts it, then we move it to ChromaDB's data path.
ADD chromadb_data.tar.gz /tmp/
RUN rm -rf /data && mv /tmp/chromadb_data /data
