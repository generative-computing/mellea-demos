"""
Local Embeddings Component for Langflow

Runs sentence-transformers models locally using PyTorch.
Produces embeddings compatible with LangChain's Embeddings interface.
"""

from langflow.custom import Component
from langflow.io import Output, StrInput


class LocalEmbeddingsComponent(Component):
    display_name = "Local Embeddings"
    description = "Generate embeddings locally using sentence-transformers models"
    icon = "binary"

    inputs = [
        StrInput(
            name="model_name",
            display_name="Model Name",
            value="sentence-transformers/multi-qa-mpnet-base-dot-v1",
            info="HuggingFace model name. Downloaded once and cached locally.",
        ),
    ]

    outputs = [
        Output(
            display_name="Embeddings",
            name="embeddings",
            method="build_embeddings",
            types=["Embeddings"],
        ),
    ]

    def build_embeddings(self):
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.status = f"Loaded {self.model_name}"
        return embeddings
