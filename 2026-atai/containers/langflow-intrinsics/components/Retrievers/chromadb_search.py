"""
ChromaDB Search Component for Langflow

Connects to a ChromaDB server, lists available collections,
and performs similarity search with precomputed embeddings.
"""

from langflow.custom import Component
from langflow.io import (
    Output,
    StrInput,
    IntInput,
    DropdownInput,
    HandleInput,
    MessageInput,
)
from langflow.schema import Data


class ChromaDBSearch(Component):
    display_name = "ChromaDB Search"
    description = "Search a ChromaDB server collection with precomputed embeddings"
    icon = "Chroma"

    inputs = [
        StrInput(
            name="host",
            display_name="Server Host",
            value="localhost",
            info="ChromaDB server hostname",
            real_time_refresh=True,
        ),
        IntInput(
            name="port",
            display_name="Server Port",
            value=8100,
            info="ChromaDB server HTTP port",
            real_time_refresh=True,
        ),
        DropdownInput(
            name="collection_name",
            display_name="Collection",
            options=[],
            value="",
            info="ChromaDB collection to search",
            refresh_button=True,
            real_time_refresh=True,
        ),
        MessageInput(
            name="search_query",
            display_name="Search Query",
            info="Text query for similarity search",
        ),
        HandleInput(
            name="embedding",
            display_name="Embedding",
            input_types=["Embeddings"],
            info="Embedding model for query vectorization",
        ),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            value=5,
            info="Number of documents to return",
        ),
        DropdownInput(
            name="search_type",
            display_name="Search Type",
            options=["Similarity", "MMR"],
            value="Similarity",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Search Results",
            name="search_results",
            method="search_documents",
        ),
    ]

    def _fetch_collection_names(self, host, port):
        import chromadb
        client = chromadb.HttpClient(host=host, port=int(port))
        return sorted(c.name for c in client.list_collections())

    async def update_build_config(self, build_config, field_value=None, field_name=None):
        try:
            host = build_config["host"]["value"]
            port = build_config["port"]["value"]
            names = self._fetch_collection_names(host, port)
            build_config["collection_name"]["options"] = names
            current = build_config["collection_name"]["value"]
            if not current or current not in names:
                build_config["collection_name"]["value"] = names[0] if names else ""
        except Exception:
            build_config["collection_name"]["options"] = []
            build_config["collection_name"]["value"] = ""
        return build_config

    def _extract_query_text(self) -> str:
        if self.search_query is None:
            return ""
        if hasattr(self.search_query, 'text'):
            return self.search_query.text
        if hasattr(self.search_query, 'content'):
            return self.search_query.content
        return str(self.search_query)

    def search_documents(self) -> list[Data]:
        from langchain_chroma import Chroma
        import chromadb

        query = self._extract_query_text()
        if not query or not query.strip():
            self.status = "No search query provided"
            return []

        client = chromadb.HttpClient(host=self.host, port=int(self.port))

        vector_store = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embedding,
        )

        self.log(f"Query: {query}", name="search input")
        self.log(f"Collection: {self.collection_name}, Results: {self.number_of_results}", name="search params")

        docs = vector_store.search(
            query=query,
            search_type=self.search_type.lower(),
            k=self.number_of_results,
        )

        results = []
        for doc in docs:
            data = Data(
                text=doc.page_content,
                data={
                    "text": doc.page_content,
                    **doc.metadata,
                },
            )
            results.append(data)

        self.log(f"Returned {len(results)} results", name="search output")
        self.status = results
        return results
