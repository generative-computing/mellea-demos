"""
ELSER Retriever Component for Langflow

Retrieves documents from Elasticsearch using ELSER semantic search.
Takes chat input and returns relevant documents.
"""

from langflow.custom import Component
from langflow.io import (
    StrInput,
    Output,
    IntInput,
    MessageInput,
    DataInput,
    DropdownInput,
)
from langflow.schema import Message, Data
import json


class ElserRetriever(Component):
    """
    ELSER Retriever Component for Langflow.

    Retrieves documents from Elasticsearch using ELSER semantic search.
    Takes chat messages as input and returns relevant documents.
    """

    display_name = "ELSER Retriever"
    description = "Retrieve documents from Elasticsearch using ELSER semantic search"
    category = "retrieval"
    icon = "database"

    # ========== INPUTS ==========
    inputs = [
        MessageInput(
            name="chat_input",
            display_name="Chat Input",
            info="Chat message to use as a single query",
            required=False,
        ),
        DropdownInput(
            name="corpus_name",
            display_name="Corpus Name",
            info="Elasticsearch index to search",
            options=[
                "md2d-es-elser-elser-index",
                "mt-rag-banking-elser-512-100-20250205",
                "mt-rag-clapnq-elser-512-100-20240503",
                "mt-rag-fiqa-beir-elser-512-100-20240501",
                "mt-rag-govt-elser-512-100-20240611",
                "mt-rag-ibmcloud-elser-512-100-20240502",
                "mt-rag-scifact-beir-elser-512-100-20240501",
                "mt-rag-telco-elser-512-100-20241210",
                "quac-es-elser-elser-index",
            ],
            value="mt-rag-clapnq-elser-512-100-20240503",
        ),
        DataInput(
            name="query_list",
            display_name="Query List",
            info="Optional list of queries as Data (e.g. from Query Expansion). Each query is searched and results are merged and deduplicated.",
            required=False,
        ),
        IntInput(
            name="top_k",
            display_name="Top K Results",
            info="Number of documents to retrieve per query",
            value=5,
        ),
    ]

    # ========== OUTPUTS ==========
    outputs = [
        Output(
            name="documents",
            display_name="Documents",
            method="retrieve_documents",
        ),
    ]

    # ========== HELPER METHODS ==========
    def _extract_message_text(self, message: Message) -> str:
        """Extract text content from a Message object."""
        if hasattr(message, 'text'):
            return message.text
        if hasattr(message, 'content'):
            return message.content
        return str(message)

    def _create_es_body(self, limit: int, query: str) -> dict:
        """
        Create Elasticsearch query body for ELSER semantic search.

        :param limit: Max number of documents to retrieve
        :param query: Query string for retrieving documents
        """
        body = {
            "size": limit,
            "query": {
                "bool": {
                    "must": {
                        "text_expansion": {
                            "ml.tokens": {
                                "model_id": ".elser_model_1",
                                "model_text": query,
                            }
                        }
                    }
                }
            },
        }
        return body

    def _retrieve_from_elasticsearch(self, query: str) -> list[dict]:
        """
        Retrieve documents from Elasticsearch using ELSER.

        :param query: Query string
        :return: List of documents with doc_id, text, and score
        """
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            raise ValueError(
                "elasticsearch package is required. Install it with: pip install elasticsearch"
            )

        try:
            # Elasticsearch host from mellea configuration
            elasticsearch_host = "https://ibm_cloud_4ae4bca5_f6aa_43b6_93a3_befbd8fcb0e7:7d325be7af5de8c018b3284d754eb264995a56e4dfeba63fdeb6db1ff37dbd19@dbcc936c-8274-450e-9cb1-44a30ec26d88.c13paqsd05a0ept695ng.databases.appdomain.cloud:32765"

            # Create Elasticsearch client
            es = Elasticsearch(
                hosts=[elasticsearch_host],
                verify_certs=False,
                ssl_show_warn=False
            )

            # Build query body
            body = self._create_es_body(self.top_k, query)

            # Execute search
            retriever_results = es.search(
                index=self.corpus_name,
                body=body,
            )

            hits = retriever_results["hits"]["hits"]

            # Format results
            documents = []
            for hit in hits:
                document = {
                    "doc_id": hit["_id"],
                    "text": hit["_source"]["text"],
                    "score": str(hit["_score"]),
                }
                documents.append(document)

            return documents

        except Exception as e:
            raise ValueError(f"Elasticsearch retrieval failed: {e}")

    def _get_queries(self) -> list[str]:
        """Get list of queries from query_list Data input and/or chat_input."""
        queries = []
        if hasattr(self, 'query_list') and self.query_list:
            if isinstance(self.query_list, Data):
                data_dict = self.query_list.data if hasattr(self.query_list, 'data') else self.query_list
                if isinstance(data_dict, dict):
                    query_items = data_dict.get("queries", [])
                    if isinstance(query_items, list):
                        queries = [str(q).strip() for q in query_items if str(q).strip()]
            elif isinstance(self.query_list, dict):
                query_items = self.query_list.get("queries", [])
                if isinstance(query_items, list):
                    queries = [str(q).strip() for q in query_items if str(q).strip()]
        if hasattr(self, 'chat_input') and self.chat_input:
            query = self._extract_message_text(self.chat_input)
            if query and query.strip():
                queries.append(query.strip())
        return queries

    def _merge_and_deduplicate(self, all_documents: list[list[dict]]) -> list[dict]:
        """Merge documents from multiple queries and deduplicate by doc_id, keeping highest score."""
        seen = {}
        for docs in all_documents:
            for doc in docs:
                doc_id = doc["doc_id"]
                if doc_id not in seen or float(doc["score"]) > float(seen[doc_id]["score"]):
                    seen[doc_id] = doc
        return sorted(seen.values(), key=lambda d: float(d["score"]), reverse=True)

    # ========== OUTPUT METHODS ==========
    def retrieve_documents(self) -> Data:
        """Retrieve documents and return as Data."""
        queries = self._get_queries()
        if not queries:
            raise ValueError("No queries provided")

        all_documents = []
        for query in queries:
            docs = self._retrieve_from_elasticsearch(query)
            all_documents.append(docs)

        if len(all_documents) == 1:
            documents = all_documents[0]
        else:
            documents = self._merge_and_deduplicate(all_documents)

        return Data(data={"documents": documents})
