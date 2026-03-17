"""
Answerability Component for Langflow

Predicts whether a question can be answered from the provided documents.
"""

from langflow.custom import Component
from langflow.io import (
    SecretStrInput,
    StrInput,
    Output,
    DropdownInput,
    MessageInput,
    HandleInput,
    DataFrameInput,
    SliderInput,
)
from langflow.schema import Message, Data
from langflow.field_typing.range_spec import RangeSpec
import httpx
import json

from openai import OpenAI

# Global backend cache - stores initialized Mellea backends per model
_mellea_backend_cache = {}

# Model name mapping - maps user-facing names to actual model IDs
_MODEL_NAME_MAPPING = {
    "granite-4.0-micro": "ibm-granite/granite-4.0-micro",
    "ibm-granite/granite-4.0-micro": "ibm-granite/granite-4.0-micro",
    "granite4:micro": "ibm-granite/granite-4.0-micro",
}

_DEFAULT_MODEL = "ibm-granite/granite-4.0-micro"


class Answerability(Component):
    display_name = "Answerability"
    description = "Predict whether a question can be answered from the provided documents"
    category = "tools"
    icon = "message-square"

    intrinsic_name = "answerability"

    BACKEND_CONFIG = {
        "LocalHF": {
            "models": ["ibm-granite/granite-4.0-micro"],
            "default_model": "ibm-granite/granite-4.0-micro",
            "default_url": "local",
            "url_editable": False,
        },
        "vLLM": {
            "models": ["granite-4.0-micro"],
            "default_model": "granite-4.0-micro",
            "default_url": "http://localhost:8000",
            "url_editable": True,
        },
        "Ollama": {
            "models": ["granite4:micro"],
            "default_model": "granite4:micro",
            "default_url": "http://localhost:11434",
            "url_editable": True,
        },
        "IntrinsicsAPI (mellea + RITS)": {
            "models": ["granite-4.0-micro", "gpt-oss-20b"],
            "default_model": "granite-4.0-micro",
            "default_url": "https://intrinsics-api.intrinsics.vpc-int.res.ibm.com",
            "url_editable": False,
        },
    }

    CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"

    inputs = [
        DropdownInput(
            name="model_backend",
            display_name="Model Backend",
            info="Backend service for Granite model",
            options=["IntrinsicsAPI (mellea + RITS)", "vLLM", "Ollama", "LocalHF"],
            value="IntrinsicsAPI (mellea + RITS)",
            real_time_refresh=True,
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            info="Model name (options change based on backend)",
            options=["granite-4.0-micro", "gpt-oss-20b"],
            value="granite-4.0-micro",
        ),
        StrInput(
            name="url",
            display_name="API URL",
            info="Base URL for the model backend (e.g., http://localhost:8000)",
            value="https://intrinsics-api.intrinsics.vpc-int.res.ibm.com",
            required=True,
            show=False,
        ),
        SecretStrInput(
            name="api_key",
            display_name="RITS API Key",
            info="API key for authentication",
            required=False,
            load_from_db=False,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            info="Controls randomness in responses",
            value=0.0,
            range_spec=RangeSpec(min=0, max=2, step=0.01),
            advanced=True,
        ),
        MessageInput(
            name="chat_input",
            display_name="Chat Input",
            info="User message to process",
            required=True,
        ),
        DataFrameInput(
            name="message_history",
            display_name="Message History",
            info="Previous conversation messages as a DataFrame with 'role' (user/assistant) and 'content' columns",
            required=False,
        ),
        HandleInput(
            name="documents",
            display_name="Documents",
            info="Documents to use with the intrinsic operation (accepts Data from vector stores)",
            input_types=["Data", "Search Results"],
            required=False,
            is_list=True,
        ),
    ]

    outputs = [
        Output(
            name="answerability_label",
            display_name="Answerability Label",
            method="get_answerability_label",
        ),
        Output(
            name="response",
            display_name="Response",
            method="get_response",
        ),
        Output(
            name="full_response",
            display_name="Full Response",
            method="get_full_response",
        ),
    ]

    def _extract_message_text(self, message: Message) -> str:
        if hasattr(message, 'text'):
            return message.text
        if hasattr(message, 'content'):
            return message.content
        return str(message)

    def _build_chat_messages(self) -> list:
        messages = []
        if hasattr(self, 'message_history') and self.message_history is not None:
            df = self.message_history
            for _, row in df.iterrows():
                sender = row.get("sender", "")
                role = "user" if sender == "User" else "assistant"
                content = row.get("text", "")
                if content and str(content).strip():
                    messages.append({"role": role, "content": str(content)})
        user_message = self._extract_message_text(self.chat_input)
        if user_message and user_message.strip():
            messages.append({"role": "user", "content": user_message})
        return messages

    def _build_request(self) -> dict:
        messages = self._build_chat_messages()
        if not messages:
            raise ValueError("No messages to send")
        request_body = {
            "messages": messages,
            "model": self.intrinsic_name,
        }
        if hasattr(self, 'temperature') and self.temperature is not None:
            request_body["temperature"] = self.temperature
        document_texts = self._extract_documents()
        if document_texts:
            if "extra_body" not in request_body:
                request_body["extra_body"] = {}
            request_body["extra_body"]["documents"] = [{"text": t} for t in document_texts]
        return request_body

    def _extract_documents(self) -> list:
        """Extract document texts from the documents input.

        Supports:
        - list[Data] from vector store search results (each Data has text/page_content)
        - Single Data with data["documents"] list
        - dict with "documents" key
        """
        document_texts = []
        if not hasattr(self, 'documents') or not self.documents:
            return []

        docs_input = self.documents
        if not isinstance(docs_input, list):
            docs_input = [docs_input]

        for item in docs_input:
            if isinstance(item, Data):
                # Try text attribute first (vector store search results)
                if hasattr(item, 'text') and item.text:
                    document_texts.append(item.text)
                # Then try nested documents list
                elif hasattr(item, 'data') and isinstance(item.data, dict):
                    nested = item.data.get("documents")
                    if nested and isinstance(nested, list):
                        for doc in nested:
                            if isinstance(doc, dict):
                                text = doc.get("text") or doc.get("content") or doc.get("page_content") or str(doc)
                            else:
                                text = str(doc)
                            document_texts.append(text)
            elif isinstance(item, dict):
                nested = item.get("documents")
                if nested and isinstance(nested, list):
                    for doc in nested:
                        if isinstance(doc, dict):
                            text = doc.get("text") or doc.get("content") or doc.get("page_content") or str(doc)
                        else:
                            text = str(doc)
                        document_texts.append(text)
                else:
                    text = item.get("text") or item.get("content") or item.get("page_content") or str(item)
                    document_texts.append(text)
            else:
                document_texts.append(str(item))

        if not document_texts:
            raise ValueError("Documents are required for answerability check")

        return document_texts

    def _get_api_url(self) -> str:
        if self.model_backend == "IntrinsicsAPI (mellea + RITS)":
            return f"{self.url.rstrip('/')}/{self.model_name}/{self.intrinsic_name}"
        else:
            base_url = self.url.rstrip('/')
            return f"{base_url}{self.CHAT_COMPLETIONS_ENDPOINT}"

    def _call_api(self) -> dict:
        if self.model_backend == "LocalHF":
            return self._call_with_mellea()
        if not self.url or not self.url.strip():
            raise ValueError("API URL is required")
        if self.model_backend in ["vLLM", "Ollama"]:
            return self._call_with_granite_rewriter()
        else:
            return self._call_intrinsics_api()

    def _call_with_mellea(self) -> dict:
        """Call the Mellea backend using mellea directly."""
        # Lazy import mellea modules to avoid heavy imports during
        # Langflow's component loading/introspection.
        from mellea.backends.huggingface import LocalHFBackend
        from mellea.stdlib.context import ChatContext
        from mellea.stdlib.components import Document as MelleaDocument
        from mellea.stdlib.components.chat import Message as MelleaMessage
        from mellea.stdlib.components.intrinsic import rag as mellea_rag

        query = self._extract_message_text(self.chat_input)
        if not query or not query.strip():
            raise ValueError("Query is required")

        document_texts = self._extract_documents()

        # Build context from message history
        context_messages = []
        if hasattr(self, 'message_history') and self.message_history is not None:
            df = self.message_history
            for _, row in df.iterrows():
                sender = row.get("sender", "")
                role = "user" if sender == "User" else "assistant"
                content = row.get("text", "")
                if content and str(content).strip():
                    context_messages.append({"role": role, "content": str(content)})

        actual_model_id = _MODEL_NAME_MAPPING.get(self.model_name, self.model_name or _DEFAULT_MODEL)
        self.log(json.dumps({
            "model": actual_model_id,
            "query": query,
            "documents": document_texts,
            "context_messages": context_messages,
        }, indent=2), name="Mellea request body")
        if actual_model_id in _mellea_backend_cache:
            backend = _mellea_backend_cache[actual_model_id]
        else:
            backend = LocalHFBackend(model_id=actual_model_id)
            _mellea_backend_cache[actual_model_id] = backend

        mellea_documents = [MelleaDocument(doc) for doc in document_texts]

        context = ChatContext()
        for msg in context_messages:
            context = context.add(MelleaMessage(msg["role"], msg["content"]))

        answerability_score = float(mellea_rag.check_answerability(query, mellea_documents, context, backend))

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": str(answerability_score)
                    }
                }
            ],
            "model": self.model_name,
            "backend": "LocalHF",
            "answerability_likelihood": answerability_score
        }

    def _call_with_granite_rewriter(self) -> dict:
        try:
            from mellea.formatters import granite as granite_common
            request_body = self._build_request()
            self.log(json.dumps(request_body, indent=2), name="mellea rewriter request body")
            intrinsics_repo_name = "ibm-granite/granite-lib-rag-r1.0"
            use_alora = False
            io_yaml_file = granite_common.intrinsics.obtain_io_yaml(
                self.intrinsic_name, self.model_name, intrinsics_repo_name, alora=use_alora
            )
            rewriter = granite_common.IntrinsicsRewriter(config_file=io_yaml_file)
            result_processor = granite_common.IntrinsicsResultProcessor(config_file=io_yaml_file)
            rewritten_request = rewriter.transform(request_body)
            api_url = self._get_api_url()
            client = OpenAI(
                base_url=api_url.replace("/chat/completions", ""),
                api_key=self.api_key or "no-key",
            )
            completion = client.chat.completions.create(**rewritten_request.model_dump())
            processed = result_processor.transform(completion, rewritten_request)
            return json.loads(processed.model_dump_json())
        except Exception as e:
            raise ValueError(f"vLLM/Ollama call with mellea failed: {e}")

    def _call_intrinsics_api(self) -> dict:
        request_body = self._build_request()
        self.log(json.dumps(request_body, indent=2), name="Intrinsics API request body")
        api_url = self._get_api_url()
        headers = {"Content-Type": "application/json"}
        if self.model_backend == "IntrinsicsAPI (mellea + RITS)":
            if self.api_key and self.api_key.strip():
                headers["RITS_API_KEY"] = self.api_key
        try:
            with httpx.Client() as client:
                response = client.post(
                    api_url, json=request_body, headers=headers, timeout=30.0,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_detail = e.response.json()
            except:
                pass
            raise ValueError(f"API error {e.response.status_code}: {error_detail}")
        except httpx.ConnectError:
            raise ValueError(f"Could not connect to API at {api_url}")
        except httpx.RequestError as e:
            raise ValueError(f"Request failed: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error: {e}")
        try:
            response_data = response.json()
        except Exception as e:
            raise ValueError(f"Failed to parse API response as JSON: {e}")
        return response_data

    def _extract_content_from_response(self, response_data: dict) -> str:
        content = ""
        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            if "message" in choice:
                message = choice["message"]
                if isinstance(message, dict):
                    content = message.get("content", "")
                else:
                    content = str(message)
        elif "answerability_likelihood" in response_data:
            return str(response_data["answerability_likelihood"])

        # If content is a JSON string containing answerability_likelihood, extract it
        if content:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "answerability_likelihood" in parsed:
                    return str(parsed["answerability_likelihood"])
            except (json.JSONDecodeError, TypeError):
                pass
        return content

    async def update_build_config(self, build_config, field_value=None, field_name=None):
        if field_name == "model_backend":
            if field_value and field_value in self.BACKEND_CONFIG:
                backend_config = self.BACKEND_CONFIG[field_value]
                build_config["model_name"]["options"] = backend_config["models"]
                build_config["model_name"]["value"] = backend_config["default_model"]
                build_config["url"]["value"] = backend_config["default_url"]
                build_config["url"]["show"] = backend_config.get("url_editable", True)
                if field_value == "IntrinsicsAPI (mellea + RITS)":
                    build_config["api_key"]["display_name"] = "RITS API Key"
                    build_config["api_key"]["show"] = True
                elif field_value == "LocalHF":
                    build_config["url"]["show"] = False
                    build_config["api_key"]["show"] = False
                    build_config["api_key"]["required"] = False
                else:
                    build_config["api_key"]["display_name"] = "API Key"
                    build_config["api_key"]["show"] = True
        return build_config

    def get_response(self) -> Message:
        response_data = self._call_api()
        content = self._extract_content_from_response(response_data)
        return Message(text=content)

    def get_full_response(self) -> str:
        response_data = self._call_api()
        return json.dumps(response_data, indent=2)

    def get_answerability_label(self) -> Message:
        response_data = self._call_api()

        # Get answerability likelihood from response
        likelihood = response_data.get("answerability_likelihood")

        # If not at top level, try extracting from response content
        if likelihood is None:
            content = self._extract_content_from_response(response_data)
            if content:
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        likelihood = parsed.get("answerability_likelihood")
                except (json.JSONDecodeError, TypeError):
                    pass
                if likelihood is None:
                    try:
                        likelihood = float(content)
                    except (ValueError, TypeError):
                        pass

        if likelihood is not None:
            result = "answerable" if float(likelihood) >= 0.5 else "unanswerable"
            return Message(text=result)

        return Message(text="unknown")
