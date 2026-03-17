"""
Query Expansion Component for Langflow

Expands a user query into multiple variations for improved retrieval.

Produces 5 queries:
  0. Original query
  1. Rewritten query (via query_rewrite intrinsic)
  2. Synonymous query (Granite prompted to rephrase with synonyms)
  3. Reverse-engineered question (from a sampled answer)
  4. Sampled answer
"""

from langflow.custom import Component
from langflow.io import (
    SecretStrInput,
    StrInput,
    Output,
    DropdownInput,
    MessageInput,
    DataFrameInput,
    SliderInput,
)
from langflow.schema import Message, Data
from langflow.field_typing.range_spec import RangeSpec
import httpx
import json

from openai import OpenAI


class QueryExpansion(Component):
    display_name = "Query Expansion"
    description = "Expand queries into multiple variations for retrieval"
    category = "tools"
    icon = "message-square"

    BACKEND_CONFIG = {
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
            "models": ["granite-4.0-micro"],
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
            options=["IntrinsicsAPI (mellea + RITS)", "vLLM", "Ollama"],
            value="IntrinsicsAPI (mellea + RITS)",
            real_time_refresh=True,
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            info="Model name (options change based on backend)",
            options=["granite-4.0-micro"],
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
            info="User query to expand",
            required=True,
        ),
        DataFrameInput(
            name="message_history",
            display_name="Message History",
            info="Previous conversation messages as a DataFrame with 'role' (user/assistant) and 'content' columns",
            required=False,
        ),
    ]

    outputs = [
        Output(
            name="all_queries",
            display_name="All Queries",
            method="get_all_queries",
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

    def _format_chat_history(self, messages: list) -> str:
        formatted = []
        for msg in messages:
            formatted.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(formatted)

    def _get_base_url(self) -> str:
        if self.model_backend == "IntrinsicsAPI (mellea + RITS)":
            return f"{self.url.rstrip('/')}/{self.model_name}"
        else:
            base_url = self.url.rstrip('/')
            return f"{base_url}{self.CHAT_COMPLETIONS_ENDPOINT}"

    def _get_intrinsic_url(self, intrinsic_name: str) -> str:
        if self.model_backend == "IntrinsicsAPI (mellea + RITS)":
            return f"{self.url.rstrip('/')}/{self.model_name}/{intrinsic_name}"
        else:
            base_url = self.url.rstrip('/')
            return f"{base_url}{self.CHAT_COMPLETIONS_ENDPOINT}"

    def _call_base_model(self, request_body: dict) -> dict:
        """Call Granite base model."""
        if self.model_backend in ["vLLM", "Ollama"]:
            api_url = self._get_base_url()
            client = OpenAI(
                base_url=api_url.replace("/chat/completions", ""),
                api_key=self.api_key or "no-key",
            )
            completion = client.chat.completions.create(**request_body)
            return json.loads(completion.model_dump_json())
        else:
            api_url = self._get_base_url()
            headers = {"Content-Type": "application/json"}
            if self.api_key and self.api_key.strip():
                headers["RITS_API_KEY"] = self.api_key
            with httpx.Client() as client:
                response = client.post(
                    api_url, json=request_body, headers=headers, timeout=30.0,
                )
                response.raise_for_status()
            return response.json()

    def _call_query_rewrite(self, request_body: dict) -> dict:
        """Call query_rewrite intrinsic."""
        from mellea.formatters import granite as granite_common
        if self.model_backend in ["vLLM", "Ollama"]:
            intrinsics_repo_name = "ibm-granite/granite-lib-rag-r1.0"
            io_yaml_file = granite_common.intrinsics.obtain_io_yaml(
                "query_rewrite", self.model_name, intrinsics_repo_name, alora=False
            )
            rewriter = granite_common.IntrinsicsRewriter(config_file=io_yaml_file)
            result_processor = granite_common.IntrinsicsResultProcessor(config_file=io_yaml_file)
            rewritten_request = rewriter.transform(request_body)
            api_url = self._get_intrinsic_url("query_rewrite")
            client = OpenAI(
                base_url=api_url.replace("/chat/completions", ""),
                api_key=self.api_key or "no-key",
            )
            completion = client.chat.completions.create(**rewritten_request.model_dump())
            processed = result_processor.transform(completion, rewritten_request)
            return json.loads(processed.model_dump_json())
        else:
            api_url = self._get_intrinsic_url("query_rewrite")
            headers = {"Content-Type": "application/json"}
            if self.api_key and self.api_key.strip():
                headers["RITS_API_KEY"] = self.api_key
            with httpx.Client() as client:
                response = client.post(
                    api_url, json=request_body, headers=headers, timeout=30.0,
                )
                response.raise_for_status()
            return response.json()

    def _extract_content(self, response_data: dict) -> str:
        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            if "message" in choice:
                message = choice["message"]
                if isinstance(message, dict):
                    return message.get("content", "")
                return str(message)
        return ""

    def _generate_query_variations(self) -> list:
        messages = self._build_chat_messages()
        if not messages:
            raise ValueError("No messages to send")

        user_message = messages[-1]["content"]
        conversation_str = self._format_chat_history(messages)

        # 1. Query rewrite via intrinsic
        rewrite_request = {"messages": messages, "model": "query_rewrite"}
        self.log(json.dumps(rewrite_request, indent=2), name="Query rewrite request")
        rewrite_response = self._call_query_rewrite(rewrite_request)
        rewrite_content = self._extract_content(rewrite_response)
        try:
            parsed = json.loads(rewrite_content)
            rewritten_query = parsed.get("rewritten_question", rewrite_content)
        except (json.JSONDecodeError, TypeError):
            rewritten_query = rewrite_content

        # 2. Synonymous query via Granite prompt
        synonym_prompt = (
            "You are given a multi-turn conversation between a user and an assistant. "
            "Reformulate the last-turn user query into a synonymous standalone query "
            "by replacing key terms with appropriate synonyms or closely related "
            "phrases, while preserving the original intent and meaning. "
            "This rewritten query will be used to retrieve relevant passages "
            "from a corpus, so it must remain faithful to the user's information need. "
            "Only output the rewritten query.\n\n[[Input]]\n"
            f"{conversation_str}\n\n[[Output]]\n"
        )
        synonym_request = {
            "messages": [{"role": "user", "content": synonym_prompt}],
            "model": "query_rewrite",
            "max_tokens": 512,
            "stop": ["[[Input]]"],
        }
        self.log(json.dumps(synonym_request, indent=2), name="Synonym request")
        synonym_response = self._call_base_model(synonym_request)
        synonymous_query = self._extract_content(synonym_response).strip()

        # 3. Sample an answer from Granite, then reverse-engineer a question
        answer_request = {
            "messages": messages,
            "model": "query_rewrite",
            "max_tokens": 512,
        }
        self.log(json.dumps(answer_request, indent=2), name="Answer sampling request")
        answer_response = self._call_base_model(answer_request)
        sampled_answer = self._extract_content(answer_response).strip()

        reverse_prompt = (
            "Generate a single question for the given answer.\n"
            "[[Answer]]\n"
            "Albert Einstein was born in Germany.\n"
            "[[Question]]\n"
            "Where was Albert Einstein born?\n"
            "[[Answer]]\n"
            f"{sampled_answer}\n"
            "[[Question]]\n"
        )
        reverse_request = {
            "messages": [{"role": "user", "content": reverse_prompt}],
            "model": "query_rewrite",
            "max_tokens": 512,
            "stop": ["[[Answer]]"],
        }
        self.log(json.dumps(reverse_request, indent=2), name="Reverse question request")
        reverse_response = self._call_base_model(reverse_request)
        reverse_question = self._extract_content(reverse_response).strip()

        return [
            user_message,       # 0: Original query
            rewritten_query,    # 1: Rewritten query (intrinsic)
            synonymous_query,   # 2: Synonymous query (prompted)
            reverse_question,   # 3: Reverse-engineered question
            sampled_answer,     # 4: Sampled answer
        ]

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
                else:
                    build_config["api_key"]["display_name"] = "API Key"
        return build_config

    def get_all_queries(self) -> Data:
        return Data(data={"queries": self._generate_query_variations()})
