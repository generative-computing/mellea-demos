"""
Granite Base Model Component for Langflow

Calls the Granite base model via multiple backends (vLLM, Ollama, IntrinsicsAPI).
Takes chat input and returns model-generated responses.
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
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from openai import OpenAI

# Global model cache for LocalHF backend
_localhf_model_cache = {}

# Model name mapping to HuggingFace identifiers
_MODEL_NAME_MAPPING = {
    "granite-4.0-micro": "ibm-granite/granite-4.0-micro",
    "ibm-granite/granite-4.0-micro": "ibm-granite/granite-4.0-micro",
    "granite4:micro": "ibm-granite/granite-4.0-micro",
}


class GraniteBaseModel(Component):
    display_name = "Base Model Chat"
    description = "Call Granite base models for text generation"
    category = "llm"
    icon = "sparkles"

    BACKEND_CONFIG = {
        "LocalHF": {
            "models": ["granite-4.0-micro"],
            "default_model": "granite-4.0-micro",
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
            required=False,
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
            info="Documents to include in the request (accepts Data from vector stores)",
            input_types=["Data", "Search Results"],
            required=False,
            is_list=True,
        ),
    ]

    outputs = [
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
        if message is None:
            return ""
        # Handle DataFrame case - should not happen but safety check
        import pandas as pd
        if isinstance(message, pd.DataFrame):
            return ""
        if hasattr(message, 'text'):
            return message.text
        if hasattr(message, 'content'):
            return message.content
        return str(message)

    def _build_chat_messages(self) -> list:
        messages = []
        # Check if message_history exists - handle both pandas and Langflow DataFrames
        if hasattr(self, 'message_history'):
            # Don't use isinstance check or boolean evaluation as it could be lfx.schema.dataframe.DataFrame
            try:
                df = self.message_history
                # Try to iterate - if it's None or invalid, this will fail
                for _, row in df.iterrows():
                    sender = row.get("sender", "")
                    role = "user" if sender == "User" else "assistant"
                    content = row.get("text", "")
                    if content and str(content).strip():
                        messages.append({"role": role, "content": str(content)})
            except (TypeError, AttributeError, ValueError) as e:
                # None, invalid type, or DataFrame boolean evaluation error - skip silently
                pass
            except Exception as e:
                self.log(f"Error processing message_history: {e}", name="message_history warning")
        # Check if chat_input exists
        if hasattr(self, 'chat_input'):
            try:
                user_message = self._extract_message_text(self.chat_input)
                if user_message and user_message.strip():
                    messages.append({"role": "user", "content": user_message})
            except (TypeError, AttributeError, ValueError):
                # None, invalid type, or DataFrame boolean evaluation error - skip silently
                pass

        return messages

    def _build_request(self) -> dict:
        messages = self._build_chat_messages()
        if not messages:
            raise ValueError("No messages to send")
        request_body = {
            "messages": messages,
            "model": self.model_name,
        }
        # Add temperature if provided
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
                if hasattr(item, 'text') and item.text:
                    document_texts.append(item.text)
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

        return document_texts

    def _get_api_url(self) -> str:
        if self.model_backend == "IntrinsicsAPI (mellea + RITS)":
            return f"{self.url.rstrip('/')}/{self.model_name}"
        else:
            base_url = self.url.rstrip('/')
            return f"{base_url}{self.CHAT_COMPLETIONS_ENDPOINT}"

    def _call_api(self) -> dict:
        if self.model_backend == "LocalHF":
            return self._call_with_localhf()
        if not self.url or not self.url.strip():
            raise ValueError("API URL is required")
        if self.model_backend in ["vLLM", "Ollama"]:
            return self._call_openai()
        else:
            return self._call_intrinsics_api()

    def _call_with_localhf(self) -> dict:
        """Call local HuggingFace model directly using transformers."""
        messages = self._build_chat_messages()
        if not messages:
            raise ValueError("No messages to send")

        hf_model_id = _MODEL_NAME_MAPPING.get(self.model_name, self.model_name)
        self.log(json.dumps({"model": hf_model_id, "messages": messages}, indent=2), name="LocalHF request")

        if hf_model_id in _localhf_model_cache:
            tokenizer, model = _localhf_model_cache[hf_model_id]
        else:
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_id, torch_dtype=torch.bfloat16, device_map="auto"
            )
            model.eval()
            _localhf_model_cache[hf_model_id] = (tokenizer, model)

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        input_length = inputs["input_ids"].shape[1]

        temperature = self.temperature if hasattr(self, "temperature") and self.temperature is not None else 0.0
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        self.log(response_text, name="LocalHF response")

        return {
            "choices": [{"message": {"role": "assistant", "content": response_text}}],
            "model": self.model_name,
            "backend": "LocalHF",
        }

    def _call_openai(self) -> dict:
        try:
            request_body = self._build_request()

            self.log(json.dumps(request_body, indent=2), name="vLLM/Ollama request")
            api_url = self._get_api_url()
            client = OpenAI(
                base_url=api_url.replace("/chat/completions", ""),
                api_key=self.api_key or "no-key",
            )
            completion = client.chat.completions.create(**request_body)
            response_data = json.loads(completion.model_dump_json())
            self.log(json.dumps(response_data, indent=2), name="vLLM/Ollama response")
            return response_data
        except Exception as e:
            raise ValueError(f"vLLM/Ollama call failed: {e}")

    def _call_intrinsics_api(self) -> dict:
        request_body = self._build_request()

        self.log(json.dumps(request_body, indent=2), name="Intrinsics API request")
        api_url = self._get_api_url()
        headers = {"Content-Type": "application/json"}
        if self.model_backend == "IntrinsicsAPI (mellea + RITS)":
            if self.api_key and self.api_key.strip():
                headers["RITS_API_KEY"] = self.api_key
        self._last_request = request_body
        try:
            with httpx.Client() as client:
                response = client.post(
                    api_url, json=request_body, headers=headers, timeout=300.0,
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
        self.log(json.dumps(response_data, indent=2), name="Intrinsics API response")
        return response_data

    def _extract_content_from_response(self, response_data: dict) -> str:
        """Extract text content from API response (full response already printed in _call_* methods)"""
        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            if "message" in choice:
                message = choice["message"]
                if isinstance(message, dict):
                    return message.get("content", "")
                return str(message)
        return ""

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
