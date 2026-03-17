"""
Prompted Base Model Component for Langflow

Generate answers based on context and user questions using base models
via multiple backends (vLLM, Ollama, IntrinsicsAPI).
"""

from langflow.custom import Component
from langflow.io import (
    SecretStrInput,
    StrInput,
    Output,
    DropdownInput,
    MessageInput,
    DataInput,
    DataFrameInput,
    SliderInput,
)
from langflow.schema import Message, Data
from langflow.field_typing.range_spec import RangeSpec
import httpx
import json
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


class PromptedBaseModel(Component):
    display_name = "Prompted Base Model"
    description = "Generate answers using retrieved context and base models via vLLM, Ollama, or IntrinsicsAPI"
    category = "base_models"
    icon = "message-circle"

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

    # Default prompt template for RAG generation
    GENERATION_PROMPT = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Keep the answer concise.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

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
        SliderInput(
            name="max_tokens",
            display_name="Max Tokens",
            info="Maximum number of tokens to generate in the response",
            value=512,
            range_spec=RangeSpec(min=1, max=4096, step=1),
            advanced=True,
        ),
        MessageInput(
            name="question",
            display_name="Question",
            info="User question to answer",
            required=True,
        ),
        DataInput(
            name="context",
            display_name="Context",
            info="Retrieved documents/context for answering",
            required=True,
        ),
        DataFrameInput(
            name="message_history",
            display_name="Message History",
            info="Previous conversation messages as a DataFrame with 'role' (user/assistant) and 'content' columns",
            required=False,
        ),
        StrInput(
            name="custom_prompt",
            display_name="Custom Prompt Template (Optional)",
            info="Override default prompt template. Use {context} and {question} placeholders.",
            required=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            name="answer",
            display_name="Answer",
            method="generate_answer",
        ),
    ]

    def _extract_message_text(self, message) -> str:
        """Extract text content from a Message object or other input types"""
        if message is None:
            return ""
        import pandas as pd
        if isinstance(message, pd.DataFrame):
            return ""
        if hasattr(message, 'text'):
            return message.text
        if hasattr(message, 'content'):
            return message.content
        if isinstance(message, str):
            return message
        return str(message)

    def _extract_context(self) -> str:
        """Extract and format context from documents"""
        context_parts = []

        if hasattr(self, 'context') and self.context:
            if isinstance(self.context, Data):
                data_dict = self.context.data if hasattr(self.context, 'data') else self.context
                if isinstance(data_dict, dict):
                    documents = data_dict.get("documents", [])
                    if documents and isinstance(documents, list):
                        for doc in documents:
                            if isinstance(doc, dict):
                                text = doc.get("text") or doc.get("content") or doc.get("page_content") or str(doc)
                            else:
                                text = str(doc)
                            context_parts.append(text)
            elif isinstance(self.context, str):
                context_parts.append(self.context)
            elif isinstance(self.context, list):
                for doc in self.context:
                    if isinstance(doc, dict):
                        text = doc.get("text") or doc.get("content") or doc.get("page_content") or str(doc)
                    else:
                        text = str(doc)
                    context_parts.append(text)
            else:
                context_parts.append(str(self.context))

        return "\n\n".join(context_parts) if context_parts else "No context provided."

    def _build_chat_messages(self, prompt: str) -> list:
        """Build chat messages list including history and the RAG prompt"""
        messages = []

        # Add message history from DataFrame
        if hasattr(self, 'message_history'):
            try:
                df = self.message_history
                for _, row in df.iterrows():
                    sender = row.get("sender", "")
                    role = "user" if sender == "User" else "assistant"
                    content = row.get("text", "")
                    if content and str(content).strip():
                        messages.append({"role": role, "content": str(content)})
            except (TypeError, AttributeError, ValueError):
                pass
            except Exception as e:
                self.log(f"Error processing message_history: {e}", name="message_history warning")

        # Add the RAG prompt as the current user message
        messages.append({"role": "user", "content": prompt})

        return messages

    def _build_rag_prompt(self) -> str:
        """Build the RAG prompt string with context and question"""
        question_text = self._extract_message_text(self.question)
        context_text = self._extract_context()

        prompt_template = (
            self.custom_prompt
            if hasattr(self, 'custom_prompt') and self.custom_prompt
            else self.GENERATION_PROMPT
        )

        return prompt_template.format(context=context_text, question=question_text)

    def _build_request(self, messages: list) -> dict:
        request_body = {
            "messages": messages,
            "model": self.model_name,
        }
        if hasattr(self, 'temperature') and self.temperature is not None:
            request_body["temperature"] = self.temperature
        if hasattr(self, 'max_tokens') and self.max_tokens is not None:
            request_body["max_tokens"] = int(self.max_tokens)
        return request_body

    def _get_api_url(self) -> str:
        if self.model_backend == "IntrinsicsAPI (mellea + RITS)":
            return f"{self.url.rstrip('/')}/{self.model_name}"
        else:
            base_url = self.url.rstrip('/')
            return f"{base_url}{self.CHAT_COMPLETIONS_ENDPOINT}"

    def _call_api(self, messages: list) -> dict:
        if self.model_backend == "LocalHF":
            return self._call_with_localhf(messages)
        if not self.url or not self.url.strip():
            raise ValueError("API URL is required")
        if self.model_backend in ["vLLM", "Ollama"]:
            return self._call_openai(messages)
        else:
            return self._call_intrinsics_api(messages)

    def _call_with_localhf(self, messages: list) -> dict:
        """Call local HuggingFace model directly using transformers."""
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
        max_tokens = int(self.max_tokens) if hasattr(self, "max_tokens") and self.max_tokens is not None else 512
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
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

    def _call_openai(self, messages: list) -> dict:
        try:
            request_body = self._build_request(messages)
            self.log(json.dumps(request_body, indent=2), name="vLLM/Ollama request")
            if self.model_backend == "Ollama":
                print(f"[Ollama request]\n{json.dumps(request_body, indent=2)}")
            api_url = self._get_api_url()
            client = OpenAI(
                base_url=api_url.replace("/chat/completions", ""),
                api_key=self.api_key or "no-key",
            )
            completion = client.chat.completions.create(**request_body)
            response_data = json.loads(completion.model_dump_json())
            self.log(json.dumps(response_data, indent=2), name="vLLM/Ollama response")
            if self.model_backend == "Ollama":
                print(f"[Ollama response]\n{json.dumps(response_data, indent=2)}")
            return response_data
        except Exception as e:
            raise ValueError(f"vLLM/Ollama call failed: {e}")

    def _call_intrinsics_api(self, messages: list) -> dict:
        request_body = self._build_request(messages)
        self.log(json.dumps(request_body, indent=2), name="Intrinsics API request")
        api_url = self._get_api_url()
        headers = {"Content-Type": "application/json"}
        if self.model_backend == "IntrinsicsAPI (mellea + RITS)":
            if self.api_key and self.api_key.strip():
                headers["RITS_API_KEY"] = self.api_key
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
            except Exception:
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

    def generate_answer(self) -> Message:
        """Build the RAG prompt and call the model via the selected backend"""
        if not hasattr(self, 'question') or self.question is None:
            return Message(text="")
        if not hasattr(self, 'context') or self.context is None:
            return Message(text="")

        try:
            question_text = self._extract_message_text(self.question)
            if not question_text or not question_text.strip():
                return Message(text="")
        except Exception:
            return Message(text="")

        rag_prompt = self._build_rag_prompt()

        print("\n" + "=" * 80)
        print("GENERATION PROMPT:")
        print(rag_prompt)
        print("=" * 80)

        messages = self._build_chat_messages(rag_prompt)
        response_data = self._call_api(messages)
        answer = self._extract_content_from_response(response_data)

        print("=" * 80)
        print("GENERATED ANSWER:")
        print(answer)
        print("=" * 80 + "\n")

        return Message(text=answer)
