"""
Query Rewrite with Prompt Component for Langflow

Rewrite a query using a default prompt template for improved retrieval performance.
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
from langflow.schema import Message
from langflow.field_typing.range_spec import RangeSpec
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import httpx
from openai import OpenAI


class QueryRewritePrompt(Component):
    display_name = "Query Rewrite (Prompt)"
    description = "Rewrite a query using a default prompt template for improved retrieval"
    category = "tools"
    icon = "message-square"

    intrinsic_name = "query_rewrite"

    BACKEND_CONFIG = {
        "LocalHF": {
            "models": ["granite-4.0-micro"],
            "default_model": "granite-4.0-micro",
            "default_url": "",
            "url_editable": False,
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

    # Model name mapping to HuggingFace identifiers
    MODEL_NAME_MAPPING = {
        "granite-4.0-micro": "ibm-granite/granite-4.0-micro",
        "ibm-granite/granite-4.0-micro": "ibm-granite/granite-4.0-micro",
        "granite4:micro": "ibm-granite/granite-4.0-micro",
    }

    # Default prompt template for query rewriting
    REWRITE_PROMPT = (
        "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
        "Here is the initial question:"
        "\n ------- \n"
        "{question}"
        "\n ------- \n"
        "Formulate an improved last-turn question. "
        "Return ONLY the improved question without any explanation, preamble, or additional text."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = None
        self._tokenizer = None

    inputs = [
        DropdownInput(
            name="model_backend",
            display_name="Model Backend",
            info="Backend service for Granite model",
            options=["IntrinsicsAPI (mellea + RITS)", "Ollama", "LocalHF"],
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
            info="Base URL for the model backend",
            value="https://intrinsics-api.intrinsics.vpc-int.res.ibm.com",
            required=False,
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
            info="User message/query to rewrite",
            required=True,
        ),
        DataFrameInput(
            name="message_history",
            display_name="Message History",
            info="Previous conversation messages as a DataFrame with 'sender' (User/AI) and 'text' columns",
            required=False,
        ),
    ]

    outputs = [
        Output(
            name="rewritten_query",
            display_name="Rewritten Query",
            method="rewrite_query",
        ),
        Output(
            name="full_response",
            display_name="Full Response",
            method="get_full_response",
        ),
    ]

    def _extract_message_text(self, message) -> str:
        """Extract text content from a Message object or other input types"""
        # Handle Message objects
        if hasattr(message, 'text'):
            return message.text
        if hasattr(message, 'content'):
            return message.content

        # Handle Data objects that might come from If-Else
        if hasattr(message, 'data'):
            data = message.data
            if isinstance(data, dict):
                # Try common text fields
                return data.get('text', data.get('content', data.get('message', str(data))))
            return str(data)

        # Handle plain strings
        if isinstance(message, str):
            return message

        # Fallback to string representation
        return str(message)

    def _build_conversation_text(self) -> str:
        """Build conversation text with alternating user/assistant messages from DataFrame"""
        conversation_lines = []

        # Get current user input first to filter it from history
        # Handle build phase where chat_input might not be set
        if not hasattr(self, 'chat_input') or self.chat_input is None:
            raise ValueError("Chat input (current query) is required")

        current_query = self._extract_message_text(self.chat_input)
        if not current_query or not current_query.strip():
            raise ValueError("Chat input (current query) is required")
        current_query_text = current_query.strip()

        # Add chat history from DataFrame (excluding current message to avoid duplicates)
        if hasattr(self, 'message_history') and self.message_history is not None:
            df = self.message_history
            print("history DataFrame shape:", df.shape if hasattr(df, 'shape') else "unknown")

            # Iterate over DataFrame rows
            for _, row in df.iterrows():
                sender = row.get("sender", "")
                content = row.get("text", "")
                content_str = str(content).strip() if content else ""

                # Skip if this is the current message (avoid duplicates from Memory component)
                if content_str and content_str != current_query_text:
                    # Format as "User:" or "AI:" based on sender
                    role_label = "User" if sender == "User" else "AI"
                    conversation_lines.append(f"{role_label}: {content_str}")

        # Add current user input as the last turn (must be user role)
        conversation_lines.append(f"User: {current_query_text}")

        return "\n".join(conversation_lines)

    def _build_prompt(self) -> str:
        """Build the full prompt with the conversation"""
        conversation_text = self._build_conversation_text()
        print("Conversation Text for Prompt:", conversation_text)

        prompt = self.REWRITE_PROMPT.format(question=conversation_text)
        return prompt

    def _load_model(self):
        """Lazy load the model and tokenizer"""
        if self._model is None or self._tokenizer is None:
            # Map model name to HuggingFace identifier
            hf_model_name = self.MODEL_NAME_MAPPING.get(self.model_name, self.model_name)
            print(f"Loading model: {self.model_name} -> {hf_model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                hf_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self._model.eval()
            print(f"Model loaded on device: {self._model.device}")

    def _invoke_with_string_prompt(self, prompt: str) -> str:
        """Invoke model with a string prompt"""
        self._load_model()

        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True).to(self._model.device)

        # Generate response
        # Use temperature if provided, otherwise use 0.0 for deterministic output
        temperature = self.temperature if hasattr(self, 'temperature') and self.temperature is not None else 0.0
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode response (only the new tokens)
        response_text = self._tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response_text.strip()

    def _invoke_with_chat_messages(self, messages: list) -> str:
        """Invoke model with chat messages using chat template

        Args:
            messages: List of dicts with 'role' and 'content' keys
                     e.g., [{"role": "user", "content": "..."}]
        """
        self._load_model()

        # Apply chat template to convert messages to string
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True).to(self._model.device)
        input_length = inputs['input_ids'].shape[1]

        # Generate response
        # Use temperature if provided, otherwise use 0.0 for deterministic output
        temperature = self.temperature if hasattr(self, 'temperature') and self.temperature is not None else 0.0
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode response (only the new tokens)
        response_text = self._tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )

        return response_text.strip()

    def _build_chat_messages_for_api(self) -> list:
        """Build chat messages for API call from conversation"""
        messages = []

        # Build conversation text with history and current input
        conversation_text = self._build_conversation_text()

        # Apply the rewrite prompt template
        prompt = self.REWRITE_PROMPT.format(question=conversation_text)

        # Create a single user message with the formatted prompt
        messages.append({"role": "user", "content": prompt})

        return messages

    def _build_request(self) -> dict:
        """Build request body for IntrinsicsAPI"""
        messages = self._build_chat_messages_for_api()
        if not messages:
            raise ValueError("No messages to send")
        request_body = {
            "messages": messages,
            "model": self.model_name,  # Use model_name, not intrinsic_name
        }
        # Add temperature if provided
        if hasattr(self, 'temperature') and self.temperature is not None:
            request_body["temperature"] = self.temperature
        return request_body

    def _get_api_url(self) -> str:
        """Construct API URL based on backend"""
        if self.model_backend == "IntrinsicsAPI (mellea + RITS)":
            # Call the model endpoint directly (not the intrinsic endpoint)
            # This sends the REWRITE_PROMPT to the base model
            return f"{self.url.rstrip('/')}/{self.model_name}"
        elif self.model_backend == "Ollama":
            base_url = self.url.rstrip('/')
            return f"{base_url}/v1/chat/completions"
        else:
            raise ValueError(f"Unsupported backend for API URL: {self.model_backend}")

    def _call_intrinsics_api(self) -> dict:
        """Call the IntrinsicsAPI endpoint"""
        if not self.url or not self.url.strip():
            raise ValueError("API URL is required for IntrinsicsAPI backend")

        request_body = self._build_request()
        self.log(json.dumps(request_body, indent=2), name="Intrinsics API request body")
        api_url = self._get_api_url()

        headers = {"Content-Type": "application/json"}
        # Only include RITS_API_KEY header if provided
        if self.model_backend == "IntrinsicsAPI (mellea + RITS)":
            if hasattr(self, 'api_key') and self.api_key and self.api_key.strip():
                headers["RITS_API_KEY"] = self.api_key.strip()

        try:
            with httpx.Client() as client:
                response = client.post(
                    api_url, json=request_body, headers=headers, timeout=120.0,
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

    def _call_ollama_api(self) -> dict:
        """Call Ollama API using OpenAI-compatible endpoint"""
        try:
            request_body = self._build_request()

            # Print model input
            print(f"\n{'='*80}")
            print(f"[Query Rewrite Prompt - Ollama] MODEL INPUT:")
            print(f"{'='*80}")
            print(json.dumps(request_body, indent=2))
            print(f"{'='*80}\n")

            self.log(json.dumps(request_body, indent=2), name="Ollama request body")
            api_url = self._get_api_url()
            client = OpenAI(
                base_url=api_url.replace("/chat/completions", ""),
                api_key=self.api_key or "no-key",
            )
            completion = client.chat.completions.create(**request_body)
            response_data = json.loads(completion.model_dump_json())

            # Print model output
            print(f"\n{'='*80}")
            print(f"[Query Rewrite Prompt - Ollama] MODEL OUTPUT:")
            print(f"{'='*80}")
            print(json.dumps(response_data, indent=2))
            print(f"{'='*80}\n")

            return response_data
        except Exception as e:
            raise ValueError(f"Ollama call failed: {e}")

    def _extract_content_from_response(self, response_data: dict) -> str:
        """Extract content from API response"""
        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            if "message" in choice:
                message = choice["message"]
                if isinstance(message, dict):
                    return message.get("content", "")
                return str(message)
        return ""

    def _call_backend(self) -> str:
        """Call the appropriate backend (LocalHF, Ollama, or IntrinsicsAPI)"""
        if self.model_backend == "LocalHF":
            return self._call_localhf_backend()
        elif self.model_backend == "Ollama":
            return self._call_ollama_backend()
        elif self.model_backend == "IntrinsicsAPI (mellea + RITS)":
            return self._call_intrinsics_backend()
        else:
            raise ValueError(f"Unsupported backend: {self.model_backend}")

    def _call_localhf_backend(self) -> str:
        """Call local HuggingFace model with the constructed prompt"""
        prompt = self._build_prompt()

        print("\n" + "=" * 80)
        print("INPUT PROMPT:")
        print(prompt)
        print("=" * 80)

        # Use chat messages format with chat template (Granite models expect this)
        messages = [{"role": "user", "content": prompt}]
        response_text = self._invoke_with_chat_messages(messages)

        # Alternative: Use string prompt directly (uncomment to test)
        # response_text = self._invoke_with_string_prompt(prompt)

        print("=" * 80)
        print("OUTPUT:")
        print(response_text)
        print("=" * 80 + "\n")

        return response_text

    def _call_ollama_backend(self) -> str:
        """Call Ollama endpoint"""
        response_data = self._call_ollama_api()
        content = self._extract_content_from_response(response_data)

        if not content:
            raise ValueError("No content received from Ollama")

        print("\n" + "=" * 80)
        print("OLLAMA API RESPONSE:")
        print(content)
        print("=" * 80 + "\n")

        return content

    def _call_intrinsics_backend(self) -> str:
        """Call IntrinsicsAPI endpoint"""
        response_data = self._call_intrinsics_api()
        content = self._extract_content_from_response(response_data)

        if not content:
            raise ValueError("No content received from IntrinsicsAPI")

        print("\n" + "=" * 80)
        print("INTRINSICS API RESPONSE:")
        print(content)
        print("=" * 80 + "\n")

        return content

    async def update_build_config(self, build_config, field_value=None, field_name=None):
        """Update build config when model_backend changes"""
        if field_name == "model_backend":
            if field_value and field_value in self.BACKEND_CONFIG:
                backend_config = self.BACKEND_CONFIG[field_value]
                build_config["model_name"]["options"] = backend_config["models"]
                build_config["model_name"]["value"] = backend_config["default_model"]
                build_config["url"]["value"] = backend_config["default_url"]
                build_config["url"]["show"] = backend_config.get("url_editable", False)
                if field_value == "IntrinsicsAPI (mellea + RITS)":
                    build_config["api_key"]["display_name"] = "RITS API Key"
                    build_config["api_key"]["show"] = True
                    build_config["api_key"]["required"] = False
                elif field_value == "LocalHF":
                    build_config["api_key"]["show"] = False
                    build_config["api_key"]["required"] = False
                else:
                    build_config["api_key"]["display_name"] = "API Key"
                    build_config["api_key"]["show"] = True
                    build_config["api_key"]["required"] = False
        return build_config

    def rewrite_query(self) -> Message:
        """Main method that returns the rewritten query as a Message"""
        print("\n" + "🔄" * 40)
        print(">>> STAGE 1: QUERY REWRITE COMPONENT EXECUTED")
        print("🔄" * 40)

        # Handle case where chat_input is not yet connected (build phase)
        if not hasattr(self, 'chat_input') or self.chat_input is None:
            print("⚠️  No chat_input - returning empty")
            return Message(text="")

        # Check if chat_input has valid content
        try:
            current_query = self._extract_message_text(self.chat_input)
            if not current_query or not current_query.strip():
                print("⚠️  Empty chat_input - returning empty")
                return Message(text="")
            print(f"📥 Input Query: '{current_query}'")
        except Exception:
            print("⚠️  Exception extracting chat_input - returning empty")
            return Message(text="")

        rewritten = self._call_backend()

        if not rewritten:
            raise ValueError("No rewritten query received from model")

        # Post-process: Try to extract "rewritten_question" from JSON response
        try:
            parsed = json.loads(rewritten)
            rewritten = parsed.get("rewritten_question", rewritten)
        except (json.JSONDecodeError, TypeError):
            # If not JSON or doesn't have the field, use the raw response
            pass

        print(f"📤 Rewritten Query: '{rewritten}'")
        print("🔄" * 40 + "\n")
        return Message(text=rewritten)

    def get_full_response(self) -> str:
        """Return the full response as JSON"""
        # Handle case where chat_input is not yet connected (build phase)
        if not hasattr(self, 'chat_input') or self.chat_input is None:
            return json.dumps({"error": "chat_input not connected"}, indent=2)

        # Check if chat_input has valid content
        try:
            current_query = self._extract_message_text(self.chat_input)
            if not current_query or not current_query.strip():
                return json.dumps({"error": "chat_input is empty"}, indent=2)
        except Exception as e:
            return json.dumps({"error": f"failed to extract text: {e}"}, indent=2)

        rewritten = self._call_backend()

        # Format as JSON for compatibility
        response_data = {
            "rewritten_query": rewritten,
            "model": self.model_name,
            "backend": self.model_backend
        }
        return json.dumps(response_data, indent=2)
