"""
Hallucination Removal with Prompt Component for Langflow

Remove hallucinated claims from agent responses using a prompt-based approach.
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
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import httpx
from openai import OpenAI


class HallucinationRemovalPrompt(Component):
    display_name = "Hallucination Removal (Prompt)"
    description = "Remove hallucinated claims from the last agent response by checking against retrieved documents"
    category = "tools"
    icon = "shield-check"

    intrinsic_name = "hallucination_removal"

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
        "IntrinsicsAPI (granite-common + RITS)": {
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

    # Default prompt template for hallucination removal
    HALLUCINATION_PROMPT = (
        "Documents:\n{documents}\n\n"
        "Conversation:\n{conversation}\n\n"
        "Rewrite the last assistant response to only include claims supported by the documents. "
        "Rephrase partially supported claims to match what the documents say. "
        "If nothing is supported, respond only with: \"I don't know the answer to the question.\"\n"
        "Return ONLY the corrected assistant response without any explanation, preamble, or meta-commentary."
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
            options=["IntrinsicsAPI (granite-common + RITS)", "Ollama", "LocalHF"],
            value="IntrinsicsAPI (granite-common + RITS)",
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
            info="Current user query",
            required=True,
        ),
        DataFrameInput(
            name="message_history",
            display_name="Message History",
            info="Previous conversation messages as a DataFrame with 'sender' (User/AI) and 'text' columns",
            required=False,
        ),
        MessageInput(
            name="agent_response",
            display_name="Agent Response",
            info="The agent's response to check for hallucinations (this will be the last turn)",
            required=True,
        ),
        DataInput(
            name="documents",
            display_name="Documents",
            info="Retrieved documents to verify claims against",
            required=True,
        ),
    ]

    outputs = [
        Output(
            name="corrected_response",
            display_name="Corrected Response",
            method="remove_hallucinations",
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
        """Build conversation text including history and current turn"""
        conversation_lines = []

        # Get current user input first
        if hasattr(self, 'chat_input') and self.chat_input is not None:
            current_query = self._extract_message_text(self.chat_input)
            current_query_text = current_query.strip() if current_query else ""
        else:
            current_query_text = ""

        # Get agent response
        if hasattr(self, 'agent_response') and self.agent_response is not None:
            agent_text = self._extract_message_text(self.agent_response)
            agent_response_text = agent_text.strip() if agent_text else ""
        else:
            agent_response_text = ""

        # Add chat history from DataFrame (excluding current message to avoid duplicates)
        if hasattr(self, 'message_history') and self.message_history is not None:
            df = self.message_history
            for _, row in df.iterrows():
                sender = row.get("sender", "")
                content = row.get("text", "")
                content_str = str(content).strip() if content else ""

                # Skip if this is the current message (avoid duplicates from Memory component)
                if content_str and content_str not in [current_query_text, agent_response_text]:
                    # Format as "User:" or "Agent:" based on sender
                    role_label = "User" if sender == "User" else "Agent"
                    conversation_lines.append(f"{role_label}: {content_str}")

        # Add current user input
        if current_query_text:
            conversation_lines.append(f"User: {current_query_text}")

        # Add agent response as the last turn
        if agent_response_text:
            conversation_lines.append(f"Agent: {agent_response_text}")

        return "\n".join(conversation_lines)

    def _extract_documents(self) -> list:
        """Extract document texts from documents input"""
        documents = []
        if hasattr(self, 'documents') and self.documents:
            if isinstance(self.documents, Data):
                data_dict = self.documents.data if hasattr(self.documents, 'data') else self.documents
                if isinstance(data_dict, dict):
                    docs = data_dict.get("documents")
                    if docs and isinstance(docs, list):
                        documents = docs
            elif isinstance(self.documents, dict):
                docs = self.documents.get("documents")
                if docs and isinstance(docs, list):
                    documents = docs
            elif isinstance(self.documents, list):
                documents = self.documents

        # During build phase, return empty list instead of raising error
        if not documents:
            return []

        # Extract text from documents
        document_texts = []
        for doc in documents:
            if isinstance(doc, dict):
                text = doc.get("text") or doc.get("content") or doc.get("page_content") or str(doc)
            else:
                text = str(doc)
            document_texts.append(text)

        return document_texts

    def _build_prompt(self, conversation: str, documents: str) -> str:
        """Build the full prompt with the conversation and documents"""
        prompt = self.HALLUCINATION_PROMPT.format(conversation=conversation, documents=documents)
        return prompt

    def _load_model(self):
        """Lazy load the model and tokenizer"""
        if self._model is None or self._tokenizer is None:
            # Map model name to HuggingFace identifier
            hf_model_name = self.MODEL_NAME_MAPPING.get(self.model_name, self.model_name)
            self.log(f"Loading model: {self.model_name} -> {hf_model_name}", name="model loading")

            self._tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                hf_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self._model.eval()
            self.log(f"Model loaded on device: {self._model.device}", name="model loading")

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

    def _build_chat_messages_for_api(self, conversation: str, documents: str) -> list:
        """Build chat messages for API call"""
        messages = []

        # Apply the hallucination removal prompt template
        prompt = self.HALLUCINATION_PROMPT.format(conversation=conversation, documents=documents)

        # Create a single user message with the formatted prompt
        messages.append({"role": "user", "content": prompt})

        return messages

    def _build_request(self, conversation: str, documents: str) -> dict:
        """Build request body for API"""
        messages = self._build_chat_messages_for_api(conversation, documents)
        if not messages:
            raise ValueError("No messages to send")
        request_body = {
            "messages": messages,
            "model": self.model_name,
        }
        # Add temperature if provided
        if hasattr(self, 'temperature') and self.temperature is not None:
            request_body["temperature"] = self.temperature
        return request_body

    def _get_api_url(self) -> str:
        """Construct API URL based on backend"""
        if self.model_backend == "IntrinsicsAPI (granite-common + RITS)":
            # Call the model endpoint directly (not the intrinsic endpoint)
            return f"{self.url.rstrip('/')}/{self.model_name}"
        elif self.model_backend == "Ollama":
            base_url = self.url.rstrip('/')
            return f"{base_url}/v1/chat/completions"
        else:
            raise ValueError(f"Unsupported backend for API URL: {self.model_backend}")

    def _call_intrinsics_api(self, conversation: str, documents: str) -> dict:
        """Call the IntrinsicsAPI endpoint"""
        if not self.url or not self.url.strip():
            raise ValueError("API URL is required for IntrinsicsAPI backend")

        request_body = self._build_request(conversation, documents)

        self.log(json.dumps(request_body, indent=2), name="Intrinsics API request")
        api_url = self._get_api_url()

        headers = {"Content-Type": "application/json"}
        # Only include RITS_API_KEY header if provided
        if self.model_backend == "IntrinsicsAPI (granite-common + RITS)":
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
        self.log(json.dumps(response_data, indent=2), name="Intrinsics API response")
        return response_data

    def _call_ollama_api(self, conversation: str, documents: str) -> dict:
        """Call Ollama API using OpenAI-compatible endpoint"""
        try:
            request_body = self._build_request(conversation, documents)

            self.log(json.dumps(request_body, indent=2), name="Ollama request")
            api_url = self._get_api_url()
            client = OpenAI(
                base_url=api_url.replace("/chat/completions", ""),
                api_key=self.api_key or "no-key",
            )
            completion = client.chat.completions.create(**request_body)
            response_data = json.loads(completion.model_dump_json())
            self.log(json.dumps(response_data, indent=2), name="Ollama response")
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

    def _call_backend(self, conversation: str, documents: str) -> str:
        """Call the appropriate backend (LocalHF, Ollama, or IntrinsicsAPI)"""
        if self.model_backend == "LocalHF":
            return self._call_localhf_backend(conversation, documents)
        elif self.model_backend == "Ollama":
            return self._call_ollama_backend(conversation, documents)
        elif self.model_backend == "IntrinsicsAPI (granite-common + RITS)":
            return self._call_intrinsics_backend(conversation, documents)
        else:
            raise ValueError(f"Unsupported backend: {self.model_backend}")

    def _call_localhf_backend(self, conversation: str, documents: str) -> str:
        """Call local HuggingFace model with the constructed prompt"""
        prompt = self._build_prompt(conversation, documents)

        self.log(prompt, name="LocalHF input prompt")

        # Use chat messages format with chat template (Granite models expect this)
        messages = [{"role": "user", "content": prompt}]
        response_text = self._invoke_with_chat_messages(messages)

        self.log(response_text, name="LocalHF output")
        return response_text

    def _call_ollama_backend(self, conversation: str, documents: str) -> str:
        """Call Ollama endpoint"""
        response_data = self._call_ollama_api(conversation, documents)
        content = self._extract_content_from_response(response_data)

        if not content:
            raise ValueError("No content received from Ollama")

        self.log(content, name="Ollama corrected response")
        return content

    def _call_intrinsics_backend(self, conversation: str, documents: str) -> str:
        """Call IntrinsicsAPI endpoint"""
        response_data = self._call_intrinsics_api(conversation, documents)
        content = self._extract_content_from_response(response_data)

        if not content:
            raise ValueError("No content received from IntrinsicsAPI")

        self.log(content, name="Intrinsics API corrected response")
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
                if field_value == "IntrinsicsAPI (granite-common + RITS)":
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

    def remove_hallucinations(self) -> Message:
        """Main method that returns the corrected response with hallucinations removed"""
        # Handle case where inputs are not yet connected (build phase)
        if not hasattr(self, 'agent_response') or self.agent_response is None:
            self.log("No agent_response - returning empty", name="hallucination removal")
            return Message(text="")

        if not hasattr(self, 'documents') or self.documents is None:
            self.log("No documents - returning empty", name="hallucination removal")
            return Message(text="")

        # Build conversation text
        conversation_text = self._build_conversation_text()
        if not conversation_text:
            self.log("Empty conversation - returning empty", name="hallucination removal")
            return Message(text="")

        self.log(f"Conversation:\n{conversation_text[:200]}...", name="hallucination removal input")

        # Extract documents
        document_texts = self._extract_documents()
        if not document_texts:
            self.log("No documents extracted - returning original response", name="hallucination removal")
            agent_text = self._extract_message_text(self.agent_response)
            return Message(text=agent_text if agent_text else "")

        # Format documents as context
        documents_str = "\n\n".join(document_texts)

        # Call backend to remove hallucinations (will print MODEL INPUT)
        corrected = self._call_backend(conversation_text, documents_str)

        if not corrected:
            raise ValueError("No corrected response received from model")

        self.log(f"Corrected Response: '{corrected[:100]}...'", name="hallucination removal output")
        return Message(text=corrected)

    def get_full_response(self) -> str:
        """Return the full response as JSON"""
        # Handle case where inputs are not yet connected (build phase)
        if not hasattr(self, 'agent_response') or self.agent_response is None:
            return json.dumps({"error": "agent_response not connected"}, indent=2)

        if not hasattr(self, 'documents') or self.documents is None:
            return json.dumps({"error": "documents not connected"}, indent=2)

        # Build conversation text
        conversation_text = self._build_conversation_text()
        if not conversation_text:
            return json.dumps({"error": "conversation is empty"}, indent=2)

        # Extract documents
        document_texts = self._extract_documents()
        if not document_texts:
            return json.dumps({"error": "no documents available"}, indent=2)

        # Format documents as context
        documents_str = "\n\n".join(document_texts)

        # Call backend to remove hallucinations
        corrected = self._call_backend(conversation_text, documents_str)

        # Get original response
        original = self._extract_message_text(self.agent_response)

        # Format as JSON for compatibility
        response_data = {
            "original_response": original,
            "corrected_response": corrected,
            "model": self.model_name,
            "backend": self.model_backend
        }
        return json.dumps(response_data, indent=2)
