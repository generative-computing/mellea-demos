"""
Grade Documents Component for Langflow

Assess relevance of retrieved documents to a user question.
Uses local HF backend (Granite model) for document relevance assessment.
"""

from langflow.custom import Component
from langflow.io import MessageInput, HandleInput, Output, SecretStrInput, StrInput, DropdownInput, DataFrameInput, SliderInput
from langflow.schema import Message, Data
from langflow.field_typing.range_spec import RangeSpec
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import httpx
import json
from openai import OpenAI


GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


class GradeDocumentsSchema(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class GradeDocuments(Component):
    display_name = "Grade Documents"
    description = "Assess relevance of retrieved documents using local Granite model"
    category = "tools"
    icon = "check-circle"

    intrinsic_name = "answerability"

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
            info="Documents to grade for relevance (accepts Data from vector stores)",
            input_types=["Data", "Search Results"],
            required=True,
            is_list=True,
        ),
    ]

    outputs = [
        Output(
            name="response",
            display_name="Response",
            method="get_response",
        ),
    ]

    def _extract_message_text(self, message: Message) -> str:
        """Extract text from Message object"""
        if hasattr(message, 'text'):
            return message.text
        if hasattr(message, 'content'):
            return message.content
        return str(message)

    def _build_conversation_text(self) -> str:
        """Build conversation text for grading prompt"""
        conversation_lines = []

        # Get current user input first to filter it from history
        current_query = self._extract_message_text(self.chat_input)
        if not current_query or not current_query.strip():
            raise ValueError("Chat input is required")
        current_query_text = current_query.strip()

        # Add chat history from DataFrame (excluding current message to avoid duplicates)
        if hasattr(self, 'message_history') and self.message_history is not None:
            df = self.message_history
            for _, row in df.iterrows():
                role = row.get("role", "")
                content = row.get("content", "")
                content_str = str(content).strip() if content else ""
                # Skip if this is the current message (avoid duplicates from Memory component)
                if content_str and content_str != current_query_text:
                    # Format as "User:" or "Assistant:" based on role
                    role_label = "User" if role == "user" else "Assistant"
                    conversation_lines.append(f"{role_label}: {content_str}")

        # Add current user input
        conversation_lines.append(f"User: {current_query_text}")

        return "\n".join(conversation_lines)

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

    def _parse_binary_score(self, response_text: str) -> str:
        """Parse yes/no from model response

        Looks for 'yes' or 'no' as standalone words at the start of the response,
        not just anywhere in the text (to avoid matching 'no' in 'information', etc.)
        """
        import re

        response_text = response_text.strip()

        # Check first 50 characters for standalone yes/no
        first_part = response_text[:50].lower()

        # Match yes/no as whole words at the beginning (with word boundaries)
        if re.search(r'\byes\b', first_part):
            return 'yes'
        elif re.search(r'\bno\b', first_part):
            return 'no'

        # Fallback: check if first non-empty line starts with yes/no
        first_line = response_text.split('\n')[0].strip().lower()
        if first_line.startswith('yes'):
            return 'yes'
        elif first_line.startswith('no'):
            return 'no'

        # Default to yes if unclear
        return 'yes'

    def _build_chat_messages_for_api(self, conversation_text: str, context: str) -> list:
        """Build chat messages for API call"""
        messages = []

        # Apply the grading prompt template
        prompt = GRADE_PROMPT.format(question=conversation_text, context=context)

        # Create a single user message with the formatted prompt
        messages.append({"role": "user", "content": prompt})

        return messages

    def _build_request(self, conversation_text: str, context: str) -> dict:
        """Build request body for IntrinsicsAPI"""
        messages = self._build_chat_messages_for_api(conversation_text, context)
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
            # This sends the GRADE_PROMPT to the base model
            return f"{self.url.rstrip('/')}/{self.model_name}"
        elif self.model_backend == "Ollama":
            base_url = self.url.rstrip('/')
            return f"{base_url}/v1/chat/completions"
        else:
            raise ValueError(f"Unsupported backend for API URL: {self.model_backend}")

    def _call_intrinsics_api(self, conversation_text: str, context: str) -> dict:
        """Call the IntrinsicsAPI endpoint"""
        if not self.url or not self.url.strip():
            raise ValueError("API URL is required for IntrinsicsAPI backend")

        request_body = self._build_request(conversation_text, context)
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

    def _call_ollama_api(self, conversation_text: str, context: str) -> dict:
        """Call Ollama API using OpenAI-compatible endpoint"""
        try:
            request_body = self._build_request(conversation_text, context)

            # Print model input
            print(f"\n{'='*80}")
            print(f"[Grade Documents - Ollama] MODEL INPUT:")
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
            print(f"[Grade Documents - Ollama] MODEL OUTPUT:")
            print(f"{'='*80}")
            print(json.dumps(response_data, indent=2))
            print(f"{'='*80}\n")

            return response_data
        except Exception as e:
            raise ValueError(f"Ollama call failed: {e}")

    def _extract_content_from_response(self, response_data: dict) -> str:
        """Extract content from API response"""
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

    def get_response(self) -> Message:
        """Grade documents and return relevance score"""
        print("\n" + "📊" * 40)
        print(">>> STAGE 3: GRADE DOCUMENTS COMPONENT EXECUTED")
        print("📊" * 40)

        # Handle build phase where inputs might not be set
        if not hasattr(self, 'chat_input') or self.chat_input is None:
            print("⚠️  No chat_input - returning empty")
            return Message(text="")
        if not hasattr(self, 'documents') or self.documents is None:
            print("⚠️  No documents - returning empty")
            return Message(text="")

        # Check if chat_input has valid content
        try:
            current_query = self._extract_message_text(self.chat_input)
            if not current_query or not current_query.strip():
                return Message(text="")
        except Exception:
            return Message(text="")

        # Build conversation text
        conversation_text = self._build_conversation_text()

        # Extract documents
        document_texts = self._extract_documents()

        # If no documents during build phase, return empty
        if not document_texts:
            return Message(text="")

        # Format documents as context
        context = "\n\n".join(document_texts)

        # Route to appropriate backend
        if self.model_backend == "LocalHF":
            score = self._grade_with_localhf(conversation_text, context)
        elif self.model_backend == "Ollama":
            score = self._grade_with_ollama(conversation_text, context)
        elif self.model_backend == "IntrinsicsAPI (mellea + RITS)":
            score = self._grade_with_intrinsics_api(conversation_text, context)
        else:
            raise ValueError(f"Unsupported backend: {self.model_backend}")

        return Message(text=score)

    def _grade_with_localhf(self, conversation_text: str, context: str) -> str:
        """Grade documents using LocalHF backend"""
        # Build grading prompt
        prompt = GRADE_PROMPT.format(question=conversation_text, context=context)

        print("\n" + "=" * 80)
        print("INPUT PROMPT FOR GRADING:")
        print(prompt)
        print("=" * 80)

        # Generate response using chat template
        messages = [{"role": "user", "content": prompt}]
        response_text = self._invoke_with_chat_messages(messages)

        # Parse binary score
        score = self._parse_binary_score(response_text)

        print("=" * 80)
        print("OUTPUT FOR GRADING:")
        print(f"Binary Score: {score}")
        print(f"Raw Response: {response_text}")
        print("=" * 80 + "\n")

        return score

    def _grade_with_ollama(self, conversation_text: str, context: str) -> str:
        """Grade documents using Ollama backend"""
        response_data = self._call_ollama_api(conversation_text, context)
        content = self._extract_content_from_response(response_data)

        if not content:
            raise ValueError("No content received from Ollama")

        # Parse binary score
        score = self._parse_binary_score(content)

        print("\n" + "=" * 80)
        print("OLLAMA API RESPONSE:")
        print(f"Binary Score: {score}")
        print(f"Raw Response: {content}")
        print("=" * 80 + "\n")

        return score

    def _grade_with_intrinsics_api(self, conversation_text: str, context: str) -> str:
        """Grade documents using IntrinsicsAPI backend"""
        response_data = self._call_intrinsics_api(conversation_text, context)
        content = self._extract_content_from_response(response_data)

        if not content:
            raise ValueError("No content received from IntrinsicsAPI")

        # Parse binary score
        score = self._parse_binary_score(content)

        print("\n" + "=" * 80)
        print("INTRINSICS API RESPONSE:")
        print(f"Binary Score: {score}")
        print(f"Raw Response: {content}")
        print("=" * 80 + "\n")

        return score
