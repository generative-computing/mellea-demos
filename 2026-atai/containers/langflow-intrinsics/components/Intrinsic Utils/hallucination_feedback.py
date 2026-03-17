"""
Hallucination Feedback Component for Langflow

Constructs a feedback message based on hallucination detection results.
Follows pattern 3 from granite_3_3_composites.py where hallucination
feedback is used to create a user message for revision.

This component takes:
- Message history (optional) - Full conversation context from Memory
- Last user query (required)
- Agent response from first Base Model Chat (required)
- Hallucination feedback from detection (required)

Returns a DataFrame with complete conversation history plus three new messages:
1. Previous conversation history (if provided)
2. Last user query
3. Agent response
4. Hallucination feedback instruction
"""

from langflow.custom import Component
from langflow.io import (
    Output,
    MessageInput,
    DataFrameInput,
)
from langflow.schema import Message
from langflow.schema.dataframe import DataFrame
import pandas as pd


class HallucinationFeedback(Component):
    display_name = "Hallucination Feedback"
    description = "Create a feedback message for hallucination revision"
    category = "tools"
    icon = "message-square"

    inputs = [
        DataFrameInput(
            name="message_history",
            display_name="Message History",
            info="Full conversation history from Memory (optional)",
            required=False,
        ),
        MessageInput(
            name="last_user_query",
            display_name="Last User Query",
            info="The last user query",
            required=True,
        ),
        MessageInput(
            name="agent_response",
            display_name="Agent Response",
            info="Response from first Base Model Chat",
            required=True,
        ),
        MessageInput(
            name="hallucination_feedback",
            display_name="Hallucination Feedback",
            info="Raw hallucination feedback from detection",
            required=True,
        ),
    ]

    outputs = [
        Output(
            name="messages_dataframe",
            display_name="Messages DataFrame",
            method="get_messages_dataframe",
        ),
    ]

    def _extract_message_text(self, message) -> str:
        """Extract text content from a Message object."""
        if message is None:
            return ""
        # Handle DataFrame case
        if isinstance(message, pd.DataFrame):
            return ""
        # Handle Message objects
        if hasattr(message, 'text'):
            return message.text
        if hasattr(message, 'content'):
            return message.content
        return str(message)

    def get_messages_dataframe(self) -> DataFrame:
        """
        Return complete conversation history with appended messages as a DataFrame.

        This DataFrame contains:
        1. All previous conversation history (if message_history is provided)
        2. Last user query
        3. Agent response
        4. Hallucination feedback instruction

        Connect this output to the second Granite Base Model's message_history input.
        Leave chat_input empty on the second Base Model.
        """
        self.log(
            f"message_history: {type(self.message_history).__name__ if hasattr(self, 'message_history') else 'None'}, "
            f"last_user_query: {type(self.last_user_query).__name__}, "
            f"agent_response: {type(self.agent_response).__name__}, "
            f"hallucination_feedback: {type(self.hallucination_feedback).__name__}",
            name="input types",
        )

        # Start with existing message history if provided
        # Don't use boolean evaluation on DataFrame - use try/except instead
        history_df = None
        if hasattr(self, 'message_history'):
            try:
                history_df = self.message_history.copy()
                self.log(f"{len(history_df)} messages", name="message history")
            except (TypeError, AttributeError, ValueError):
                # None, invalid, or DataFrame boolean evaluation error - create empty DataFrame
                pass

        if history_df is None:
            history_df = pd.DataFrame(columns=["sender", "text"])
            self.log("No message history provided", name="message history")

        user_query = self._extract_message_text(self.last_user_query)
        agent_resp = self._extract_message_text(self.agent_response)
        raw_hals = self._extract_message_text(self.hallucination_feedback)

        self.log(
            f"User Query: {user_query[:100]}...\n"
            f"Agent Response: {agent_resp[:100]}...\n"
            f"Hallucination Feedback: {raw_hals[:100]}...",
            name="extracted text",
        )

        
        labeled_hals = raw_hals
        all_unfaithful = False

        inference_signals = [
            "does not explicitly state",
            "does not explicitly mention",
            "not explicitly state",
            "not explicitly mention",
            "is not explicitly",
            "reasonable to infer",
            "based on general knowledge",
            "general knowledge",
            "common knowledge",
            "not directly supported",
        ]

        def _label_from_explanation(explanation: str) -> str:
            exp_lower = explanation.lower()
            if any(sig in exp_lower for sig in inference_signals):
                return "unfaithful"
            return "faithful"

        try:
            import json
            hals_list = json.loads(raw_hals)
            for item in hals_list:
                score = item.get("faithfulness_likelihood")
                if score is not None and score >= 0.9:
                    item["faithfulness_label"] = "faithful"
                else:
                    explanation = item.get("explanation", "")
                    item["faithfulness_label"] = _label_from_explanation(explanation)
            # Reorder keys so faithfulness_label appears immediately after response_text
            reordered = []
            for item in hals_list:
                new_item = {}
                for k, v in item.items():
                    if k == "faithfulness_label":
                        continue
                    new_item[k] = v
                    if k == "response_text" and "faithfulness_label" in item:
                        new_item["faithfulness_label"] = item["faithfulness_label"]
                reordered.append(new_item)
            hals_list = reordered
            labeled_items = [item for item in hals_list if item.get("faithfulness_label") is not None]
            all_unfaithful = bool(labeled_items) and all(
                item["faithfulness_label"] == "unfaithful" for item in labeled_items
            )
            # Only pass faithful claims to the LLM — unfaithful response_text must not be visible
            faithful_only = [item for item in labeled_items if item["faithfulness_label"] == "faithful"]
            labeled_hals = json.dumps(faithful_only)
        except Exception:
            pass

        all_unfaithful_clause = " Since all claims lack direct document support, respond only with \"I don't know the answer to the question.\"" if all_unfaithful else ""

        feedback_instruction = f"""Based on the hallucination analysis below, revise only the last assistant response to include only claims that are directly supported by the provided document.

For each claim, examine its "faithfulness_likelihood" score and "explanation" field:
- Always keep the claim if its "faithfulness_likelihood" score is 0.9 or above.
- For claims below 0.9, keep the claim only if the explanation states the document directly supports or explicitly states the information.
- Remove the claim if the explanation states the document does not explicitly state the information, or if the explanation indicates the claim is supported by reasonable inference rather than the document, or if the explanation states the claim is based on general factual knowledge or common knowledge about the world rather than information present in the document.
You may rewrite or combine the kept claims into natural sentences, but do not introduce any new factual information not contained in the kept claims.

{labeled_hals}

Finally, examine each distinct part of only the last user question: {user_query}
- Do not answer anything else besides last user question.
- For any sub-question of the last-turn question that is not directly answered by the kept claims, address it naturally in the flow of the response by saying you don't know the answer to that specific part.
- If after removing unsupported claims the remaining content does not directly answer the last-turn user's question, respond only with "I don't know the answer to the question."

Write the full response as natural, flowing sentences. Do not repeat the question and follow the same writing style as the last agent response.  Do not provide any explanation or commentary about the document or the revision.{all_unfaithful_clause}

"""

        # Create DataFrame with three new messages
        new_messages_data = [
            {"sender": "User", "text": user_query},
            {"sender": "Machine", "text": agent_resp},
            {"sender": "User", "text": feedback_instruction},
        ]
        new_messages_df = pd.DataFrame(new_messages_data)

        # Concatenate history with new messages
        df = pd.concat([history_df, new_messages_df], ignore_index=True)

        self.log(f"Total messages: {len(df)}", name="output dataframe")

        # Convert DataFrame records to Message objects for Langflow DataFrame wrapper
        messages = []
        for _, row in df.iterrows():
            msg = Message(
                text=row["text"],
                sender=row["sender"]
            )
            messages.append(msg)

        # Return Langflow DataFrame wrapper (same as Memory component)
        return DataFrame(messages)
