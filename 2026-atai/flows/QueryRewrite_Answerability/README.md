# Query Rewrite + Answerability Detection RAG Flow

## Overview

This folder contains two Langflow flows configured for government domain documents, with the goal of demonstrating the pitfalls with a fully prompt-based RAG flow available on the LangChain tutorial page [LangChain Agentic RAG tutorial](https://docs.langchain.com/oss/python/langgraph/agentic-rag) and then swapping its components with corresponding intrinsics (specialized trained models for the purpose), and demonstrating why it's better. This uses two key components:

1. **Query Rewriting (QR)**: Query reformulation to resolve coreference and make query standalone for better retrieval
2. **Answerability Detection (AD)**: Evaluating whether retrieved documents contain sufficient evidence to answer the query

### Configuration

- **Retriever**: `sentence-transformers/multi-qa-mpnet-base-dot-v1` embeddings
- **Collection**: `mt-rag-govt`
- **Models used**: `ibm-granite/granite-4.0-micro`, `ibm-granite/granite-4.0-h-small`, and `openai/gpt-oss-20b`

### Flow Architecture

```
User Query w/ MT conversation history
    ↓
Query Rewrite
    ↓
Vector Search
    ↓
Answerability Check
    ↓
    ├─→ Documents Relevant → Generate Answer
    │
    └─→ Documents Not Relevant → Return "I don't know the answer to the question"
```

---

## Methods

### 1. Prompt-Based
**File**: `QueryRewrite_Answerability (Prompted with LangChain Tutorial).json`

The [LangChain Agentic RAG tutorial](https://docs.langchain.com/oss/python/langgraph/agentic-rag) goes into cycle of deciding, whether to rewrite the question and re-query or to simply generate based on the already retrieved context  with the help of a context grader. This cycle goes into an infinite loop whenever it encounters an unanswerable question for the corpus. We break this cycle with the model abstaining to respond if the grader deemed the retrieved context to be not relevant for the query. This implementation uses prompt-based approaches for QR and AD using the same prompts as the LangChain tutorial.

**Components**:
- Query Rewrite (Prompt)
- Grade Documents (Prompt)
- Base Model Chat
- ChromaDB Search with Local Embeddings

**Prompts**:

*Query Rewrite Prompt*:
```
Look at the input and try to reason about the underlying semantic intent / meaning.
Here is the initial question:
 -------
{question}
 -------
Formulate an improved question:
```

*Answerability Determination Prompt*:
```
You are a grader assessing relevance of a retrieved document to a user question.
Here is the retrieved document:

{context}

Here is the user question: {question}
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
```


### 2. Our Intrinsic Models
**File**: `QueryRewrite_Answerability (Intrinsic).json`

This implementation uses specialized trained models for QR and AD.

**Components**:
- Query Rewrite (Intrinsic)
- Answerability (Intrinsic)
- Base Model Chat
- ChromaDB Search with Local Embeddings

---

## Examples

### Example 1: Retrieval Failure from Over-Specific Query Rewriting

**Scenario**: The prompt-based query rewrite adds excessive context from conversation history, leading to overly specific queries that fail to retrieve relevant documents. The intrinsic model's minimal rewrite approach maintains the core question while keeping it general enough to retrieve relevant information.

**Models used**: `granite-4.0-micro`, `granite-4.0-h-small`

**Conversation**:
- **Turn 1**: `Which county did the Midway Fire occur in?`
- **Turn 2**: `what role does FEMA play? and does it cover san diego county?`

**What Happened**:

Both flows successfully answered Turn 1 with `The Midway Fire occurred in Kern County.` This establishes the wildfire context for the conversation.

However, in Turn 2, the approaches diverged:

**Prompt-Based Flow (Turn 2)**:
- **Query Rewrite**: `"What is FEMA's role in wildfire response, and does it provide coverage for wildfires in San Diego County?"`
- The rewrite pulls in excessive context from Turn 1, adding "wildfire response" and "wildfires in San Diego County"
- **Retrieval Result**: No FEMA documents retrieved
- **Answer**: `"I don't know the answer to the question"`
- **Result**: The overly specific query matches wildfire response related topics instead of FEMA documents, causing the system to abstain

**Intrinsic Flow (Turn 2)**:
- **Query Rewrite**: `"What role does FEMA play in wildfire recovery and does it cover San Diego County?"`
- The minimal rewrite preserves the core question with appropriate wildfire context without over-specifying
- **Retrieval Result**: Relevant FEMA documents retrieved successfully
- **Answer** (with `granite-4.0-micro`):
```
FEMA, or the Federal Emergency Management Agency, is a U.S. government organization that is responsible for coordinating the response to natural and human-caused disasters within the United States. Their mission is to reduce the loss of life and property and protect the nation from all hazards by leading and supporting the nation in a risk-based, comprehensive emergency management system of preparedness, protection, response, recovery, and mitigation. FEMA does cover San Diego County, as it is a part of the U.S. They provide assistance and resources to help the county prepare for, respond to, and recover from disasters.
```
- **Result**: Successfully answers the question with grounded information from retrieved documents

*Note: This example was tested with both `granite-4.0-micro` and `granite-4.0-h-small`. The prompt-based flow returned "I don't know the answer to the question" with both models, while the intrinsic flow successfully retrieved and answered with both models.*

**Key Insight**:

The prompt-based approach's tendency to over-incorporate conversational context during query rewriting can backfire by making queries too specific for the corpus. This can cause prompt-based rewrite to incorrectly narrow the search space too much and miss relevant documents. The intrinsic model's minimal rewrite strategy focuses on making the query standalone without adding speculative details, leading to more robust retrieval.

---

### Example 2: Answerability Detection Failure on Cross-Document Comparison

**Scenario**: The prompt-based answerability detection fails to recognize when information can be synthesized across multiple documents to answer a question. The intrinsic model understands that comparing information from different documents can provide valid answers.

**Models used**: `granite-4.0-micro`, `granite-4.0-h-small`

**Conversation**:
- **Turn 1**: `Which county did the Midway Fire occur in?`
- **Turn 2**: `was it the same place as avocado fire?`

**What Happened**:

Both flows successfully answered Turn 1 with `The Midway Fire occurred in Kern County.` However, in Turn 2, the approaches diverged:

**Prompt-Based Flow (Turn 2)**:
- **Retrieval Result**: Relevant documents about both Midway Fire and Avocado Fire were retrieved
- **Answerability Determination**: Despite having the necessary information in retrieved documents, the system abstains because the grader cannot synthesize cross-document information
- **Answer**: `"I don't know the answer to the question"`

**Intrinsic Flow (Turn 2)**:
- **Retrieval Result**: Relevant documents about both Midway Fire and Avocado Fire were retrieved
- **Answerability Determination**: Recognized that comparing location information from the documents can answer the question
- **Answer** (with `granite-4.0-micro`):
```
No, the Midway Fire and the Avocado Fire did not occur in the same place. The Midway Fire was located in Kern County, while the Avocado Fire was near Avocado Mesa Road in La Cresta, also in Riverside County.
```

*Note: This example was tested with both `granite-4.0-micro` and `granite-4.0-h-small`. The prompt-based flow returned "I don't know the answer to the question" with both models, while the intrinsic flow successfully synthesized information across documents and answered with both models.*

**Key Insight**:

Prompt-based answerability detection often fails when a question requires synthesizing or comparing information across multiple documents. The intrinsic model understands that answering questions often requires reasoning across multiple pieces of evidence, enabling it to handle comparative questions and cross-document synthesis naturally.

---

### Example 3: Answerability Detection Failure on Negation and Inference

**Scenario**: The prompt-based answerability detection only determines based on what's explicitly stated in the documents. The intrinsic flow, however, makes reasonable, qualified inferences from available information to provide helpful answers.

**Models used**: `granite-4.0-micro`, `granite-4.0-h-small`

**Conversation**:
- **Turn 1**: `When did the Midway Fire start?`
- **Turn 2**: `When did the Midway Fire NOT start?`

**What Happened**:

Both flows successfully answered Turn 1 with `The Midway Fire started on November 12, 2023.` However, in Turn 2, the approaches diverged:

**Prompt-Based Flow (Turn 2)**:
- **Retrieval Result**: Documents containing the Midway Fire start date were retrieved
- **Answerability Determination**: Failed to recognize that knowing when the fire started allows inferring when it did not start. Despite having the necessary information, the system abstains because the grader looks for explicit negative statements rather than making logical inferences
- **Answer**: `"I don't know the answer to the question"`

**Intrinsic Flow (Turn 2)**:
- **Retrieval Result**: Documents containing the Midway Fire start date were retrieved
- **Answerability Determination**: Recognized that the affirmative statement allows answering the negation through logical inference. Successfully uses logical inference to answer the negation question
- **Answer** (with `granite-4.0-micro`): `The Midway Fire did not start on any date other than November 12, 2023.`

*Note: This example was tested with both `granite-4.0-micro` and `granite-4.0-h-small`. The prompt-based flow returned "I don't know the answer to the question" with both models, while the intrinsic flow successfully inferred the answer with both models.*

**Key Insight**:

The prompt-based answerability detection is strictly literal—it only determines answerability based on what's explicitly stated in documents. When documents describe what is covered but don't explicitly list what is NOT covered, it refuses to answer. The intrinsic flow makes qualified inferences from available information.

---

### Example 4: Hallucination Beyond Retrieved Information

**Scenario**: When documents are topically relevant but lack specific details, the prompt-based grader may deem the question answerable based on "general common knowledge," causing the model to hallucinate rather than answering based on corpus. The intrinsic model enforces strict document grounding.

**Models used**: `openai/gpt-oss-20b`

**Query**: `How often do large dust storms cover the entire planet on Mars?`

**What's in Corpus**: A NASA document mentioning "Mars" large storms sometimes cover the entire planet" - but NO information about storm frequency or timing

**What Happened**:

**Prompt-Based Flow (with `gpt-oss-20b`)**:
- **Retrieval Result**: Retrieved the Mars NASA document containing information about dust storms
- **Answerability Determination**: Incorrectly graded as answerable because the grader likely considers "general common knowledge" acceptable, not just document content.
- **Answer** (truncated):
```
Large dust storms that blanket the entire Martian surface—often called global dust storms—are relatively rare events. Based on observations from orbiters and landers over the past few decades, the consensus is:

Typical Frequency: Global dust storm roughly once every 2–3 Martian years (≈ 1–1.5 Earth years)
Duration: They can persist for weeks to a few months once they start
Seasonal trigger: Most global storms begin in the southern hemisphere's summer...
```

**Intrinsic Flow**:
- **Retrieval Result**: Retrieved the Mars NASA document
- **Answerability Determination**: Correctly identified that the specific frequency information is not present in the corpus
- **Answer**: `"I don't know the answer to the question"`
- **Result**: Properly abstained, enforcing strict document grounding rather than allowing general knowledge

**Key Insight**:

When documents mention a topic (Mars dust storms) but lack the specific information requested (frequency), the grader may pass answerability based on the model's background knowledge rather than document content. This defeats the core purpose of RAG: ensuring answers are grounded in the provided corpus. The intrinsic answerability model distinguishes between topical relevance and true answerability, ensuring the system abstains when the corpus lacks the specific information requested rather than allowing the generation model to hallucinate from its parametric knowledge.

---

## Notes

The answerability detection operates before generation, determining whether to answer or abstain, but does not validate the generated response after it's created.

For handling mixed hallucination scenarios where some portions of the answer are accurate while others are fabricated, refer to [Hallucination Mitigation](../Hallucination_Mitigation/README.md), which uses post-generation hallucination detection to identify and correct fabricated content in model responses.

---

