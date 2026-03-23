# General Domain (ClapNQ) RAG Flows with Query Clarification

## Overview

This folder contains three Langflow flows configured for general domain documents (from ClapNQ), demonstrating different approaches to handling ambiguous queries in RAG:

1. **RAG no Clarification - ClapNQ.json**: RAG flow with intrinsics but without query clarification
2. **Query Clarification (Intrinsic) - ClapNQ.json**: RAG flow enhanced with a trained Query Clarification intrinsic
3. **Query Clarification (Prompted) - ClapNQ.json**: RAG flow using prompt-based clarification with a base model

### Configuration

- **Retriever**: [ChromaDB](https://www.trychroma.com/) with `sentence-transformers/multi-qa-mpnet-base-dot-v1` embeddings
- **Collection**: `mt-rag-clapnq`
- **Models**: Ollama (default), using `http://host.docker.internal:11434`

### Key Differences

The Query Clarification flow applies the trained Query Clarification intrinsic **after retrieval** but **before answering**. This allows the system to identify ambiguity in the user's query based on the retrieved documents and ask clarifying questions when needed. The RAG prompted Clarification flow attempts to do something similar by just prompting the base model (e.g., granite-4.0-micro or gpt-oss-20b) with explicit instructions to detect if ambiguity is present in the user query and ask back clarification questions to the user.

---

## Running Example: Ambiguous Query "When was the university established in Washington?"

### User Query
> "When was the university established in Washington?"

This query is inherently ambiguous - it doesn't specify which university the user is asking about, as there are multiple universities in Washington. Additionally, "Washington" itself is ambiguous, as it could refer to Washington state or Washington, D.C.

### RAG Flow Behavior without Clarification

The Regular RAG flow retrieves documents about various universities but arbitrarily picks one to answer about (in this case, Howard University) without acknowledging the ambiguity:

```
USER: When was the university established in Washington?

ASSISTANT: The university established in Washington is Howard University, which was 
founded in 1867.
```

The system assumed the user meant Howard University and provided a specific answer. If the user actually meant a different university, this response would be incorrect and unhelpful.

### Query Clarification Flow Behavior

The Query Clarification flow, after retrieving documents about multiple universities, recognizes the ambiguity and asks the user to specify which university they mean:

```
USER: When was the university established in Washington?

ASSISTANT: There are many universities you could be referring to, such as 1) Howard
University, which was founded in Washington, D.C., in 1867, and 2) Trinity Western
University, which began its journey as a two-year college in 1962. Which one are you
talking about?

USER: I meant George Washington University

ASSISTANT: George Washington University was established in 1821. It is named after the
first President of the United States, George Washington. The university has its main
campus on Capitol Hill in Washington, D.C., and it offers a wide range of undergraduate
and graduate programs across various fields of study.
```

By asking for clarification, the system engages in a meaningful conversation and provides an accurate answer about the university the user actually intended.

---

### RAG prompted Clarification Flow

One might consider prompting an existing base model to ask clarifying questions when it encounters ambiguous queries. The **RAG prompted Clarification** flow explores this approach using a RAG prompt that **explicitly instructs the model to ask for clarification**:

```
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
If the question is ambiguous with unclear or multiple interpretations, do not attempt to answer it.
Instead produce a clarifying request back to the user, outlining the different possible
interpretations based on the given context.
Keep the answer concise.

Context:
{context}

Question: {question}

Answer:
```

Note that the prompt **explicitly tells the model** to identify ambiguous questions and then to ask for clarification instead of answering. This flow can be configured to use either `granite-4.0-micro` or a larger open-source model like `gpt-oss-20b`.

### Results with Base Models

Despite being **explicitly prompted to ask for clarification** and having access to documents about multiple universities, the model fails to follow the instruction and immediately assumes Howard University:

**granite-4.0-micro response:**
```
USER: When was the university established in Washington?

ASSISTANT: Howard University was founded in Washington, D.C., in 1867.
```

Similar behavior can be observed with both `granite4:micro` and `gpt-oss-20b` - they will not reliably clarify, and answer for the wrong choice.

Note: `gpt-oss-20b` can only be run through internal RITS, not Ollama.

### Observations

Base models - even larger ones like gpt-oss-20b - pick Howard University without acknowledging that the retrieved documents contain information about multiple universities. Neither model follows the explicit instruction to ask for clarification, despite the prompt clearly stating: *"If the question is ambiguous with unclear or multiple interpretations, do not attempt to answer it. Instead produce a clarifying request back to the user."*

This suggests that simply prompting base models to ask clarifying questions - even with explicit instructions - does not reliably produce this behavior. The models tend to default to answering based on their training patterns rather than following the clarification directive.

In contrast, the Query Clarification intrinsic (shown in the previous section) was trained specifically to:
1. Analyze retrieved documents for conflicting or multiple valid answers
2. Recognize when user queries are underspecified
3. Generate appropriate clarifying questions before providing an answer
