# General Domain (ClapNQ) RAG Flows with Query Clarification

## Overview

This folder contains three Langflow flows configured for general domain documents (from ClapNQ), demonstrating different approaches to handling ambiguous queries in RAG:

1. **RAG no Clarification - ClapNQ.json**: RAG flow with intrinsics but without query clarification
2. **Query Clarification (Intrinsic) - ClapNQ.json**: RAG flow enhanced with a trained Query Clarification intrinsic
3. **Query Clarification (Prompted) - ClapNQ.json**: RAG flow using prompt-based clarification with a base model

### Configuration

- **Retriever**: ELSER (Elastic Learned Sparse Encoder Retrieval)
- **Collection**: `mt-rag-clapnq`
- **Models**: IBM internal RITS deployment

### Key Differences

The Query Clarification flow applies the trained Query Clarification intrinsic **after retrieval** but **before answering**. This allows the system to identify ambiguity in the user's query based on the retrieved documents and ask clarifying questions when needed. The RAG prompted Clarification flow attempts to do something similar by just prompting the base model (e.g., granite-4.0-micro or gpt-oss-20b) with explicit instructions to detect if ambiguity is present in the user query and ask back clarification questions to the user.

---

## Running Example: Ambiguous Query "when did the war end"

### User Query
> "when did the war end"

This query is inherently ambiguous - it doesn't specify which war the user is asking about.

### RAG Flow Behavior without Clarification

The Regular RAG flow retrieves documents about various wars but arbitrarily picks one to answer about (in this case, World War II) without acknowledging the ambiguity:

```
USER: when did the war end?

ASSISTANT: The war ended on V-E Day, 8 May 1945 in Europe and on V-J Day, 15 August
1945 in the Pacific.
```

The system assumed the user meant World War II and provided a specific answer. If the user actually meant a different war, this response would be incorrect and unhelpful.

### Query Clarification Flow Behavior

The Query Clarification flow, after retrieving documents about multiple wars, recognizes the ambiguity and asks the user to specify which war they mean:

```
USER: when did the war end?

ASSISTANT: There are many wars you could be referring to, such as: 1) World War II,
which ended with the armistice of 14 August 1945 (V-J Day) or the formal surrender
of Japan on 2 September 1945, 2) the European Theatre of World War II, which ended
with the German surrender to the Soviet Union and the Western Allies in late April
and early May 1945, 3) the Second Sino-Japanese War, which ended on 9 September 1945,
4) the Vietnam War, which ended with the French surrender at Dien Bien Phu on 7 May
1954, and 5) World War I, which ended with the signing of the Treaty of Versailles
on 28 June 1919. Which one are you talking about?

USER: I meant the US civil war

ASSISTANT: The American Civil War ended on May 9, 1865, when President Andrew Johnson
officially declared the end of the insurrection. However, the last Confederate surrender
occurred on November 6, 1865, when the Confederate warship CSS Shenandoah surrendered
at Liverpool, England.
```

By asking for clarification, the system engages in a meaningful conversation and provides an accurate answer about the war the user actually intended.

---

### Note on Corpus Content

The clarification response references "the Vietnam War, which ended with the French surrender at Dien Bien Phu on 7 May 1954." This refers to the First Indochina War (French colonial war), which is labeled as "Vietnam War" in the ClapNQ corpus documents - distinct from what Americans typically call "the Vietnam War" (which ended in 1975).

This illustrates an important aspect of RAG systems: the responses are grounded in the retrieved documents and faithfully reflect the corpus content. The purpose of this demo is to showcase the query clarification behavior - identifying ambiguity and asking clarifying questions - rather than to validate the factual accuracy of the underlying corpus.

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

Despite being **explicitly prompted to ask for clarification** and having access to documents about multiple wars, both models fail to follow the instruction and immediately assume World War II:

**granite-4.0-micro response:**
```
USER: when did the war end?

ASSISTANT: The war in Europe ended on V-E Day, 8 May 1945, with the German surrender.
The war in Asia ended on V-J Day, 15 August 1945, with Japan's surrender following
the atomic bombings of Hiroshima and Nagasaki. The formal end of World War II was
marked by the signing of peace treaties, with the final treaty with Japan signed in 1951.
```

**gpt-oss-20b response:**
```
USER: when did the war end?

ASSISTANT: The war ended on 15 August 1945 (V‑J Day), when Japan formally surrendered,
marking the conclusion of World War II.
```

### Observations

Both base models - even the larger gpt-oss-20b - pick WWII without acknowledging that the retrieved documents contain information about multiple wars. Neither model follows the explicit instruction to ask for clarification, despite the prompt clearly stating: *"If the question is ambiguous with unclear or multiple interpretations, do not attempt to answer it. Instead produce a clarifying request back to the user."*

This suggests that simply prompting base models to ask clarifying questions - even with explicit instructions - does not reliably produce this behavior. The models tend to default to answering based on their training patterns rather than following the clarification directive.

In contrast, the Query Clarification intrinsic (shown in the previous section) was trained specifically to:
1. Analyze retrieved documents for conflicting or multiple valid answers
2. Recognize when user queries are underspecified
3. Generate appropriate clarifying questions before providing an answer
