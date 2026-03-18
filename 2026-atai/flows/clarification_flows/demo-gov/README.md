# Gov Domain RAG Flows with Query Clarification

## Overview

This folder contains two contrasting Langflow flows configured for government domain documents, with the goal of demonstrating the benefits of incorporating query clarification capabilities into RAG:

1. **Regular RAG with Intrinsics - Gov.json**: RAG flow with intrinsics (including Citation) but without query clarification
2. **Query Clarification (post-retriever) RAG - Gov.json**: RAG flow enhanced with the Query Clarification intrinsic

### Configuration

- **Retriever**: ELSER (Elastic Learned Sparse Encoder Retrieval)
- **Collection**: `mt-rag-gov`
- **Model Backend**: Ollama (default), using `http://host.docker.internal:11434`

### Key Difference

The Query Clarification flow applies the Query Clarification intrinsic **after retrieval** but **before answering**. This allows the system to identify when critical context is missing from the user's query and ask clarifying questions to provide accurate, location-specific guidance.

---

## Running Example: DMV License Renewal Query

### User Query
> "My driver's license is about to expire. What do I need to renew it at the DMV?"

This query is missing critical context - DMV requirements vary significantly by state, and the user hasn't specified their location.

### RAG Flow Behavior without Clarification

The Regular RAG flow retrieves documents about driver's license renewal from various states but provides a generic answer without asking which state the user is in:

```
USER: My driver's license is about to expire. What do I need to renew it at the DMV?

ASSISTANT: To renew your driver's license at the DMV, you will need to provide proof
of identity, residency, and Social Security number. If you are renewing by mail, you
will need to submit a renewal application, a photo of your license, and payment. If
you are renewing in person, you will need to bring your current license, proof of
identity, residency, and Social Security number, and pay the renewal fee.
```

The final response is a generic answer that may not apply to the specific state where the user is in. Furthermore, when taking a closer look at the quality of the response using the **Citation intrinsic**, it can be seen that the model response is an aggregation of facts that are specific to one state (e.g., California) with other facts that are specific to a different state (e.g., New York).

#### Citation Analysis

The Citation intrinsic annotates the response with references to the source documents:

```
To renew your driver's license at the DMV, you will need to provide proof of
identity, residency, and Social Security number. [1] [2] If you are renewing
by mail, you will need to submit a renewal application, a photo of your
license, and payment. If you are renewing in person, you will need to bring
your current license, proof of identity, residency, and Social Security
number, and pay the renewal fee. [2]

Sources:
[1] California DMV - "Renew Your Vehicle's Registration"
[2] New York DMV - "MyDMV"
```

This reveals that the model combined information from both California DMV and New York DMV documents into a single response, even though these states have different requirements. Without clarifying the user's location, the system cannot provide accurate, state-specific guidance.

### Query Clarification Flow Behavior

The Query Clarification flow recognizes that state-specific information is required to answer accurately and asks a clarifying question:

```
USER: My driver's license is about to expire. What do I need to renew it at the DMV?

ASSISTANT: There are many DMVs, like the California DMV, which offers online services
for renewing your driver's license, and the New York DMV, which also provides online
renewal options. Which one are you referring to?

USER: I am in upstate NY

ASSISTANT: If you are in upstate New York and your driver's license is about to expire,
you can renew it online through the New York DMV's MyDMV service. You will need to
provide proof of identification and take a written test. If you are unsure about the
specific requirements or the process, you can visit the New York DMV's website or
contact them directly for assistance.
```

By asking for the user's location first, the system provides accurate, state-specific guidance rather than generic information that may not apply.
