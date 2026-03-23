# Gov Domain RAG Flows with Query Clarification

## Overview

This folder contains two contrasting Langflow flows configured for government domain documents, with the goal of demonstrating the benefits of incorporating query clarification capabilities into RAG:

1. **RAG no Clarification - Gov.json**: RAG flow with intrinsics (including Citation) but without query clarification
2. **Query Clarification (Intrinsic) - Gov.json**: RAG flow enhanced with the Query Clarification intrinsic

### Configuration

- **Retriever**: ChromaDB
- **Collection**: `mt-rag-gov`
- **Models**: Ollama (default), using `http://host.docker.internal:11434`

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

ASSISTANT: Renewing your driver's license typically involves several steps:

1. **Check Your License Status**: First, you should check if your current driver's
license has expired or will soon expire. You can usually find this information on
your state's Department of Motor Vehicles (DMV) website.

2. **Prepare Required Documents**: Most states require that you bring proof of identity
and residency when renewing a driver's license. This could include documents like a
birth certificate, passport, utility bill, or lease agreement. Check with your local
DMV for specific requirements.

3. **Schedule an Appointment**: Some DMVs allow walk-in renewals, while others may
require you to schedule an appointment in advance. You can usually find this
information on the DMV's website.

4. **Complete the Renewal Process**: This typically involves filling out a renewal form
and possibly taking a vision test or written test if your license has been expired for
more than a certain number of years.

5. **Pay the Fees**: There will be a fee to renew your driver's license, which can vary
by state. Some DMVs allow you to pay this fee in person when you get your new license,
while others require payment online or over the phone.

6. **Receive Your New License**: Once everything is complete, you'll receive your new
driver's license in the mail (usually within a few weeks).

Please note that these are general steps and may vary by state. It's always best to
check with your local DMV for specific requirements and procedures.
```

The response is generic and may not apply to the specific state where the user is located. In the absence of a specific state, the model response may mix facts from California DMV documents with facts from New York DMV documents, or even draw from its own general knowledge rather than the retrieved corpus. Thus, the response is not directly helpful.

### Query Clarification Flow Behavior

Based on the retrieved documents about driver's license renewal, the Query Clarification flow detects that there are multiple states that may apply to the user question, and hence the it recognizes that there is ambiguity based on the underlying content. As a result, it produces a clarification request back to the user that state-specific information is required to answer accurately:

```
USER: My driver's license is about to expire. What do I need to renew it at the DMV?

ASSISTANT: Which state are you referring to?

USER: I am in upstate New York

ASSISTANT: In Upstate New York, specifically in New York State, here are the general
steps to renew your driver's license:

1. **Prepare Required Documents**: You will need to bring proof of identity and date
of birth such as a valid passport or birth certificate. If you have an out-of-state
license, also bring that.

2. **Check Your Eligibility**: Make sure your current license is not suspended or
revoked. Also, ensure it's not expired by more than 1 year (for non-commercial drivers).

3. **Visit the DMV Website**: Go to the New York State Department of Motor Vehicles
website and look for information on renewing a driver's license.

4. **Schedule an Appointment**: If required in your area, you may need to schedule an
appointment online or by phone.

5. **Attend the Renewal Process**: This usually involves taking a vision test, providing
biometric data (fingerprints), and possibly taking a written test if it has been more
than 1 year since your last renewal.

6. **Pay the Fees**: There will be a fee for renewing your license. Check the DMV
website for current fees.

Please note that these are general steps and the process may vary slightly depending
on your specific situation and location within New York State. It's always best to
check with the local DMV office or visit their official website for the most accurate
information.
```

By asking for the user's location first, the system provides accurate, state-specific guidance rather than generic information that may not apply.
