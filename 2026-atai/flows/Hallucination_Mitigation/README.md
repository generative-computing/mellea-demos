# Hallucination Mitigation RAG Flow

## Overview

This folder contains two Langflow flows configured for government domain documents, with the goal of demonstrating the utility of hallucination detection intrinsic component post-generation for hallucination removal. The [Query Rewrite + Answerability Detection](../QueryRewrite_Answerability/README.md) flow determines answerability of the question with respect to retrieved context before generation and completely abstains when the question is unanswerable, while this flow addresses hallucinations post-generation. We provide both prompt-based approach and intrinsic approaches. This uses two key components:

1. **Query Rewriting (QR)**: Reformulation to resolve coreference and make query standalone for better retrieval
2. **Hallucination Detection/Mitigation (HD)**: Identifying and correcting hallucinated content in generated responses that is not supported by retrieved documents

This pattern addresses one of the most critical challenges in RAG systems: ensuring that generated responses remain grounded in the retrieved evidence and do not fabricate information.

### Configuration

- **Retriever**: `sentence-transformers/multi-qa-mpnet-base-dot-v1` embeddings
- **Collection**: `mt-rag-govt`
- **Models used**: `openai/gpt-oss-20b`

### Flow Architecture

```
User Query w/ MT conversation history
    ↓
Query Rewrite
    ↓
Vector Search
    ↓
Generate Initial Response
    ↓
Feedback from Hallucination Detection 
    ↓
Corrected Response
```

---

## Methods

We provide two implementations of this RAG pattern:

### 1. Prompt-Based
**File**: `Prompt_QR_HD.json`

This implementation uses prompt-based approaches for both query rewriting (From the LangChain tutorial page [LangChain Agentic RAG tutorial](https://docs.langchain.com/oss/python/langgraph/agentic-rag)) and hallucination removal.

**Components**:
- Query Rewrite (Prompt)
- Hallucination Removal (Prompt)
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

*Hallucination Removal Prompt*:
```
Documents:
{documents}

Conversation:
{conversation}

Rewrite the last assistant response to only include claims supported by the documents. Rephrase partially supported claims to match what the documents say. If nothing is supported, respond only with: "I don't know the answer to the question."
Return ONLY the corrected assistant response without any explanation, preamble, or meta-commentary.
```


### 2. Our Intrinsic Models
**File**: `Intrinsic_QR_HD.json`

This implementation uses specialized trained models for query rewriting and hallucination detection with a feedback-based correction mechanism.

**Components**:
- Query Rewrite (Intrinsic)
- Hallucination Detection (Intrinsic)
- Hallucination Feedback (Merge ouput from Hallucination Detection in Conversation Chat)
- Base Model Chat
- ChromaDB Search with Local Embeddings

---

## Examples

### Example 1: Removing Hallucinations Post-Generation

**Scenario**: When documents are topically relevant but lack specific details requested in the question, models may generate answers using their internal parametric knowledge rather than the corpus. The LangChain tutorial's approach produced such a hallucinated answer. The intrinsic flow's hallucination detection checks the answer against retrieved documents and abstains when content is unsupported, while the prompt-based hallucination removal attempts to strip fabrications but may still leave unsupported content.

**Models used**: `openai/gpt-oss-20b`

**Question**: `How often do large dust storms cover the entire planet on Mars?`

**What's in Corpus**: A NASA document mentioning "Mars' large storms sometimes cover the entire planet" - but NO information about storm frequency or timing

**What Happened**:

Both flows generated initial answers containing detailed statistics not present in the retrieved documents, using the model's internal knowledge about Mars.

**LangChain Tutorial Flow**:
- **Answerability Determination**: The [LangChain tutorial approach](https://docs.langchain.com/oss/python/langgraph/agentic-rag) lets hallucination pass at the answerability stage (see [Query Rewrite + Answerability Detection](../QueryRewrite_Answerability/README.md) Example 4)
- **Answer**:
```
Large, planet-wide dust storms on Mars are relatively rare events. In the past few decades the planet has experienced a global dust storm roughly once every two to three Martian years (≈ 1–2 global storms per 1.5–2 Earth years). They are not guaranteed to occur each cycle; many Martian years pass without a storm that covers the entire globe. When they do happen, the storms can persist for several weeks to a few months, obscuring the surface and affecting the planet's climate and surface temperature. (See NASA's Mars Climate Orbiter and Mars Reconnaissance Orbiter studies for detailed statistics.)
```
- **Result**: The hallucinated answer with fabricated statistics is passed through to the user

**Prompt-Based Flow**:
- **Initial Answer**:
```
Large, planet-wide dust storms on Mars are relatively rare events. In the past few decades the planet has experienced a global dust storm roughly once every two to three Martian years (≈ 1–2 global storms per 1.5–2 Earth years). They are not guaranteed to occur each cycle; many Martian years pass without a storm that covers the entire globe. When they do happen, the storms can persist for several weeks to a few months, obscuring the surface and affecting the planet's climate and surface temperature. (See NASA's Mars Climate Orbiter and Mars Reconnaissance Orbiter studies for detailed statistics.)
```

- **Final Answer**: `"Large dust storms on Mars sometimes cover the entire planet."`

- **Hallucination Detection**: Hallucination removal stripped out most fabricated details, but the final answer still doesn't acknowledge that frequency information is unavailable in the corpus

**Intrinsic Flow**:
- **Initial Answer** (truncated):
```
Large dust storms that blanket the entire Martian surface are **relatively rare**. In practice, a global-scale dust event occurs roughly **once every 2–3 Martian years** (about 1.5–2 Earth years), but the exact interval varies from one cycle to the next.

**Why the interval is irregular**

* **Seasonal trigger** – Global storms tend to develop during the southern hemisphere's summer (when the planet is closest to the Sun), when the atmosphere is warmest and winds strongest.
* **Dust-devil seed** – Small, localized dust devils lift fine particles into the air; when enough of them overlap, a planet-wide storm can form.
* **Atmospheric conditions** – The background pressure, temperature, and wind patterns must align just right; otherwise the dust remains confined to regional storms.

**Recent examples**

| Year (Earth) | Martian year | Storm duration | Notes |
|--------------|--------------|----------------|-------|
| 2018 | 2018–2019 | ~3 months | First global storm in ~30 years; covered > 90% of the planet. |
| 2020–2021 | 2020–2021 | ~2 months | Global coverage, but less intense than 2018. |
| 2023 | 2023–2024 | ~2 months | Most recent global event; again triggered by a strong dust-devil season. |

These events illustrate the typical cadence: a global dust storm can happen, but it is not guaranteed every Martian year. When they do occur, they can persist for weeks to months, dramatically reducing surface visibility and affecting solar power for orbiters and rovers.
```
- **Hallucination Feedback** (truncated):
```json
[
  {
    "response_text": "Large dust storms that blanket the entire Martian surface are **relatively rare**.",
    "explanation": "This sentence makes a factual claim about the frequency of large dust storms on Mars. The provided context does not directly support this claim, but it does mention that 'Mars' large storms sometimes cover the entire planet.' This suggests that such storms are not common, but the context does not provide enough information to determine the exact frequency."
  },
  {
    "response_text": "In practice, a global-scale dust event occurs roughly **once every 2–3 Martian years** (about 1.5–2 Earth years), but the exact interval varies from one cycle to the next.",
    "explanation": "This sentence makes a specific factual claim about the frequency of global-scale dust events on Mars. The provided context does not directly support this claim, and there is no information about the exact interval between such events."
  },
  {
    "response_text": "**Seasonal trigger** – Global storms tend to develop during the southern hemisphere's summer (when the planet is closest to the Sun), when the atmosphere is warmest and winds strongest.",
    "explanation": "This sentence makes a factual claim about the seasonal trigger for global dust storms on Mars. The provided context does not directly support this claim, and there is no information about the relationship between the southern hemisphere's summer and the development of global dust storms."
  },
  {
    "response_text": "| 2018 | 2018–2019 | ~3 months | First global storm in ~30 years; covered > 90% of the planet. |",
    "explanation": "This sentence makes a factual claim about a specific global dust storm on Mars in 2018-2019. The provided context does not directly support this claim, and there is no information about the duration or intensity of this storm."
  },
  ...
]
```
The hallucination feedback component analyzed each segment of the response with explanations indicating that while the context mentions Mars' large storms sometimes cover the entire planet, it does not contain specific information about frequencies (once every 2-3 Martian years), durations, seasonal patterns, or historical storm records. The claim-level feedback reveals that even seemingly supported statements lack corpus grounding for their specific details. Using this segment-level analysis, the system identified that the entire response consists of hallucinated details not present in the corpus.

- **Final Answer**: `"I don't know the answer to the question"`

- **Hallucination Detection**: Analyzed the answer against retrieved documents and identified that the specific frequencies, durations, seasonal patterns, and mission observations are not present in the corpus. Detected hallucination and completely abstained from providing unsupported information.

**Key Insight**:

The intrinsic-based flow provides a way of removing hallucination post-generation. The hallucination detection checks the generated answer against retrieved documents, identifying when the model has used its internal knowledge to produce content not supported by the corpus. The intrinsic model provides fine-grained, claim-level feedback with detailed explanations for each segment of the response, revealing exactly which parts lack corpus grounding. When hallucinations are detected, the system abstains entirely - ensuring that only corpus-grounded information is returned to users.

---

### Example 2: Handling Partial Answerability

**Scenario**: When a multi-part question has some parts answerable from the corpus and other parts not, models may hallucinate the missing information. The intrinsic flow's hallucination detection can distinguish which parts are supported and provide partial answers with appropriate acknowledgment of what's unknown, while the prompt-based approach removes hallucinations but doesn't explicitly indicate what's missing.

**Models used**: `openai/gpt-oss-20b`

**Question**: `What are Mars' moon called, and how far are they from the planet?`

**What's in Corpus**: A NASA document listing Mars' moons (Phobos and Deimos) - but NO information about their distances from Mars

**What Happened**:

Both flows generated initial answers with complete details including distances, using the model's internal knowledge.

**Prompt-Based Flow**:
- **Initial Answer**:
```
Mars' moons

| Moon | Distance from Mars (average) | How far above the surface |
|------|------------------------------|---------------------------|
| Phobos | ~9 378 km from Mars' center | ~6 000 km above the surface |
| Deimos | ~23 460 km from Mars' center | ~20 000 km above the surface |

Quick facts:
- Phobos is the larger of the two, but it is still only about 22 km across.
- Deimos is smaller, roughly 12 km across, and is farther away.
- Both moons orbit very close to Mars compared with the distances of Earth's Moon (384 000 km)...
```
- **Final Answer**: `"Mars has two moons: Phobos and Deimos."`
- **Result**: Hallucination removal stripped out distance information, but the answer doesn't explicitly acknowledge that distance information was requested but unavailable

**Intrinsic Flow**:
- **Initial Answer**:
```
**Mars' moons**

| Moon | Approximate orbital distance from Mars (center to moon) | Approximate distance from Mars' surface |
|------|--------------------------------------------------------|----------------------------------------|
| **Phobos** | ~9 378 km (≈5 827 mi) | ~6 000 km (≈3 730 mi) |
| **Deimos** | ~23 460 km (≈14 590 mi) | ~20 000 km (≈12 430 mi) |

**Key points**

- **Phobos** is the larger and closer of the two. It orbits Mars in about 7.6 hours and is only a few hundred kilometers wide.
- **Deimos** is smaller and farther away, taking about 30.3 hours to complete one orbit.
- Both moons are irregularly shaped, heavily cratered, and are thought to be captured asteroids.

These distances are averages; the moons' orbits are slightly elliptical, so the actual distance can vary a little over time.
```
- **Hallucination Feedback** (truncated):
```json
[
  {
    "response_text": "**Mars' moons**\n\n...\n| **Phobos** | ~9 378 km ... | ~6 000 km ... |\n| **Deimos** | ~23 460 km ... | ~20 000 km ... |\n...\n- **Phobos** is the larger and closer of the two.",
    "explanation": "This sentence contains factual claims about the moons of Mars and their distances from the planet. The provided context does not directly support these claims, as it does not contain specific information about the moons of Mars. However, the context does mention that Mars has two moons, Phobos and Deimos, which supports the claim that Mars has moons."
  },
  {
    "response_text": "It orbits Mars in about 7.6 hours and is only a few hundred kilometers wide.",
    "explanation": "This sentence contains factual claims about the size and orbit of Phobos. The provided context does not directly support these claims, as it does not contain specific information about the size and orbit of Phobos. However, the context does mention that Phobos is one of the moons of Mars, which supports the claim that Phobos is a moon of Mars."
  },
  {
    "response_text": "These distances are averages; the moons' orbits are slightly elliptical, so the actual distance can vary a little over time.",
    "explanation": "This sentence contains factual claims about the orbits of the moons of Mars. The provided context does not directly support these claims, as it does not contain specific information about the orbits of the moons of Mars. However, the context does mention that Mars has two moons, Phobos and Deimos, which supports the claim that Mars has moons."
  },
  ...
]
```
The hallucination feedback component analyzed each segment of the response with explanations indicating that while the context mentions Mars has two moons (Phobos and Deimos), it does not contain specific information about distances, orbital periods, sizes, or other details. Using this segment-level analysis, the system identified that only the moon names are supported by the corpus and constructed a partial answer retaining only the grounded information.
- **Final Answer**: `"Mars has two moons, Phobos and Deimos. I don't know how far they are from the planet."`
- **Result**: Provided the answerable part (moon names) and explicitly acknowledged the unanswerable part (distances)

**Key Insight**:

The prompt-based hallucination removal strips out unsupported details but produces an incomplete answer that doesn't address all parts of the question or acknowledge information gaps. The intrinsic flow's hallucination detection enables nuanced handling: it can provide partial answers for what's supported by the corpus while explicitly stating what cannot be answered. Additionally, the intrinsic model provides detailed segment-level feedback for each part of the response, including faithfulness likelihood scores and explanations for why each segment was scored, giving transparency into which specific claims are grounded in the corpus and which are not.

---