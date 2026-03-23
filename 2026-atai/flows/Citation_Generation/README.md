# Help users verify model responses

## Overview

This scenario demonstrates how the citation generation intrinsic can be used to help users verify the responses and identify hallucinations.


## Flows

We provide two flows to be used as part of the demo:

#### 1. Citation Generation (Prompted)

This flow prompts a base model to generate citations for the response.

```
Retriever -> Response Generation -> Citation Generation (Prompted)
```

#### 2. Citation Generation (Intrinsic)

This flow uses the citation generation intrinsic to generate citations. 

```
Retriever -> Response Generation -> Citation Generation (Intrinsic)
```

#### Configuration

Both flows are configured as follows:

- **Retriever**: `sentence-transformers/multi-qa-mpnet-base-dot-v1` embeddings
- **Collection**: `mt-rag-govt`
- **Model Backend**: `Ollama`
- **Model for Response Generation + Prompting:** `ibm-granite/granite-4.0-micro`
- **Intrinsics:** Trained over `ibm-granite/granite-4.0-micro` 


## Demo

We demonstrate a use case where a user interested in astronomy interacts with a RAG system. 

### Step 1: Prompting out-of-the-box model to generate citations

The user starts by interacting with a RAG system, which in addition to the retrieval and generation components, also includes a citation generation component to help users verify the response and spot hallucinations. This component has been created by prompting the `granite-4.0-micro` model out of the box to generate citations.

Wanting to find out more about the moons of Venus and Mars, the user asks a corresponding question. The system responds by creating a response and adding [1] at the end (indicating a citation without any other information), thus making it unclear whether the generated response is grounded or not.

<table>
  <tr>
    <td><b>User Question</b></td>
    <td>which planet has more moons: Mars or Venus?</td>
  </tr>
  <tr>
    <td><b>Assistant Response with Citations</b></td>
    <td>
      Venus has two moons, while Mars also has two. Therefore, both planets have the same number of moons. [1]     
    </td>
    </td>
  </tr>
</table>


### Step 2: Leveraging intrinsics to generate citations

We now employ our citation generation intrinsic and let the user ask the same question against the revised system. The response in this case includes fine-grained citation information, helping the user understand which part of the response it is grounded and which is not. For the first response sentence, claiming that Venus and Mars have both two moons, it produces a single citation, showing that only the part about Mars having two moons is grounded, while the part about Venus having two moons is ungrounded (in reality, as mentioned  in [Wikipedia](https://en.wikipedia.org/wiki/Venus), Venus has no moons)). It also does not include any citation about the second response sentence, indicating that this is also ungrounded. 


<table>
  <tr>
    <td><b>User Question</b></td>
    <td>which planet has more moons: Mars or Venus?</td>
  </tr>
  <tr><td><b>Assistant Response with Citations</b></td>
  <td>
Venus has two moons, while Mars has two as well. <b>[1]</b> Therefore, both planets have the same number of moons.<br><br>

Citations:

<b>[1]</b> Mars' Neighbors

Mars has two moons.</td>
  </tr>
</table>