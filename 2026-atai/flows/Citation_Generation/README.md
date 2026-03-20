# Help users verify model responses

## Overview

This scenario demonstrates how the citation generation and hallucination intrinsics can be used to help users verify the responses and identify hallucinations.


## Flows

We provide two flows to be used as part of the demo:

#### 1. Citation Generation (Prompted)

This flow prompts a base model to generate citations for the response.

```
Retriever -> Response Generation -> Citation Generation (Prompted)
```

#### 2. Citation Generation (Intrinsic)

This flow uses the citation generation and hallucination intrinsics to generate citations and identify hallucinations, respectively. 

```
Retriever -> Response Generation -> Citation Generation (Intrinsic)
                                 -> Hallucination Detection (Intrinsic)
```

#### Configuration

Both flows are configured as follows:

- **Retriever**: `sentence-transformers/multi-qa-mpnet-base-dot-v1` embeddings
- **Collection**: `mt-rag-govt`
- **Model Backend**: `Ollama`
- **Model for Response Generation + Prompting:** `ibm-granite/granite-4.0-h-small`
- **Intrinsics:** Trained over `ibm-granite/granite-4.0-micro` 


## Demo

We demonstrate a use case where the user is a property owner in California who wants to find more information about the defensible space zones in order to better protect their property from wildfires. 

### Step 1: Prompting out-of-the-box model to generate citations

The user starts by interacting with a RAG system, which in addition to the retrieve and generate components, also contains a citation generation component to help users verify the response and spot hallucinations. This component has been created by prompting the `granite-4.0-h-small` model out of the box to generate citations.

---
**Question 1 (Successful citation generation):**

---

Wanting to understand more about the defensible space zones, the user asks a corresponding question. The system responds by creating a response with citations increasing the user's confidence in the response's faithfulness.

<table>
  <tr>
    <td><b>User Question</b></td>
    <td>What are the defensible space zones and their requirements?</td>
  </tr>
  <tr>
    <td><b>Assistant Response with Citations</b></td>
    <td>
      California has three defensible space zones. Zone 0, which extends five feet from your home, requires ember-resistant materials and the removal of all dead and dying plants, weeds, and debris. [1] Zone 1, which extends 30 feet from your home, requires regularly clearing dead or dry vegetation, creating space between trees, and removing dead or dry leaves and pine needles. [2] Zone 2, which extends 100 feet from your home or to the property line, requires reducing potential fuel by cutting or mowing annual grass, creating horizontal and vertical space between shrubs and trees, and removing fallen leaves, needles, twigs, bark, cones, and small branches. [3]<br><br>

Citations: [1] Defensible Space | CAL FIRE. Zone 0: Start Closest to Your Home to be Ember-resistant. The first five feet from your home is the most important. Keeping the area closest to buildings, structures, and decks clear will prevent embers from igniting materials that can spread the fire to your home. Why? The majority of homes lost to wildfire are ignited by flying embers. Embers can travel miles ahead of the active front of wildfires. What to do: Use hardscape like gravel, pavers, or concrete. No combustible bark or mulch. Remove all dead and dying plants, weeds, and debris (leaves, needles, etc.) from your roof, gutter, deck, porch, stairways, and under any areas of your home. Remove all branches within 10 feet of any chimney or stovepipe outlet. Limit combustible items (like outdoor furniture and planters) on top of decks. Relocate firewood and lumber to Zone 2. Replace combustible fencing, gates, and arbors attached to the home with noncombustible alternatives. Consider relocating garbage and recycling containers outside this zone. Consider relocating boats, RVs, vehicles, and other combustible items outside this zone. [2] Defensible Space | CAL FIRE. Zone 1: Keep it Lean, Clean and Green Within 30 feet. Regularly clear dead or dry vegetation and create space between trees. During times of drought when watering is limited, pay special attention to clearing dead or dying material. Why? Removing dead plants and creating space between trees and shrubs creates a buffer for your property and reduces potential fuel for fire. What to do: Remove all dead plants, grass, and weeds. Remove dead or dry leaves and pine needles. Trim trees regularly to keep branches a minimum of 10 feet from other trees. Create a separation between trees, shrubs, and items that could catch fire, such as patio furniture, wood piles, swing sets, etc. [3] Defensible Space | CAL FIRE. Zone 2: Reduce Potential Fuel Within 100 feet. Continue reducing potential fuel within 100 feet or the property line. Why? 100 feet of defensible space is required by law. Public Resources Code (PRC) 4291. What to do: Cut or mow annual grass down to a maximum height of four inches. Create horizontal space between shrubs and trees. (See diagram) Create vertical space between grass, shrubs and trees. (See diagram) Remove fallen leaves, needles, twigs, bark, cones, and small branches. However, they may be permitted to a depth of three inches. Keep 10 feet of clearance around exposed wood piles, down to bare mineral soil, in all directions. Clear areas around outbuildings and propane tanks. Keep 10 feet of clearance to bare mineral soil and no flammable vegetation for an additional 10 feet around their exterior.     
    </td>
    </td>
  </tr>
</table>

---
**Question 2 (Unsuccessful citation generation):**

---

Confident about the system's citation abilities, the user asks another question on defensible space zones. The system responds with a response and associated citations. However, the problem is that the response is in this case not grounded in the retrieved passages. While, there are a couple of retrieved passages about defensible space, they do not contain any information about grants and assistance programs. The generated response is essentially hallucinated by the user gets the impression that it is not.

<table>
  <tr>
    <td><b>User Question</b></td>
    <td>Are there grants or financial assistance programs for defensible space compliance?</td>
  </tr>
  <tr>
    <td><b>Assistant Response with Citations</b></td>
    <td>Yes, there are grants and financial assistance programs available for defensible space compliance. [1] The U.S. Department of Agriculture (USDA) Forest Service offers the Forest Stewardship Program, which provides technical and financial assistance to private landowners for creating and maintaining defensible space. [1] Additionally, some states and local governments offer grants or cost-share programs to help property owners with defensible space projects. [1] It's recommended to check with your state forestry agency or local fire department for more information on available programs in your area. [1]<br><br>

Citations: [1] Defensible Space | CAL FIRE</td>
  </tr>
</table>

### Step 2: Leveraging intrinsics to generate citations and identify hallucinations

To address this issue, we employ our citation generation and hallucination identification intrinsics and let the user ask the same question against the revised system. Since the response is ungrounded, the system does not include any citations. The system is also returning a per-sentence hallucination analysis, explaining that each sentence is not supported by the retrieved passages.  


<table>
  <tr>
    <td><b>User Question</b></td>
    <td>Are there grants or financial assistance programs for defensible space compliance?</td>
  </tr>
  <tr><td><b>Assistant Response with Citations</b></td>
  <td>
Yes, there are grants and financial assistance programs available for defensible space compliance. The U.S. Department of Agriculture (USDA) Forest Service offers the Forest Stewardship Program, which provides technical and financial assistance to private landowners for creating and maintaining defensible space. Additionally, some states and local governments offer grants or cost-share programs to help property owners with defensible space projects. It's recommended to check with your state forestry agency or local fire department for more information on available programs in your area.</td>
  </tr>
</table>

**Hallucination Analysis:**
<pre style="background-color: #f0f0f0; padding: 10px; overflow-x: auto; margin: 0; white-space: pre-wrap;">[
    {
        "response_begin": 0,
        "response_end": 99,
        "response_text": "Yes, there are grants and financial assistance programs available for defensible space compliance. ",
        "faithfulness_likelihood": 0.3433651380848655,
        "explanation": "This sentence makes a factual claim about the availability of grants and financial assistance programs for defensible space compliance. The provided context does not directly support this claim, as it does not mention any grants or financial assistance programs for defensible space compliance. However, the sentence is a general statement about the availability of such programs, which is not contradicted by the provided context."
    },
    {
        "response_begin": 99,
        "response_end": 108,
        "response_text": "The U.S. ",
        "faithfulness_likelihood": 0.3614338195387987,
        "explanation": "This sentence makes a factual claim about the Forest Stewardship Program offered by the USDA Forest Service. The provided context does not directly support this claim, as it does not mention the Forest Stewardship Program or any technical and financial assistance provided by the USDA Forest Service for creating and maintaining defensible space. However, the sentence is a general statement about the availability of such a program, which is not contradicted by the provided context."
    },
    {
        "response_begin": 108,
        "response_end": 314,
        "response_text": "Department of Agriculture (USDA) Forest Service offers the Forest Stewardship Program, which provides technical and financial assistance to private landowners for creating and maintaining defensible space. ",
        "faithfulness_likelihood": 0.5036262253080092,
        "explanation": "This sentence makes a factual claim about the availability of grants or cost-share programs for defensible space projects offered by some states and local governments. The provided context does not directly support this claim, as it does not mention any grants or cost-share programs for defensible space projects offered by states or local governments. However, the sentence is a general statement about the availability of such programs, which is not contradicted by the provided context."
    },
    {
        "response_begin": 314,
        "response_end": 454,
        "response_text": "Additionally, some states and local governments offer grants or cost-share programs to help property owners with defensible space projects. ",
        "faithfulness_likelihood": 0.6506826029373467,
        "explanation": "This sentence makes a factual claim about the recommendation to check with the state forestry agency or local fire department for more information on available programs in the user's area. The provided context does not directly support this claim, as it does not mention any recommendation to check with the state forestry agency or local fire department for more information on available programs. However, the sentence is a general statement about the availability of such information, which is not contradicted by the provided context."
    }
]</pre>
