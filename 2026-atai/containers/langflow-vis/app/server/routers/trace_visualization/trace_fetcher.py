"""
Trace Fetcher Module

This module provides functionality to:
1. Fetch traces from LangFuse API
2. Fetch flow schema from LangFlow API
3. Parse and match components
4. Build flowConfig JSON for visualization
"""

import json
import requests
from typing import Dict, List, Any, Optional, Tuple


# =============================================================================
# LangFuse API Functions
# =============================================================================

def fetch_trace_from_api(
    trace_id: str,
    langfuse_base_url: str,
    langfuse_public_key: str,
    langfuse_secret_key: str
) -> Dict[str, Any]:
    """
    Fetch LangFuse trace from API by trace ID.

    Args:
        trace_id: The trace ID to fetch
        langfuse_base_url: LangFuse API base URL
        langfuse_public_key: LangFuse public API key
        langfuse_secret_key: LangFuse secret API key

    Returns:
        Trace data dictionary with 'trace' and 'observations' keys
    """
    # Fetch the main trace
    trace_endpoint = f"{langfuse_base_url}/api/public/traces/{trace_id}"
    trace_response = requests.get(
        trace_endpoint,
        auth=(langfuse_public_key, langfuse_secret_key)
    )
    trace_response.raise_for_status()
    trace_info = trace_response.json()

    # Fetch observations for this trace
    observations_endpoint = f"{langfuse_base_url}/api/public/observations"
    observations_response = requests.get(
        observations_endpoint,
        params={"traceId": trace_id},
        auth=(langfuse_public_key, langfuse_secret_key)
    )
    observations_response.raise_for_status()
    observations_data = observations_response.json()

    return {
        "trace": trace_info,
        "observations": observations_data.get("data", [])
    }


def fetch_latest_trace_from_api(
    langfuse_base_url: str,
    langfuse_public_key: str,
    langfuse_secret_key: str,
    user_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Fetch the latest trace from LangFuse API.

    Args:
        langfuse_base_url: LangFuse API base URL
        langfuse_public_key: LangFuse public API key
        langfuse_secret_key: LangFuse secret API key
        user_id: Optional user ID to filter traces by

    Returns:
        Latest trace data dictionary or None
    """
    endpoint = f"{langfuse_base_url}/api/public/traces"

    params = {
        "limit": 1,
    }
    if user_id:
        params["userId"] = user_id

    response = requests.get(
        endpoint,
        params=params,
        auth=(langfuse_public_key, langfuse_secret_key)
    )
    response.raise_for_status()

    result = response.json()
    traces = result.get("data", [])

    if traces:
        trace_id = traces[0]["id"]
        return fetch_trace_from_api(
            trace_id,
            langfuse_base_url,
            langfuse_public_key,
            langfuse_secret_key
        )

    return None


def list_recent_traces(
    langfuse_base_url: str,
    langfuse_public_key: str,
    langfuse_secret_key: str,
    limit: int = 15,
    page: int = 1,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    List recent traces from LangFuse API with pagination.

    Args:
        langfuse_base_url: LangFuse API base URL
        langfuse_public_key: LangFuse public API key
        langfuse_secret_key: LangFuse secret API key
        limit: Maximum number of traces to return per page
        page: Page number (1-indexed)
        session_id: Optional session ID to filter traces by conversation session
        user_id: Optional user ID to filter traces by

    Returns:
        Dict with 'traces' list and 'hasMore' boolean
    """
    endpoint = f"{langfuse_base_url}/api/public/traces"

    params = {
        "limit": limit,
        "page": page,
    }

    # Add session filter if provided
    if session_id:
        params["sessionId"] = session_id

    # Add user filter if provided
    if user_id:
        params["userId"] = user_id

    response = requests.get(
        endpoint,
        params=params,
        auth=(langfuse_public_key, langfuse_secret_key)
    )
    response.raise_for_status()

    result = response.json()
    traces = result.get("data", [])

    # Extract summary info for each trace
    trace_list = []
    for trace in traces:
        # Get input text (user query) from trace
        # The input is structured as {"Chat Input (component-id)": {"input_value": "..."}}
        input_text = ""
        trace_input = trace.get("input")
        if trace_input:
            if isinstance(trace_input, str):
                input_text = trace_input
            elif isinstance(trace_input, dict):
                # Look for Chat Input component in the input dict
                for key, value in trace_input.items():
                    if "Chat Input" in key or "ChatInput" in key:
                        if isinstance(value, dict):
                            # Try input_value first, then text
                            input_text = value.get("input_value", value.get("text", ""))
                            if input_text:
                                break
                # Fallback: try other common patterns
                if not input_text:
                    input_text = trace_input.get("text", trace_input.get("input", ""))

        # Get flow name from metadata
        flow_name = "Unknown Flow"
        metadata = trace.get("metadata")
        if metadata:
            if isinstance(metadata, dict):
                flow_name = metadata.get("flow_name", "Unknown Flow")

        trace_list.append({
            "id": trace.get("id"),
            "timestamp": trace.get("timestamp"),
            "name": trace.get("name", flow_name),
            "flow_name": flow_name,
            "session_id": trace.get("sessionId"),
            "input_preview": input_text[:50] + "..." if len(input_text) > 50 else input_text
        })

    # Determine if there are more traces (if we got a full page, there might be more)
    has_more = len(traces) >= limit

    return {
        "traces": trace_list,
        "hasMore": has_more,
        "page": page
    }


def extract_flow_id_from_trace(trace_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract flow_id from trace metadata.

    Args:
        trace_data: Trace data dictionary

    Returns:
        Flow ID string or None
    """
    if not trace_data or 'trace' not in trace_data:
        return None

    trace_info = trace_data.get('trace', {})
    metadata = trace_info.get('metadata')

    if metadata:
        if isinstance(metadata, str):
            try:
                metadata_dict = json.loads(metadata)
                return metadata_dict.get('flow_id')
            except:
                pass
        elif isinstance(metadata, dict):
            return metadata.get('flow_id')

    return None


# =============================================================================
# LangFlow API Functions
# =============================================================================

def fetch_flow_schema(
    flow_id: str,
    base_url: str,
    api_key: str
) -> Dict[str, Any]:
    """
    Fetch flow schema from LangFlow API.

    Args:
        flow_id: The flow ID to fetch
        base_url: LangFlow API base URL
        api_key: API authentication key

    Returns:
        Flow schema dictionary
    """
    endpoint = f"{base_url}/api/v1/flows/{flow_id}"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }

    response = requests.get(endpoint, headers=headers)
    response.raise_for_status()

    return response.json()


# =============================================================================
# Parsing Functions
# =============================================================================

def is_message_history_node(node: Dict[str, Any]) -> bool:
    """Check if a node is a Message History component."""
    node_id = node.get('id', '')
    node_data = node.get('data', {})
    node_info = node_data.get('node', {})
    display_name = node_info.get('display_name', '')
    node_type = node.get('type', '')

    return ('MessageHistory' in node_id or
            'Message History' in display_name or
            'MessageHistory' in node_type or
            'message_history' in node_id.lower())


def is_chat_output_node(node: Dict[str, Any]) -> bool:
    """Check if a node is a Chat Output component."""
    node_id = node.get('id', '')
    node_data = node.get('data', {})
    node_info = node_data.get('node', {})
    display_name = node_info.get('display_name', '')
    node_type = node.get('type', '')

    return ('ChatOutput' in node_id or
            'Chat Output' in display_name or
            'ChatOutput' in node_type or
            'Output' in display_name or
            'chat_output' in node_id.lower())


def parse_langflow_schema(flow_schema: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse LangFlow schema to extract nodes and edges.
    Filters out Message History components. Chat Output nodes are kept but have no outgoing edges.

    Args:
        flow_schema: Raw flow schema from LangFlow API

    Returns:
        Tuple of (nodes, edges)
    """
    if not flow_schema:
        return [], []

    data = flow_schema.get('data', {})
    all_nodes = data.get('nodes', [])
    all_edges = data.get('edges', [])

    # Filter out Message History nodes (but keep Chat Output nodes)
    nodes = [node for node in all_nodes
             if not is_message_history_node(node)]

    # Get IDs of filtered nodes and identify Chat Output nodes
    node_ids = {node.get('id') for node in nodes}
    chat_output_ids = {node.get('id') for node in nodes if is_chat_output_node(node)}

    # Filter out edges that connect to/from filtered nodes
    # Also filter out edges that originate FROM Chat Output nodes (they have no outgoing edges)
    edges = [
        edge for edge in all_edges
        if edge.get('source') in node_ids
        and edge.get('target') in node_ids
        and edge.get('source') not in chat_output_ids
    ]

    return nodes, edges


def is_message_history_observation(observation: Dict[str, Any]) -> bool:
    """Check if an observation is for a Message History component."""
    obs_name = observation.get('name', '')
    metadata = observation.get('metadata', {})
    component_id = metadata.get('component_id', '')

    return ('MessageHistory' in obs_name or
            'Message History' in obs_name or
            'MessageHistory' in component_id or
            'message_history' in component_id.lower())


def is_chat_output_observation(observation: Dict[str, Any]) -> bool:
    """Check if an observation is for a Chat Output component."""
    obs_name = observation.get('name', '')
    metadata = observation.get('metadata', {})
    component_id = metadata.get('component_id', '')

    return ('ChatOutput' in obs_name or
            'Chat Output' in obs_name or
            'ChatOutput' in component_id or
            'Output' in obs_name or
            'chat_output' in component_id.lower())


def parse_langfuse_trace(trace_data: Dict[str, Any]) -> List[Dict]:
    """
    Parse LangFuse trace to extract and sort observations.
    Filters out Message History (but keeps Chat Output).

    Args:
        trace_data: Raw trace data from LangFuse

    Returns:
        List of observations sorted by start time
    """
    all_observations = trace_data.get('observations', [])

    # Filter out Message History observations (but keep Chat Output)
    observations = [obs for obs in all_observations
                   if not is_message_history_observation(obs)]

    # Sort by start time
    sorted_observations = sorted(
        observations,
        key=lambda x: x.get('startTime', '')
    )

    return sorted_observations


# =============================================================================
# Component Matching
# =============================================================================

def match_components(
    nodes: List[Dict],
    observations: List[Dict]
) -> List[Tuple[Optional[Dict], Optional[Dict]]]:
    """
    Match components between LangFlow schema and LangFuse trace.

    Args:
        nodes: List of nodes from LangFlow schema
        observations: List of observations from LangFuse trace

    Returns:
        List of (node, observation) tuples
    """
    matches = []

    # Create maps for quick lookup
    node_map = {node['id']: node for node in nodes}
    obs_map = {}
    for obs in observations:
        metadata = obs.get('metadata', {})
        component_id = metadata.get('component_id')
        if component_id:
            obs_map[component_id] = obs

    # Match nodes with observations
    for node in nodes:
        node_id = node.get('id')
        observation = obs_map.get(node_id)
        matches.append((node, observation))

    return matches


# =============================================================================
# Logical Flow Computation
# =============================================================================

def dependency_based_flow(edges: List[Dict]) -> List[Tuple[str, str]]:
    """
    Compute logical flow using dependency-based execution order.
    Removes redundant edges where dependencies are satisfied transitively.

    Args:
        edges: List of edge dictionaries from schema

    Returns:
        List of (source, target) tuples representing minimal logical flow
    """
    # Build adjacency list
    adjacency = {}
    all_edges = set()

    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        if source and target:
            if source not in adjacency:
                adjacency[source] = set()
            adjacency[source].add(target)
            all_edges.add((source, target))

    # For each edge, check if there's an alternative path
    reduced_edges = set()

    for source, target in all_edges:
        has_alternative_path = False

        # BFS to find if target is reachable through intermediates
        visited = set()
        queue = list(adjacency.get(source, set()) - {target})

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            if node == target:
                has_alternative_path = True
                break

            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    queue.append(neighbor)

        if not has_alternative_path:
            reduced_edges.add((source, target))

    return list(reduced_edges)


# =============================================================================
# Component Args Extraction
# =============================================================================

def extract_component_args(
    observation: Dict[str, Any],
    component_type: str
) -> Dict[str, Any]:
    """
    Extract component arguments from trace execution data.

    Args:
        observation: Observation from trace
        component_type: Type of component

    Returns:
        Dictionary of component arguments
    """
    args = {}

    if not observation:
        return args

    obs_input = observation.get('input', {})
    obs_output = observation.get('output', {})
    obs_name = observation.get('name', '')

    # Check for errors in the output (applies to all component types)
    if isinstance(obs_output, dict) and 'error' in obs_output:
        error_value = obs_output.get('error')
        if error_value:
            args['error'] = str(error_value)

    if 'Chat Input' in obs_name or 'ChatInput' in obs_name:
        if 'input_value' in obs_input:
            args['query'] = obs_input.get('input_value', '')
        elif 'message' in obs_output:
            message_data = obs_output.get('message', {})
            if isinstance(message_data, dict):
                args['query'] = message_data.get('text', message_data.get('data', {}).get('text', ''))

    elif 'Query Rewrite' in obs_name or 'QueryRewrite' in obs_name:
        if 'chat_input' in obs_input:
            chat_input_data = obs_input.get('chat_input', {})
            if isinstance(chat_input_data, dict):
                args['original_query'] = chat_input_data.get('text', chat_input_data.get('data', {}).get('text', ''))

        if 'rewritten_query' in obs_output:
            rewritten_data = obs_output.get('rewritten_query', {})
            if isinstance(rewritten_data, dict):
                args['rewritten_query'] = rewritten_data.get('text', rewritten_data.get('data', {}).get('text', ''))

    elif 'Query Expansion' in obs_name or 'QueryExpansion' in obs_name:
        # Extract query variations from output
        # The Query Expansion component outputs: {"all_queries": {"data": {"queries": [...]}}}
        if 'all_queries' in obs_output:
            all_queries_data = obs_output.get('all_queries', {})
            if isinstance(all_queries_data, dict):
                # Get the nested data structure
                data = all_queries_data.get('data', all_queries_data)
                if isinstance(data, dict) and 'queries' in data:
                    queries = data.get('queries', [])
                    if isinstance(queries, list):
                        args['query_variations'] = [
                            str(q) for q in queries if q
                        ]
        else:
            # Fallback: try common output key names
            for key in ['query_variations', 'expanded_queries', 'variations', 'queries']:
                if key in obs_output:
                    variations_data = obs_output.get(key, {})
                    if isinstance(variations_data, dict):
                        # Handle nested structure like {'data': {'text': '...'}} or list
                        data = variations_data.get('data', variations_data)
                        if isinstance(data, dict) and 'text' in data:
                            # Text might be newline-separated variations
                            text = data.get('text', '')
                            if text:
                                args['query_variations'] = [v.strip() for v in text.split('\n') if v.strip()]
                        elif isinstance(data, list):
                            # Direct list of variations
                            args['query_variations'] = [
                                v.get('text', v) if isinstance(v, dict) else str(v)
                                for v in data
                            ]
                    elif isinstance(variations_data, list):
                        args['query_variations'] = [
                            v.get('text', v) if isinstance(v, dict) else str(v)
                            for v in variations_data
                        ]
                    break

    elif 'Query Clarification' in obs_name or 'QueryClarification' in obs_name:
        # Extract clarification result from output
        # Output is either "CLEAR" (query is non-ambiguous) or a clarifying question
        if 'clarification' in obs_output:
            clarification_data = obs_output.get('clarification', {})
            if isinstance(clarification_data, dict):
                clarification_text = clarification_data.get('text', clarification_data.get('data', {}).get('text', ''))
                if clarification_text:
                    args['clarification_result'] = clarification_text

    elif 'Answerability' in obs_name:
        # Extract answerability label from output
        if 'answerability_label' in obs_output:
            label_data = obs_output.get('answerability_label', {})
            if isinstance(label_data, dict):
                label_text = label_data.get('text', label_data.get('data', {}).get('text', ''))
                if label_text:
                    args['answerability_label'] = label_text
            elif isinstance(label_data, str):
                args['answerability_label'] = label_data

        # Extract answerability likelihood from response output
        # The response contains a JSON string like: {"answerability_likelihood": 0.123}
        if 'response' in obs_output:
            response_data = obs_output.get('response', {})
            if isinstance(response_data, dict):
                response_text = response_data.get('text', response_data.get('data', {}).get('text', ''))
                if response_text:
                    try:
                        # Parse the JSON string to extract answerability_likelihood
                        parsed = json.loads(response_text)
                        if isinstance(parsed, dict) and 'answerability_likelihood' in parsed:
                            args['answerability_likelihood'] = parsed['answerability_likelihood']
                    except (json.JSONDecodeError, TypeError):
                        # If not valid JSON, try to parse as plain float
                        try:
                            args['answerability_likelihood'] = float(response_text)
                        except (ValueError, TypeError):
                            pass

        # Fallback: check observation metadata for answerability_score
        if 'answerability_likelihood' not in args:
            obs_metadata = observation.get('metadata', {})
            if isinstance(obs_metadata, dict) and 'answerability_score' in obs_metadata:
                args['answerability_likelihood'] = obs_metadata.get('answerability_score')

    elif 'Hallucination Detection' in obs_name or 'HallucinationDetection' in obs_name:
        # Extract hallucination detection results from output
        # Output is a JSON array string with sentence-level faithfulness assessments
        if 'response' in obs_output:
            response_data = obs_output.get('response', {})
            if isinstance(response_data, dict):
                response_text = response_data.get('text', response_data.get('data', {}).get('text', ''))
                if response_text:
                    try:
                        # Parse the JSON array string
                        parsed = json.loads(response_text)
                        if isinstance(parsed, list):
                            args['hallucination_results'] = parsed
                    except (json.JSONDecodeError, TypeError):
                        pass

    elif 'Retriever' in obs_name or 'ELSER' in obs_name:
        if 'documents' in obs_output:
            docs_data = obs_output.get('documents', {})
            if isinstance(docs_data, dict):
                documents = docs_data.get('data', {}).get('documents', [])
                args['passages'] = [
                    {
                        'id': doc.get('doc_id', ''),
                        'text': doc.get('text', ''),
                        'score': doc.get('score', '')
                    }
                    for doc in documents
                ]

    elif 'Base Model' in obs_name or 'LLM' in obs_name or 'Granite' in obs_name:
        if 'response' in obs_output:
            response_data = obs_output.get('response', {})
            if isinstance(response_data, dict):
                args['response'] = response_data.get('text', response_data.get('data', {}).get('text', ''))

    elif 'Citation' in obs_name:
        if 'response' in obs_output:
            response_data = obs_output.get('response', {})
            if isinstance(response_data, dict):
                citation_text = response_data.get('text', response_data.get('data', {}).get('text', ''))
                args['citations'] = citation_text

        if 'pretty_response' in obs_output:
            pretty_data = obs_output.get('pretty_response', {})
            if isinstance(pretty_data, dict):
                args['response_with_citations'] = pretty_data.get('text', pretty_data.get('data', {}).get('text', ''))

    elif 'Chat Output' in obs_name or 'ChatOutput' in obs_name or 'Output' in obs_name:
        if 'message' in obs_output:
            message_data = obs_output.get('message', {})
            if isinstance(message_data, dict):
                args['text'] = message_data.get('text', message_data.get('data', {}).get('text', ''))

    elif 'If-Else' in obs_name or 'ConditionalRouter' in obs_name:
        # Extract operator and match_text from observation metadata
        obs_metadata = observation.get('metadata', {})
        args['operator'] = obs_metadata.get('operator', '')

        # Extract match_text from input
        obs_input = observation.get('input', {})
        args['match_text'] = obs_input.get('match_text', '')

        # Extract both true_value and false_value
        true_result = obs_output.get('true_result', {})
        false_result = obs_output.get('false_result', {})

        # Helper to extract text from result dict
        def extract_text(result):
            if not isinstance(result, dict):
                return None
            text = result.get('text', '')
            if not text:
                data = result.get('data', {})
                if isinstance(data, dict):
                    text = data.get('text', '')
            return text if text else None

        true_value = extract_text(true_result)
        false_value = extract_text(false_result)

        # Determine which branch was activated (has data)
        if true_value:
            args['true_value'] = true_value
            args['activated_branch'] = 'true'
            args['output_value'] = true_value  # Keep for speech bubble
        if false_value:
            args['false_value'] = false_value
            if 'activated_branch' not in args:
                args['activated_branch'] = 'false'
                args['output_value'] = false_value  # Keep for speech bubble

    else:
        # Generic component - extract first valid output
        # Skip internal keys that don't contain useful output
        skip_keys = {'logs', 'error', 'full_response'}

        for key, value in obs_output.items():
            if key in skip_keys:
                continue

            extracted_value = None

            if isinstance(value, str) and value:
                # Direct string value
                extracted_value = value
            elif isinstance(value, list):
                # Skip list values (complex structures like dataframes, tools)
                continue
            elif isinstance(value, dict):
                # Try to extract text from dict structure
                # Priority 1: value["data"]["text"]
                data = value.get('data')
                if isinstance(data, dict):
                    if 'text' in data and data['text']:
                        extracted_value = data['text']
                    else:
                        # Priority 3: entire value["data"] as string (fallback)
                        extracted_value = json.dumps(data)
                elif 'text' in value and value['text']:
                    # Priority 2: value["text"]
                    extracted_value = value['text']
                # If no data key and no text key, skip this output

            if extracted_value:
                args['output'] = extracted_value
                args['output_key'] = key
                break  # Use first valid output found

    return args


# =============================================================================
# FlowConfig Building
# =============================================================================

def get_node_display_name(node: Dict[str, Any]) -> str:
    """Extract display name from node."""
    node_data = node.get('data', {})
    node_info = node_data.get('node', {})

    display_name = node_info.get('display_name')
    if display_name:
        return display_name

    template = node_info.get('template', {})
    display_name = template.get('display_name')
    if display_name:
        return display_name

    return node.get('type', 'Unknown')


def get_component_type_from_node(node: Dict[str, Any]) -> str:
    """Determine component type from node for visualization."""
    node_id = node.get('id', '')
    node_data = node.get('data', {})
    node_info = node_data.get('node', {})
    display_name = node_info.get('display_name', '')

    if 'ChatInput' in node_id or 'ChatInput' in display_name:
        return 'chat_input'
    elif 'Retriever' in node_id or 'Retriever' in display_name or 'ELSER' in node_id:
        return 'retriever'
    elif 'BaseModel' in node_id or 'LLM' in node_id or 'Granite' in node_id or 'Model' in display_name:
        return 'llm'
    elif 'Citation' in node_id or 'Citation' in display_name:
        return 'citations'
    elif 'QueryRewrite' in node_id or 'Query Rewrite' in display_name or 'QueryRewrite' in display_name:
        return 'query_rewrite'
    elif 'QueryExpansion' in node_id or 'Query Expansion' in display_name or 'QueryExpansion' in display_name:
        return 'query_expansion'
    elif 'QueryClarification' in node_id or 'Query Clarification' in display_name or 'QueryClarification' in display_name:
        return 'query_clarification'
    elif 'HallucinationDetection' in node_id or 'Hallucination Detection' in display_name or 'HallucinationDetection' in display_name:
        return 'hallucination_detection'
    elif 'Answerability' in node_id or 'Answerability' in display_name:
        return 'answerability'
    elif 'ChatOutput' in node_id or 'ChatOutput' in display_name or 'Chat Output' in display_name:
        return 'chat_output'
    elif 'ConditionalRouter' in node_id or 'If-Else' in display_name or 'If Else' in display_name:
        return 'conditional'
    elif 'Text' in node_id:
        return 'text'
    else:
        return 'generic'


def get_output_components_from_edges(
    node_id: str,
    edge_list: List[Tuple[str, str]]
) -> List[str]:
    """Get list of output component IDs for a node from edge list."""
    output_ids = []
    for source, target in edge_list:
        if source == node_id:
            output_ids.append(target)
    return output_ids


def get_executed_component_ids(observations: List[Dict]) -> set:
    """
    Extract the set of component IDs that were actually executed from Langfuse observations.

    Args:
        observations: List of observations from LangFuse trace

    Returns:
        Set of component IDs that were executed
    """
    executed_ids = set()
    for obs in observations:
        metadata = obs.get('metadata', {})
        component_id = metadata.get('component_id')
        if component_id:
            executed_ids.add(component_id)
    return executed_ids


def build_flow_components(
    nodes: List[Dict],
    edge_list: List[Tuple[str, str]],
    observations: List[Dict],
    matches: List[Tuple[Optional[Dict], Optional[Dict]]]
) -> List[Dict]:
    """
    Build component objects for flowConfig.

    Args:
        nodes: Nodes from LangFlow schema
        edge_list: Logical flow edges
        observations: Observations from LangFuse trace
        matches: Matched (node, observation) tuples

    Returns:
        List of component dictionaries for flowConfig
    """
    # Get the set of executed component IDs from observations
    executed_ids = get_executed_component_ids(observations)

    components = []

    for node, obs in matches:
        if node is None:
            continue

        node_id = node.get('id')
        comp_type = get_component_type_from_node(node)
        label = get_node_display_name(node)
        output_components = get_output_components_from_edges(node_id, edge_list)

        args = {}
        if obs:
            args = extract_component_args(obs, comp_type)

        # Determine if this component was activated (executed) during the trace
        # A component is considered activated if it appears in the Langfuse observations
        activated = node_id in executed_ids

        # Determine if this component uses a specialized model
        # Components of type citations, query_rewrite, query_clarification, hallucination_detection, or answerability use specialized models
        # unless they contain "prompt" in the node ID or display name
        specialized_model = False
        if comp_type in ('citations', 'query_rewrite', 'query_clarification', 'hallucination_detection', 'answerability'):
            node_id_lower = node_id.lower() if node_id else ''
            label_lower = label.lower() if label else ''
            if 'prompt' not in node_id_lower and 'prompt' not in label_lower:
                specialized_model = True

        component = {
            'id': node_id,
            'type': comp_type,
            'label': label,
            'output_components': output_components,
            'args': args,
            'activated': activated,
            'specialized_model': specialized_model
        }

        components.append(component)

    return components


def build_flow_config(flow_name: str, components: List[Dict], trace_id: Optional[str] = None) -> Dict[str, Any]:
    """Assemble the complete flowConfig object."""
    config = {
        'name': flow_name,
        'components': components
    }
    if trace_id:
        config['trace_id'] = trace_id
    return config


def get_flow_name_from_trace(trace_data: Dict[str, Any]) -> str:
    """Extract flow name from trace metadata."""
    if not trace_data or 'trace' not in trace_data:
        return "Unknown Flow"

    trace_info = trace_data.get('trace', {})
    metadata = trace_info.get('metadata')

    if metadata:
        if isinstance(metadata, str):
            try:
                metadata_dict = json.loads(metadata)
                return metadata_dict.get('flow_name', 'Unknown Flow')
            except:
                pass
        elif isinstance(metadata, dict):
            return metadata.get('flow_name', 'Unknown Flow')

    return "Unknown Flow"


# =============================================================================
# Main Pipeline Function
# =============================================================================

def save_debug_file(data: Any, filepath: str) -> None:
    """Save data to a JSON file for debugging."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def get_flow_config(
    langfuse_base_url: str,
    langfuse_public_key: str,
    langfuse_secret_key: str,
    langflow_base_url: str,
    langflow_api_key: str,
    trace_id: Optional[str] = None,
    debug_mode: bool = False,
    debug_dir: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function to fetch trace and build flowConfig.

    Args:
        langfuse_base_url: LangFuse API base URL
        langfuse_public_key: LangFuse public API key
        langfuse_secret_key: LangFuse secret API key
        langflow_base_url: LangFlow API base URL
        langflow_api_key: LangFlow API key
        trace_id: Optional specific trace ID (defaults to latest)
        debug_mode: If True, save raw API responses to debug files
        debug_dir: Directory to save debug files (required if debug_mode is True)
        user_id: Optional user ID to filter traces by

    Returns:
        Complete flowConfig dictionary

    Raises:
        Exception: If trace or flow cannot be fetched
    """
    import os

    # Step 1: Fetch trace (specific or latest)
    if trace_id:
        trace_data = fetch_trace_from_api(
            trace_id,
            langfuse_base_url,
            langfuse_public_key,
            langfuse_secret_key
        )
    else:
        trace_data = fetch_latest_trace_from_api(
            langfuse_base_url,
            langfuse_public_key,
            langfuse_secret_key,
            user_id=user_id
        )

    if not trace_data:
        raise Exception("No trace found")

    # Get the actual trace_id for debug file naming
    actual_trace_id = trace_data.get('trace', {}).get('id', trace_id or 'unknown')

    # Save LangFuse trace data if debug mode is enabled
    if debug_mode and debug_dir:
        langfuse_debug_path = os.path.join(debug_dir, f"{actual_trace_id}-langfuse.json")
        save_debug_file(trace_data, langfuse_debug_path)
        print(f"[DEBUG] Saved LangFuse trace to: {langfuse_debug_path}")

    # Step 2: Extract flow_id from trace metadata
    flow_id = extract_flow_id_from_trace(trace_data)
    if not flow_id:
        raise Exception("Could not extract flow_id from trace metadata")

    # Step 3: Fetch flow schema from LangFlow
    flow_schema = fetch_flow_schema(flow_id, langflow_base_url, langflow_api_key)

    # Save LangFlow schema if debug mode is enabled
    if debug_mode and debug_dir:
        langflow_debug_path = os.path.join(debug_dir, f"{actual_trace_id}-langflow.json")
        save_debug_file(flow_schema, langflow_debug_path)
        print(f"[DEBUG] Saved LangFlow schema to: {langflow_debug_path}")

    # Step 4: Parse schema and trace
    nodes, edges = parse_langflow_schema(flow_schema)
    observations = parse_langfuse_trace(trace_data)

    # Step 5: Match components
    matches = match_components(nodes, observations)

    # Step 6: Compute logical flow edges
    logical_flow_edges = dependency_based_flow(edges)

    # Step 7: Build components
    components = build_flow_components(nodes, logical_flow_edges, observations, matches)

    # Step 8: Build flowConfig
    flow_name = get_flow_name_from_trace(trace_data)
    if flow_name == "Unknown Flow" and flow_schema:
        flow_name = flow_schema.get('name', flow_name)

    flow_config = build_flow_config(flow_name, components, actual_trace_id)

    return flow_config
