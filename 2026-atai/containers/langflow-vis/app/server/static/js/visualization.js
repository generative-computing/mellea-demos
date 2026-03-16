/* =============================================================================
   Flow Visualization JavaScript
   ============================================================================= */

// Configuration parameters - Clean Professional Infographic Style
const config = {
    enableAnimations: false,
    enableTypingAnimation: true,
    bubbleFontSize: 7,           // Readable size
    bubbleFontColor: '#333333', //'#475569',  // Slightly darker slate-600 for readability
    bubbleFontFamily: 'Inter, -apple-system, sans-serif',
    bubbleFontStyle: 'normal',
    componentLabelFontSize: 8,   // Elegant label size
    componentLabelSpacing: 14,   // More breathing room below icon
    componentLabelBackgroundPaddingX: 2,  // Horizontal padding for label background
    componentLabelBackgroundPaddingY: 2,  // Vertical padding for label background
    bubbleWidth: 54,             // Slightly larger for readability
    bubbleHeight: 42,
    bubbleVerticalGap: 15,
    bubbleFillColor: '#f0f9ff', //'#f5f5f5', //#f8fafc',  // Very light slate-50
    bubbleBorderWidth: 0.5,      // Thin, elegant border
    bubbleBorderColor: '#cbd5e1', // Soft slate-300
    bubbleShadowEnabled: false,
    bubbleShadowColor: 'rgba(15, 23, 42, 0.08)',
    bubbleShadowBlur: 6,
    bubbleShadowOffsetX: 0,
    bubbleShadowOffsetY: 2,
    typingSpeed: 40,
    iconSize: 30,                // Balanced circle size
    ringStrokeWidth: 2,          // Slightly thinner for elegance
    arrowWidth: 1.5,             // Elegant thin arrows
    arrowColor: '#9ca3af',       // Soft gray-400
    arrowGap: 7,
    arrowDashArray: '',
    componentSpacing: 50,
    branchSpacing: 100,
    branchingArrowStyle: 'curved',  // 'orthogonal' for L-shape, 'curved' for smooth curves, 'diagonal' for straight
    documentIconSize: 14,        // Larger for visibility
    documentBorderWidth: 1.2,
    documentBorderColor: '#cbd5e1',  // Same as bubble border (slate-300)
    variationIconSize: 12,       // Size for query variation bubble icons
    variationIconColor: '#06b6d4', // Cyan to match query_expansion
    documentShadowEnabled: false,
    documentShadowColor: 'rgba(0,0,0,0)',
    documentShadowBlur: 0,
    documentShadowOffsetX: 0,
    documentShadowOffsetY: 0,
    animationCycleDuration: 12000,
    animationRestartDelay: 2000,
    enableAnimationLoop: false,  // Whether animation should loop continuously
    queryDelay: 1000,
    documentStagger: 400,
    documentDelay: 800,
    maxTextLength: 200,  // Safety net for DOM - CSS line-clamp handles visual truncation
    showZoomControls: true,
    zoomIncrement: 0.1,
    zoomMin: 0.5,
    zoomMax: 1.5,
    inactiveComponentOpacity: 0.45,  // Opacity for non-activated components (0-1)
    showSpecializedModelBadge: true  // Whether to show badge on components using specialized models
};

// Helper function to truncate text for display (safety net - CSS handles visual ellipsis)
function truncateForDisplay(text) {
    if (!text) return text;
    if (text.length <= config.maxTextLength) return text;
    return text.substring(0, config.maxTextLength);
}

// Professional color palette for ring borders (distinct, harmonious colors)
const componentColors = {
    chat_input: '#3b82f6',     // blue-500 (clear blue)
    chat_output: '#6366f1',    // indigo-500 (distinct indigo - pairs with blue input)
    query_rewrite: '#ec4899',  // pink-500 (distinct pink/magenta)
    query_expansion: '#06b6d4', // cyan-500 (distinct cyan/teal)
    query_clarification: '#14b8a6', // teal-500 (distinct teal for clarification)
    retriever: '#22c55e',      // green-500 (fresh green)
    llm: '#f59e0b',            // amber-500 (warm amber)
    citations: '#f97316',      // orange-500 (warm orange)
    answerability: '#8b5cf6',  // violet-500 (distinct violet)
    conditional: '#f43f5e',    // rose-500 (distinct rose for branching/decisions)
    text: '#64748b',           // slate-500 (neutral gray)
    hallucination_detection: '#dc2626',  // red-600 (red for hallucination/risk detection)
    generic: '#9ca3af'         // gray-400 (neutral gray for unknown types)
};

// Flow configuration - will be loaded dynamically from API
let flowConfig = null;
let flowComponents = [];

// State tracking
let isLoading = false;
let visualizationInitialized = false;
let selectedComponent = null;

// Trace selector state
let currentTraceId = null;
let traceList = [];
let traceDropdownOpen = false;
let traceCurrentPage = 1;
let traceHasMore = true;
let traceLoadingMore = false;

// Properties panel DOM references (set on DOMContentLoaded)
let propertiesPanel = null;
let propertiesPanelContent = null;
let closePanelBtn = null;

// Validate graph and compute layout (depth and row for each component)
// Supports:
// - Branching (one component -> multiple outputs)
// - Merging (multiple inputs -> one component)
// - Multiple starting nodes (multiple entry points)
// - Filters out isolated/disconnected nodes (like notes)
function validateAndComputeLayout() {
    // Create a map of component ID to component
    const componentMap = {};
    flowComponents.forEach(comp => {
        componentMap[comp.id] = comp;
    });

    // Count incoming and outgoing edges for each component
    const incomingEdgeCount = {};
    const outgoingEdgeCount = {};
    const incomingSources = {}; // { componentId: [sourceId1, sourceId2, ...] }
    flowComponents.forEach(comp => {
        if (!incomingEdgeCount[comp.id]) incomingEdgeCount[comp.id] = 0;
        if (!outgoingEdgeCount[comp.id]) outgoingEdgeCount[comp.id] = 0;
        if (!incomingSources[comp.id]) incomingSources[comp.id] = [];

        outgoingEdgeCount[comp.id] = comp.output_components.length;

        comp.output_components.forEach(outputId => {
            incomingEdgeCount[outputId] = (incomingEdgeCount[outputId] || 0) + 1;
            if (!incomingSources[outputId]) incomingSources[outputId] = [];
            incomingSources[outputId].push(comp.id);
        });
    });

    // Filter out isolated nodes (nodes with no incoming AND no outgoing edges)
    // These are typically notes or disconnected components
    const connectedComponents = flowComponents.filter(comp => {
        const hasIncoming = incomingEdgeCount[comp.id] > 0;
        const hasOutgoing = outgoingEdgeCount[comp.id] > 0;
        return hasIncoming || hasOutgoing;
    });

    // Log filtered out components
    const filteredOut = flowComponents.filter(comp => {
        const hasIncoming = incomingEdgeCount[comp.id] > 0;
        const hasOutgoing = outgoingEdgeCount[comp.id] > 0;
        return !hasIncoming && !hasOutgoing;
    });
    if (filteredOut.length > 0) {
        console.log(`Filtered out ${filteredOut.length} isolated component(s): ${filteredOut.map(c => c.id).join(', ')}`);
    }

    // Update flowComponents to only include connected components
    flowComponents = connectedComponents;

    // Find starting nodes (nodes with no incoming edges)
    const startNodes = flowComponents.filter(comp => !incomingEdgeCount[comp.id] || incomingEdgeCount[comp.id] === 0);

    // Validate: must have at least one start node
    if (startNodes.length === 0) {
        throw new Error('Graph validation failed: No starting node found (all nodes have incoming edges). Cannot draw graph.');
    }

    // Log multiple start nodes info
    if (startNodes.length > 1) {
        console.log(`Graph has ${startNodes.length} starting nodes: ${startNodes.map(n => n.id).join(', ')}`);
    }

    // Phase 1: Compute depth using topological sort (Kahn's algorithm)
    // Initialize all start nodes at depth 0
    const layout = {}; // { componentId: { depth, row, component } }
    const remainingIncoming = { ...incomingEdgeCount }; // Track remaining unprocessed incoming edges
    const queue = startNodes.map(node => node.id);
    const depthMap = {}; // { componentId: depth }

    // Initialize all start nodes at depth 0
    startNodes.forEach(node => {
        depthMap[node.id] = 0;
    });

    while (queue.length > 0) {
        const id = queue.shift();
        const component = componentMap[id];
        const currentDepth = depthMap[id];

        // Process all children
        component.output_components.forEach(childId => {
            // Skip children that aren't in our connected components
            if (!componentMap[childId] || !flowComponents.find(c => c.id === childId)) {
                return;
            }

            // Update child's depth to be max of all incoming paths + 1
            const newDepth = currentDepth + 1;
            if (depthMap[childId] === undefined || depthMap[childId] < newDepth) {
                depthMap[childId] = newDepth;
            }

            // Decrement remaining incoming edges for child
            remainingIncoming[childId]--;

            // If all incoming edges are processed, add to queue
            if (remainingIncoming[childId] === 0) {
                queue.push(childId);
            }
        });
    }

    // Check for cycles (nodes with remaining incoming edges weren't processed)
    const processedCount = Object.keys(depthMap).length;
    if (processedCount !== flowComponents.length) {
        const unprocessed = flowComponents.filter(c => depthMap[c.id] === undefined).map(c => c.id);
        throw new Error(`Graph validation failed: Cycle detected or unreachable nodes: ${unprocessed.join(', ')}. Cannot draw graph.`);
    }

    // Phase 2: DFS-based row assignment with subtree-aware ordering
    // Key insight: order children by their "max reachable depth" so nodes that
    // terminate at similar depths are grouped together (minimizing crossings with final arrows)

    const rowMap = {}; // { componentId: row }
    let nextRow = 0;
    const maxDepth = Math.max(...Object.values(depthMap));

    // Identify merge nodes (nodes with multiple incoming edges)
    const mergeNodes = new Set();
    for (const [id, count] of Object.entries(incomingEdgeCount)) {
        if (count > 1) {
            mergeNodes.add(id);
        }
    }

    // Group nodes by depth for later reference
    const nodesByDepth = {};
    for (const [id, depth] of Object.entries(depthMap)) {
        if (!nodesByDepth[depth]) nodesByDepth[depth] = [];
        nodesByDepth[depth].push(id);
    }

    // Step 1: Compute max reachable depth for each node using DFS (memoized)
    const maxReachableDepth = {}; // { nodeId: maxDepth reachable from this node }

    function computeMaxReachableDepth(nodeId) {
        if (maxReachableDepth[nodeId] !== undefined) {
            return maxReachableDepth[nodeId];
        }

        const comp = componentMap[nodeId];
        if (!comp) {
            maxReachableDepth[nodeId] = depthMap[nodeId] || 0;
            return maxReachableDepth[nodeId];
        }

        // Filter to valid children
        const validChildren = comp.output_components.filter(childId => componentMap[childId]);

        if (validChildren.length === 0) {
            // Terminal node - max depth is its own depth (plus 1 for final arrow)
            maxReachableDepth[nodeId] = depthMap[nodeId] + 1;
        } else {
            // Max of children's max reachable depths
            let maxChild = depthMap[nodeId];
            for (const childId of validChildren) {
                const childMax = computeMaxReachableDepth(childId);
                if (childMax > maxChild) maxChild = childMax;
            }
            maxReachableDepth[nodeId] = maxChild;
        }

        return maxReachableDepth[nodeId];
    }

    // Compute for all nodes
    for (const comp of flowComponents) {
        computeMaxReachableDepth(comp.id);
    }

    // Step 2: DFS to assign rows with row reuse to minimize vertical spread
    // Track which rows are occupied at each depth
    const occupiedAtDepth = {}; // { depth: Set of occupied rows }
    for (let d = 0; d <= maxDepth; d++) {
        occupiedAtDepth[d] = new Set();
    }

    // Check if a node will have a final arrow (no children and not chat_output)
    function hasFinalArrow(nodeId) {
        const comp = componentMap[nodeId];
        if (!comp) return false;
        const validChildren = comp.output_components.filter(childId => componentMap[childId]);
        return validChildren.length === 0 && comp.type !== 'chat_output';
    }

    // Mark row as occupied - for nodes with final arrows, also reserve the NEXT depth
    // (Final arrows only extend one componentSpacing, i.e., to the next depth level)
    function markRowOccupied(nodeId, row) {
        const nodeDepth = depthMap[nodeId];
        occupiedAtDepth[nodeDepth].add(row);

        // If this node has a final arrow, reserve this row for the NEXT depth only
        // This prevents nodes at the immediately adjacent depth from appearing at the arrow's endpoint
        if (hasFinalArrow(nodeId) && nodeDepth + 1 <= maxDepth) {
            occupiedAtDepth[nodeDepth + 1].add(row);
        }
    }

    // Find the closest available row to targetRow for a node
    // For nodes with final arrows, also check that no node exists at the NEXT depth on that row
    function findClosestAvailableRow(nodeId, targetRow) {
        const nodeDepth = depthMap[nodeId];
        const nodeHasFinalArrow = hasFinalArrow(nodeId);

        function isRowAvailable(row) {
            // Always check the node's own depth
            if (occupiedAtDepth[nodeDepth].has(row)) return false;

            // If this node will have a final arrow, check the NEXT depth only
            // (Final arrow only extends one componentSpacing to the right)
            if (nodeHasFinalArrow && nodeDepth + 1 <= maxDepth) {
                if (occupiedAtDepth[nodeDepth + 1].has(row)) return false;
            }
            return true;
        }

        // Try targetRow first
        if (isRowAvailable(targetRow)) return targetRow;

        // Search outward from targetRow
        for (let offset = 1; offset <= 100; offset++) {
            // Try below (higher row number)
            if (isRowAvailable(targetRow + offset)) {
                return targetRow + offset;
            }
            // Try above (lower row number) - but not negative
            if (targetRow - offset >= 0 && isRowAvailable(targetRow - offset)) {
                return targetRow - offset;
            }
        }

        // Fallback
        return nextRow++;
    }

    const rowAssignVisited = new Set();

    function assignRowsDFS(nodeId, preferredRow) {
        if (rowAssignVisited.has(nodeId)) return;
        rowAssignVisited.add(nodeId);

        // Find closest available row (considers final arrow constraints)
        const assignedRow = findClosestAvailableRow(nodeId, preferredRow);
        rowMap[nodeId] = assignedRow;
        markRowOccupied(nodeId, assignedRow);

        // Update nextRow if we used a higher row
        if (assignedRow >= nextRow) {
            nextRow = assignedRow + 1;
        }

        const comp = componentMap[nodeId];
        if (!comp) return;

        // Get valid children (excluding merge nodes which we'll handle later)
        const validChildren = comp.output_components.filter(childId =>
            componentMap[childId] && !mergeNodes.has(childId) && !rowAssignVisited.has(childId)
        );

        if (validChildren.length === 0) return;

        // Sort children by max reachable depth (descending - deeper subtrees first)
        // This groups nodes that terminate at similar depths together
        validChildren.sort((a, b) => maxReachableDepth[b] - maxReachableDepth[a]);

        // Assign rows: first child prefers same row as parent, others get offset rows
        validChildren.forEach((childId, index) => {
            assignRowsDFS(childId, assignedRow + index);
        });
    }

    // Start from all start nodes for row assignment
    const rowAssignStartNodes = flowComponents.filter(comp =>
        !incomingEdgeCount[comp.id] || incomingEdgeCount[comp.id] === 0
    );

    for (const startNode of rowAssignStartNodes) {
        assignRowsDFS(startNode.id, nextRow);
    }

    // Step 3: Handle merge nodes - assign them to the row of their topmost source
    // Process in depth order to ensure sources are assigned first
    for (let d = 0; d <= maxDepth; d++) {
        const nodesAtDepthList = nodesByDepth[d] || [];
        for (const nodeId of nodesAtDepthList) {
            if (mergeNodes.has(nodeId) && rowMap[nodeId] === undefined) {
                const sourceRows = incomingSources[nodeId]
                    .filter(srcId => rowMap[srcId] !== undefined)
                    .map(srcId => rowMap[srcId]);
                const preferredRow = sourceRows.length > 0 ? Math.min(...sourceRows) : nextRow;
                const assignedRow = findClosestAvailableRow(nodeId, preferredRow);
                rowMap[nodeId] = assignedRow;
                markRowOccupied(nodeId, assignedRow);
                if (assignedRow >= nextRow) nextRow = assignedRow + 1;
            }
        }
    }

    // Ensure all nodes have rows assigned
    // Process in depth order so sources are assigned before their targets
    for (let d = 0; d <= maxDepth; d++) {
        const nodesAtDepthList = nodesByDepth[d] || [];
        for (const nodeId of nodesAtDepthList) {
            if (rowMap[nodeId] === undefined) {
                // Calculate preferred row based on sources (like merge nodes do)
                const sourceRows = (incomingSources[nodeId] || [])
                    .filter(srcId => rowMap[srcId] !== undefined)
                    .map(srcId => rowMap[srcId]);
                const preferredRow = sourceRows.length > 0 ? Math.min(...sourceRows) : nextRow;
                const assignedRow = findClosestAvailableRow(nodeId, preferredRow);
                rowMap[nodeId] = assignedRow;
                markRowOccupied(nodeId, assignedRow);
                if (assignedRow >= nextRow) nextRow = assignedRow + 1;
            }
        }
    }

    // Phase 3: Collect edges and log layout info
    const allEdges = [];
    const FINAL_ARROW_DEPTH = maxDepth + 1;

    for (const comp of flowComponents) {
        if (comp.output_components.length > 0) {
            for (const targetId of comp.output_components) {
                if (componentMap[targetId]) {
                    allEdges.push({ source: comp.id, target: targetId, isFinal: false });
                }
            }
        } else if (comp.type !== 'chat_output') {
            allEdges.push({ source: comp.id, target: null, isFinal: true });
        }
    }

    // Count crossings between two edges
    function edgesCross(e1, e2, rowMapToUse) {
        const r1 = rowMapToUse[e1.source];
        const r2 = e1.isFinal ? rowMapToUse[e1.source] : rowMapToUse[e1.target];
        const s1 = rowMapToUse[e2.source];
        const s2 = e2.isFinal ? rowMapToUse[e2.source] : rowMapToUse[e2.target];

        const d1src = depthMap[e1.source];
        const d1tgt = e1.isFinal ? FINAL_ARROW_DEPTH : depthMap[e1.target];
        const d2src = depthMap[e2.source];
        const d2tgt = e2.isFinal ? FINAL_ARROW_DEPTH : depthMap[e2.target];

        const minD1 = Math.min(d1src, d1tgt);
        const maxD1 = Math.max(d1src, d1tgt);
        const minD2 = Math.min(d2src, d2tgt);
        const maxD2 = Math.max(d2src, d2tgt);

        if (maxD1 < minD2 || maxD2 < minD1) return false;
        return (r1 < s1 && r2 > s2) || (r1 > s1 && r2 < s2);
    }

    function countTotalCrossings(rowMapToUse) {
        let count = 0;
        for (let i = 0; i < allEdges.length; i++) {
            for (let j = i + 1; j < allEdges.length; j++) {
                if (edgesCross(allEdges[i], allEdges[j], rowMapToUse)) {
                    count++;
                }
            }
        }
        return count;
    }

    function findCrossingEdges(rowMapToUse) {
        const pairs = [];
        for (let i = 0; i < allEdges.length; i++) {
            for (let j = i + 1; j < allEdges.length; j++) {
                if (edgesCross(allEdges[i], allEdges[j], rowMapToUse)) {
                    const e1 = allEdges[i], e2 = allEdges[j];
                    pairs.push({
                        e1: e1.isFinal ? `${e1.source}→[final]` : `${e1.source}→${e1.target}`,
                        e2: e2.isFinal ? `${e2.source}→[final]` : `${e2.source}→${e2.target}`,
                        e1_rows: `${rowMapToUse[e1.source]}→${e1.isFinal ? rowMapToUse[e1.source] : rowMapToUse[e1.target]}`,
                        e2_rows: `${rowMapToUse[e2.source]}→${e2.isFinal ? rowMapToUse[e2.source] : rowMapToUse[e2.target]}`
                    });
                }
            }
        }
        return pairs;
    }

    const crossings = countTotalCrossings(rowMap);
    console.log(`[Layout] DFS-based layout - edge crossings: ${crossings}`);
    console.log(`[Layout] Max reachable depths:`, maxReachableDepth);
    console.log(`[Layout] Node positions:`, Object.fromEntries(
        flowComponents.map(c => [c.id, { depth: depthMap[c.id], row: rowMap[c.id], maxReach: maxReachableDepth[c.id] }])
    ));
    if (crossings > 0) {
        console.log(`[Layout] Crossing edge pairs:`, findCrossingEdges(rowMap));
    }

    // Build final layout
    for (const comp of flowComponents) {
        layout[comp.id] = {
            depth: depthMap[comp.id],
            row: rowMap[comp.id],
            component: comp
        };
    }

    // Check if all components are reachable
    const visited = new Set(Object.keys(layout));
    if (visited.size !== flowComponents.length) {
        const unreachable = flowComponents.filter(c => !visited.has(c.id)).map(c => c.id);
        console.warn(`Warning: Some components are not reachable from the start nodes: ${unreachable.join(', ')}`);
    }

    return layout;
}

// Component layout variables - will be computed when data is loaded
let componentLayout;
let orderedFlowComponents;

// Compute and validate layout - called when flowConfig is loaded
function computeLayout() {
    componentLayout = validateAndComputeLayout();
    // Create ordered list sorted by depth then row for rendering
    orderedFlowComponents = Object.values(componentLayout)
        .sort((a, b) => {
            if (a.depth !== b.depth) return a.depth - b.depth;
            return a.row - b.row;
        })
        .map(item => item.component);
}

// Icon templates - Professional ring style with line icons
const iconTemplates = {
    chat_input: function(iconGroup, color) {
        // Ring circle (stroke only, no fill)
        const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        ring.setAttribute('cx', '20');
        ring.setAttribute('cy', '20');
        ring.setAttribute('r', '18');
        ring.setAttribute('fill', 'none');
        ring.setAttribute('stroke', color);
        ring.setAttribute('stroke-width', config.ringStrokeWidth || '2.5');

        // User head (stroke outline)
        const head = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        head.setAttribute('cx', '20');
        head.setAttribute('cy', '16');
        head.setAttribute('r', '5');
        head.setAttribute('fill', 'none');
        head.setAttribute('stroke', color);
        head.setAttribute('stroke-width', '1.5');

        // User body (stroke outline)
        const body = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        body.setAttribute('d', 'M 12 32 C 12 27 15 24 20 24 C 25 24 28 27 28 32');
        body.setAttribute('fill', 'none');
        body.setAttribute('stroke', color);
        body.setAttribute('stroke-width', '1.5');
        body.setAttribute('stroke-linecap', 'round');

        iconGroup.appendChild(ring);
        iconGroup.appendChild(head);
        iconGroup.appendChild(body);
    },
    chat_output: function(iconGroup, color) {
        // Ring circle (stroke only, no fill)
        const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        ring.setAttribute('cx', '20');
        ring.setAttribute('cy', '20');
        ring.setAttribute('r', '18');
        ring.setAttribute('fill', 'none');
        ring.setAttribute('stroke', color);
        ring.setAttribute('stroke-width', config.ringStrokeWidth || '2.5');

        // Chat bubble outline
        const bubble = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        bubble.setAttribute('d', 'M 12 14 L 28 14 Q 30 14 30 16 L 30 24 Q 30 26 28 26 L 18 26 L 14 30 L 14 26 L 12 26 Q 10 26 10 24 L 10 16 Q 10 14 12 14 Z');
        bubble.setAttribute('fill', 'none');
        bubble.setAttribute('stroke', color);
        bubble.setAttribute('stroke-width', '1.5');
        bubble.setAttribute('stroke-linejoin', 'round');

        // Three dots inside bubble (representing text output)
        const dot1 = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        dot1.setAttribute('cx', '15');
        dot1.setAttribute('cy', '20');
        dot1.setAttribute('r', '1.5');
        dot1.setAttribute('fill', color);

        const dot2 = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        dot2.setAttribute('cx', '20');
        dot2.setAttribute('cy', '20');
        dot2.setAttribute('r', '1.5');
        dot2.setAttribute('fill', color);

        const dot3 = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        dot3.setAttribute('cx', '25');
        dot3.setAttribute('cy', '20');
        dot3.setAttribute('r', '1.5');
        dot3.setAttribute('fill', color);

        iconGroup.appendChild(ring);
        iconGroup.appendChild(bubble);
        iconGroup.appendChild(dot1);
        iconGroup.appendChild(dot2);
        iconGroup.appendChild(dot3);
    },
    retriever: function(iconGroup, color) {
        // Ring circle
        const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        ring.setAttribute('cx', '20');
        ring.setAttribute('cy', '20');
        ring.setAttribute('r', '18');
        ring.setAttribute('fill', 'none');
        ring.setAttribute('stroke', color);
        ring.setAttribute('stroke-width', config.ringStrokeWidth || '2.5');

        // Search glass circle (stroke)
        const glass = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        glass.setAttribute('cx', '17');
        glass.setAttribute('cy', '17');
        glass.setAttribute('r', '6');
        glass.setAttribute('fill', 'none');
        glass.setAttribute('stroke', color);
        glass.setAttribute('stroke-width', '1.5');

        // Search handle (stroke line)
        const handle = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        handle.setAttribute('x1', '22');
        handle.setAttribute('y1', '22');
        handle.setAttribute('x2', '28');
        handle.setAttribute('y2', '28');
        handle.setAttribute('stroke', color);
        handle.setAttribute('stroke-width', '2');
        handle.setAttribute('stroke-linecap', 'round');

        iconGroup.appendChild(ring);
        iconGroup.appendChild(glass);
        iconGroup.appendChild(handle);
    },
    llm: function(iconGroup, color) {
        // Ring circle
        const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        ring.setAttribute('cx', '20');
        ring.setAttribute('cy', '20');
        ring.setAttribute('r', '18');
        ring.setAttribute('fill', 'none');
        ring.setAttribute('stroke', color);
        ring.setAttribute('stroke-width', config.ringStrokeWidth || '2');

        // Brain icon with organic curves
        // Left hemisphere curve
        const leftBrain = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        leftBrain.setAttribute('d', 'M 15 14 Q 12 17 12 20 Q 12 23 15 26');
        leftBrain.setAttribute('fill', 'none');
        leftBrain.setAttribute('stroke', color);
        leftBrain.setAttribute('stroke-width', '1.5');
        leftBrain.setAttribute('stroke-linecap', 'round');

        // Right hemisphere curve
        const rightBrain = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        rightBrain.setAttribute('d', 'M 25 14 Q 28 17 28 20 Q 28 23 25 26');
        rightBrain.setAttribute('fill', 'none');
        rightBrain.setAttribute('stroke', color);
        rightBrain.setAttribute('stroke-width', '1.5');
        rightBrain.setAttribute('stroke-linecap', 'round');

        // Top connection
        const topConnection = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        topConnection.setAttribute('d', 'M 15 14 Q 20 11 25 14');
        topConnection.setAttribute('fill', 'none');
        topConnection.setAttribute('stroke', color);
        topConnection.setAttribute('stroke-width', '1.5');
        topConnection.setAttribute('stroke-linecap', 'round');

        // Bottom connection
        const bottomConnection = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        bottomConnection.setAttribute('d', 'M 15 26 Q 20 29 25 26');
        bottomConnection.setAttribute('fill', 'none');
        bottomConnection.setAttribute('stroke', color);
        bottomConnection.setAttribute('stroke-width', '1.5');
        bottomConnection.setAttribute('stroke-linecap', 'round');

        // Neural pathways (small curves inside)
        const pathway1 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        pathway1.setAttribute('d', 'M 16 18 Q 18 17 20 18');
        pathway1.setAttribute('fill', 'none');
        pathway1.setAttribute('stroke', color);
        pathway1.setAttribute('stroke-width', '1');

        const pathway2 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        pathway2.setAttribute('d', 'M 20 22 Q 22 23 24 22');
        pathway2.setAttribute('fill', 'none');
        pathway2.setAttribute('stroke', color);
        pathway2.setAttribute('stroke-width', '1');

        iconGroup.appendChild(ring);
        iconGroup.appendChild(leftBrain);
        iconGroup.appendChild(rightBrain);
        iconGroup.appendChild(topConnection);
        iconGroup.appendChild(bottomConnection);
        iconGroup.appendChild(pathway1);
        iconGroup.appendChild(pathway2);
    },
    query_rewrite: function(iconGroup, color) {
        // Ring circle
        const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        ring.setAttribute('cx', '20');
        ring.setAttribute('cy', '20');
        ring.setAttribute('r', '18');
        ring.setAttribute('fill', 'none');
        ring.setAttribute('stroke', color);
        ring.setAttribute('stroke-width', config.ringStrokeWidth || '2.5');

        // Pencil icon (stroke style, tilted)
        const pencilGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        pencilGroup.setAttribute('transform', 'rotate(-45 20 20)');

        const pencilBody = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        pencilBody.setAttribute('x', '17');
        pencilBody.setAttribute('y', '10');
        pencilBody.setAttribute('width', '6');
        pencilBody.setAttribute('height', '16');
        pencilBody.setAttribute('fill', 'none');
        pencilBody.setAttribute('stroke', color);
        pencilBody.setAttribute('stroke-width', '1.5');
        pencilBody.setAttribute('rx', '1');

        const pencilTip = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        pencilTip.setAttribute('d', 'M 17 26 L 20 31 L 23 26');
        pencilTip.setAttribute('fill', 'none');
        pencilTip.setAttribute('stroke', color);
        pencilTip.setAttribute('stroke-width', '1.5');
        pencilTip.setAttribute('stroke-linejoin', 'round');

        pencilGroup.appendChild(pencilBody);
        pencilGroup.appendChild(pencilTip);
        iconGroup.appendChild(ring);
        iconGroup.appendChild(pencilGroup);
    },
    query_expansion: function(iconGroup, color) {
        // Ring circle
        const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        ring.setAttribute('cx', '20');
        ring.setAttribute('cy', '20');
        ring.setAttribute('r', '18');
        ring.setAttribute('fill', 'none');
        ring.setAttribute('stroke', color);
        ring.setAttribute('stroke-width', config.ringStrokeWidth || '2.5');

        // Expansion icon: one arrow splitting into three (centered at x=20)
        // Stem from left to center
        const stem = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        stem.setAttribute('d', 'M 11 20 L 20 20');
        stem.setAttribute('fill', 'none');
        stem.setAttribute('stroke', color);
        stem.setAttribute('stroke-width', '1.5');
        stem.setAttribute('stroke-linecap', 'round');

        // Top branch (from center, diagonal up-right)
        const branch1 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        branch1.setAttribute('d', 'M 20 20 L 26 14 L 29 14');
        branch1.setAttribute('fill', 'none');
        branch1.setAttribute('stroke', color);
        branch1.setAttribute('stroke-width', '1.5');
        branch1.setAttribute('stroke-linecap', 'round');
        branch1.setAttribute('stroke-linejoin', 'round');

        // Middle branch (from center, straight right)
        const branch2 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        branch2.setAttribute('d', 'M 20 20 L 29 20');
        branch2.setAttribute('fill', 'none');
        branch2.setAttribute('stroke', color);
        branch2.setAttribute('stroke-width', '1.5');
        branch2.setAttribute('stroke-linecap', 'round');

        // Bottom branch (from center, diagonal down-right)
        const branch3 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        branch3.setAttribute('d', 'M 20 20 L 26 26 L 29 26');
        branch3.setAttribute('fill', 'none');
        branch3.setAttribute('stroke', color);
        branch3.setAttribute('stroke-width', '1.5');
        branch3.setAttribute('stroke-linecap', 'round');
        branch3.setAttribute('stroke-linejoin', 'round');

        iconGroup.appendChild(ring);
        iconGroup.appendChild(stem);
        iconGroup.appendChild(branch1);
        iconGroup.appendChild(branch2);
        iconGroup.appendChild(branch3);
    },
    query_clarification: function(iconGroup, color) {
        // Ring circle
        const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        ring.setAttribute('cx', '20');
        ring.setAttribute('cy', '20');
        ring.setAttribute('r', '18');
        ring.setAttribute('fill', 'none');
        ring.setAttribute('stroke', color);
        ring.setAttribute('stroke-width', config.ringStrokeWidth || '2.5');

        // Speech bubble outline (positioned lower in circle)
        const bubble = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        bubble.setAttribute('d', 'M 12 12 L 28 12 Q 30 12 30 14 L 30 24 Q 30 26 28 26 L 18 26 L 14 30 L 14 26 L 12 26 Q 10 26 10 24 L 10 14 Q 10 12 12 12 Z');
        bubble.setAttribute('fill', 'none');
        bubble.setAttribute('stroke', color);
        bubble.setAttribute('stroke-width', '1.5');
        bubble.setAttribute('stroke-linejoin', 'round');

        // Smaller question mark inside bubble
        const questionMark = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        questionMark.setAttribute('d', 'M 18 16 Q 18 14.5 20 14.5 Q 22 14.5 22 16 Q 22 17.5 20 18.5 L 20 20');
        questionMark.setAttribute('fill', 'none');
        questionMark.setAttribute('stroke', color);
        questionMark.setAttribute('stroke-width', '1.3');
        questionMark.setAttribute('stroke-linecap', 'round');

        // Question mark dot
        const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        dot.setAttribute('cx', '20');
        dot.setAttribute('cy', '22.5');
        dot.setAttribute('r', '0.9');
        dot.setAttribute('fill', color);

        iconGroup.appendChild(ring);
        iconGroup.appendChild(bubble);
        iconGroup.appendChild(questionMark);
        iconGroup.appendChild(dot);
    },
    hallucination_detection: function(iconGroup, color) {
        // Ring circle
        const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        ring.setAttribute('cx', '20');
        ring.setAttribute('cy', '20');
        ring.setAttribute('r', '18');
        ring.setAttribute('fill', 'none');
        ring.setAttribute('stroke', color);
        ring.setAttribute('stroke-width', config.ringStrokeWidth || '2.5');

        // Vertical dividing line
        const divider = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        divider.setAttribute('x1', '20');
        divider.setAttribute('y1', '8');
        divider.setAttribute('x2', '20');
        divider.setAttribute('y2', '32');
        divider.setAttribute('stroke', color);
        divider.setAttribute('stroke-width', '1.5');

        // Checkmark on left side (matching X height)
        const checkmark = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        checkmark.setAttribute('d', 'M 8 19 L 11 22 L 16 16');
        checkmark.setAttribute('fill', 'none');
        checkmark.setAttribute('stroke', color);
        checkmark.setAttribute('stroke-width', '1.5');
        checkmark.setAttribute('stroke-linecap', 'round');
        checkmark.setAttribute('stroke-linejoin', 'round');

        // X mark on right side (square proportions, vertically centered with checkmark)
        const xmark = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        xmark.setAttribute('d', 'M 25 16 L 31 22 M 31 16 L 25 22');
        xmark.setAttribute('fill', 'none');
        xmark.setAttribute('stroke', color);
        xmark.setAttribute('stroke-width', '1.5');
        xmark.setAttribute('stroke-linecap', 'round');

        iconGroup.appendChild(ring);
        iconGroup.appendChild(divider);
        iconGroup.appendChild(checkmark);
        iconGroup.appendChild(xmark);
    },
    citations: function(iconGroup, color) {
        // Ring circle
        const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        ring.setAttribute('cx', '20');
        ring.setAttribute('cy', '20');
        ring.setAttribute('r', '18');
        ring.setAttribute('fill', 'none');
        ring.setAttribute('stroke', color);
        ring.setAttribute('stroke-width', config.ringStrokeWidth || '2.5');

        // Document outline (stroke style)
        const doc = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        doc.setAttribute('x', '14');
        doc.setAttribute('y', '12');
        doc.setAttribute('width', '12');
        doc.setAttribute('height', '16');
        doc.setAttribute('fill', 'none');
        doc.setAttribute('stroke', color);
        doc.setAttribute('stroke-width', '1.5');
        doc.setAttribute('rx', '1');

        // Opening quotation mark (left)
        const quote1 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        quote1.setAttribute('d', 'M 17 18 Q 16 17 16 16 M 17 18 L 17 21');
        quote1.setAttribute('fill', 'none');
        quote1.setAttribute('stroke', color);
        quote1.setAttribute('stroke-width', '1.5');
        quote1.setAttribute('stroke-linecap', 'round');

        const quote2 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        quote2.setAttribute('d', 'M 19.5 18 Q 18.5 17 18.5 16 M 19.5 18 L 19.5 21');
        quote2.setAttribute('fill', 'none');
        quote2.setAttribute('stroke', color);
        quote2.setAttribute('stroke-width', '1.5');
        quote2.setAttribute('stroke-linecap', 'round');

        // Closing quotation mark (right)
        const quote3 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        quote3.setAttribute('d', 'M 21 23 Q 22 24 22 25 M 21 23 L 21 20');
        quote3.setAttribute('fill', 'none');
        quote3.setAttribute('stroke', color);
        quote3.setAttribute('stroke-width', '1.5');
        quote3.setAttribute('stroke-linecap', 'round');

        const quote4 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        quote4.setAttribute('d', 'M 23.5 23 Q 24.5 24 24.5 25 M 23.5 23 L 23.5 20');
        quote4.setAttribute('fill', 'none');
        quote4.setAttribute('stroke', color);
        quote4.setAttribute('stroke-width', '1.5');
        quote4.setAttribute('stroke-linecap', 'round');

        iconGroup.appendChild(ring);
        iconGroup.appendChild(doc);
        iconGroup.appendChild(quote1);
        iconGroup.appendChild(quote2);
        iconGroup.appendChild(quote3);
        iconGroup.appendChild(quote4);
    },
    answerability: function(iconGroup, color) {
        // Ring circle
        const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        ring.setAttribute('cx', '20');
        ring.setAttribute('cy', '20');
        ring.setAttribute('r', '18');
        ring.setAttribute('fill', 'none');
        ring.setAttribute('stroke', color);
        ring.setAttribute('stroke-width', config.ringStrokeWidth || '2');

        // Large question mark
        const qTop = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        qTop.setAttribute('d', 'M 15 15 Q 15 11 20 11 Q 25 11 25 15 Q 25 19 20 21');
        qTop.setAttribute('fill', 'none');
        qTop.setAttribute('stroke', color);
        qTop.setAttribute('stroke-width', '2');
        qTop.setAttribute('stroke-linecap', 'round');

        const qDot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        qDot.setAttribute('cx', '20');
        qDot.setAttribute('cy', '27');
        qDot.setAttribute('r', '2');
        qDot.setAttribute('fill', color);

        iconGroup.appendChild(ring);
        iconGroup.appendChild(qTop);
        iconGroup.appendChild(qDot);
    },
    conditional: function(iconGroup, color) {
        // Ring circle
        const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        ring.setAttribute('cx', '20');
        ring.setAttribute('cy', '20');
        ring.setAttribute('r', '18');
        ring.setAttribute('fill', 'none');
        ring.setAttribute('stroke', color);
        ring.setAttribute('stroke-width', config.ringStrokeWidth || '2.5');

        // Decision diamond
        const diamond = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        diamond.setAttribute('d', 'M 20 10 L 28 20 L 20 30 L 12 20 Z');
        diamond.setAttribute('fill', 'none');
        diamond.setAttribute('stroke', color);
        diamond.setAttribute('stroke-width', '1.5');
        diamond.setAttribute('stroke-linejoin', 'round');

        // Question mark inside diamond
        const qMark = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        qMark.setAttribute('x', '20');
        qMark.setAttribute('y', '23');
        qMark.setAttribute('fill', color);
        qMark.setAttribute('font-family', 'Arial, sans-serif');
        qMark.setAttribute('font-size', '10');
        qMark.setAttribute('font-weight', '600');
        qMark.setAttribute('text-anchor', 'middle');
        qMark.textContent = '?';

        iconGroup.appendChild(ring);
        iconGroup.appendChild(diamond);
        iconGroup.appendChild(qMark);
    },
    text: function(iconGroup, color) {
        // Ring circle
        const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        ring.setAttribute('cx', '20');
        ring.setAttribute('cy', '20');
        ring.setAttribute('r', '18');
        ring.setAttribute('fill', 'none');
        ring.setAttribute('stroke', color);
        ring.setAttribute('stroke-width', config.ringStrokeWidth || '2.5');

        // Typography "Aa" symbol - centered
        const letterGroup = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        letterGroup.setAttribute('x', '20');
        letterGroup.setAttribute('y', '24');
        letterGroup.setAttribute('fill', color);
        letterGroup.setAttribute('font-family', 'Arial, sans-serif');
        letterGroup.setAttribute('font-weight', '600');
        letterGroup.setAttribute('text-anchor', 'middle');
        letterGroup.setAttribute('dominant-baseline', 'middle');

        // Create tspan for capital A
        const spanA = document.createElementNS('http://www.w3.org/2000/svg', 'tspan');
        spanA.setAttribute('font-size', '13');
        spanA.textContent = 'A';

        // Create tspan for lowercase a
        const spanLowerA = document.createElementNS('http://www.w3.org/2000/svg', 'tspan');
        spanLowerA.setAttribute('font-size', '11');
        spanLowerA.textContent = 'a';

        letterGroup.appendChild(spanA);
        letterGroup.appendChild(spanLowerA);

        iconGroup.appendChild(ring);
        iconGroup.appendChild(letterGroup);
    },
    generic: function(iconGroup, color) {
        // Ring circle
        const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        ring.setAttribute('cx', '20');
        ring.setAttribute('cy', '20');
        ring.setAttribute('r', '18');
        ring.setAttribute('fill', 'none');
        ring.setAttribute('stroke', color);
        ring.setAttribute('stroke-width', config.ringStrokeWidth || '2.5');

        // Gear/cog icon for generic component
        const gear = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        // Simple gear shape
        gear.setAttribute('d', 'M20 13 L21.5 13 L22 15 L24 15.5 L25.5 14 L27 15.5 L25.5 17 L26 19 L28 19.5 L28 21.5 L26 22 L25.5 24 L27 25.5 L25.5 27 L24 25.5 L22 26 L21.5 28 L19.5 28 L19 26 L17 25.5 L15.5 27 L14 25.5 L15.5 24 L15 22 L13 21.5 L13 19.5 L15 19 L15.5 17 L14 15.5 L15.5 14 L17 15.5 L19 15 L19.5 13 Z');
        gear.setAttribute('fill', 'none');
        gear.setAttribute('stroke', color);
        gear.setAttribute('stroke-width', '1.5');
        gear.setAttribute('stroke-linejoin', 'round');

        // Center circle of gear
        const centerCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        centerCircle.setAttribute('cx', '20.5');
        centerCircle.setAttribute('cy', '20.5');
        centerCircle.setAttribute('r', '3');
        centerCircle.setAttribute('fill', 'none');
        centerCircle.setAttribute('stroke', color);
        centerCircle.setAttribute('stroke-width', '1.5');

        iconGroup.appendChild(ring);
        iconGroup.appendChild(gear);
        iconGroup.appendChild(centerCircle);
    }
};

// Dynamically create components from flowComponents JSON
function createComponents() {
    const componentsGroup = document.getElementById('componentsGroup');
    componentsGroup.innerHTML = '';

    orderedFlowComponents.forEach((component, index) => {
        const componentId = `component-${component.id}`;
        const iconId = `icon-${component.id}`;

        // Create component group
        const componentGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        componentGroup.classList.add('component');
        componentGroup.setAttribute('id', componentId);
        componentGroup.setAttribute('data-type', component.type);
        componentGroup.setAttribute('data-index', index);
        componentGroup.setAttribute('data-component-id', component.id);

        // Check if component was activated (executed) during the trace
        // Default to true if the activated flag is not present (backwards compatibility)
        const isActivated = component.activated !== false;
        componentGroup.setAttribute('data-activated', isActivated ? 'true' : 'false');

        // Apply dimming for non-activated components
        if (!isActivated) {
            componentGroup.classList.add('component-inactive');
            componentGroup.setAttribute('opacity', String(config.inactiveComponentOpacity));
        }

        // Create icon group
        const iconGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        iconGroup.classList.add('component-icon', 'normalize-icon');
        iconGroup.setAttribute('id', iconId);

        // Add invisible hit area circle (must be first so it's behind the icon)
        const hitArea = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        hitArea.setAttribute('cx', '20');
        hitArea.setAttribute('cy', '20');
        hitArea.setAttribute('r', '18');
        hitArea.setAttribute('fill', 'transparent');
        hitArea.classList.add('hit-area');
        iconGroup.appendChild(hitArea);

        // Add icon based on component type with color
        if (iconTemplates[component.type]) {
            const color = componentColors[component.type] || '#94a3b8';
            iconTemplates[component.type](iconGroup, color);
        }

        // Add error badge if component has an error
        if (component.args && component.args.error) {
            const errorBadge = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            errorBadge.classList.add('error-badge');

            // Red circle background
            const badgeCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            badgeCircle.setAttribute('cx', '35');
            badgeCircle.setAttribute('cy', '5');
            badgeCircle.setAttribute('r', '8');
            badgeCircle.setAttribute('fill', '#ef4444');
            badgeCircle.setAttribute('stroke', '#ffffff');
            badgeCircle.setAttribute('stroke-width', '2');

            // Exclamation mark
            const exclamation = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            exclamation.setAttribute('x', '35');
            exclamation.setAttribute('y', '9');
            exclamation.setAttribute('fill', '#ffffff');
            exclamation.setAttribute('font-family', 'Arial, sans-serif');
            exclamation.setAttribute('font-size', '12');
            exclamation.setAttribute('font-weight', '700');
            exclamation.setAttribute('text-anchor', 'middle');
            exclamation.textContent = '!';

            errorBadge.appendChild(badgeCircle);
            errorBadge.appendChild(exclamation);
            iconGroup.appendChild(errorBadge);
        }

        // Add specialized model badge if component uses a specialized model
        if (config.showSpecializedModelBadge && component.specialized_model) {
            const modelBadge = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            modelBadge.classList.add('model-badge');

            // Light grey circle background (bottom-right corner)
            const badgeCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            badgeCircle.setAttribute('cx', '35');
            badgeCircle.setAttribute('cy', '35');
            badgeCircle.setAttribute('r', '8');
            badgeCircle.setAttribute('fill', '#9ca3af');  // Light grey color
            badgeCircle.setAttribute('stroke', '#ffffff');
            badgeCircle.setAttribute('stroke-width', '2');

            // Star symbol to indicate specialized model
            const star = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            star.setAttribute('x', '35');
            star.setAttribute('y', '38');
            star.setAttribute('fill', '#ffffff');
            star.setAttribute('font-family', 'Arial, sans-serif');
            star.setAttribute('font-size', '8');
            star.setAttribute('font-weight', '700');
            star.setAttribute('text-anchor', 'middle');
            star.textContent = '★';

            modelBadge.appendChild(badgeCircle);
            modelBadge.appendChild(star);
            iconGroup.appendChild(modelBadge);
        }

        componentGroup.appendChild(iconGroup);

        componentsGroup.appendChild(componentGroup);
    });
}

// Dynamically create labels (in separate group for proper z-ordering)
function createLabels() {
    const labelsGroup = document.getElementById('labelsGroup');
    labelsGroup.innerHTML = '';

    // Create two subgroups: backgrounds first (lower z-index), then text on top
    // This prevents later label backgrounds from covering earlier label text
    const backgroundsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    backgroundsGroup.classList.add('label-backgrounds-group');
    const textsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    textsGroup.classList.add('label-texts-group');

    orderedFlowComponents.forEach((component, index) => {
        if (component.label) {
            const componentId = `component-${component.id}`;
            const iconId = `icon-${component.id}`;

            // Check if component was activated (executed) during the trace
            const isActivated = component.activated !== false;

            // Create background group (will be positioned later)
            const bgGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            bgGroup.classList.add('label-bg-group');
            bgGroup.setAttribute('data-component-id', componentId);
            bgGroup.setAttribute('data-icon', iconId);

            // Apply dimming for non-activated components
            if (!isActivated) {
                bgGroup.setAttribute('opacity', String(config.inactiveComponentOpacity));
            }

            // Create background rect for label
            const labelBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            labelBg.classList.add('label-background');
            labelBg.setAttribute('data-icon', iconId);
            labelBg.setAttribute('fill', 'white');
            labelBg.setAttribute('stroke', 'none');

            bgGroup.appendChild(labelBg);
            backgroundsGroup.appendChild(bgGroup);

            // Create text group (will be positioned later)
            const textGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            textGroup.classList.add('label-text-group');
            textGroup.setAttribute('data-component-id', componentId);
            textGroup.setAttribute('data-icon', iconId);

            // Apply dimming for non-activated components
            if (!isActivated) {
                textGroup.setAttribute('opacity', String(config.inactiveComponentOpacity));
            }

            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.classList.add('component-label');
            label.setAttribute('data-icon', iconId);
            label.setAttribute('fill', '#475569');  // slate-600 for professional look
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('font-weight', '600');
            label.textContent = component.label;

            textGroup.appendChild(label);
            textsGroup.appendChild(textGroup);
        }
    });

    // Append backgrounds first, then text - ensures text is always on top
    labelsGroup.appendChild(backgroundsGroup);
    labelsGroup.appendChild(textsGroup);
}

// Dynamically create arrows between components based on output_components
function createArrows() {
    const arrowsGroup = document.getElementById('arrowsGroup');
    arrowsGroup.innerHTML = '';

    // Create a map for quick lookup of component activation status
    const activationMap = {};
    orderedFlowComponents.forEach(comp => {
        activationMap[comp.id] = comp.activated !== false;
    });

    // Create arrows by following output_components
    orderedFlowComponents.forEach((component, index) => {
        const fromId = `component-${component.id}`;
        const fromActivated = component.activated !== false;

        // Create arrow to each output component
        if (component.output_components.length > 0) {
            component.output_components.forEach((outputId, outputIndex) => {
                const toId = `component-${outputId}`;
                const toActivated = activationMap[outputId] !== false;

                // Arrow is inactive if either source or target is inactive
                const arrowActivated = fromActivated && toActivated;

                const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                arrow.classList.add('dynamic-arrow');
                arrow.setAttribute('data-from', fromId);
                arrow.setAttribute('data-to', toId);
                arrow.setAttribute('data-arrow-index', String(index));
                arrow.setAttribute('data-output-index', String(outputIndex));
                arrow.setAttribute('data-from-component-id', component.id);
                arrow.setAttribute('data-to-component-id', outputId);
                arrow.setAttribute('d', 'M 0 0 L 0 0'); // Will be set by positionArrows
                arrow.setAttribute('fill', 'none');
                arrow.setAttribute('stroke', config.arrowColor);
                arrow.setAttribute('stroke-width', config.arrowWidth);
                arrow.setAttribute('stroke-dasharray', config.arrowDashArray);
                arrow.setAttribute('marker-end', 'url(#arrowhead)');

                // Dim arrow if connecting to/from inactive components
                if (!arrowActivated) {
                    arrow.setAttribute('opacity', String(config.inactiveComponentOpacity));
                }

                arrowsGroup.appendChild(arrow);
            });
        } else if (component.type !== 'chat_output') {
            // Component has no outputs - create a final arrow extending to the right
            // (Skip for chat_output components which intentionally have no outputs)
            const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            arrow.classList.add('dynamic-arrow', 'final-arrow');
            arrow.setAttribute('data-from', fromId);
            arrow.setAttribute('data-arrow-index', String(index));
            arrow.setAttribute('data-from-component-id', component.id);
            arrow.setAttribute('d', 'M 0 0 L 0 0'); // Will be set by positionArrows
            arrow.setAttribute('fill', 'none');
            arrow.setAttribute('stroke', config.arrowColor);
            arrow.setAttribute('stroke-width', config.arrowWidth);
            arrow.setAttribute('stroke-dasharray', config.arrowDashArray);
            arrow.setAttribute('marker-end', 'url(#arrowhead)');

            // Dim arrow if source component is inactive
            if (!fromActivated) {
                arrow.setAttribute('opacity', String(config.inactiveComponentOpacity));
            }

            arrowsGroup.appendChild(arrow);
        }
    });
}

// Dynamically create speech bubbles for user, llm, and query_rewrite components
// Creates a bubble for each outgoing edge that leads to an activated component
function createSpeechBubbles() {
    const speechBubblesGroup = document.getElementById('speechBubblesGroup');
    speechBubblesGroup.innerHTML = '';

    // Create a map for quick lookup of component activation status
    const activationMap = {};
    orderedFlowComponents.forEach(comp => {
        activationMap[comp.id] = comp.activated !== false;
    });

    orderedFlowComponents.forEach((component, index) => {
        if (component.type === 'chat_input' || component.type === 'llm' || component.type === 'query_rewrite' || component.type === 'query_clarification' || component.type === 'citations' || component.type === 'answerability' || component.type === 'hallucination_detection' || component.type === 'text' || component.type === 'conditional' || component.type === 'generic') {

            // Determine base bubble ID suffix based on type
            let bubbleIdSuffix;
            if (component.type === 'chat_input') {
                bubbleIdSuffix = 'query';
            } else if (component.type === 'llm') {
                bubbleIdSuffix = 'response';
            } else if (component.type === 'query_rewrite') {
                bubbleIdSuffix = 'rewritten';
            } else if (component.type === 'citations') {
                bubbleIdSuffix = 'citations';
            } else if (component.type === 'answerability') {
                bubbleIdSuffix = 'decision';
            } else if (component.type === 'query_clarification') {
                bubbleIdSuffix = 'clarification';
            } else if (component.type === 'hallucination_detection') {
                bubbleIdSuffix = 'detection';
            } else if (component.type === 'text') {
                bubbleIdSuffix = 'text';
            } else if (component.type === 'conditional') {
                bubbleIdSuffix = 'conditional';
            } else if (component.type === 'generic') {
                bubbleIdSuffix = 'output';
            }

            // Only create bubbles for activated source components
            const sourceActivated = component.activated !== false;
            if (!sourceActivated) {
                return; // Skip creating bubbles for non-activated components
            }

            // Determine which output indices should get bubbles
            let outputIndices = [];

            if (component.output_components && component.output_components.length > 0) {
                // Create bubble for each output that leads to an activated component
                component.output_components.forEach((outputId, outputIndex) => {
                    const targetActivated = activationMap[outputId] !== false;
                    if (targetActivated) {
                        outputIndices.push(outputIndex);
                    }
                });
                // Fallback: if no activated targets, still create one bubble on first output
                if (outputIndices.length === 0) {
                    outputIndices.push(0);
                }
            } else {
                // Terminal node - create one bubble for the final arrow
                outputIndices.push(-1);
            }

            // Create a bubble for each output index
            outputIndices.forEach((outputIndex, bubbleIdx) => {
                // Use old ID format (without index suffix) for backwards compatibility when:
                // - It's a terminal node (outputIndex === -1), OR
                // - It's the first (or only) bubble for this component
                const useSimpleId = outputIndex === -1 || (bubbleIdx === 0 && outputIndices.length === 1);
                const bubbleId = useSimpleId
                    ? `bubble-${component.id}-${bubbleIdSuffix}`
                    : `bubble-${component.id}-${bubbleIdSuffix}-${outputIndex}`;
                const textId = useSimpleId
                    ? `text-${component.id}`
                    : `text-${component.id}-${outputIndex}`;

                // Get target component ID for sorting during animation
                const targetId = outputIndex >= 0 ? component.output_components[outputIndex] : '';

                const bubble = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                bubble.classList.add('speech-bubble');
                bubble.setAttribute('id', bubbleId);
                bubble.setAttribute('data-component-index', index);
                bubble.setAttribute('data-component-id', component.id);
                bubble.setAttribute('data-type', component.type);
                bubble.setAttribute('data-output-index', String(outputIndex));
                bubble.setAttribute('data-target-id', targetId);

                // Create unified speech bubble path (rounded rectangle + subtle tail)
                const bubblePath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                const w = config.bubbleWidth;
                const h = config.bubbleHeight;
                const r = 4; // corner radius
                const tailW = 6; // tail width (slightly larger for visibility)
                const tailH = 5; // tail height
                const tailX = w / 2; // center tail
                const d = `M ${r} 0 H ${w-r} A ${r} ${r} 0 0 1 ${w} ${r} V ${h-r} A ${r} ${r} 0 0 1 ${w-r} ${h} H ${tailX+tailW/2} L ${tailX} ${h+tailH} L ${tailX-tailW/2} ${h} H ${r} A ${r} ${r} 0 0 1 0 ${h-r} V ${r} A ${r} ${r} 0 0 1 ${r} 0 Z`;
                bubblePath.setAttribute('d', d);
                bubblePath.setAttribute('fill', config.bubbleFillColor);
                bubblePath.setAttribute('stroke', config.bubbleBorderColor);
                bubblePath.setAttribute('stroke-width', config.bubbleBorderWidth);
                bubblePath.classList.add('bubble-path');

                const foreignObject = document.createElementNS('http://www.w3.org/2000/svg', 'foreignObject');
                foreignObject.setAttribute('x', '4');
                foreignObject.setAttribute('y', '4');
                foreignObject.setAttribute('width', String(w - 8));
                foreignObject.setAttribute('height', String(h - 8));

                const textDiv = document.createElement('div');
                textDiv.setAttribute('id', textId);
                textDiv.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');
                // Inline styles for line-clamp (CSS handles visual truncation with ellipsis)
                textDiv.style.display = '-webkit-box';
                textDiv.style.webkitBoxOrient = 'vertical';
                textDiv.style.webkitLineClamp = '4';
                textDiv.style.overflow = 'hidden';

                foreignObject.appendChild(textDiv);
                bubble.appendChild(bubblePath);
                bubble.appendChild(foreignObject);

                speechBubblesGroup.appendChild(bubble);
            });
        }
    });
}

// Normalize icon sizes and positions
function normalizeIcons() {
    const TARGET_SIZE = config.iconSize;
    const TARGET_Y = 25; // Target Y position for all icons

    const icons = document.querySelectorAll('.normalize-icon');

    icons.forEach(icon => {
        // Get the bounding box of the icon
        const bbox = icon.getBBox();

        // Calculate the scale factor to fit the icon in TARGET_SIZE
        const maxDimension = Math.max(bbox.width, bbox.height);
        const scale = TARGET_SIZE / maxDimension;

        // Calculate center of the icon
        const iconCenterX = bbox.x + bbox.width / 2;
        const iconCenterY = bbox.y + bbox.height / 2;

        // Calculate translation to center the icon at TARGET_SIZE/2 and TARGET_Y
        const targetCenterX = TARGET_SIZE / 2;
        const targetCenterY = TARGET_Y + TARGET_SIZE / 2;

        const translateX = targetCenterX - (iconCenterX * scale);
        const translateY = targetCenterY - (iconCenterY * scale);

        // Apply transform to normalize size and position
        icon.setAttribute('transform', `translate(${translateX}, ${translateY}) scale(${scale})`);
    });
}

// Position component labels under their icons
function positionLabels() {
    const TARGET_SIZE = config.iconSize;
    const TARGET_Y = 25; // Icon Y position

    // Position text groups (and apply same transform to corresponding bg groups)
    const textGroups = document.querySelectorAll('.label-text-group');

    // Process text groups (and apply same transform to corresponding bg groups)
    textGroups.forEach(textGroup => {
        const componentId = textGroup.getAttribute('data-component-id');
        const iconId = textGroup.getAttribute('data-icon');
        const component = document.getElementById(componentId);

        if (!component) return;

        // Get the component's transform
        const componentTransform = component.getAttribute('transform');
        const match = componentTransform.match(/translate\(([^,]+),\s*([^)]+)\)/);
        if (!match) return;

        const componentX = parseFloat(match[1]);
        const componentY = parseFloat(match[2]);

        // Calculate label position relative to component
        const centerX = TARGET_SIZE / 2;
        const bottomY = TARGET_Y + TARGET_SIZE;
        const textY = bottomY + config.componentLabelSpacing;

        // Apply component transform to text group
        textGroup.setAttribute('transform', `translate(${componentX}, ${componentY})`);

        // Find and transform the corresponding background group
        const bgGroup = document.querySelector(`.label-bg-group[data-icon="${iconId}"]`);
        if (bgGroup) {
            bgGroup.setAttribute('transform', `translate(${componentX}, ${componentY})`);
        }

        // Position label within the group (relative positioning)
        const label = textGroup.querySelector('.component-label');
        if (label) {
            label.setAttribute('x', centerX);
            label.setAttribute('y', textY);
        }
    });
}

// Measure text width using canvas (independent of SVG coordinate transforms)
function measureTextWidth(text, fontWeight, fontSize, fontFamily) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    ctx.font = `${fontWeight} ${fontSize}px ${fontFamily}`;
    return ctx.measureText(text).width;
}

// Position and size label backgrounds based on text measurements
function positionLabelBackgrounds() {
    const labels = document.querySelectorAll('.component-label');
    const paddingX = config.componentLabelBackgroundPaddingX;
    const paddingY = config.componentLabelBackgroundPaddingY;

    labels.forEach(label => {
        const iconId = label.getAttribute('data-icon');
        const labelBg = document.querySelector(`.label-background[data-icon="${iconId}"]`);

        if (labelBg) {
            // Get text properties
            const text = label.textContent;
            const fontSize = config.componentLabelFontSize;
            const fontWeight = label.getAttribute('font-weight') || '600';
            const fontFamily = 'Inter, -apple-system, sans-serif';

            // Measure text width using canvas (avoids SVG coordinate system issues)
            const textWidth = measureTextWidth(text, fontWeight, fontSize, fontFamily);

            // Calculate background dimensions
            const tightHeight = fontSize * 1.2; // Height based on font size

            // Get the text position (x is center due to text-anchor: middle)
            const textX = parseFloat(label.getAttribute('x'));
            const textY = parseFloat(label.getAttribute('y'));

            // Position background centered on text
            const bgX = textX - (textWidth / 2) - paddingX;
            const bgY = textY - fontSize - paddingY; // Position above baseline
            const bgWidth = textWidth + (paddingX * 2);
            const bgHeight = tightHeight + (paddingY * 2);

            labelBg.setAttribute('x', bgX);
            labelBg.setAttribute('y', bgY);
            labelBg.setAttribute('width', bgWidth);
            labelBg.setAttribute('height', bgHeight);
        }
    });
}

// Position arrows between components
function positionArrows() {
    const TARGET_SIZE = config.iconSize;
    const ARROW_GAP = config.arrowGap;
    const TARGET_Y = 25; // Icon Y position within component group
    const iconCenterY = TARGET_Y + (TARGET_SIZE / 2);

    // Build a map to count incoming edges for each component (to detect merge nodes)
    const incomingEdgeCount = {};
    flowComponents.forEach(comp => {
        comp.output_components.forEach(outputId => {
            incomingEdgeCount[outputId] = (incomingEdgeCount[outputId] || 0) + 1;
        });
    });

    const arrows = document.querySelectorAll('.dynamic-arrow');

    // Group arrows by target component for merge nodes (to calculate staggered entry points)
    const arrowsByTarget = {};
    arrows.forEach(arrow => {
        const toComponentId = arrow.getAttribute('data-to-component-id');
        if (toComponentId && incomingEdgeCount[toComponentId] > 1) {
            if (!arrowsByTarget[toComponentId]) {
                arrowsByTarget[toComponentId] = [];
            }
            const fromId = arrow.getAttribute('data-from');
            // Extract component ID from "component-XXX"
            const fromComponentId = fromId.replace('component-', '');
            const sourceRow = componentLayout[fromComponentId]?.row ?? 0;
            arrowsByTarget[toComponentId].push({
                arrow,
                sourceRow
            });
        }
    });

    // Sort arrows for each merge node by source row (top to bottom)
    for (const targetId in arrowsByTarget) {
        arrowsByTarget[targetId].sort((a, b) => a.sourceRow - b.sourceRow);
    }

    arrows.forEach(arrow => {
        const fromId = arrow.getAttribute('data-from');
        const toId = arrow.getAttribute('data-to');
        const toComponentId = arrow.getAttribute('data-to-component-id');
        const isFinalArrow = arrow.classList.contains('final-arrow');

        const fromComponent = document.getElementById(fromId);
        if (!fromComponent) return;

        // Get the transform of the from component
        const fromTransform = fromComponent.getAttribute('transform');
        const fromMatch = fromTransform.match(/translate\(([^,]+),\s*([^)]+)\)/);
        if (!fromMatch) return;

        const fromX = parseFloat(fromMatch[1]);
        const fromY = parseFloat(fromMatch[2]);

        // Calculate right edge of from component icon and Y position at icon center
        const x1 = fromX + TARGET_SIZE + ARROW_GAP;
        const y1 = fromY + iconCenterY;

        if (isFinalArrow) {
            // For final arrow, use the same spacing as arrows between components
            const x2 = x1 + config.componentSpacing;
            const y2 = y1;

            // Create straight horizontal path
            arrow.setAttribute('d', `M ${x1} ${y1} L ${x2} ${y2}`);

            // Store coordinates for bubble positioning
            arrow.setAttribute('data-x1', x1);
            arrow.setAttribute('data-y1', y1);
            arrow.setAttribute('data-x2', x2);
            arrow.setAttribute('data-y2', y2);
        } else if (toId) {
            const toComponent = document.getElementById(toId);
            if (!toComponent) return;

            // Get the transform of the to component
            const toTransform = toComponent.getAttribute('transform');
            const toMatch = toTransform.match(/translate\(([^,]+),\s*([^)]+)\)/);
            if (!toMatch) return;

            const toX = parseFloat(toMatch[1]);
            const toY = parseFloat(toMatch[2]);

            // Check if this is an incoming branch (target has multiple incoming edges)
            const isIncomingBranch = toComponentId && incomingEdgeCount[toComponentId] > 1;

            // Get source and target rows for incoming branches
            const fromComponentId = fromId.replace('component-', '');
            const sourceRow = componentLayout[fromComponentId]?.row ?? 0;
            const targetRow = componentLayout[toComponentId]?.row ?? 0;

            // Only use special incoming branch handling if source and target are on different rows
            const needsCurvedEntry = isIncomingBranch && arrowsByTarget[toComponentId] && sourceRow !== targetRow;

            // Calculate endpoint based on whether this is an incoming branch from a different row
            let x2, y2;

            if (needsCurvedEntry) {
                // For incoming branches from different rows, arrows enter from top or bottom
                // Arrow enters from top if source is above target, bottom if below
                const entersFromTop = sourceRow < targetRow;

                // Entry point is center-top or below the label for bottom entry
                x2 = toX + (TARGET_SIZE / 2);
                if (entersFromTop) {
                    y2 = toY + TARGET_Y - ARROW_GAP; // Top of icon with gap
                } else {
                    // Bottom entry: arrow stops just below the target's label
                    // The arrow curves UP from the source to this point
                    const labelOffset = 24; // Small gap below label
                    y2 = toY + TARGET_Y + TARGET_SIZE + labelOffset;
                }

                // Store entry direction for speech bubble positioning
                arrow.setAttribute('data-entry-direction', entersFromTop ? 'top' : 'bottom');
            } else {
                // Standard entry from left side
                x2 = toX - ARROW_GAP;
                y2 = toY + iconCenterY;
            }

            // Determine if arrow is diagonal
            const isDiagonal = Math.abs(y2 - y1) > 1 || needsCurvedEntry;

            let pathData;
            if (needsCurvedEntry) {
                // Incoming branches: horizontal line, rounded corner, then vertical line
                // Similar to rounded rectangle edges
                const cornerRadius = 15; // Radius of the rounded corner
                const cornerStartX = x2 - cornerRadius;

                if (config.branchingArrowStyle === 'curved') {
                    // Determine direction: going up or down?
                    const goingUp = y1 > y2; // Source is below target (in SVG coords)

                    if (goingUp) {
                        // Arrow curves up: horizontal → arc (counter-clockwise) → vertical up
                        const arcEndY = y1 - cornerRadius;
                        // SVG arc: A rx ry x-rotation large-arc-flag sweep-flag x y
                        // sweep-flag 0 = counter-clockwise
                        pathData = `M ${x1} ${y1} L ${cornerStartX} ${y1} A ${cornerRadius} ${cornerRadius} 0 0 0 ${x2} ${arcEndY} L ${x2} ${y2}`;
                    } else {
                        // Arrow curves down: horizontal → arc (clockwise) → vertical down
                        const arcEndY = y1 + cornerRadius;
                        // sweep-flag 1 = clockwise
                        pathData = `M ${x1} ${y1} L ${cornerStartX} ${y1} A ${cornerRadius} ${cornerRadius} 0 0 1 ${x2} ${arcEndY} L ${x2} ${y2}`;
                    }

                    arrow.setAttribute('data-is-curved', 'true');
                    arrow.setAttribute('data-is-orthogonal', 'false');
                    // Store horizontal segment info for bubble positioning
                    arrow.setAttribute('data-horizontal-segment-x1', x1);
                    arrow.setAttribute('data-horizontal-segment-x2', cornerStartX);
                    arrow.setAttribute('data-horizontal-segment-y', y1);
                } else {
                    // Orthogonal path: horizontal then vertical (no rounded corner)
                    pathData = `M ${x1} ${y1} L ${x2} ${y1} L ${x2} ${y2}`;
                    arrow.setAttribute('data-is-orthogonal', 'true');
                    arrow.setAttribute('data-horizontal-segment-x1', x1);
                    arrow.setAttribute('data-horizontal-segment-x2', x2);
                    arrow.setAttribute('data-horizontal-segment-y', y1);
                }
            } else if (isDiagonal && config.branchingArrowStyle === 'orthogonal') {
                // Outgoing branch: vertical first, then horizontal (diverge from source)
                const bendY = y2;
                pathData = `M ${x1} ${y1} L ${x1} ${bendY} L ${x2} ${bendY}`;
                arrow.setAttribute('data-is-orthogonal', 'true');
                arrow.setAttribute('data-horizontal-segment-x1', x1);
                arrow.setAttribute('data-horizontal-segment-x2', x2);
                arrow.setAttribute('data-horizontal-segment-y', bendY);
            } else if (isDiagonal && config.branchingArrowStyle === 'curved') {
                // Outgoing branch: curve starts vertical and ends horizontally
                const controlX = x1;
                const controlY = y2;
                pathData = `M ${x1} ${y1} Q ${controlX} ${controlY} ${x2} ${y2}`;

                arrow.setAttribute('data-is-curved', 'true');
                arrow.setAttribute('data-control-x', controlX);
                arrow.setAttribute('data-control-y', controlY);
                arrow.setAttribute('data-is-orthogonal', 'false');
            } else {
                // Create straight horizontal or diagonal path
                pathData = `M ${x1} ${y1} L ${x2} ${y2}`;
                arrow.setAttribute('data-is-orthogonal', 'false');
            }

            arrow.setAttribute('d', pathData);

            // Store coordinates for bubble positioning
            arrow.setAttribute('data-x1', x1);
            arrow.setAttribute('data-y1', y1);
            arrow.setAttribute('data-x2', x2);
            arrow.setAttribute('data-y2', y2);
        }
    });
}

// Position components based on depth (X) and row (Y)
function positionComponents() {
    const START_X = 60; // Starting X position for first component
    const START_Y = 200; // Starting Y position for first row

    // Calculate horizontal spacing based on componentSpacing
    // Each spacing includes: icon width + gap + arrow gap + gap + next icon
    const horizontalSpacing = config.iconSize + config.arrowGap * 2 + config.componentSpacing;

    // Position all components using layout information
    orderedFlowComponents.forEach((component) => {
        const componentId = `component-${component.id}`;
        const componentElement = document.getElementById(componentId);
        const layoutInfo = componentLayout[component.id];

        if (componentElement && layoutInfo) {
            const xPos = START_X + (layoutInfo.depth * horizontalSpacing);
            const yPos = START_Y + (layoutInfo.row * config.branchSpacing);
            componentElement.setAttribute('transform', `translate(${xPos}, ${yPos})`);
        }
    });
}

// Update arrow Y positions based on icon size
function updateArrowPositions() {
    const COMPONENT_Y_OFFSET = 200; // The y value in transform="translate(80, 200)"
    const TARGET_Y = 25; // Icon Y position
    const arrowY = COMPONENT_Y_OFFSET + TARGET_Y + (config.iconSize / 2);

    // Update all arrows
    const arrows = document.querySelectorAll('.dynamic-arrow');
    arrows.forEach(arrow => {
        arrow.setAttribute('y1', arrowY);
        arrow.setAttribute('y2', arrowY);
    });
}

// Position all speech bubbles over their respective arrows
// Bubbles are positioned at:
// - X: consistent offset from source component (as if all edges were horizontal)
// - Y: at the destination row's height (aligns with destination component)
function positionSpeechBubbles() {
    const bubbles = document.querySelectorAll('.speech-bubble');

    // Create component map for quick lookup
    const componentMap = {};
    flowComponents.forEach(comp => {
        componentMap[comp.id] = comp;
    });

    // Calculate horizontal spacing (same as in positionComponents)
    const START_X = 60;
    const horizontalSpacing = config.iconSize + config.arrowGap * 2 + config.componentSpacing;

    bubbles.forEach(bubble => {
        const componentId = bubble.getAttribute('data-component-id');
        const component = componentMap[componentId];
        const bubbleOutputIndex = bubble.getAttribute('data-output-index');

        if (!component) return;

        let arrow = null;

        // Find the arrow matching this bubble's output index
        if (bubbleOutputIndex === '-1') {
            // Final arrow for terminal nodes
            arrow = document.querySelector(
                `.dynamic-arrow[data-from-component-id="${componentId}"].final-arrow`
            );
        } else if (component.output_components.length > 0) {
            // Find the arrow with matching output index
            const allArrows = document.querySelectorAll(`.dynamic-arrow[data-from-component-id="${componentId}"]`);
            arrow = Array.from(allArrows).find(a => {
                const arrowOutputIndex = a.getAttribute('data-output-index');
                return arrowOutputIndex === bubbleOutputIndex;
            });
            // Fallback to first arrow if no index match
            if (!arrow && allArrows.length > 0) {
                arrow = allArrows[0];
            }
        }

        // Fallback: for components with no outputs (terminal nodes), find the final arrow
        if (!arrow) {
            arrow = document.querySelector(
                `.dynamic-arrow[data-from-component-id="${componentId}"].final-arrow`
            );
        }

        if (arrow) {
            const x1 = parseFloat(arrow.getAttribute('data-x1'));
            const x2 = parseFloat(arrow.getAttribute('data-x2'));
            const y2 = parseFloat(arrow.getAttribute('data-y2'));
            const isOrthogonal = arrow.getAttribute('data-is-orthogonal') === 'true';
            const isCurved = arrow.getAttribute('data-is-curved') === 'true';

            let centerX, bubbleY;

            // Special handling for merge node arrows (orthogonal or curved with horizontal segments)
            if (isOrthogonal) {
                // For orthogonal (L-shape) arrows going into merge nodes,
                // position bubble on the horizontal ending segment
                const horizontalY = parseFloat(arrow.getAttribute('data-horizontal-segment-y'));
                centerX = (x1 + x2) / 2;
                bubbleY = horizontalY - config.bubbleHeight - config.bubbleVerticalGap;
            } else if (isCurved && arrow.hasAttribute('data-horizontal-segment-x1')) {
                // For curved arrows with horizontal segments (incoming branches to merge nodes),
                // position bubble on the horizontal segment
                const hSegX1 = parseFloat(arrow.getAttribute('data-horizontal-segment-x1'));
                const hSegX2 = parseFloat(arrow.getAttribute('data-horizontal-segment-x2'));
                const hSegY = parseFloat(arrow.getAttribute('data-horizontal-segment-y'));
                centerX = (hSegX1 + hSegX2) / 2;
                bubbleY = hSegY - config.bubbleHeight - config.bubbleVerticalGap;
            } else {
                // For regular edges (outgoing from a component):
                // X: consistent offset from source component (as if horizontal)
                // Y: at the destination row's height
                const sourceLayout = componentLayout[componentId];

                if (sourceLayout) {
                    const sourceX = START_X + (sourceLayout.depth * horizontalSpacing) + config.iconSize / 2;
                    centerX = sourceX + horizontalSpacing / 2;
                } else {
                    centerX = (x1 + x2) / 2;
                }

                bubbleY = y2 - config.bubbleHeight - config.bubbleVerticalGap;
            }

            // Center the bubble using config.bubbleWidth
            bubble.setAttribute('transform', `translate(${centerX - config.bubbleWidth / 2}, ${bubbleY})`);
        }
    });
}

// Typing animation
function typeText(element, text, speed = 50) {
    let index = 0;
    element.textContent = '';

    return new Promise((resolve) => {
        const interval = setInterval(() => {
            if (index < text.length) {
                element.textContent += text[index];
                index++;
                // Resolve immediately after adding the last character
                if (index >= text.length) {
                    clearInterval(interval);
                    resolve();
                }
            }
        }, speed);
    });
}

// Create document icons dynamically for retriever components
// Creates documents for each outgoing arrow that leads to an activated component
// Documents are positioned at:
// - X: consistent spacing from source component (as if all edges were horizontal)
// - Y: at the destination row's height (aligns with destination component)
function createDocumentIcons() {
    const documentsGroup = document.getElementById('documentsGroup');
    documentsGroup.innerHTML = '';

    // Create a map for quick lookup of component activation status
    const activationMap = {};
    orderedFlowComponents.forEach(comp => {
        activationMap[comp.id] = comp.activated !== false;
    });

    // Find all retriever components and create documents for their outgoing arrows
    orderedFlowComponents.forEach((component, componentIndex) => {
        if (component.type === 'retriever' && component.args.passages) {
            // Get all outgoing arrows to activated targets
            const allArrows = document.querySelectorAll(`.dynamic-arrow[data-from-component-id="${component.id}"]`);
            const activatedArrows = Array.from(allArrows).filter(arrow => {
                const targetId = arrow.getAttribute('data-to-component-id');
                // Include arrow if target is activated (or if it's a final arrow with no target)
                return !targetId || activationMap[targetId] !== false;
            });

            // Create documents for each activated arrow
            activatedArrows.forEach((arrow, arrowIndex) => {
                const x1 = parseFloat(arrow.getAttribute('data-x1'));
                const x2 = parseFloat(arrow.getAttribute('data-x2'));
                const y2 = parseFloat(arrow.getAttribute('data-y2'));
                const arrowTargetId = arrow.getAttribute('data-to-component-id') || '';

                // Use actual arrow length for spacing (keeps documents compact)
                // X positioning uses arrow coordinates, Y uses destination row height
                const passages = component.args.passages;
                const arrowLength = x2 - x1;
                const spacing = arrowLength / (passages.length + 1);

                // Y position: at the destination row's height
                const y = y2;

                // Create a document icon for each passage
                passages.forEach((passage, docIndex) => {
                    const xPos = x1 + spacing * (docIndex + 1);
                    const yPos = y - 35;

                    const docIcon = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                    docIcon.classList.add('document-icon');
                    docIcon.setAttribute('id', `doc-${componentIndex}-${arrowIndex}-${docIndex}`);
                    docIcon.setAttribute('data-component-index', componentIndex);
                    docIcon.setAttribute('data-arrow-index', arrowIndex);
                    docIcon.setAttribute('data-doc-index', docIndex);
                    docIcon.setAttribute('data-target-id', arrowTargetId);
                    docIcon.setAttribute('transform', `translate(${xPos - config.documentIconSize / 2}, ${yPos})`);

                    // Apply shadow if enabled
                    if (config.documentShadowEnabled) {
                        docIcon.setAttribute('filter', 'url(#documentShadow)');
                    }

                    // Simple document outline using config.documentIconSize
                    // Height is 4/3 of width to maintain aspect ratio
                    const docWidth = config.documentIconSize;
                    const docHeight = config.documentIconSize * 4 / 3;

                    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect.setAttribute('x', '0');
                    rect.setAttribute('y', '3');
                    rect.setAttribute('width', docWidth);
                    rect.setAttribute('height', docHeight);
                    rect.setAttribute('fill', config.bubbleFillColor);
                    rect.setAttribute('stroke', config.bubbleBorderColor);
                    rect.setAttribute('stroke-width', config.bubbleBorderWidth);

                    // Folded corner (scaled proportionally)
                    const cornerSize = docWidth * 0.25;
                    const corner = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                    corner.setAttribute('d', `M ${docWidth - cornerSize} 3 L ${docWidth - cornerSize} ${3 + cornerSize} L ${docWidth} ${3 + cornerSize}`);
                    corner.setAttribute('fill', 'none');
                    corner.setAttribute('stroke', config.bubbleBorderColor);
                    corner.setAttribute('stroke-width', config.bubbleBorderWidth);

                    // Document lines (scaled proportionally)
                    const lineMargin = docWidth * 0.16;
                    const lineWidth = docWidth - (lineMargin * 2);
                    const lineY1 = 3 + docHeight * 0.375;
                    const lineY2 = 3 + docHeight * 0.5625;
                    const lineY3 = 3 + docHeight * 0.75;

                    const line1 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line1.setAttribute('x1', lineMargin);
                    line1.setAttribute('y1', lineY1);
                    line1.setAttribute('x2', lineMargin + lineWidth);
                    line1.setAttribute('y2', lineY1);
                    line1.setAttribute('stroke', config.bubbleBorderColor);
                    line1.setAttribute('stroke-width', '0.5');

                    const line2 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line2.setAttribute('x1', lineMargin);
                    line2.setAttribute('y1', lineY2);
                    line2.setAttribute('x2', lineMargin + lineWidth);
                    line2.setAttribute('y2', lineY2);
                    line2.setAttribute('stroke', config.bubbleBorderColor);
                    line2.setAttribute('stroke-width', '0.5');

                    const line3 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line3.setAttribute('x1', lineMargin);
                    line3.setAttribute('y1', lineY3);
                    line3.setAttribute('x2', lineMargin + lineWidth);
                    line3.setAttribute('y2', lineY3);
                    line3.setAttribute('stroke', config.bubbleBorderColor);
                    line3.setAttribute('stroke-width', '0.5');

                    docIcon.appendChild(rect);
                    docIcon.appendChild(corner);
                    docIcon.appendChild(line1);
                    docIcon.appendChild(line2);
                    docIcon.appendChild(line3);

                    documentsGroup.appendChild(docIcon);
                });
            });
        }
    });
}

// Create variation bubble icons dynamically for query_expansion components
// Creates variations for each outgoing arrow that leads to an activated component
// Variations are positioned at:
// - X: along the arrow using actual arrow length for spacing (keeps icons compact)
// - Y: at the destination row's height (aligns with destination component)
function createVariationIcons() {
    // Use the same group as documents for layering
    const documentsGroup = document.getElementById('documentsGroup');

    // Create a map for quick lookup of component activation status
    const activationMap = {};
    orderedFlowComponents.forEach(comp => {
        activationMap[comp.id] = comp.activated !== false;
    });

    // Find all query_expansion components and create variation bubbles for their outgoing arrows
    orderedFlowComponents.forEach((component, componentIndex) => {
        if (component.type === 'query_expansion' && component.args.query_variations) {
            // Get all outgoing arrows to activated targets
            const allArrows = document.querySelectorAll(`.dynamic-arrow[data-from-component-id="${component.id}"]`);
            const activatedArrows = Array.from(allArrows).filter(arrow => {
                const targetId = arrow.getAttribute('data-to-component-id');
                // Include arrow if target is activated (or if it's a final arrow with no target)
                return !targetId || activationMap[targetId] !== false;
            });

            // Create variations for each activated arrow
            activatedArrows.forEach((arrow, arrowIndex) => {
                const x1 = parseFloat(arrow.getAttribute('data-x1'));
                const x2 = parseFloat(arrow.getAttribute('data-x2'));
                const y2 = parseFloat(arrow.getAttribute('data-y2'));
                const arrowTargetId = arrow.getAttribute('data-to-component-id') || '';

                // Use actual arrow length for spacing (keeps variations compact)
                // X positioning uses arrow coordinates, Y uses destination row height
                const variations = component.args.query_variations;
                const arrowLength = x2 - x1;
                const spacing = arrowLength / (variations.length + 1);

                // Y position: at the destination row's height
                const y = y2;

                // Create a bubble icon for each variation
                variations.forEach((variation, varIndex) => {
                    const xPos = x1 + spacing * (varIndex + 1);
                    const yPos = y - 30;

                    const varIcon = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                    varIcon.classList.add('variation-icon');
                    varIcon.setAttribute('id', `var-${componentIndex}-${arrowIndex}-${varIndex}`);
                    varIcon.setAttribute('data-component-index', componentIndex);
                    varIcon.setAttribute('data-arrow-index', arrowIndex);
                    varIcon.setAttribute('data-var-index', varIndex);
                    varIcon.setAttribute('data-target-id', arrowTargetId);
                    varIcon.setAttribute('transform', `translate(${xPos - config.variationIconSize / 2}, ${yPos})`);

                    // Create small speech bubble shape
                    const bubbleSize = config.variationIconSize;
                    const bubbleHeight = bubbleSize * 0.8;
                    const tailHeight = 3;
                    const r = 2; // corner radius

                    const bubblePath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                    const d = `M ${r} 0 H ${bubbleSize-r} A ${r} ${r} 0 0 1 ${bubbleSize} ${r} V ${bubbleHeight-r} A ${r} ${r} 0 0 1 ${bubbleSize-r} ${bubbleHeight} H ${bubbleSize/2+2} L ${bubbleSize/2} ${bubbleHeight+tailHeight} L ${bubbleSize/2-2} ${bubbleHeight} H ${r} A ${r} ${r} 0 0 1 0 ${bubbleHeight-r} V ${r} A ${r} ${r} 0 0 1 ${r} 0 Z`;
                    bubblePath.setAttribute('d', d);
                    bubblePath.setAttribute('fill', config.bubbleFillColor);
                    bubblePath.setAttribute('stroke', config.bubbleBorderColor);
                    bubblePath.setAttribute('stroke-width', config.bubbleBorderWidth);

                    varIcon.appendChild(bubblePath);
                    documentsGroup.appendChild(varIcon);
                });
            });
        }
    });
}

// Compute all activated paths through the graph for animation
// Returns an array of paths that should be animated sequentially.
// Handles:
// - Multiple start nodes (animated top to bottom by row)
// - Multiple outgoing branches (first branch completes, then second, etc.)
// - Merge nodes (animated after all incoming branches complete)
function getAllActivatedPaths() {
    const componentMap = {};
    flowComponents.forEach(comp => {
        componentMap[comp.id] = comp;
    });

    // Count incoming edges for each component to identify merge nodes
    const incomingEdgeCount = {};
    flowComponents.forEach(comp => {
        comp.output_components.forEach(outputId => {
            incomingEdgeCount[outputId] = (incomingEdgeCount[outputId] || 0) + 1;
        });
    });

    // Identify merge nodes (nodes with multiple incoming edges)
    const mergeNodes = new Set();
    for (const [id, count] of Object.entries(incomingEdgeCount)) {
        if (count > 1) {
            mergeNodes.add(id);
        }
    }

    // Find all start nodes (nodes with no incoming edges)
    const startNodes = flowComponents.filter(comp => !incomingEdgeCount[comp.id] || incomingEdgeCount[comp.id] === 0);

    if (startNodes.length === 0) return [];

    // Sort start nodes by their row (top to bottom) using componentLayout
    startNodes.sort((a, b) => {
        const rowA = componentLayout[a.id]?.row ?? 0;
        const rowB = componentLayout[b.id]?.row ?? 0;
        return rowA - rowB;
    });

    const allPaths = [];
    const visited = new Set();
    const pendingBranches = []; // Queue of branch start IDs to process

    // Helper function to build a path from a starting point, stopping BEFORE merge nodes
    // Also queues additional branches from branching nodes for later processing
    function buildPathUntilMergeOrEnd(startId) {
        const path = [];
        let currentId = startId;

        while (currentId) {
            // Stop BEFORE merge nodes (they'll be handled after all incoming paths)
            if (mergeNodes.has(currentId) && path.length > 0) {
                break;
            }

            const currentComp = componentMap[currentId];
            if (!currentComp) break;

            // Skip if already visited
            if (visited.has(currentId)) {
                break;
            }

            path.push(currentComp);
            visited.add(currentId);

            // Handle branching: queue additional branches for later processing
            if (currentComp.output_components.length > 1) {
                // Queue branches 1, 2, ... for later (sorted by row, top to bottom)
                const additionalBranches = currentComp.output_components.slice(1)
                    .filter(branchId => !visited.has(branchId))
                    .sort((a, b) => {
                        const rowA = componentLayout[a]?.row ?? 0;
                        const rowB = componentLayout[b]?.row ?? 0;
                        return rowA - rowB;
                    });
                pendingBranches.push(...additionalBranches);

                // Continue with first branch (index 0)
                currentId = currentComp.output_components[0];
            } else if (currentComp.output_components.length === 1) {
                currentId = currentComp.output_components[0];
            } else {
                currentId = null;
            }
        }

        return path;
    }

    // Helper to process pending branches and merge nodes
    function processPendingWork() {
        // Process pending branches first (from branching nodes)
        while (pendingBranches.length > 0) {
            const branchStartId = pendingBranches.shift();
            if (!visited.has(branchStartId) && !mergeNodes.has(branchStartId)) {
                const path = buildPathUntilMergeOrEnd(branchStartId);
                if (path.length > 0) {
                    allPaths.push(path);
                }
            }
        }

        // Then check if any merge nodes are now ready (all inputs visited)
        const readyMergeNodes = Array.from(mergeNodes)
            .filter(id => !visited.has(id))
            .map(id => componentMap[id])
            .filter(comp => comp)
            .filter(comp => {
                // Check if all incoming edges have been visited
                const sources = flowComponents.filter(c => c.output_components.includes(comp.id));
                return sources.every(src => visited.has(src.id));
            })
            .sort((a, b) => {
                const depthA = componentLayout[a.id]?.depth ?? 0;
                const depthB = componentLayout[b.id]?.depth ?? 0;
                return depthA - depthB;
            });

        for (const mergeNode of readyMergeNodes) {
            if (!visited.has(mergeNode.id)) {
                const path = buildPathUntilMergeOrEnd(mergeNode.id);
                if (path.length > 0) {
                    allPaths.push(path);
                }
                // After processing a merge node, there might be new pending branches
                processPendingWork();
            }
        }
    }

    // Phase 1: Build paths from each start node
    for (const startNode of startNodes) {
        const path = buildPathUntilMergeOrEnd(startNode.id);
        if (path.length > 0) {
            allPaths.push(path);
        }
        // After each start node path, process any pending branches
        processPendingWork();
    }

    // Final pass: ensure all merge nodes are processed
    processPendingWork();

    return allPaths;
}

// Legacy function for backwards compatibility - returns first path only
function getActivatedPath() {
    const allPaths = getAllActivatedPaths();
    return allPaths.length > 0 ? allPaths[0] : [];
}

// Helper function to activate all bubbles for a component and set their text
// This handles the case where a component has multiple outgoing edges to activated components
function activateAllBubblesForComponent(componentId, componentType, displayText) {
    // Find ALL bubbles for this component (there may be multiple for multi-output components)
    const bubbles = document.querySelectorAll(`.speech-bubble[data-component-id="${componentId}"][data-type="${componentType}"]`);

    bubbles.forEach(bubble => {
        const outputIndex = bubble.getAttribute('data-output-index');
        // Text element ID depends on whether this is a single or multi-output bubble
        // Single output (or first bubble): text-{componentId}
        // Multi-output: text-{componentId}-{outputIndex}
        let textElement = document.getElementById(`text-${componentId}`);
        if (!textElement && outputIndex !== '-1') {
            textElement = document.getElementById(`text-${componentId}-${outputIndex}`);
        }

        if (textElement) {
            bubble.classList.add('active');
            textElement.textContent = displayText;
        }
    });

    return bubbles.length > 0;
}

// Async version for animated activation with typing effect
// Activates bubbles for a component sequentially by edge (top edge first, then second, etc.)
async function activateAllBubblesForComponentAnimated(componentId, componentType, displayText, enableTyping, typingSpeed) {
    const bubbles = document.querySelectorAll(`.speech-bubble[data-component-id="${componentId}"][data-type="${componentType}"]`);

    // Sort bubbles by target row (top to bottom)
    const sortedBubbles = Array.from(bubbles).sort((a, b) => {
        const targetIdA = a.getAttribute('data-target-id') || '';
        const targetIdB = b.getAttribute('data-target-id') || '';
        const rowA = componentLayout[targetIdA]?.row ?? 0;
        const rowB = componentLayout[targetIdB]?.row ?? 0;
        return rowA - rowB;
    });

    // Iterate through bubbles sequentially (sorted by target row, top to bottom)
    for (const bubble of sortedBubbles) {
        const outputIndex = bubble.getAttribute('data-output-index');
        let textElement = document.getElementById(`text-${componentId}`);
        if (!textElement && outputIndex !== '-1') {
            textElement = document.getElementById(`text-${componentId}-${outputIndex}`);
        }

        if (textElement) {
            bubble.classList.add('active');
            if (enableTyping) {
                await typeText(textElement, displayText, typingSpeed);
            } else {
                textElement.textContent = displayText;
            }
        }
    }

    return bubbles.length > 0;
}

// Run full animation sequence
async function animateFlow() {
    // Reset all animations
    const bubbles = document.querySelectorAll('.speech-bubble');
    bubbles.forEach(bubble => bubble.classList.remove('active'));

    const docs = document.querySelectorAll('.document-icon');
    docs.forEach(doc => doc.classList.remove('active'));

    // If animations are disabled, show everything immediately
    if (!config.enableAnimations) {
        // Show bubbles for ALL components that have content (not just activated path)
        // This ensures bubbles appear for all source nodes in graphs with multiple entry points
        for (let i = 0; i < flowComponents.length; i++) {
            const component = flowComponents[i];

            // Show user query bubble
            if (component.type === 'chat_input' && component.args.query) {
                const displayText = truncateForDisplay(component.args.query);
                activateAllBubblesForComponent(component.id, 'chat_input', displayText);
            }

            // Show query_rewrite bubble
            if (component.type === 'query_rewrite' && component.args.rewritten_query) {
                const displayText = truncateForDisplay(component.args.rewritten_query);
                activateAllBubblesForComponent(component.id, 'query_rewrite', displayText);
            }

            // Show retriever documents (activate on all outgoing arrows)
            if (component.type === 'retriever' && component.args.passages) {
                const passages = component.args.passages;
                // Find the component's index in orderedFlowComponents for document lookup
                const componentIndex = orderedFlowComponents.findIndex(c => c.id === component.id);
                for (let docIndex = 0; docIndex < passages.length; docIndex++) {
                    // Get all document icons for this document across all arrows
                    const docs = document.querySelectorAll(`.document-icon[data-component-index="${componentIndex}"][data-doc-index="${docIndex}"]`);
                    docs.forEach(doc => doc.classList.add('active'));
                }
            }

            // Show query_expansion variation icons (activate on all outgoing arrows)
            if (component.type === 'query_expansion' && component.args.query_variations) {
                const variations = component.args.query_variations;
                const componentIndex = orderedFlowComponents.findIndex(c => c.id === component.id);
                for (let varIndex = 0; varIndex < variations.length; varIndex++) {
                    // Get all variation icons for this variation across all arrows
                    const varIcons = document.querySelectorAll(`.variation-icon[data-component-index="${componentIndex}"][data-var-index="${varIndex}"]`);
                    varIcons.forEach(varIcon => varIcon.classList.add('active'));
                }
            }

            // Show llm response bubble
            if (component.type === 'llm' && component.args.response) {
                const displayText = truncateForDisplay(component.args.response);
                activateAllBubblesForComponent(component.id, 'llm', displayText);
            }

            // Show citations bubble
            if (component.type === 'citations' && component.args.response_with_citations) {
                const displayText = truncateForDisplay(component.args.response_with_citations);
                activateAllBubblesForComponent(component.id, 'citations', displayText);
            }

            // Show answerability bubble
            if (component.type === 'answerability' && (component.args.answerability_label || component.args.answerability_likelihood !== undefined)) {
                let text;
                if (component.args.answerability_label) {
                    text = component.args.answerability_label;
                } else {
                    const likelihoodPercent = (parseFloat(component.args.answerability_likelihood) * 100).toFixed(1);
                    text = `Likelihood: ${likelihoodPercent}%`;
                }
                const displayText = truncateForDisplay(text);
                activateAllBubblesForComponent(component.id, 'answerability', displayText);
            }

            // Show query_clarification bubble
            if (component.type === 'query_clarification' && component.args.clarification_result) {
                const displayText = truncateForDisplay(component.args.clarification_result);
                activateAllBubblesForComponent(component.id, 'query_clarification', displayText);
            }

            // Show text bubble
            if (component.type === 'text' && component.args.text) {
                const displayText = truncateForDisplay(component.args.text);
                activateAllBubblesForComponent(component.id, 'text', displayText);
            }

            // Show conditional bubble
            if (component.type === 'conditional' && component.args.output_value) {
                const displayText = truncateForDisplay(component.args.output_value);
                activateAllBubblesForComponent(component.id, 'conditional', displayText);
            }

            // Show generic bubble
            if (component.type === 'generic' && component.args.output) {
                const displayText = truncateForDisplay(component.args.output);
                activateAllBubblesForComponent(component.id, 'generic', displayText);
            }

            // Show hallucination_detection bubble
            if (component.type === 'hallucination_detection' && component.args.hallucination_results) {
                const results = component.args.hallucination_results;
                let text = 'Faithfulness:\n';
                results.forEach((result, idx) => {
                    const pct = ((result.faithfulness_likelihood || 0) * 100).toFixed(0);
                    text += `Sent ${idx + 1}: ${pct}%\n`;
                });
                const displayText = truncateForDisplay(text.trim());
                activateAllBubblesForComponent(component.id, 'hallucination_detection', displayText);
            }
        }
        return; // Exit early, no animation needed
    }

    // Wait a bit then start
    await new Promise(resolve => setTimeout(resolve, 500));

    // Get all activated paths for animation (one per start node, sorted top to bottom by row)
    const allPaths = getAllActivatedPaths();

    // Animate each path sequentially (top branch first, then second branch, etc.)
    for (let pathIndex = 0; pathIndex < allPaths.length; pathIndex++) {
        const activatedPath = allPaths[pathIndex];

        // Animate sequentially through this path
        for (let i = 0; i < activatedPath.length; i++) {
            const component = activatedPath[i];

        // Animate user query bubble
        if (component.type === 'chat_input' && component.args.query) {
            const displayText = truncateForDisplay(component.args.query);
            const activated = await activateAllBubblesForComponentAnimated(
                component.id, 'chat_input', displayText,
                config.enableTypingAnimation, config.typingSpeed
            );
            if (activated) {
                await new Promise(resolve => setTimeout(resolve, config.queryDelay));
            }
        }

        // Animate query_rewrite rewritten query bubble
        if (component.type === 'query_rewrite' && component.args.rewritten_query) {
            const displayText = truncateForDisplay(component.args.rewritten_query);
            const activated = await activateAllBubblesForComponentAnimated(
                component.id, 'query_rewrite', displayText,
                config.enableTypingAnimation, config.typingSpeed
            );
            if (activated) {
                await new Promise(resolve => setTimeout(resolve, config.queryDelay));
            }
        }

        // Animate retriever documents (edge by edge, top to bottom)
        if (component.type === 'retriever' && component.args.passages) {
            const passages = component.args.passages;
            // Find the component's index in orderedFlowComponents for document lookup
            const componentIndex = orderedFlowComponents.findIndex(c => c.id === component.id);

            // Get all document icons for this component
            const allDocs = document.querySelectorAll(`.document-icon[data-component-index="${componentIndex}"]`);

            // Get unique target IDs and sort by row (top to bottom)
            const targetIds = [...new Set(Array.from(allDocs).map(d => d.getAttribute('data-target-id')))];
            targetIds.sort((a, b) => {
                const rowA = componentLayout[a]?.row ?? 0;
                const rowB = componentLayout[b]?.row ?? 0;
                return rowA - rowB;
            });

            // Iterate by target (sorted by row, top edge first)
            for (const targetId of targetIds) {
                // For this target, show all documents with stagger
                for (let docIndex = 0; docIndex < passages.length; docIndex++) {
                    const doc = document.querySelector(`.document-icon[data-component-index="${componentIndex}"][data-target-id="${targetId}"][data-doc-index="${docIndex}"]`);
                    if (doc) {
                        doc.classList.add('active');
                        await new Promise(resolve => setTimeout(resolve, config.documentStagger));
                    }
                }
            }

            // Wait after documents appear
            await new Promise(resolve => setTimeout(resolve, config.documentDelay));
        }

        // Animate query_expansion variation icons (edge by edge, top to bottom)
        if (component.type === 'query_expansion' && component.args.query_variations) {
            const variations = component.args.query_variations;
            const componentIndex = orderedFlowComponents.findIndex(c => c.id === component.id);

            // Get all variation icons for this component
            const allVars = document.querySelectorAll(`.variation-icon[data-component-index="${componentIndex}"]`);

            // Get unique target IDs and sort by row (top to bottom)
            const targetIds = [...new Set(Array.from(allVars).map(v => v.getAttribute('data-target-id')))];
            targetIds.sort((a, b) => {
                const rowA = componentLayout[a]?.row ?? 0;
                const rowB = componentLayout[b]?.row ?? 0;
                return rowA - rowB;
            });

            // Iterate by target (sorted by row, top edge first)
            for (const targetId of targetIds) {
                // For this target, show all variations with stagger
                for (let varIndex = 0; varIndex < variations.length; varIndex++) {
                    const varIcon = document.querySelector(`.variation-icon[data-component-index="${componentIndex}"][data-target-id="${targetId}"][data-var-index="${varIndex}"]`);
                    if (varIcon) {
                        varIcon.classList.add('active');
                        await new Promise(resolve => setTimeout(resolve, config.documentStagger));
                    }
                }
            }

            // Wait after variations appear
            await new Promise(resolve => setTimeout(resolve, config.documentDelay));
        }

        // Animate llm response bubble
        if (component.type === 'llm' && component.args.response) {
            const displayText = truncateForDisplay(component.args.response);
            const activated = await activateAllBubblesForComponentAnimated(
                component.id, 'llm', displayText,
                config.enableTypingAnimation, config.typingSpeed
            );
            if (activated) {
                await new Promise(resolve => setTimeout(resolve, config.queryDelay));
            }
        }

        // Animate citations response_with_citations bubble
        if (component.type === 'citations' && component.args.response_with_citations) {
            const displayText = truncateForDisplay(component.args.response_with_citations);
            const activated = await activateAllBubblesForComponentAnimated(
                component.id, 'citations', displayText,
                config.enableTypingAnimation, config.typingSpeed
            );
            if (activated) {
                await new Promise(resolve => setTimeout(resolve, config.queryDelay));
            }
        }

        // Animate answerability label bubble
        if (component.type === 'answerability' && (component.args.answerability_label || component.args.answerability_likelihood !== undefined)) {
            let text;
            if (component.args.answerability_label) {
                text = component.args.answerability_label;
            } else {
                const likelihoodPercent = (parseFloat(component.args.answerability_likelihood) * 100).toFixed(1);
                text = `Likelihood: ${likelihoodPercent}%`;
            }
            const displayText = truncateForDisplay(text);
            const activated = await activateAllBubblesForComponentAnimated(
                component.id, 'answerability', displayText,
                config.enableTypingAnimation, config.typingSpeed
            );
            if (activated) {
                await new Promise(resolve => setTimeout(resolve, config.queryDelay));
            }
        }

        // Animate query_clarification bubble
        if (component.type === 'query_clarification' && component.args.clarification_result) {
            const displayText = truncateForDisplay(component.args.clarification_result);
            const activated = await activateAllBubblesForComponentAnimated(
                component.id, 'query_clarification', displayText,
                config.enableTypingAnimation, config.typingSpeed
            );
            if (activated) {
                await new Promise(resolve => setTimeout(resolve, config.queryDelay));
            }
        }

        // Animate text bubble
        if (component.type === 'text' && component.args.text) {
            const displayText = truncateForDisplay(component.args.text);
            const activated = await activateAllBubblesForComponentAnimated(
                component.id, 'text', displayText,
                config.enableTypingAnimation, config.typingSpeed
            );
            if (activated) {
                await new Promise(resolve => setTimeout(resolve, config.queryDelay));
            }
        }

        // Animate conditional bubble
        if (component.type === 'conditional' && component.args.output_value) {
            const displayText = truncateForDisplay(component.args.output_value);
            const activated = await activateAllBubblesForComponentAnimated(
                component.id, 'conditional', displayText,
                config.enableTypingAnimation, config.typingSpeed
            );
            if (activated) {
                await new Promise(resolve => setTimeout(resolve, config.queryDelay));
            }
        }

        // Animate generic output bubble
        if (component.type === 'generic' && component.args.output) {
            const displayText = truncateForDisplay(component.args.output);
            const activated = await activateAllBubblesForComponentAnimated(
                component.id, 'generic', displayText,
                config.enableTypingAnimation, config.typingSpeed
            );
            if (activated) {
                await new Promise(resolve => setTimeout(resolve, config.queryDelay));
            }
        }

        // Animate hallucination_detection bubble
        if (component.type === 'hallucination_detection' && component.args.hallucination_results) {
            const results = component.args.hallucination_results;
            let text = 'Faithfulness:\n';
            results.forEach((result, idx) => {
                const pct = ((result.faithfulness_likelihood || 0) * 100).toFixed(0);
                text += `Sent ${idx + 1}: ${pct}%\n`;
            });
            const displayText = truncateForDisplay(text.trim());
            const activated = await activateAllBubblesForComponentAnimated(
                component.id, 'hallucination_detection', displayText,
                config.enableTypingAnimation, config.typingSpeed
            );
            if (activated) {
                await new Promise(resolve => setTimeout(resolve, config.componentDelay));
            }
        }
        }  // End of inner for loop (activatedPath)
    }  // End of outer for loop (allPaths)
}

// Apply configuration
function applyConfig() {
    // Apply font size, color, and family to ALL text elements in speech bubbles
    // Use querySelectorAll to find all text divs inside foreignObjects in speech bubbles
    const allBubbleTextElements = document.querySelectorAll('.speech-bubble foreignObject div');
    allBubbleTextElements.forEach(textElement => {
        textElement.style.fontSize = `${config.bubbleFontSize}px`;
        textElement.style.color = config.bubbleFontColor;
        textElement.style.fontFamily = config.bubbleFontFamily;
        textElement.style.fontStyle = config.bubbleFontStyle;
    });

    // Apply font size to component labels
    const componentLabels = document.querySelectorAll('.component-label');
    componentLabels.forEach(label => {
        label.setAttribute('font-size', config.componentLabelFontSize);
    });

    // Apply arrow width, color, and style to all arrows
    const arrows = document.querySelectorAll('.dynamic-arrow');
    arrows.forEach(arrow => {
        arrow.setAttribute('stroke-width', config.arrowWidth);
        arrow.setAttribute('stroke', config.arrowColor);
        arrow.setAttribute('stroke-dasharray', config.arrowDashArray);
    });

    // Apply arrow color to arrowhead marker
    const arrowhead = document.querySelector('#arrowhead polygon');
    if (arrowhead) {
        arrowhead.setAttribute('fill', config.arrowColor);
    }

    // Configure bubble shadow filter
    const shadowFilter = document.querySelector('#bubbleShadow feDropShadow');
    if (shadowFilter) {
        shadowFilter.setAttribute('dx', config.bubbleShadowOffsetX);
        shadowFilter.setAttribute('dy', config.bubbleShadowOffsetY);
        shadowFilter.setAttribute('stdDeviation', config.bubbleShadowBlur);
        shadowFilter.setAttribute('flood-color', config.bubbleShadowColor);
    }

    // Apply shadow to all bubbles if enabled
    const allBubbles = document.querySelectorAll('.speech-bubble');
    allBubbles.forEach(bubble => {
        if (config.bubbleShadowEnabled) {
            bubble.setAttribute('filter', 'url(#bubbleShadow)');
        } else {
            bubble.removeAttribute('filter');
        }
    });

    // Configure document shadow filter
    const documentShadowFilter = document.querySelector('#documentShadow feDropShadow');
    if (documentShadowFilter) {
        documentShadowFilter.setAttribute('dx', config.documentShadowOffsetX);
        documentShadowFilter.setAttribute('dy', config.documentShadowOffsetY);
        documentShadowFilter.setAttribute('stdDeviation', config.documentShadowBlur);
        documentShadowFilter.setAttribute('flood-color', config.documentShadowColor);
    }

    // Apply bubble dimensions and fill color to all bubbles
    const rx = 4; // Corner radius
    const width = config.bubbleWidth;
    const height = config.bubbleHeight;

    // Generate unified bubble path with dynamic dimensions
    const tailW = 6; // tail width
    const tailH = 5; // tail height
    const tailX = width / 2;
    const unifiedBubblePath = `M ${rx} 0 H ${width - rx} A ${rx} ${rx} 0 0 1 ${width} ${rx} V ${height - rx} A ${rx} ${rx} 0 0 1 ${width - rx} ${height} H ${tailX + tailW/2} L ${tailX} ${height + tailH} L ${tailX - tailW/2} ${height} H ${rx} A ${rx} ${rx} 0 0 1 0 ${height - rx} V ${rx} A ${rx} ${rx} 0 0 1 ${rx} 0 Z`;

    const padding = 8;
    const textWidth = config.bubbleWidth - padding;
    const textHeight = config.bubbleHeight - padding;

    allBubbles.forEach(bubble => {
        const bubblePath = bubble.querySelector('.bubble-path');
        const bubbleFO = bubble.querySelector('foreignObject');

        if (bubblePath) {
            bubblePath.setAttribute('d', unifiedBubblePath);
            bubblePath.setAttribute('fill', config.bubbleFillColor);
            bubblePath.setAttribute('stroke-width', config.bubbleBorderWidth);
            bubblePath.setAttribute('stroke', config.bubbleBorderColor);
        }

        if (bubbleFO) {
            bubbleFO.setAttribute('width', textWidth);
            bubbleFO.setAttribute('height', textHeight);
        }
    });
}

// ========== Data Loading and Visualization Initialization ==========

// UI Element references
const loadingState = document.getElementById('loadingState');
const errorState = document.getElementById('errorState');
const errorMessage = document.getElementById('errorMessage');
const visualizationContainer = document.getElementById('visualizationContainer');
const refreshButton = document.getElementById('refreshButton');
const refreshIcon = document.getElementById('refreshIcon');

// Show loading state
function showLoading() {
    loadingState.classList.remove('hidden');
    errorState.classList.add('hidden');
    visualizationContainer.classList.add('hidden');
    const setupState = document.getElementById('setupState');
    if (setupState) setupState.classList.add('hidden');
    // Show header controls
    const header = document.querySelector('.header');
    if (header) header.classList.remove('setup-mode');
    refreshButton.disabled = true;
    refreshIcon.innerHTML = '<div class="spinner"></div>';
}

// Show error state (or setup state if API key is missing)
function showError(message) {
    loadingState.classList.add('hidden');
    visualizationContainer.classList.add('hidden');
    refreshButton.disabled = false;
    refreshIcon.textContent = '↻';

    // Check if error is about missing API key - show setup state instead
    const setupState = document.getElementById('setupState');
    const header = document.querySelector('.header');
    const isApiKeyError = message && (
        message.toLowerCase().includes('api key') ||
        message.toLowerCase().includes('apikey')
    );

    if (isApiKeyError && setupState) {
        errorState.classList.add('hidden');
        setupState.classList.remove('hidden');
        // Hide header controls in setup mode
        if (header) header.classList.add('setup-mode');
        // Clear the stored key since it's invalid (don't auto-load on refresh)
        localStorage.removeItem('langflowApiKey');
        // Keep the invalid key in the input fields so user can see/correct it
        const setupApiKeyInput = document.getElementById('setupApiKeyInput');
        // Show error message on setup screen
        const setupErrorMessage = document.getElementById('setupErrorMessage');
        if (setupErrorMessage) {
            setupErrorMessage.textContent = message;
            setupErrorMessage.classList.remove('hidden');
        }
        if (setupApiKeyInput) {
            setupApiKeyInput.classList.add('error');
        }
    } else {
        errorState.classList.remove('hidden');
        if (setupState) setupState.classList.add('hidden');
        // Show header controls for regular errors
        if (header) header.classList.remove('setup-mode');
        errorMessage.textContent = message;
    }
}

// Show visualization
function showVisualization() {
    loadingState.classList.add('hidden');
    errorState.classList.add('hidden');
    visualizationContainer.classList.remove('hidden');
    refreshButton.disabled = false;
    refreshIcon.textContent = '↻';
}

// Load visualization data from API
async function loadVisualization(traceId = null) {
    if (isLoading) return;
    isLoading = true;
    showLoading();

    try {
        // Build URL with optional trace_id parameter
        let url = '/api/refresh';
        if (traceId) {
            url += `?trace_id=${encodeURIComponent(traceId)}`;
        }
        const headers = {};
        const apiKeyInput = document.getElementById('langflowApiKey');
        if (apiKeyInput && apiKeyInput.value.trim()) {
            headers['X-Langflow-Api-Key'] = apiKeyInput.value.trim();
        }
        const response = await fetch(url, { headers });
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.message || `HTTP error ${response.status}`);
        }

        const data = await response.json();

        // Check for error in response
        if (data.error) {
            throw new Error(data.message || data.error);
        }

        // Update flowConfig and flowComponents
        flowConfig = data;
        flowComponents = flowConfig.components || [];

        // Update currentTraceId from response (for dropdown highlighting)
        if (flowConfig.trace_id) {
            currentTraceId = flowConfig.trace_id;
        }

        // Compute layout
        computeLayout();

        // Show visualization container BEFORE initializing
        // (positioning calculations require visible elements)
        showVisualization();

        // Initialize visualization
        initializeVisualization();

        visualizationInitialized = true;

        // Log the flow information
        console.log('Flow:', flowConfig.name);
        console.log('Flow configuration:', flowConfig);

    } catch (error) {
        console.error('Error loading visualization:', error);
        showError(error.message || 'Failed to load visualization');
    } finally {
        isLoading = false;
    }
}

// Initialize the visualization with loaded data
function initializeVisualization() {
    // Step 0: Set flow title
    document.getElementById('flowTitle').textContent = flowConfig.name;

    // Step 1: Create all dynamic elements
    createComponents();
    createArrows();
    createLabels();
    createSpeechBubbles();

    // Step 2: Normalize icons
    normalizeIcons();

    // Step 3: Position components and arrows
    positionComponents();
    positionArrows();

    // Step 4: Position labels (after components have transforms)
    positionLabels();
    positionLabelBackgrounds();

    // Step 5: Position speech bubbles and create documents/variations
    positionSpeechBubbles();
    createDocumentIcons();
    createVariationIcons();

    // Step 6: Apply configuration styling
    applyConfig();

    // Setup zoom controls
    let currentZoom = 1.0;
    const zoomControls = document.getElementById('zoomControls');
    const zoomInBtn = document.getElementById('zoomIn');
    const zoomOutBtn = document.getElementById('zoomOut');
    const svgElement = document.getElementById('flowChart');

    // Store original viewBox values
    const originalViewBox = svgElement.getAttribute('viewBox').split(' ').map(Number);
    const baseX = originalViewBox[0];
    const baseY = originalViewBox[1];
    const baseWidth = originalViewBox[2];
    const baseHeight = originalViewBox[3];

    // Show/hide zoom controls based on config
    if (config.showZoomControls) {
        zoomControls.classList.remove('hidden');
    }

    function updateZoom(newZoom) {
        currentZoom = Math.max(config.zoomMin, Math.min(config.zoomMax, newZoom));

        const newWidth = baseWidth / currentZoom;
        const newHeight = baseHeight / currentZoom;
        const newX = baseX;
        const newY = baseY;

        svgElement.setAttribute('viewBox', `${newX} ${newY} ${newWidth} ${newHeight}`);

        zoomInBtn.disabled = currentZoom >= config.zoomMax;
        zoomOutBtn.disabled = currentZoom <= config.zoomMin;
    }

    zoomInBtn.addEventListener('click', () => {
        updateZoom(currentZoom + config.zoomIncrement);
    });

    zoomOutBtn.addEventListener('click', () => {
        updateZoom(currentZoom - config.zoomIncrement);
    });

    // Initialize button states
    updateZoom(currentZoom);

    // Start animation loop
    async function animationLoop() {
        await new Promise(resolve => setTimeout(resolve, 1000));

        if (!config.enableAnimations) {
            await animateFlow();
            return;
        }

        do {
            await animateFlow();
            if (config.enableAnimationLoop) {
                await new Promise(resolve => setTimeout(resolve, config.animationRestartDelay));
            }
        } while (config.enableAnimationLoop);
    }

    animationLoop();

    // ========== Setup component click handlers ==========
    // Add click handlers to all components (must be done each time components are recreated)
    const componentElements = document.querySelectorAll('.component');
    componentElements.forEach(comp => {
        comp.addEventListener('click', (e) => {
            e.stopPropagation();
            const componentId = comp.getAttribute('data-component-id');
            // Toggle: if already selected, close panel; otherwise open it
            if (selectedComponent && selectedComponent.getAttribute('data-component-id') === componentId) {
                closePropertiesPanel();
            } else {
                openPropertiesPanel(componentId);
            }
        });
    });
}

// ========== Properties Panel Functions (defined at top level) ==========

function closePropertiesPanel() {
    if (!propertiesPanel) return;
    propertiesPanel.classList.remove('open');
    if (selectedComponent) {
        selectedComponent.classList.remove('selected');
        selectedComponent = null;
    }
}

function openPropertiesPanel(componentId) {
    if (!propertiesPanel || !propertiesPanelContent) return;

    // Find the component data
    const component = flowComponents.find(c => c.id === componentId);
    if (!component) return;

    // Update selection state
    if (selectedComponent) {
        selectedComponent.classList.remove('selected');
    }
    selectedComponent = document.getElementById(`component-${componentId}`);
    if (selectedComponent) {
        selectedComponent.classList.add('selected');
    }

    // Generate panel content
    propertiesPanelContent.innerHTML = generatePanelContent(component);

    // Setup section toggles
    setupSectionToggles();

    // Open panel
    propertiesPanel.classList.add('open');
}

// Parse markdown-like content for citations display
function parseMarkdown(text) {
    if (!text) return '';

    // Convert <br> tags to newlines first
    let result = text.replace(/<br\s*\/?>/gi, '\n');

    // Escape HTML entities except our allowed tags
    result = result.replace(/&/g, '&amp;')
                  .replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;');

    // Parse bold text **text**
    result = result.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Parse italic text *text* (but not inside **)
    result = result.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');

    // Parse citation references like [1], [2], etc.
    result = result.replace(/\[(\d+)\]/g, '<span class="citation-ref">[$1]</span>');

    // Parse "Citations:" header
    result = result.replace(/(Citations?:)/gi, '<div class="citations-header">$1</div>');

    // Parse citation items starting with [n] at line start
    result = result.replace(/^(<span class="citation-ref">\[(\d+)\]<\/span>)\s*(.+)$/gm,
        '<div class="citation-item"><span class="citation-number">[$2]</span> $3</div>');

    // Convert remaining newlines to paragraphs
    const paragraphs = result.split(/\n\n+/).filter(p => p.trim());
    if (paragraphs.length > 1) {
        result = paragraphs.map(p => {
            // Don't wrap if it's already a block element
            if (p.startsWith('<div')) return p;
            return `<p>${p.replace(/\n/g, '<br>')}</p>`;
        }).join('');
    } else {
        result = result.replace(/\n/g, '<br>');
    }

    return result;
}

function generatePanelContent(component) {
    const color = componentColors[component.type] || '#94a3b8';
    const typeLabel = component.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

    let html = `
        <div class="component-type-header">
            <div class="component-type-icon" style="background: ${color}20;">
                <svg width="24" height="24" viewBox="0 0 40 40">
                    ${getIconSVG(component.type, color)}
                </svg>
            </div>
            <div class="component-type-info">
                <h3>${component.label || typeLabel}</h3>
                <span style="color: ${color};">${typeLabel}</span>
            </div>
        </div>
    `;

    // Error Section (if component has an error)
    if (component.args && component.args.error) {
        html += `
            <div class="property-section error-section" style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 6px; margin-bottom: 12px;">
                <div class="property-section-header" style="background: #fee2e2;">
                    <span class="property-section-title" style="color: #dc2626; font-weight: 600;">⚠ Error</span>
                    <span class="property-section-toggle">&#9660;</span>
                </div>
                <div class="property-section-content">
                    <div class="property-value" style="color: #b91c1c; font-family: monospace; font-size: 11px; white-space: pre-wrap; word-break: break-word;">${component.args.error}</div>
                </div>
            </div>
        `;
    }

    // Basic Info Section
    html += `
        <div class="property-section">
            <div class="property-section-header">
                <span class="property-section-title">Basic Info</span>
                <span class="property-section-toggle">&#9660;</span>
            </div>
            <div class="property-section-content">
                <div class="property-row">
                    <span class="property-label">ID</span>
                    <span class="property-value">${component.id}</span>
                </div>
                <div class="property-row">
                    <span class="property-label">Type</span>
                    <span class="property-value">${typeLabel}</span>
                </div>
                ${component.label ? `
                <div class="property-row">
                    <span class="property-label">Label</span>
                    <span class="property-value">${component.label}</span>
                </div>
                ` : ''}
            </div>
        </div>
    `;

    // Connections Section
    if (component.output_components.length > 0) {
        html += `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-title">Connections</span>
                    <span class="property-section-toggle">&#9660;</span>
                </div>
                <div class="property-section-content">
                    <div class="connection-list">
        `;
        component.output_components.forEach((outputId) => {
            const targetComp = flowComponents.find(c => c.id === outputId);
            const targetLabel = targetComp ? (targetComp.label || targetComp.type.replace(/_/g, ' ')) : outputId;
            html += `
                <div class="connection-item clickable-connection" data-target-id="${outputId}">
                    <span class="arrow">&#8594;</span>
                    <span>${targetLabel}</span>
                </div>
            `;
        });
        html += `
                    </div>
                </div>
            </div>
        `;
    }

    // Args Section - varies by component type
    if (component.args && Object.keys(component.args).length > 0) {
        html += generateArgsSection(component);
    }

    return html;
}

function generateArgsSection(component) {
    let html = '';

    // Handle different arg types
    if (component.type === 'chat_input' && component.args.query) {
        html += `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-title">Query</span>
                    <span class="property-section-toggle">&#9660;</span>
                </div>
                <div class="property-section-content">
                    <div class="property-value" style="font-style: italic;">${component.args.query}</div>
                </div>
            </div>
        `;
    }

    if (component.type === 'query_rewrite' && component.args.rewritten_query) {
        html += `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-title">Rewritten Query</span>
                    <span class="property-section-toggle">&#9660;</span>
                </div>
                <div class="property-section-content">
                    <div class="property-value" style="font-style: italic;">${component.args.rewritten_query}</div>
                </div>
            </div>
        `;
    }

    if (component.type === 'query_expansion' && component.args.query_variations) {
        html += `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-title">Query Variations (${component.args.query_variations.length})</span>
                    <span class="property-section-toggle">&#9660;</span>
                </div>
                <div class="property-section-content">
        `;
        component.args.query_variations.forEach((variation, idx) => {
            html += `
                <div class="variation-item">
                    <div class="variation-number">${idx + 1}</div>
                    <div class="variation-text">${variation}</div>
                    <span class="variation-expand" data-expanded="false">Show more</span>
                </div>
            `;
        });
        html += `
                </div>
            </div>
        `;
    }

    if (component.type === 'retriever' && component.args.passages) {
        html += `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-title">Retrieved Passages (${component.args.passages.length})</span>
                    <span class="property-section-toggle">&#9660;</span>
                </div>
                <div class="property-section-content">
        `;
        component.args.passages.forEach((passage, idx) => {
            html += `
                <div class="passage-item">
                    <div class="passage-title">[${passage.id || idx + 1}] ${passage.title || 'Passage ' + (idx + 1)}</div>
                    <div class="passage-text">${passage.text || ''}</div>
                    <span class="passage-expand" data-expanded="false">Show more</span>
                </div>
            `;
        });
        html += `
                </div>
            </div>
        `;
    }

    if (component.type === 'llm' && component.args.response) {
        html += `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-title">Response</span>
                    <span class="property-section-toggle">&#9660;</span>
                </div>
                <div class="property-section-content">
                    <div class="property-value">${component.args.response}</div>
                </div>
            </div>
        `;
    }

    if (component.type === 'citations' && component.args.response_with_citations) {
        html += `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-title">Response with Citations</span>
                    <span class="property-section-toggle">&#9660;</span>
                </div>
                <div class="property-section-content">
                    <div class="markdown-content">${parseMarkdown(component.args.response_with_citations)}</div>
                </div>
            </div>
        `;
    }

    if (component.type === 'answerability') {
        // Show answerability label if present
        if (component.args.answerability_label) {
            const labelLower = component.args.answerability_label.toLowerCase();
            const labelColor = labelLower === 'answerable' ? '#22c55e' : '#ef4444';
            html += `
                <div class="property-section">
                    <div class="property-section-header">
                        <span class="property-section-title">Answerability Label</span>
                        <span class="property-section-toggle">&#9660;</span>
                    </div>
                    <div class="property-section-content">
                        <div class="property-value" style="color: ${labelColor}; font-weight: 600;">${component.args.answerability_label}</div>
                    </div>
                </div>
            `;
        }

        // Show answerability likelihood if present
        if (component.args.answerability_likelihood !== undefined && component.args.answerability_likelihood !== null) {
            const likelihood = parseFloat(component.args.answerability_likelihood);
            const likelihoodColor = likelihood >= 0.5 ? '#22c55e' : '#ef4444';
            const likelihoodPercent = (likelihood * 100).toFixed(1);
            html += `
                <div class="property-section">
                    <div class="property-section-header">
                        <span class="property-section-title">Answerability Likelihood</span>
                        <span class="property-section-toggle">&#9660;</span>
                    </div>
                    <div class="property-section-content">
                        <div class="property-value" style="color: ${likelihoodColor}; font-weight: 600;">${likelihoodPercent}%</div>
                        <div class="likelihood-bar" style="background: #e5e7eb; border-radius: 4px; height: 8px; margin-top: 8px; overflow: hidden;">
                            <div style="background: ${likelihoodColor}; height: 100%; width: ${likelihoodPercent}%; transition: width 0.3s ease;"></div>
                        </div>
                    </div>
                </div>
            `;
        }
    }

    if (component.type === 'query_clarification' && component.args.clarification_result) {
        const isClear = component.args.clarification_result.toUpperCase() === 'CLEAR';
        const resultColor = isClear ? '#22c55e' : '#f59e0b';  // Green for CLEAR, amber for question
        const resultLabel = isClear ? 'Query is Clear' : 'Clarification Needed';
        html += `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-title">${resultLabel}</span>
                    <span class="property-section-toggle">&#9660;</span>
                </div>
                <div class="property-section-content">
                    <div class="property-value" style="color: ${resultColor}; font-weight: 600;">${component.args.clarification_result}</div>
                </div>
            </div>
        `;
    }

    if (component.type === 'hallucination_detection' && component.args.hallucination_results) {
        const results = component.args.hallucination_results;
        html += `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-title">Hallucination Analysis (${results.length} sentences)</span>
                    <span class="property-section-toggle">&#9660;</span>
                </div>
                <div class="property-section-content">
        `;
        results.forEach((result, idx) => {
            const likelihood = result.faithfulness_likelihood || 0;
            const likelihoodPercent = (likelihood * 100).toFixed(0);
            // Color: green if >= 0.7, amber if >= 0.4, red if < 0.4
            const barColor = likelihood >= 0.7 ? '#22c55e' : (likelihood >= 0.4 ? '#f59e0b' : '#ef4444');
            const statusLabel = likelihood >= 0.7 ? 'Faithful' : (likelihood >= 0.4 ? 'Uncertain' : 'Likely Hallucinated');
            html += `
                <div class="hallucination-item" style="margin-bottom: 12px; padding: 8px; background: #f9fafb; border-radius: 6px; border-left: 3px solid ${barColor};">
                    <div style="font-size: 13px; font-weight: 500; margin-bottom: 4px; color: #374151;">"${result.response_text}"</div>
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                        <span style="font-size: 12px; color: ${barColor}; font-weight: 600;">${statusLabel} (${likelihoodPercent}%)</span>
                        <div style="flex: 1; background: #e5e7eb; border-radius: 4px; height: 6px; overflow: hidden;">
                            <div style="background: ${barColor}; height: 100%; width: ${likelihoodPercent}%;"></div>
                        </div>
                    </div>
                    <div class="hallucination-explanation" style="font-size: 11px; color: #6b7280; display: none;">${result.explanation || ''}</div>
                    <span class="hallucination-expand" data-expanded="false" style="font-size: 11px; color: #3b82f6; cursor: pointer;">Show explanation</span>
                </div>
            `;
        });
        html += `
                </div>
            </div>
        `;
    }

    if (component.type === 'text' && component.args.text) {
        html += `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-title">Text Output</span>
                    <span class="property-section-toggle">&#9660;</span>
                </div>
                <div class="property-section-content">
                    <div class="property-value">${component.args.text}</div>
                </div>
            </div>
        `;
    }

    if (component.type === 'conditional') {
        // Condition section (combines operator and match_text)
        if (component.args.operator || component.args.match_text) {
            const operator = component.args.operator || '?';
            const matchText = component.args.match_text || '?';
            html += `
                <div class="property-section">
                    <div class="property-section-header">
                        <span class="property-section-title">Condition</span>
                        <span class="property-section-toggle">&#9660;</span>
                    </div>
                    <div class="property-section-content">
                        <div class="property-value" style="font-family: monospace;">Input <span style="font-weight: 600; color: #8b5cf6;">${operator}</span> "${matchText}"</div>
                    </div>
                </div>
            `;
        }

        // True Value section
        const trueActivated = component.args.activated_branch === 'true';
        const trueValue = component.args.true_value;
        const trueBorderStyle = trueActivated ? 'border-left: 3px solid #22c55e; padding-left: 8px;' : 'opacity: 0.5;';
        const trueIndicator = trueActivated ? ' ✓' : '';
        html += `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-title" style="${trueActivated ? 'color: #22c55e; font-weight: 600;' : ''}">True Value${trueIndicator}</span>
                    <span class="property-section-toggle">&#9660;</span>
                </div>
                <div class="property-section-content">
                    <div class="property-value" style="${trueBorderStyle}">${trueValue || '<em style="color: #9ca3af;">not set</em>'}</div>
                </div>
            </div>
        `;

        // False Value section
        const falseActivated = component.args.activated_branch === 'false';
        const falseValue = component.args.false_value;
        const falseBorderStyle = falseActivated ? 'border-left: 3px solid #ef4444; padding-left: 8px;' : 'opacity: 0.5;';
        const falseIndicator = falseActivated ? ' ✓' : '';
        html += `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-title" style="${falseActivated ? 'color: #ef4444; font-weight: 600;' : ''}">False Value${falseIndicator}</span>
                    <span class="property-section-toggle">&#9660;</span>
                </div>
                <div class="property-section-content">
                    <div class="property-value" style="${falseBorderStyle}">${falseValue || '<em style="color: #9ca3af;">not set</em>'}</div>
                </div>
            </div>
        `;
    }

    if (component.type === 'generic' && component.args.output) {
        let outputLabel = 'Output';
        if (component.args.output_key) {
            // Format output_key: replace underscores with spaces, capitalize each word
            const formattedKey = component.args.output_key
                .replace(/_/g, ' ')
                .replace(/\b\w/g, char => char.toUpperCase());
            outputLabel = `Output (${formattedKey})`;
        }
        html += `
            <div class="property-section">
                <div class="property-section-header">
                    <span class="property-section-title">${outputLabel}</span>
                    <span class="property-section-toggle">&#9660;</span>
                </div>
                <div class="property-section-content">
                    <div class="property-value">${component.args.output}</div>
                </div>
            </div>
        `;
    }

    return html;
}

function getIconSVG(type, color) {
    // Simplified icons for the panel header
    const icons = {
        chat_input: `<circle cx="20" cy="20" r="14" fill="none" stroke="${color}" stroke-width="2"/>
               <circle cx="20" cy="16" r="4" fill="none" stroke="${color}" stroke-width="1.5"/>
               <path d="M 13 28 C 13 24 16 22 20 22 C 24 22 27 24 27 28" fill="none" stroke="${color}" stroke-width="1.5"/>`,
        chat_output: `<circle cx="20" cy="20" r="14" fill="none" stroke="${color}" stroke-width="2"/>
               <path d="M 12 13 L 28 13 Q 29 13 29 14 L 29 22 Q 29 23 28 23 L 17 23 L 14 27 L 14 23 L 12 23 Q 11 23 11 22 L 11 14 Q 11 13 12 13 Z" fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round"/>
               <circle cx="15" cy="18" r="1" fill="${color}"/>
               <circle cx="20" cy="18" r="1" fill="${color}"/>
               <circle cx="25" cy="18" r="1" fill="${color}"/>`,
        retriever: `<circle cx="20" cy="20" r="14" fill="none" stroke="${color}" stroke-width="2"/>
                    <circle cx="17" cy="17" r="5" fill="none" stroke="${color}" stroke-width="1.5"/>
                    <line x1="21" y1="21" x2="26" y2="26" stroke="${color}" stroke-width="2" stroke-linecap="round"/>`,
        llm: `<circle cx="20" cy="20" r="14" fill="none" stroke="${color}" stroke-width="2"/>
              <path d="M 15 14 Q 12 17 12 20 Q 12 23 15 26" fill="none" stroke="${color}" stroke-width="1.5"/>
              <path d="M 25 14 Q 28 17 28 20 Q 28 23 25 26" fill="none" stroke="${color}" stroke-width="1.5"/>
              <path d="M 15 14 Q 20 11 25 14" fill="none" stroke="${color}" stroke-width="1.5"/>
              <path d="M 15 26 Q 20 29 25 26" fill="none" stroke="${color}" stroke-width="1.5"/>`,
        query_rewrite: `<circle cx="20" cy="20" r="14" fill="none" stroke="${color}" stroke-width="2"/>
                        <rect x="16" y="12" width="8" height="12" fill="none" stroke="${color}" stroke-width="1.5" rx="1" transform="rotate(-45 20 20)"/>`,
        query_expansion: `<circle cx="20" cy="20" r="14" fill="none" stroke="${color}" stroke-width="2"/>
                          <path d="M 12 20 L 16 20" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round"/>
                          <path d="M 16 20 L 20 15 L 28 15" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                          <path d="M 16 20 L 28 20" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round"/>
                          <path d="M 16 20 L 20 25 L 28 25" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>`,
        query_clarification: `<circle cx="20" cy="20" r="14" fill="none" stroke="${color}" stroke-width="2"/>
                              <path d="M 12 12 L 28 12 Q 30 12 30 14 L 30 24 Q 30 26 28 26 L 18 26 L 14 30 L 14 26 L 12 26 Q 10 26 10 24 L 10 14 Q 10 12 12 12 Z" fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round"/>
                              <path d="M 18 16 Q 18 14.5 20 14.5 Q 22 14.5 22 16 Q 22 17.5 20 18.5 L 20 20" fill="none" stroke="${color}" stroke-width="1.3" stroke-linecap="round"/>
                              <circle cx="20" cy="22.5" r="0.9" fill="${color}"/>`,
        hallucination_detection: `<circle cx="20" cy="20" r="14" fill="none" stroke="${color}" stroke-width="2"/>
                          <line x1="20" y1="8" x2="20" y2="32" stroke="${color}" stroke-width="1.5"/>
                          <path d="M 8 19 L 11 22 L 16 16" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                          <path d="M 25 16 L 31 22 M 31 16 L 25 22" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round"/>`,
        citations: `<circle cx="20" cy="20" r="14" fill="none" stroke="${color}" stroke-width="2"/>
                    <rect x="14" y="12" width="12" height="16" fill="none" stroke="${color}" stroke-width="1.5" rx="1"/>`,
        answerability: `<circle cx="20" cy="20" r="14" fill="none" stroke="${color}" stroke-width="2"/>
                        <path d="M 16 16 Q 16 12 20 12 Q 24 12 24 16 Q 24 19 20 21" fill="none" stroke="${color}" stroke-width="2"/>
                        <circle cx="20" cy="26" r="1.5" fill="${color}"/>`,
        text: `<circle cx="20" cy="20" r="14" fill="none" stroke="${color}" stroke-width="2"/>
               <text x="20" y="24" fill="${color}" font-family="Arial" font-weight="600" text-anchor="middle" font-size="12">Aa</text>`,
        conditional: `<circle cx="20" cy="20" r="14" fill="none" stroke="${color}" stroke-width="2"/>
                      <path d="M 12 20 L 17 20" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round"/>
                      <path d="M 17 20 L 22 14 L 28 14" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                      <path d="M 17 20 L 22 26 L 28 26" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                      <circle cx="17" cy="20" r="2" fill="${color}"/>`,
        generic: `<circle cx="20" cy="20" r="14" fill="none" stroke="${color}" stroke-width="2"/>
                  <circle cx="20" cy="20" r="4" fill="none" stroke="${color}" stroke-width="1.5"/>
                  <path d="M20 12 L20 14 M20 26 L20 28 M12 20 L14 20 M26 20 L28 20 M14 14 L15.5 15.5 M24.5 24.5 L26 26 M14 26 L15.5 24.5 M24.5 15.5 L26 14" stroke="${color}" stroke-width="1.5" stroke-linecap="round"/>`
    };
    return icons[type] || icons.generic;
}

function setupSectionToggles() {
    if (!propertiesPanelContent) return;

    const headers = propertiesPanelContent.querySelectorAll('.property-section-header');
    headers.forEach(header => {
        header.addEventListener('click', () => {
            const section = header.parentElement;
            section.classList.toggle('collapsed');
        });
    });

    // Add click handlers for clickable connections
    const connectionItems = propertiesPanelContent.querySelectorAll('.clickable-connection');
    connectionItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent document click handler from closing panel
            const targetId = item.getAttribute('data-target-id');
            if (targetId) {
                openPropertiesPanel(targetId);
            }
        });
    });

    // Add click handlers for passage expand/collapse
    const passageExpands = propertiesPanelContent.querySelectorAll('.passage-expand');
    passageExpands.forEach(expandBtn => {
        expandBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const passageItem = expandBtn.parentElement;
            const passageText = passageItem.querySelector('.passage-text');
            const isExpanded = expandBtn.getAttribute('data-expanded') === 'true';

            if (isExpanded) {
                passageText.classList.remove('expanded');
                expandBtn.textContent = 'Show more';
                expandBtn.setAttribute('data-expanded', 'false');
            } else {
                passageText.classList.add('expanded');
                expandBtn.textContent = 'Show less';
                expandBtn.setAttribute('data-expanded', 'true');
            }
        });
    });

    // Add click handlers for variation expand/collapse
    const variationExpands = propertiesPanelContent.querySelectorAll('.variation-expand');
    variationExpands.forEach(expandBtn => {
        expandBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const variationItem = expandBtn.parentElement;
            const variationText = variationItem.querySelector('.variation-text');
            const isExpanded = expandBtn.getAttribute('data-expanded') === 'true';

            if (isExpanded) {
                variationText.classList.remove('expanded');
                expandBtn.textContent = 'Show more';
                expandBtn.setAttribute('data-expanded', 'false');
            } else {
                variationText.classList.add('expanded');
                expandBtn.textContent = 'Show less';
                expandBtn.setAttribute('data-expanded', 'true');
            }
        });
    });

    // Add click handlers for hallucination explanation expand/collapse
    const hallucinationExpands = propertiesPanelContent.querySelectorAll('.hallucination-expand');
    hallucinationExpands.forEach(expandBtn => {
        expandBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const item = expandBtn.parentElement;
            const explanation = item.querySelector('.hallucination-explanation');
            const isExpanded = expandBtn.getAttribute('data-expanded') === 'true';

            if (isExpanded) {
                explanation.style.display = 'none';
                expandBtn.textContent = 'Show explanation';
                expandBtn.setAttribute('data-expanded', 'false');
            } else {
                explanation.style.display = 'block';
                expandBtn.textContent = 'Hide explanation';
                expandBtn.setAttribute('data-expanded', 'true');
            }
        });
    });
}

// ========== Trace Selector Functions ==========

// Format timestamp as relative time (e.g., "2 min ago")
function formatRelativeTime(timestamp) {
    const now = new Date();
    const date = new Date(timestamp);
    const diffMs = now - date;
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHour = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHour / 24);

    if (diffSec < 60) return 'just now';
    if (diffMin < 60) return `${diffMin} min ago`;
    if (diffHour < 24) return `${diffHour}h ago`;
    if (diffDay < 7) return `${diffDay}d ago`;

    // Fall back to date format
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

// Fetch list of recent traces from API with pagination
async function fetchTraceList(page = 1, append = false) {
    try {
        const traceHeaders = {};
        const traceApiKeyInput = document.getElementById('langflowApiKey');
        if (traceApiKeyInput && traceApiKeyInput.value.trim()) {
            traceHeaders['X-Langflow-Api-Key'] = traceApiKeyInput.value.trim();
        }
        const response = await fetch(`/api/traces?limit=15&page=${page}`, { headers: traceHeaders });
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        const result = await response.json();

        if (append) {
            // Append new traces to existing list
            traceList = [...traceList, ...result.traces];
        } else {
            // Replace trace list (first page)
            traceList = result.traces;
        }

        traceCurrentPage = result.page;
        traceHasMore = result.hasMore;

        return result;
    } catch (error) {
        console.error('Error fetching trace list:', error);
        throw error;
    }
}

// Load more traces when scrolling
async function loadMoreTraces() {
    if (traceLoadingMore || !traceHasMore) return;

    traceLoadingMore = true;
    const dropdownContent = document.getElementById('traceDropdownContent');

    // Add loading indicator at the bottom
    const loadingEl = document.createElement('div');
    loadingEl.className = 'trace-dropdown-loading-more';
    loadingEl.innerHTML = 'Loading more...';
    dropdownContent.appendChild(loadingEl);

    try {
        await fetchTraceList(traceCurrentPage + 1, true);
        // Remove loading indicator
        loadingEl.remove();
        // Append new items to dropdown
        appendTracesToDropdown();
    } catch (error) {
        loadingEl.innerHTML = 'Failed to load more';
        setTimeout(() => loadingEl.remove(), 2000);
    } finally {
        traceLoadingMore = false;
    }
}

// Append new traces to the dropdown (for infinite scroll)
function appendTracesToDropdown() {
    const dropdownContent = document.getElementById('traceDropdownContent');
    if (!dropdownContent) return;

    // Get the index to start from (traces already in DOM)
    const existingItems = dropdownContent.querySelectorAll('.trace-item').length;

    // Only add new traces
    const newTraces = traceList.slice(existingItems);

    newTraces.forEach((trace, i) => {
        const index = existingItems + i;
        const isActive = trace.id === currentTraceId;
        const timeStr = formatRelativeTime(trace.timestamp);
        const displayName = trace.flow_name || trace.name || 'Unknown Flow';
        const preview = trace.input_preview || 'No query';

        const item = document.createElement('div');
        item.className = `trace-item ${isActive ? 'active' : ''}`;
        item.setAttribute('data-trace-id', trace.id);
        item.setAttribute('data-index', index);
        item.innerHTML = `
            <div class="trace-item-header">
                <span class="trace-item-name" title="${displayName}">${displayName}</span>
                <span class="trace-item-time">${timeStr}</span>
            </div>
            <div class="trace-item-preview" title="${preview}">${preview}</div>
        `;

        item.addEventListener('click', () => {
            selectTrace(trace.id);
        });

        dropdownContent.appendChild(item);
    });
}

// Populate the trace dropdown with trace items
function populateTraceDropdown() {
    const dropdownContent = document.getElementById('traceDropdownContent');
    if (!dropdownContent) return;

    if (traceList.length === 0) {
        dropdownContent.innerHTML = '<div class="trace-dropdown-loading">No traces found</div>';
        return;
    }

    let html = '';
    traceList.forEach((trace, index) => {
        const isActive = trace.id === currentTraceId;
        const timeStr = formatRelativeTime(trace.timestamp);
        const displayName = trace.flow_name || trace.name || 'Unknown Flow';
        const preview = trace.input_preview || 'No query';

        html += `
            <div class="trace-item ${isActive ? 'active' : ''}" data-trace-id="${trace.id}" data-index="${index}">
                <div class="trace-item-header">
                    <span class="trace-item-name" title="${displayName}">${displayName}</span>
                    <span class="trace-item-time">${timeStr}</span>
                </div>
                <div class="trace-item-preview" title="${preview}">${preview}</div>
            </div>
        `;
    });

    dropdownContent.innerHTML = html;

    // Add click handlers to trace items
    dropdownContent.querySelectorAll('.trace-item').forEach(item => {
        item.addEventListener('click', () => {
            const traceId = item.getAttribute('data-trace-id');
            selectTrace(traceId);
        });
    });
}

// Toggle the trace dropdown open/closed
async function toggleTraceDropdown() {
    const dropdown = document.getElementById('traceDropdown');
    const dropdownContent = document.getElementById('traceDropdownContent');
    if (!dropdown) return;

    traceDropdownOpen = !traceDropdownOpen;

    if (traceDropdownOpen) {
        dropdown.classList.remove('hidden');

        // Reset pagination state
        traceCurrentPage = 1;
        traceHasMore = true;
        traceLoadingMore = false;
        traceList = [];

        // Show loading state
        dropdownContent.innerHTML = '<div class="trace-dropdown-loading">Loading traces...</div>';

        // Fetch fresh trace list (first page)
        try {
            await fetchTraceList(1, false);
            populateTraceDropdown();
            setupDropdownScrollListener();
        } catch (error) {
            dropdownContent.innerHTML = `<div class="trace-dropdown-error">Failed to load traces</div>`;
        }
    } else {
        dropdown.classList.add('hidden');
    }
}

// Setup scroll listener for infinite scrolling
function setupDropdownScrollListener() {
    const dropdownContent = document.getElementById('traceDropdownContent');
    if (!dropdownContent) return;

    // Remove any existing listener first (to prevent duplicates)
    dropdownContent.removeEventListener('scroll', handleDropdownScroll);
    dropdownContent.addEventListener('scroll', handleDropdownScroll);
}

// Handle scroll event for infinite loading
function handleDropdownScroll(e) {
    const el = e.target;
    // Check if scrolled near bottom (within 50px)
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 50;

    if (nearBottom && traceHasMore && !traceLoadingMore) {
        loadMoreTraces();
    }
}

// Close the trace dropdown
function closeTraceDropdown() {
    const dropdown = document.getElementById('traceDropdown');
    if (dropdown) {
        dropdown.classList.add('hidden');
        traceDropdownOpen = false;
    }
}

// Select a specific trace and load it
async function selectTrace(traceId) {
    closeTraceDropdown();

    // Update current trace ID
    currentTraceId = traceId;

    // Load the selected trace (button label stays as "Select trace")
    await loadVisualization(traceId);
}

// ========== One-time initialization ==========
// Initialize DOM references and set up event handlers once on DOMContentLoaded
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM references
    propertiesPanel = document.getElementById('propertiesPanel');
    propertiesPanelContent = document.getElementById('propertiesPanelContent');
    closePanelBtn = document.getElementById('closePanelBtn');

    // Close panel button handler (set up once)
    if (closePanelBtn) {
        closePanelBtn.addEventListener('click', () => {
            closePropertiesPanel();
        });
    }

    // Close panel when clicking outside (set up once)
    document.addEventListener('click', (e) => {
        if (propertiesPanel && propertiesPanel.classList.contains('open')) {
            // Check if click is outside panel and not on a component
            const isClickInPanel = propertiesPanel.contains(e.target);
            const isClickOnComponent = e.target.closest('.component');
            if (!isClickInPanel && !isClickOnComponent) {
                closePropertiesPanel();
            }
        }
    });

    // Trace selector setup
    const traceSelectorButton = document.getElementById('traceSelectorButton');
    const traceDropdown = document.getElementById('traceDropdown');

    if (traceSelectorButton) {
        traceSelectorButton.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleTraceDropdown();
        });
    }

    // Config dropdown setup (must be defined before API key handlers that use closeConfigDropdown)
    const configButton = document.getElementById('configButton');
    const configDropdown = document.getElementById('configDropdown');
    let configDropdownOpen = false;

    function toggleConfigDropdown() {
        configDropdownOpen = !configDropdownOpen;
        if (configDropdownOpen) {
            configDropdown.classList.remove('hidden');
            configButton.classList.add('active');
        } else {
            configDropdown.classList.add('hidden');
            configButton.classList.remove('active');
        }
    }

    function closeConfigDropdown() {
        if (configDropdownOpen) {
            configDropdownOpen = false;
            configDropdown.classList.add('hidden');
            configButton.classList.remove('active');
        }
    }

    if (configButton) {
        configButton.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleConfigDropdown();
        });
    }

    // API key input setup (both config dropdown and setup state inputs)
    const apiKeyInput = document.getElementById('langflowApiKey');
    const apiKeyToggle = document.getElementById('apiKeyToggle');
    const setupApiKeyInput = document.getElementById('setupApiKeyInput');
    const setupApiKeyToggle = document.getElementById('setupApiKeyToggle');
    const setupSubmitButton = document.getElementById('setupSubmitButton');
    const setupState = document.getElementById('setupState');
    const setupErrorMessage = document.getElementById('setupErrorMessage');

    // Helper to sync both inputs
    function syncApiKeyInputs(sourceInput) {
        const value = sourceInput.value;
        if (apiKeyInput && apiKeyInput !== sourceInput) apiKeyInput.value = value;
        if (setupApiKeyInput && setupApiKeyInput !== sourceInput) setupApiKeyInput.value = value;
        localStorage.setItem('langflowApiKey', value);
    }

    // Helper to show/hide setup error
    function showSetupError(message) {
        if (setupErrorMessage) {
            setupErrorMessage.textContent = message;
            setupErrorMessage.classList.remove('hidden');
        }
        if (setupApiKeyInput) {
            setupApiKeyInput.classList.add('error');
        }
    }

    function hideSetupError() {
        if (setupErrorMessage) {
            setupErrorMessage.classList.add('hidden');
        }
        if (setupApiKeyInput) {
            setupApiKeyInput.classList.remove('error');
        }
    }

    // Helper to handle API key submission with validation
    async function handleApiKeySubmit(key) {
        if (!key || !key.trim()) return;

        const trimmedKey = key.trim();

        // Hide any previous error
        hideSetupError();

        // Show loading state on button
        if (setupSubmitButton) {
            setupSubmitButton.disabled = true;
            setupSubmitButton.innerHTML = '<span class="spinner"></span> Validating...';
        }

        try {
            // Validate the key by making a lightweight API call
            const response = await fetch('/api/traces?limit=1', {
                headers: { 'X-Langflow-Api-Key': trimmedKey }
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || 'Invalid API key');
            }

            // Key is valid - proceed
            localStorage.setItem('langflowApiKey', trimmedKey);
            // Sync both inputs
            if (apiKeyInput) apiKeyInput.value = trimmedKey;
            if (setupApiKeyInput) setupApiKeyInput.value = trimmedKey;
            // Hide setup state if visible
            if (setupState) setupState.classList.add('hidden');
            // Close config dropdown if open
            closeConfigDropdown();
            // Load visualization
            loadVisualization();

        } catch (error) {
            // Show inline error on setup screen
            showSetupError(error.message || 'Invalid API key. Please check and try again.');
        } finally {
            // Reset button state
            if (setupSubmitButton) {
                setupSubmitButton.disabled = false;
                setupSubmitButton.textContent = 'Load Visualization';
            }
        }
    }

    // Config dropdown API key input (simpler flow - just save and load)
    if (apiKeyInput) {
        const savedKey = localStorage.getItem('langflowApiKey');
        if (savedKey) apiKeyInput.value = savedKey;
        // Auto-refresh on blur if key changed
        let lastValue = apiKeyInput.value;
        apiKeyInput.addEventListener('blur', () => {
            if (apiKeyInput.value !== lastValue && apiKeyInput.value.trim()) {
                lastValue = apiKeyInput.value;
                // For config dropdown, just save and load (errors shown via showError)
                localStorage.setItem('langflowApiKey', apiKeyInput.value.trim());
                if (setupApiKeyInput) setupApiKeyInput.value = apiKeyInput.value.trim();
                closeConfigDropdown();
                loadVisualization();
            }
        });
        apiKeyInput.addEventListener('change', () => {
            syncApiKeyInputs(apiKeyInput);
        });
    }
    if (apiKeyToggle && apiKeyInput) {
        apiKeyToggle.addEventListener('click', () => {
            apiKeyInput.type = apiKeyInput.type === 'password' ? 'text' : 'password';
        });
    }

    // Setup state API key input
    if (setupApiKeyInput) {
        const savedKey = localStorage.getItem('langflowApiKey');
        if (savedKey) setupApiKeyInput.value = savedKey;
        // Clear error when user starts typing
        setupApiKeyInput.addEventListener('input', () => {
            hideSetupError();
        });
        // Submit on Enter key
        setupApiKeyInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                handleApiKeySubmit(setupApiKeyInput.value);
            }
        });
    }
    if (setupApiKeyToggle && setupApiKeyInput) {
        setupApiKeyToggle.addEventListener('click', () => {
            setupApiKeyInput.type = setupApiKeyInput.type === 'password' ? 'text' : 'password';
        });
    }
    if (setupSubmitButton && setupApiKeyInput) {
        setupSubmitButton.addEventListener('click', () => {
            handleApiKeySubmit(setupApiKeyInput.value);
        });
    }

    // Close dropdowns when clicking outside
    document.addEventListener('click', (e) => {
        if (traceDropdownOpen) {
            const isClickInDropdown = traceDropdown && traceDropdown.contains(e.target);
            const isClickOnButton = traceSelectorButton && traceSelectorButton.contains(e.target);
            if (!isClickInDropdown && !isClickOnButton) {
                closeTraceDropdown();
            }
        }
        if (configDropdownOpen) {
            const isClickInConfig = configDropdown && configDropdown.contains(e.target);
            const isClickOnConfigButton = configButton && configButton.contains(e.target);
            if (!isClickInConfig && !isClickOnConfigButton) {
                closeConfigDropdown();
            }
        }
    });
});

// Auto-load visualization on page load (or show setup state if no API key)
window.addEventListener('load', async function() {
    const setupState = document.getElementById('setupState');
    const loadingState = document.getElementById('loadingState');
    const header = document.querySelector('.header');

    // Check for server-configured API key
    let savedKey = localStorage.getItem('langflowApiKey');
    if (!savedKey || !savedKey.trim()) {
        try {
            const resp = await fetch('/api/config');
            if (resp.ok) {
                const cfg = await resp.json();
                if (cfg.langflow_api_key) {
                    localStorage.setItem('langflowApiKey', cfg.langflow_api_key);
                    savedKey = cfg.langflow_api_key;
                    // Populate input fields so API calls pick up the key
                    document.querySelectorAll('#langflowApiKey, #setupApiKey').forEach(el => {
                        el.value = cfg.langflow_api_key;
                    });
                }
            }
        } catch (e) {
            // Ignore - fall through to manual entry
        }
    }

    if (savedKey && savedKey.trim()) {
        // Has API key, load visualization
        loadVisualization();
    } else {
        // No API key, show setup state
        if (loadingState) loadingState.classList.add('hidden');
        if (setupState) setupState.classList.remove('hidden');
        // Hide header controls in setup mode
        if (header) header.classList.add('setup-mode');
    }
});

// Log display configuration
console.log('Display configuration:', config);
