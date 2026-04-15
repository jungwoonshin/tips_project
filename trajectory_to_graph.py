"""Generate agent dependency graphs (.graph DOT files + PNG) from trajectory JSONs.

Consecutive same-agent steps are merged into single nodes.
Dependency detection determines whether agent tasks are chained (dependent on
prior agent output) or branched (independent siblings under the same
orchestrator plan node).

Uses LLM-based detection when API is available, falls back to token-overlap
heuristic otherwise.
"""
import json
import re
import sys
import httpx
from pathlib import Path
from itertools import groupby

import graphviz

AGENT_NAME_MAP = {
    "MagenticOneOrchestrator": "Orchestrator",
    "user": "human",
}

LLM_API_URL = "http://100.124.141.69:8000/v1/chat/completions"
LLM_MODEL = "QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ"


def map_agent_name(raw: str) -> str:
    return AGENT_NAME_MAP.get(raw, raw)


# ---------------------------------------------------------------------------
# Step 1 & 2: Parse and merge
# ---------------------------------------------------------------------------

def merge_steps(steps: list[dict]) -> list[dict]:
    """Merge consecutive same-agent steps into grouped nodes."""
    merged = []
    for agent, group in groupby(steps, key=lambda s: s["agent"]):
        group_list = list(group)
        merged.append({
            "agent": agent,
            "mapped": map_agent_name(agent),
            "steps": [s["step"] for s in group_list],
            "contents": [s["content"] for s in group_list],
        })
    return merged


def assign_node_ids(merged: list[dict]) -> list[dict]:
    """Assign {AgentType}_{N} identifiers to each merged node."""
    counters: dict[str, int] = {}
    nodes = []
    for m in merged:
        name = m["mapped"]
        counters[name] = counters.get(name, 0) + 1
        node_id = f"{name}_{counters[name]}"
        nodes.append({
            "id": node_id,
            "agent": m["agent"],
            "mapped": name,
            "steps": m["steps"],
            "contents": m["contents"],
        })
    return nodes


# ---------------------------------------------------------------------------
# Step 3: Dependency detection
# ---------------------------------------------------------------------------

def _extract_new_tokens(agent_content: str, user_query: str) -> set[str]:
    """Extract tokens from agent output that are NOT in the original user query.

    Focuses on numbers, URLs, and multi-char words that represent new
    information the agent discovered.
    """
    # Extract numbers (integers and decimals)
    agent_numbers = set(re.findall(r'\b\d[\d,]*\.?\d*\b', agent_content))
    query_numbers = set(re.findall(r'\b\d[\d,]*\.?\d*\b', user_query))
    new_numbers = agent_numbers - query_numbers

    # Extract URLs
    new_urls = set(re.findall(r'https?://[^\s\)\]>"\']+', agent_content))
    query_urls = set(re.findall(r'https?://[^\s\)\]>"\']+', user_query))
    new_urls = new_urls - query_urls

    # Extract capitalized words/phrases (proper nouns, names) that are new
    agent_caps = set(re.findall(r'\b[A-Z][a-z]{2,}\b', agent_content))
    query_caps = set(re.findall(r'\b[A-Z][a-z]{2,}\b', user_query))
    new_caps = agent_caps - query_caps

    new_tokens = set()
    for n in new_numbers:
        # Only keep numbers with 3+ digits (significant data, not step indices)
        cleaned = n.replace(",", "")
        if len(cleaned.replace(".", "")) >= 3:
            new_tokens.add(n)
    new_tokens |= new_urls
    new_tokens |= new_caps

    return new_tokens


def detect_dependency_heuristic(
    user_query: str,
    orch_content: str,
    prev_agent_content: str,
) -> bool:
    """Heuristic: check if orchestrator delegation references new info from prev agent.

    Returns True if dependent (should chain), False if independent (can branch).
    """
    new_tokens = _extract_new_tokens(prev_agent_content, user_query)
    if not new_tokens:
        return False  # No new info found → independent

    orch_lower = orch_content.lower()
    for token in new_tokens:
        if token.lower() in orch_lower:
            return True

    return False


def detect_dependency_llm(
    user_query: str,
    orch_content: str,
    prev_agent_name: str,
    prev_agent_content: str,
) -> bool | None:
    """Use LLM to determine if orchestrator delegation depends on previous agent output.

    Returns True (dependent/chain), False (independent/branch), or None on failure.
    """
    prompt = f"""You are analyzing an orchestrator's delegation message in a multi-agent system.

The original user question was:
"{user_query}"

The previous agent ({prev_agent_name}) produced this output:
"{prev_agent_content[:1500]}"

The orchestrator then wrote this delegation message:
"{orch_content[:1500]}"

Does this orchestrator delegation USE or REFERENCE specific information, data, or results that were obtained by the previous agent ({prev_agent_name})?

Answer ONLY "DEPENDENT" if the delegation references specific new information from the previous agent's output (data, numbers, URLs found, content discovered, etc.).
Answer ONLY "INDEPENDENT" if the delegation could have been written based solely on the original user question and general knowledge, without needing the previous agent's output.

Answer:"""

    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                LLM_API_URL,
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 32,
                    "temperature": 0,
                },
            )
            answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
            return "DEPENDENT" in answer
    except Exception:
        return None  # Signal caller to use fallback


def detect_dependency(
    user_query: str,
    orch_content: str,
    prev_agent_name: str,
    prev_agent_content: str,
    use_llm: bool = True,
) -> bool:
    """Detect whether orchestrator delegation depends on previous agent output.

    Tries LLM first (if use_llm=True), falls back to heuristic.
    Returns True if dependent (chain), False if independent (branch).
    """
    if use_llm:
        result = detect_dependency_llm(
            user_query, orch_content, prev_agent_name, prev_agent_content,
        )
        if result is not None:
            return result
        # LLM failed, fall through to heuristic

    return detect_dependency_heuristic(user_query, orch_content, prev_agent_content)


# ---------------------------------------------------------------------------
# Step 4: Build edges
# ---------------------------------------------------------------------------

def build_edges(nodes: list[dict], user_query: str, use_llm: bool = True) -> list[tuple[str, str]]:
    """Build dependency edges between nodes.

    For each worker agent node, check whether its orchestrator delegation
    depends on the previous worker agent's output. If independent, the
    worker branches from the orchestrator plan node instead of chaining
    through the previous worker.
    """
    edges: list[tuple[str, str]] = []

    for i in range(1, len(nodes)):
        curr = nodes[i]
        prev = nodes[i - 1]

        # Skip: always chain orchestrator/human nodes to their predecessor
        if curr["mapped"] in ("Orchestrator", "human"):
            edges.append((prev["id"], curr["id"]))
            continue

        # curr is a worker agent, prev should be an orchestrator
        if prev["mapped"] != "Orchestrator":
            edges.append((prev["id"], curr["id"]))
            continue

        # Find the previous worker agent (before this orchestrator)
        prev_worker_idx = None
        for j in range(i - 2, -1, -1):
            if nodes[j]["mapped"] not in ("Orchestrator", "human"):
                prev_worker_idx = j
                break

        if prev_worker_idx is None:
            # No previous worker → just chain from orchestrator
            edges.append((prev["id"], curr["id"]))
            continue

        prev_worker = nodes[prev_worker_idx]
        orch_content = "\n".join(prev["contents"])
        prev_agent_content = "\n".join(prev_worker["contents"])[:2000]

        is_dep = detect_dependency(
            user_query=user_query,
            orch_content=orch_content,
            prev_agent_name=prev_worker["mapped"],
            prev_agent_content=prev_agent_content,
            use_llm=use_llm,
        )

        if is_dep:
            # Dependent chain: prev_worker → orch → curr
            edges.append((prev["id"], curr["id"]))
            print(f"  {prev_worker['id']} -> {prev['id']} -> {curr['id']}  [DEPENDENT]")
        else:
            # Independent branch: find the common orchestrator plan parent
            # (the orchestrator node right before the previous worker)
            plan_orch_idx = None
            for j in range(prev_worker_idx - 1, -1, -1):
                if nodes[j]["mapped"] == "Orchestrator":
                    plan_orch_idx = j
                    break

            if plan_orch_idx is not None:
                edges.append((nodes[plan_orch_idx]["id"], curr["id"]))
                print(f"  {nodes[plan_orch_idx]['id']} -> {curr['id']}  [INDEPENDENT branch]")
            else:
                # Fallback: chain from immediate predecessor
                edges.append((prev["id"], curr["id"]))

    return edges


# ---------------------------------------------------------------------------
# Step 5: Generate output
# ---------------------------------------------------------------------------

def trajectory_to_graph(json_path: str, use_llm: bool = True) -> None:
    path = Path(json_path)
    print(f"\nProcessing: {path.name}")

    with open(path) as f:
        data = json.load(f)

    user_query = data.get("query", "")
    steps = data.get("failure_log", [])

    # Filter out system steps
    steps = [s for s in steps if s["agent"] != "system"]

    # Merge and assign IDs
    merged = merge_steps(steps)
    nodes = assign_node_ids(merged)

    # Build edges with dependency detection
    edges = build_edges(nodes, user_query, use_llm=use_llm)

    # Deduplicate edges
    seen = set()
    unique_edges = []
    for src, dst in edges:
        if (src, dst) not in seen:
            unique_edges.append((src, dst))
            seen.add((src, dst))

    # Build DOT string
    lines = [
        "digraph agent_dag {",
        "    rankdir=LR;",
        "    node [shape=box, style=rounded];",
        "",
    ]
    for n in nodes:
        step_str = ",".join(str(s) for s in n["steps"])
        lines.append(f'    "{n["id"]}" [label="{n["id"]}" steps="{step_str}"];')
    lines.append("")
    for src, dst in unique_edges:
        lines.append(f'    "{src}" -> "{dst}";')
    lines.append("}")

    dot_content = "\n".join(lines) + "\n"

    # Write .graph file
    graph_path = path.with_suffix(".graph")
    graph_path.write_text(dot_content)
    print(f"  Written: {graph_path}")

    # Render PNG
    g = graphviz.Source(dot_content)
    png_path = str(path.with_suffix(""))
    g.render(png_path, format="png", cleanup=True)
    print(f"  Written: {png_path}.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trajectory_to_graph.py [--no-llm] <trajectory.json> [...]")
        sys.exit(1)

    use_llm = True
    paths = []
    for arg in sys.argv[1:]:
        if arg == "--no-llm":
            use_llm = False
        else:
            paths.append(arg)

    for p in paths:
        trajectory_to_graph(p, use_llm=use_llm)

