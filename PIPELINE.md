# TIPS: Task Improvement via Plan Steering

## Goal

Build a labeled error-step dataset from failed multi-agent (Magentic-One) trajectories on the GAIA benchmark. Validate labels by rerunning agents with fixes applied — if a wrong answer flips to correct, the identified error is meaningful.

---

## Pipeline Overview

```
Trajectory Generation ──► Error Identification ──► Critical Node Selection ──► Fix Generation ──► Rerun & Validate
     (Phase 0)                 (Phase 1)               (Phase 2)                (Phase 3)         (Phase 4 & 5)
```

### Phase 0: Trajectory Generation

Run Magentic-One agent team (WebSurfer, FileSurfer, Coder, Executor) on GAIA tasks. Each task produces a trajectory JSON with step-by-step agent actions, tool calls, and a final answer scored against ground truth.

- **Script**: `scripts/run_benchmark.py --framework magentic-one --split validation --max-tasks N`
- **Agent setup**: `runners/magentic_one_runner.py` — 4 agents + orchestrator, shared LLM client, max 100 turns
- **Output**: `results/magentic_one/validation/{task_id}.json`

### Phase 1: Error Identification (LLM)

An LLM reviews each failed trajectory (knowing the answer was wrong) and identifies ALL error steps with type and explanation.

- **Error types**: REASONING, INFORMATION, SEARCH, PLANNING, TOOL, PREMATURE_TERMINATION
- **Output**: `error_analysis/step1_all_errors.json`

### Phase 2: Critical Node Selection

Find the root-cause errors — the ones worth fixing.

| Approach | Algorithm | Script |
|----------|-----------|--------|
| **Non-graph** | Earliest error step (linear scan) | `identify_and_fix.py` |
| **Graph** | Error nodes with no error-node ancestors in the dependency DAG | `graph_identify_and_fix.py` |

Graph edges are built by `build_graphs.py` (LLM identifies per-step parent dependencies). Stored in `graphs/problem_XX.json`.

- **Output**: `{graph_}error_analysis/step2_critical_nodes.json`

### Phase 3: Fix Generation (LLM Replay)

For each critical node, the LLM re-generates the agent's response with knowledge of what went wrong. Prior steps include any earlier fixes already applied.

- **Output**: `{graph_}error_analysis/step3_fixes.json`

### Phase 4: Rerun with Fixed History

Inject the corrected steps into the conversation history up to the last critical node, then resume the Magentic-One team from that point. The agents execute fresh from the corrected state.

- **Timeout**: 300s per task, max 100 turns
- **Output**: `{graph_}error_analysis/step4_reruns.json`, live logs in `logs/{non_graph,graph}_reruns/`

### Phase 5: Scoring & Validation

Compare original and rerun answers against ground truth. A "flipped_to_correct" result validates that the identified error steps were genuine root causes.

- **Evaluator**: Normalized exact match (lowercase, strip punctuation/articles, numeric tolerance)
- **Output**: `{graph_}error_analysis/step5_results.json`

---

## Directory Structure

```
tips_project/
├── scripts/run_benchmark.py       # Phase 0: run agents on GAIA
├── identify_and_fix.py            # Phases 1-5 (non-graph, linear critical nodes)
├── graph_identify_and_fix.py      # Phases 1-5 (graph-based critical nodes)
├── build_graphs.py                # Build dependency DAGs from trajectories
├── runners/
│   ├── base_runner.py             # Abstract runner with task loading & scoring
│   └── magentic_one_runner.py     # Magentic-One agent team setup & event capture
├── trajectory/
│   ├── writer.py                  # TrajectoryWriter: step accumulation & live logging
│   ├── evaluator.py               # GAIA-style answer evaluation
│   └── schema.py                  # Pydantic models for trajectory data
├── dataset/gaia/                  # GAIA benchmark data (validation.jsonl + files/)
├── results/magentic_one/          # Raw trajectory outputs (Phase 0)
├── graphs/                        # Dependency DAGs (Phase 2 input)
├── error_analysis/                # Non-graph pipeline outputs (Phases 1-5)
├── graph_error_analysis/          # Graph pipeline outputs (Phases 1-5)
└── logs/                          # Live rerun logs
```

---

## Key Design Decision

The rerun acts as a **label validation mechanism**, not a production correction system. Oracle access (knowing the answer is wrong) is acceptable during labeling. The resulting dataset — trajectories annotated with validated error steps — can later train a runtime error detector that operates without oracle knowledge.
