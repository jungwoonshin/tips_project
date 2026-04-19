# Plan — TIPS AutoResearch Benchmark

## Context

TIPS v4 currently flips ~25% of GAIA failed trajectories (7/28) via a hand-tuned Sonnet-fixer + Gemma-replay pipeline. Karpathy's AutoResearch pattern — give an LLM autonomous access to its own experiment, let it iterate with a tight read/edit/evaluate loop — is a natural next step to push beyond human-tuned performance. The target is to measure how much flip-rate headroom an autonomous research agent can extract from prompt + hyperparameter search alone, keeping the replay harness and oracle-blindness invariants untouched. This answers two questions at once: (1) how much of TIPS performance is hand-tuning vs. fundamental ceiling; (2) does Karpathy's "local-search-trap / ratchet-conservatism / RLHF-cagy" failure taxonomy actually show up empirically when we put an agent on TIPS itself?

The single non-negotiable invariant: **the fix prompt (Stage 2b) and the meta-research-agent must both be oracle-blind**. Stage 2b is already enforced by `assert_oracle_absent`. The meta-agent must additionally never see `oracle_answer` or `agent_original_final_answer` strings in any artifact it reads, because agent_original often contains a near-match of the oracle (e.g. `0017`: agent said `"key, the value is 26.4"`, oracle `"26.4"`).

## Goal (objective function)

```
score(pipeline) = flip_rate(pipeline, DEV_SET)
                = #{t ∈ DEV | process_one(t).outcome == "published"} / |DEV|
```

**Baseline to beat**: v4 flip rate on DEV = measured once, frozen. Expected ≈ 0.4 given stratified picks below.

**Secondary metrics** (reported, not optimized): average `|M|`, wall-clock per trajectory, total Sonnet API tokens.

## Dev / Test split (stratified from the existing 28)

Dev (5, visible to agent):
| Trajectory | Class |
| --- | --- |
| `gaia_validation_0006` | Tool-swap-flippable (canary) |
| `gaia_validation_0002` | Terminal-reasoning |
| `gaia_validation_0011` | Info-not-reachable |
| `gaia_validation_0016` | Prior-lock-in |
| `gaia_validation_0028` | Borderline (flipped in v3, not v4) |

Test (23, held out, evaluated only at end + every 20 iters as overfit watchdog): remaining `gaia_validation_*` IDs.

Rationale: agent can't win by specializing to one failure class; overfit to 5 trajectories is inevitable without the test watchdog.

## Editable surface (allowlist)

**Allowed for the meta-agent to edit**:
- `tips_v3/llm/prompts/detect_2a.py`
- `tips_v3/llm/prompts/fix_2b.py`
- `tips_v3/llm/prompts/expand.py`
- `tips_v3/llm/prompts/null_fix.py`
- `tips_v3/config.py` — only the following constants:
  `DETECTION_TEMPERATURE`, `FIX_TEMPERATURE`, `FIX_SAMPLES_PER_NODE`, `DETECTION_K_SAMPLES`,
  `TAU_DROP`, `UNCERTAIN_LOWER`, `UNCERTAIN_UPPER`, `N_MAX_CANDIDATES`, `M_CAP`,
  `ADAPTIVE_K_INITIAL`, `ADAPTIVE_K_TIEBREAK`, `REDUCTION_K`,
  `GREEDY_STEP`, `REDUCTION_STEP`, `TYPE_PRIORITY`, `TYPE_ALLOWED_FIELDS`, `FINAL_FLIP_THRESHOLD`

**Forbidden** (structural integrity):
- All of `tips_v3/replay/`, `tips_v3/eval.py`, `tips_v3/stages/*`, `tips_v3/run.py`
- `SEEDS_*`, `SONNET_MODEL_ID`, `GEMMA_MODEL_ID`, `FINAL_VERIFY_K`
- `assert_oracle_absent` in `fix_2b.py`
- `owl_counterfactual_rerun.py`

Enforcement: the patch applier validates that every diff touches only allowlisted paths + only whitelisted config symbols. Anything else → reject the patch, record "invalid_patch" in history.

## Oracle-blindness protocol for the meta-agent

The agent's context per iteration includes:
1. **Current pipeline state** (allowlist files, read-only copies)
2. **Iteration history** (last 10 proposals, their scores, reverted flag, agent-supplied hypothesis)
3. **Redacted dev-set artifacts**: for each dev trajectory, a redacted JSON derived from `output/owl_counterfactual_v4/answers/<tid>.json`:
   - Keep: `trajectory_id`, `published_sufficient_set`, `fixes` (node_id, type, field, content), `per-seed replay `final_answer` strings, per-seed `flipped` bool, `outcome`
   - **Strip**: `ground_truth`, `agent_original_final_answer`, any `oracle_*` field. Replace with constant sentinels (`"<REDACTED>"`).
   - Justification: flipped/unflipped + replay strings give enough signal to reason about failure modes without disclosing the target answer.

The `autoresearch/redact.py` module builds these artifacts at the start of each iteration. A single unit test asserts no oracle-answer substring leaks into the agent's prompt.

## Architecture (files under `autoresearch/`)

`autoresearch/` already exists (empty). Proposed contents:

| File | Responsibility |
| --- | --- |
| `__init__.py` | empty |
| `spec.md` | This plan, copied and linked from TIPS docs |
| `allowlist.py` | File paths + config symbols the agent may edit; validator function |
| `redact.py` | Strip oracle from answers/*.json → agent-safe artifacts |
| `patch.py` | Apply/revert unified-diff patches; rejects non-allowlisted edits |
| `evaluator.py` | Call `tips_v3.run.process_one` on dev or test set, return flip_rate |
| `agent.py` | Propose-patch call to Sonnet 4.5 via existing `SonnetClient` |
| `orchestrator.py` | Main iteration loop with ratchet + watchdog |
| `history.py` | Append-only JSONL log of (iteration, hypothesis, patch, score, kept/reverted) |
| `cli.py` | Entry: `python -m autoresearch.cli --budget 100 --fast` |
| `runs/<run_id>/` | Per-iteration snapshots: patch.diff, score.json, agent_transcript.txt |

### Iteration loop (orchestrator.py, pseudocode)

```python
def run(budget=100, ratchet_slack=0.02, watchdog_every=20, overfit_threshold=0.15):
    baseline = evaluator.evaluate(DEV_SET, k_seeds=3)
    history.append({"iter": 0, "score": baseline, "kept": True})

    for i in range(1, budget + 1):
        agent_ctx = {
            "allowlist_files": read_allowlist(),
            "recent_history": history.tail(10),
            "dev_artifacts": redact.build_dev_snapshot(DEV_SET),
        }
        hypothesis, patch = agent.propose(agent_ctx)

        if not patch.is_valid_against(ALLOWLIST):
            history.append({"iter": i, "score": None, "kept": False,
                             "reason": "invalid_patch"})
            continue

        patch.apply()
        score = evaluator.evaluate(DEV_SET, k_seeds=3)
        kept = (score >= baseline - ratchet_slack)
        if kept:
            baseline = max(baseline, score)
        else:
            patch.revert()
        history.append({"iter": i, "score": score, "kept": kept,
                         "hypothesis": hypothesis})

        # Overfit watchdog
        if i % watchdog_every == 0:
            test_score = evaluator.evaluate(TEST_SET, k_seeds=3)
            if (baseline - test_score) > overfit_threshold:
                log.warning("OVERFIT detected; halting at iter %d", i)
                break

    final_test = evaluator.evaluate(TEST_SET, k_seeds=5)  # strict
    return {"dev": baseline, "test": final_test}
```

### Evaluator oracle-blindness guarantee

`evaluator.evaluate(trajectories, k_seeds)` internally calls `tips_v3.run.process_one` (reusing existing function at `tips_v3/run.py:160`), which calls `BoundedReplay.run` — the *only* place `traj.oracle_answer` is consulted (via `gaia_scorer`). The evaluator returns just `{flip_rate: float, per_tid_flipped: dict[str,bool]}`. No oracle strings cross from `process_one`'s return into history or agent context.

### Runtime optimizations (fast mode)

Target: ≤5 minutes per iteration on DEV_SET.

1. **Parallelize dev trajectories**: wrap `process_one` calls in a ThreadPoolExecutor (5 workers = 5 trajectories in parallel). Replay cache (sqlite) is already thread-safe via connection per thread.
2. **Stage 2a caching across iterations**: if the detection prompt hash is unchanged between iterations, reuse the previous run's stage2a.json (content-addressed under `autoresearch/cache/stage2a/<hash>/<tid>.json`). Only re-invoke Sonnet when the prompt changed.
3. **k=3 during iteration, k=5 only at final test.** Saves ~40% replay time per iteration.
4. **Per-iteration replay cache**: reuse `output/owl_counterfactual_v4/_replay_cache.sqlite` across iterations unless fix content changes the cache key. Backwards-elimination shares rows cleanly.

Expected iteration wall-clock: 3–6 min depending on whether a prompt changed invalidates the stage2a cache.

## Anti-patterns (Karpathy failure modes) and mitigations

| Failure mode | Mitigation |
| --- | --- |
| Local-search trap (agent tweaks single words forever) | Every 20 iters, orchestrator injects a "paradigm-shift" meta-prompt forcing the agent to propose a structurally different change (e.g. swap TYPE_PRIORITY completely, rewrite fix_2b from scratch). |
| Ratchet over-conservatism | 2% slack on regressions + optional paired-hypothesis mode: agent can request "accept this regression; next iter will pay it back". |
| Overfit to 5-trajectory DEV | Test-set watchdog every 20 iters; halt if `dev - test > 0.15`. |
| Reward hacking via oracle leak | Redaction + `gaia_scorer` is non-editable. No pathway for the agent to cheat. |
| Agent invents non-existent tools / file paths | Patch validator: every file path in a proposed patch must be in ALLOWLIST. |

## Deliverables

1. `autoresearch/` module with files listed above
2. `autoresearch/runs/<run_id>/history.jsonl` — full iteration trace (publishable as supplementary material)
3. `autoresearch/runs/<run_id>/report.md` — auto-generated summary: baseline vs final dev/test flip rates, what prompts/configs changed, hypotheses accepted, Karpathy-failure-mode incidents
4. A single-page table: Baseline (human-tuned v4) flip rate | After N iters flip rate | Top 3 agent-discovered changes

## Critical files to read before implementation (reference, not edit)

- `tips_v3/run.py:160` — `process_one(traj, client, replay, summary)` signature
- `tips_v3/io/ingest.py` — `iter_trajectories()` and `load_trajectory(tid)`
- `tips_v3/replay/bounded_replay.py` — `BoundedReplay.run` and the gaia_scorer call site
- `tips_v3/llm/sonnet_client.py` — reuse `SonnetClient` for the research-agent call
- `tips_v3/config.py` — full list of symbols for allowlist validation
- `tips_v3/__init__.py` — `PROMPT_VERSION` string; iterations should bump a suffix (v3.5.0-ar001 etc.)
- `output/owl_counterfactual_v4/answers/*.json` — dev-set source; redact.py consumes these

## Verification

After implementing:
1. **Unit tests**:
   - `test_redact.py`: assert oracle_answer and agent_original_final_answer never appear in agent-facing artifacts (regex check on every DEV trajectory's snapshot).
   - `test_allowlist.py`: feed a patch that touches `tips_v3/replay/bounded_replay.py`; assert rejection. Feed a patch that modifies `SEEDS_GREEDY_INITIAL`; assert rejection.
   - `test_evaluator.py`: call `evaluator.evaluate` on a 1-trajectory dev set with a known-flipping fix; assert flip_rate = 1.0.

2. **Smoke run**: `python -m autoresearch.cli --budget 3 --dev-only`. Verify 3 agent proposals, at least one reverted, history.jsonl written, no oracle substrings in run transcripts.

3. **End-to-end dry run (10 iterations)**: measure dev-flip trajectory; confirm:
   - Iteration wall-clock ≤ 8 min
   - At least 50% of proposals produce a valid patch
   - Overfit watchdog fires as expected if we artificially force an overfitting patch

4. **Full run** (budget=100): compare final test flip rate vs v4 baseline. Expected outcome: +5 to +15 percentage points if the loop works as designed.

## Out of scope (explicitly)

- Scaling to full GAIA (450 problems) — separate effort; this plan stays on the 28 already-processed
- Swapping the agent model for the research agent — use Sonnet 4.5 throughout for apples-to-apples
- Modifying the replay harness or scorer — these are frozen to preserve label integrity
- Fine-tuning Gemma or Sonnet — purely prompt + config search
