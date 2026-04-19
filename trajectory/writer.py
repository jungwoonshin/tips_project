"""TrajectoryWriter: accumulates steps during a task run and writes to JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluator import evaluate_answer
from .schema import (
    AgenticSystem,
    AgentInfo,
    Annotation,
    FailureLogEntry,
    ToolCall,
    ToolResult,
    TrajectoryOutput,
)


# Map agent names to their available tools
AGENT_TOOLS: dict[str, list[str]] = {}


class TrajectoryWriter:
    """Accumulates trajectory steps for a single task and writes the result."""

    def __init__(
        self,
        task: dict,
        framework: str,
        model: str,
        output_dir: str,
        instance_id: str = "",
        problem_number: int = 0,
        agents: list[dict] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.task = task
        self.task_id = task["task_id"]
        self.instance_id = instance_id or task["task_id"]
        self.problem_number = problem_number
        self.query = task["Question"]
        self.ground_truth = task.get("Final answer", "")
        self.framework = framework
        self.model = model

        agent_infos = [AgentInfo(**a) for a in (agents or [])]
        self.agentic_system = AgenticSystem(
            system_type="algorithm_generated",
            generator=framework,
            base_model=model,
            agents=agent_infos,
        )

        # Build agent name -> tools lookup
        self._agent_tools: dict[str, list[str]] = {}
        for a in (agents or []):
            self._agent_tools[a["name"]] = a.get("tools", [])

        self.steps: list[FailureLogEntry] = []
        self._step_counter = 0
        self._live_log_path: Path | None = None

    def add_step(
        self,
        agent: str = "",
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> FailureLogEntry:
        parsed_calls = []
        if tool_calls:
            for tc in tool_calls:
                parsed_calls.append(ToolCall(
                    tool=tc.get("tool", ""),
                    arguments=tc.get("arguments", {}),
                ))

        parsed_results = []
        if tool_results:
            for tr in tool_results:
                parsed_results.append(ToolResult(
                    tool=tr.get("tool", ""),
                    output=tr.get("output", ""),
                    is_error=tr.get("is_error", False),
                ))

        # Look up available tools for this agent
        available_tools = self._agent_tools.get(agent, [])

        entry = FailureLogEntry(
            step=self._step_counter,
            agent=agent,
            content=content,
            available_tools=available_tools,
            tool_calls=parsed_calls,
            tool_results=parsed_results,
        )
        self.steps.append(entry)
        self._step_counter += 1

        # Write intermediate snapshot if enabled
        if self._live_log_path is not None:
            self._write_live_log()

        return entry

    def enable_live_log(self, path: Path | str | None = None) -> None:
        """Enable writing intermediate trajectory JSON after every step.

        If *path* is ``None``, the live log is written to the same location
        as the final output (``output_dir/problem_NN.json``).
        """
        if path is None:
            self._live_log_path = self.output_dir / f"problem_{self.problem_number:02d}.json"
        else:
            self._live_log_path = Path(path)
        self._live_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_live_log(self) -> None:
        """Write current trajectory state to the live log file."""
        output = TrajectoryOutput(
            problem_number=self.problem_number,
            instance_id=self.instance_id,
            source_benchmark="GAIA",
            query=self.query,
            ground_truth=self.ground_truth,
            agentic_system=self.agentic_system,
            failure_log=self.steps,
            final_answer=None,
            task_outcome="in_progress",
            annotation=Annotation(),
        )
        self._live_log_path.write_text(
            output.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def finalize(
        self,
        final_answer: str | None = None,
        error: str | None = None,
    ) -> Path:
        """Evaluate, build output, and write JSON."""
        is_correct = evaluate_answer(final_answer, self.ground_truth)
        task_outcome = "success" if is_correct else "failure"

        if error:
            self.add_step(
                agent="system",
                content=f"Error: {error}",
            )

        output = TrajectoryOutput(
            problem_number=self.problem_number,
            instance_id=self.instance_id,
            source_benchmark="GAIA",
            query=self.query,
            ground_truth=self.ground_truth,
            agentic_system=self.agentic_system,
            failure_log=self.steps,
            final_answer=final_answer,
            task_outcome=task_outcome,
            annotation=Annotation(),
        )

        out_path = self.output_dir / f"problem_{self.problem_number:02d}.json"
        tmp_path = out_path.with_suffix(".tmp")
        tmp_path.write_text(
            output.model_dump_json(indent=2),
            encoding="utf-8",
        )
        tmp_path.rename(out_path)
        return out_path
