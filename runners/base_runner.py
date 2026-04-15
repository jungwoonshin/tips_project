"""Abstract base runner for GAIA benchmark."""
from __future__ import annotations

import asyncio
import json
import traceback
from abc import ABC, abstractmethod
from pathlib import Path

from trajectory.writer import TrajectoryWriter


class BaseRunner(ABC):
    FRAMEWORK: str = ""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        model: str = "gpt-4o",
        gaia_files_dir: str | None = None,
        timeout: int = 600,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.model = model
        self.gaia_files_dir = Path(gaia_files_dir) if gaia_files_dir else self.data_dir / "files"
        self.timeout = timeout

    def load_tasks(
        self, split: str = "validation", levels: list[int] | None = None
    ) -> list[dict]:
        path = self.data_dir / f"{split}.jsonl"
        tasks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                task = json.loads(line.strip())
                lvl = int(task["Level"])
                if levels and lvl not in levels:
                    continue
                tasks.append(task)
        return tasks

    def _is_completed(self, task_id: str) -> bool:
        return (self.output_dir / f"{task_id}.json").exists()

    def _build_prompt(self, task: dict) -> str:
        question = task["Question"]
        file_name = task.get("file_name", "")
        if file_name:
            file_path = self.gaia_files_dir / task.get("file_path", file_name)
            question = (
                f"The following file is provided for this task: {file_path}\n\n"
                f"{question}"
            )
        return question

    def _get_agent_definitions(self) -> list[dict]:
        """Override in subclass to provide agent definitions."""
        return []

    @abstractmethod
    async def run_single_task(self, task: dict, writer: TrajectoryWriter) -> str | None:
        """Run one GAIA task. Return the final answer string."""
        ...

    async def run_all(
        self,
        split: str = "validation",
        levels: list[int] | None = None,
        max_tasks: int | None = None,
        task_ids: list[str] | None = None,
    ) -> dict:
        tasks = self.load_tasks(split, levels)
        if task_ids:
            tasks = [t for t in tasks if t["task_id"] in task_ids]
        if max_tasks:
            tasks = tasks[:max_tasks]

        completed = 0
        skipped = 0
        errors = 0
        correct = 0
        total = len(tasks)

        def _bar(i: int, status: str = "") -> str:
            done = completed + skipped + errors
            pct = done * 100 // total if total else 0
            filled = done * 20 // total if total else 0
            bar = "█" * filled + "░" * (20 - filled)
            return f"[{bar}] {done}/{total} ({pct}%) | ✓{correct} ✗{errors} ⏭{skipped} {status}"

        for i, task in enumerate(tasks):
            tid = task["task_id"]
            short_id = tid[:8]

            if self._is_completed(tid):
                skipped += 1
                print(f"  ⏭ [{i+1}/{total}] {short_id} SKIP (already done)")
                print(_bar(i))
                continue

            q_preview = task["Question"][:60].replace("\n", " ")
            print(f"\n  ▶ [{i+1}/{total}] {short_id}… (L{task['Level']}) {q_preview}…")

            instance_id = f"gaia_{split}_{i:04d}"
            writer = TrajectoryWriter(
                task=task,
                framework=self.FRAMEWORK,
                model=self.model,
                output_dir=str(self.output_dir),
                instance_id=instance_id,
                problem_number=i,
                agents=self._get_agent_definitions(),
            )

            try:
                answer = await asyncio.wait_for(
                    self.run_single_task(task, writer),
                    timeout=self.timeout,
                )
                out = writer.finalize(final_answer=answer)
                from trajectory.evaluator import evaluate_answer
                is_correct = evaluate_answer(answer, task.get("Final answer", ""))
                if is_correct:
                    correct += 1
                    print(f"  ✓ CORRECT | '{answer}' == '{task.get('Final answer', '?')}'")
                else:
                    print(f"  ✗ WRONG   | '{answer}' != '{task.get('Final answer', '?')}'")
                completed += 1
            except asyncio.TimeoutError:
                writer.finalize(error="Timeout")
                print(f"  ⏱ TIMEOUT after {self.timeout}s")
                errors += 1
            except Exception as e:
                writer.finalize(error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
                print(f"  ✗ ERROR: {e}")
                errors += 1

            print(_bar(i))

        summary = self.generate_summary(split)
        print(f"\n{'='*50}")
        print(f"  DONE  {completed} completed, {skipped} skipped, {errors} errors")
        print(f"  ACC   {correct}/{completed} ({correct*100//completed if completed else 0}%)")
        print(f"{'='*50}")
        return summary

    def generate_summary(self, split: str = "validation") -> dict:
        results = []
        for p in self.output_dir.glob("*.json"):
            if p.name == "summary.json":
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            results.append({
                "instance_id": data.get("instance_id", ""),
                "task_outcome": data.get("task_outcome", "failure"),
                "final_answer": data.get("final_answer"),
                "ground_truth": data.get("ground_truth", ""),
            })

        total = len(results)
        correct = sum(1 for r in results if r["task_outcome"] == "success")

        summary = {
            "framework": self.FRAMEWORK,
            "model": self.model,
            "split": split,
            "total_tasks": total,
            "correct": correct,
            "accuracy": correct / total if total else 0,
            "per_task": results,
        }

        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary
