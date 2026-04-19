#!/usr/bin/env python3
"""CLI entry point for running GAIA benchmarks with Magentic-One or OWL."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Run GAIA benchmark")
    parser.add_argument(
        "--framework",
        required=True,
        choices=["magentic-one", "owl"],
        help="Agent framework to use",
    )
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--levels", nargs="+", type=int, default=None, help="Filter by GAIA level (1, 2, 3)")
    parser.add_argument("--model", default="google/gemma-4-31b-it", help="Model name")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--task-ids", nargs="+", default=None, help="Run specific task IDs only")
    parser.add_argument("--timeout", type=int, default=600, help="Per-task timeout in seconds")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "gaia"),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: results/<framework>/<split>)",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.output_dir = os.path.join(project_root, "results", args.framework.replace("-", "_"), args.split)

    gaia_files_dir = os.path.join(args.data_dir, "files")

    if args.framework == "magentic-one":
        from runners.magentic_one_runner import MagenticOneRunner

        runner = MagenticOneRunner(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model=args.model,
            gaia_files_dir=gaia_files_dir,
            timeout=args.timeout,
        )
    else:
        from runners.owl_runner import OWLRunner

        runner = OWLRunner(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model=args.model,
            gaia_files_dir=gaia_files_dir,
            timeout=args.timeout,
        )

    print(f"=== GAIA Benchmark: {args.framework} ===")
    print(f"  Split: {args.split}")
    print(f"  Levels: {args.levels or 'all'}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output_dir}")
    print(f"  Timeout: {args.timeout}s per task")
    print()

    summary = asyncio.run(
        runner.run_all(
            split=args.split,
            levels=args.levels,
            max_tasks=args.max_tasks,
            task_ids=args.task_ids,
        )
    )

    print(f"\n=== Summary ===")
    print(f"  Total: {summary['total_tasks']}")
    print(f"  Correct: {summary['correct']}")
    print(f"  Accuracy: {summary['accuracy']:.2%}")
    for lvl, stats in summary.get("by_level", {}).items():
        print(f"  {lvl}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")


if __name__ == "__main__":
    main()
