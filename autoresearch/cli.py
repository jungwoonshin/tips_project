"""CLI entry: python -m autoresearch.cli --budget 100 [--dev-only]"""

from __future__ import annotations

import argparse
import logging
import sys

from autoresearch.orchestrator import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--ratchet-slack", type=float, default=0.02)
    ap.add_argument("--watchdog-every", type=int, default=20)
    ap.add_argument("--overfit-threshold", type=float, default=0.15)
    ap.add_argument("--paradigm-shift-every", type=int, default=20)
    ap.add_argument("--dev-only", action="store_true",
                    help="Skip the held-out test watchdog + final test eval.")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    result = run(
        budget=args.budget,
        ratchet_slack=args.ratchet_slack,
        watchdog_every=args.watchdog_every,
        overfit_threshold=args.overfit_threshold,
        paradigm_shift_every=args.paradigm_shift_every,
        dev_only=args.dev_only,
        run_id=args.run_id,
    )
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
