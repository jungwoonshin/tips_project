"""Typed objects used throughout the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Node:
    step_id: str
    level: str                         # "planner" | "worker"
    role: str
    action_type: str
    action_content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    observation: str | None = None
    parent_step_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trajectory:
    trajectory_id: str
    gaia_task_id: str
    query: str
    oracle_answer: str
    agent_final_answer: str
    agent_model: str
    nodes: list[Node]
    raw: dict = field(repr=False, default_factory=dict)


@dataclass
class Candidate:
    node_id: str
    level: str
    role: str
    predicted_type: str
    confidence: float
    rationale: str
    diagnostic_hint: str


@dataclass
class Fix:
    node_id: str
    predicted_type: str
    modified_field: str
    new_content: str
    fixer_rationale: str
    agent_plausible: bool | None = None   # None if not audited


@dataclass
class ReplayResult:
    seed: int
    final_answer: str
    flipped: bool
    error: str | None = None


@dataclass
class SuffSetState:
    trajectory_id: str
    stage: str
    M: list[str]
    flip_seeds: dict[int, str]       # seed -> "flip" | "no_flip" | "error"
    flip_rate: float
    k_used: int


@dataclass
class Record:
    trajectory_id: str
    gaia_task_id: str
    agent_framework: str
    agent_model: str
    agent_final_answer: str
    oracle_answer: str
    trajectory: list[dict]
    published_sufficient_set: list[str]
    per_node: dict[str, dict]
    fixes_supplementary: dict[str, dict]
    validity_metadata: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)
