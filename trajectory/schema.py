"""Unified trajectory schema for agent benchmark runs."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool: str
    output: str = ""
    is_error: bool = False


class AgentInfo(BaseModel):
    name: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)


class AgenticSystem(BaseModel):
    system_type: str = "algorithm_generated"
    generator: str = ""
    base_model: str = ""
    agents: list[AgentInfo] = Field(default_factory=list)


class FailureLogEntry(BaseModel):
    step: int
    agent: str = ""
    content: str = ""
    available_tools: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)


class Annotation(BaseModel):
    failure_responsible_agent: Optional[str] = None
    decisive_error_step: Optional[int] = None
    reason: Optional[str] = None
    annotator_confidence: Optional[str] = None
    annotators: list[str] = Field(default_factory=list)


class TrajectoryOutput(BaseModel):
    problem_number: int = 0
    instance_id: str
    source_benchmark: str = "GAIA"
    query: str = ""
    ground_truth: str = ""
    agentic_system: AgenticSystem = Field(default_factory=AgenticSystem)
    failure_log: list[FailureLogEntry] = Field(default_factory=list)
    final_answer: Optional[str] = None
    task_outcome: str = "failure"
    annotation: Annotation = Field(default_factory=Annotation)
