"""Magentic-One runner for GAIA benchmark with trajectory capture."""
from __future__ import annotations

import json
import logging
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import (
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
    TextMessage,
    ThoughtEvent,
)
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_core.logging import LLMCallEvent
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.web_surfer._events import WebSurferEvent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

from runners.base_runner import BaseRunner
from trajectory.writer import TrajectoryWriter


AGENT_DEFINITIONS = [
    {
        "name": "WebSurfer",
        "system_prompt": (
            "A helpful assistant with access to a web browser. "
            "Performs web searches, opens pages, and interacts with content "
            "(clicking links, scrolling, filling forms). "
            "Can summarize pages or answer questions based on page content."
        ),
        "tools": ["visit_url", "web_search", "click", "input_text", "hover",
                  "scroll_up", "scroll_down", "history_back", "answer_question",
                  "summarize_page", "sleep"],
    },
    {
        "name": "FileSurfer",
        "system_prompt": "An agent that can handle local files, read their contents, and extract information.",
        "tools": ["open_path", "page_down", "page_up", "find_on_page_ctrl_f", "find_next"],
    },
    {
        "name": "Coder",
        "system_prompt": (
            "A helpful and general-purpose AI assistant with strong language skills, "
            "Python skills, and Linux command line skills. Writes code for the "
            "Executor agent to run."
        ),
        "tools": [],
    },
    {
        "name": "Executor",
        "system_prompt": "An agent that executes code blocks (Python, shell) and returns the output.",
        "tools": ["python_executor", "bash"],
    },
    {
        "name": "MagenticOneOrchestrator",
        "system_prompt": (
            "The orchestrator that coordinates the team. "
            "Creates plans, delegates tasks to agents, and synthesizes results."
        ),
        "tools": [],
    },
]

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


class MessageLogger:
    """Logs every prompt input/output message to a JSONL file in real time."""

    def __init__(self, task_id: str):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"{task_id}.jsonl"
        self.log_path.write_text("", encoding="utf-8")
        self._step = 0

    def log(self, role: str, agent: str, content: str, **extra: Any) -> None:
        entry = {
            "step": self._step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "agent": agent,
            "content": content,
        }
        entry.update(extra)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
        self._step += 1


class AgentEventHandler(logging.Handler):
    """Captures WebSurferEvent (tool actions) and LLMCallEvent from autogen_core logger.

    WebSurferEvent is the only way to get actual tool name + args for WebSurfer,
    because WebSurfer handles tools internally and only emits a MultiModalMessage
    summary at the run_stream level.
    """

    def __init__(self, msg_logger: MessageLogger, writer: TrajectoryWriter):
        super().__init__()
        self.msg_logger = msg_logger
        self.writer = writer
        # Track the last tool call so we can link it to the next MultiModalMessage result
        self.last_tool_name: str | None = None
        self.last_tool_args: dict | None = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event = record.msg

            # --- WebSurferEvent: actual tool action ---
            if isinstance(event, WebSurferEvent):
                tool_name = event.action or "unknown"
                tool_args = event.arguments or {}

                # Store for linking to the result
                self.last_tool_name = tool_name
                self.last_tool_args = tool_args

                self.msg_logger.log(
                    role="tool_call",
                    agent=event.source,
                    content=event.message[:3000],
                    tool_name=tool_name,
                    tool_args=tool_args,
                    url=event.url or "",
                )
                self.writer.add_step(
                    agent=event.source,
                    content=event.message,
                    tool_calls=[{"tool": tool_name, "arguments": tool_args}],
                )
                return

            # --- LLMCallEvent: LLM prompt input/output ---
            if isinstance(event, LLMCallEvent):
                if event.messages:
                    for msg in event.messages:
                        if isinstance(msg, dict):
                            self.msg_logger.log(
                                role=msg.get("role", "unknown"),
                                agent="llm_input",
                                content=str(msg.get("content", ""))[:3000],
                            )

                response_content = ""
                if isinstance(event.response, dict):
                    choices = event.response.get("choices", [])
                    if choices:
                        msg_data = choices[0].get("message", {})
                        response_content = msg_data.get("content", "")

                self.msg_logger.log(
                    role="assistant",
                    agent="llm_output",
                    content=response_content[:3000] if response_content else "",
                    prompt_tokens=event.prompt_tokens,
                    completion_tokens=event.completion_tokens,
                )
                return

        except Exception:
            pass

    def pop_last_tool(self) -> tuple[str | None, dict | None]:
        """Return and clear the last tool call info for linking to result."""
        name, args = self.last_tool_name, self.last_tool_args
        self.last_tool_name = None
        self.last_tool_args = None
        return name, args


def _parse_args(args_str: str) -> dict:
    try:
        return json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        return {"raw": str(args_str)} if args_str else {}


class MagenticOneRunner(BaseRunner):
    FRAMEWORK = "magentic-one"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAIChatCompletionClient(
            model="QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ",
            base_url="http://localhost:8000/v1",
            api_key="unused",
            max_tokens=32768,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
            },
        )

    def _get_agent_definitions(self) -> list[dict]:
        return AGENT_DEFINITIONS

    async def run_single_task(self, task: dict, writer: TrajectoryWriter) -> str | None:
        prompt = self._build_prompt(task)
        task_id = task["task_id"]

        msg_logger = MessageLogger(task_id)
        msg_logger.log(role="user", agent="user", content=prompt)

        web_surfer = MultimodalWebSurfer(
            "WebSurfer",
            model_client=self.client,
            headless=True,
        )
        file_surfer = FileSurfer("FileSurfer", model_client=self.client)
        coder = MagenticOneCoderAgent("Coder", model_client=self.client)

        work_dir = Path(tempfile.mkdtemp(prefix=f"gaia_{task_id[:8]}_"))
        code_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
        executor = CodeExecutorAgent("Executor", code_executor=code_executor)

        team = MagenticOneGroupChat(
            participants=[web_surfer, file_surfer, coder, executor],
            model_client=self.client,
            max_turns=100,
        )

        logger = logging.getLogger("autogen_core")
        agent_handler = AgentEventHandler(msg_logger, writer)
        logger.addHandler(agent_handler)
        logger.setLevel(logging.INFO)

        final_answer = None
        try:
            async for event in team.run_stream(task=prompt):
                event_type = getattr(event, "type", "")
                agent_name = str(getattr(event, "source", "unknown"))

                # --- ToolCallRequestEvent (from Coder/Executor agents) ---
                if isinstance(event, ToolCallRequestEvent):
                    tool_calls = []
                    for fc in event.content:
                        tool_calls.append({
                            "tool": fc.name,
                            "arguments": _parse_args(fc.arguments),
                        })
                    content = "; ".join(
                        f"{tc['tool']}({json.dumps(tc['arguments'])})"
                        for tc in tool_calls
                    )
                    msg_logger.log(
                        role="tool_call_request", agent=agent_name,
                        content=content[:3000], tool_calls=tool_calls,
                    )
                    writer.add_step(
                        agent=agent_name, content=content,
                        tool_calls=tool_calls,
                    )

                # --- ToolCallExecutionEvent (tool execution result) ---
                elif isinstance(event, ToolCallExecutionEvent):
                    for result in event.content:
                        result_content = result.content if result.content else ""
                        is_error = getattr(result, "is_error", False)
                        msg_logger.log(
                            role="tool_result", agent=agent_name,
                            content=result_content[:3000],
                            tool_name=result.name, is_error=is_error,
                        )
                        writer.add_step(
                            agent=agent_name,
                            content=f"[{result.name}] {'ERROR: ' if is_error else ''}{result_content}",
                            tool_results=[{
                                "tool": result.name,
                                "output": result_content,
                                "is_error": is_error,
                            }],
                        )

                # --- ToolCallSummaryMessage ---
                elif isinstance(event, ToolCallSummaryMessage):
                    content = event.content if isinstance(event.content, str) else str(event.content)
                    msg_logger.log(role="tool_summary", agent=agent_name, content=content[:3000])
                    writer.add_step(agent=agent_name, content=content)

                # --- TextMessage ---
                elif isinstance(event, TextMessage):
                    msg_logger.log(role="message", agent=agent_name, content=event.content[:3000])
                    writer.add_step(agent=agent_name, content=event.content)

                # --- ThoughtEvent ---
                elif isinstance(event, ThoughtEvent):
                    msg_logger.log(role="thought", agent=agent_name, content=event.content[:3000])

                # --- TaskResult ---
                elif hasattr(event, "messages"):
                    if event.messages:
                        last_msg = event.messages[-1]
                        last_content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
                        final_answer = await self._extract_answer(last_content, prompt)

                    stop_reason = getattr(event, "stop_reason", "unknown")
                    msg_logger.log(
                        role="system", agent="system",
                        content=f"Task completed. Stop reason: {stop_reason}",
                        final_answer=final_answer,
                    )
                    writer.add_step(agent="system", content=f"Task completed. Stop reason: {stop_reason}")

                # --- Fallback: MultiModalMessage (WebSurfer tool result) ---
                elif hasattr(event, "source") and hasattr(event, "content"):
                    content = event.content if isinstance(event.content, str) else str(event.content)

                    # Check if this is a tool result from WebSurfer
                    last_tool, last_args = agent_handler.pop_last_tool()
                    if last_tool and agent_name == "WebSurfer":
                        msg_logger.log(
                            role="tool_result", agent=agent_name,
                            content=content[:3000],
                            tool_name=last_tool,
                        )
                        writer.add_step(
                            agent=agent_name, content=content,
                            tool_results=[{
                                "tool": last_tool,
                                "output": content,
                                "is_error": False,
                            }],
                        )
                    else:
                        msg_logger.log(
                            role=event_type or "agent_message",
                            agent=agent_name, content=content[:3000],
                        )
                        writer.add_step(agent=agent_name, content=content)

        finally:
            logger.removeHandler(agent_handler)
            await team.reset()

        return final_answer

    async def _extract_answer(self, text: str, question: str) -> str:
        """Use the LLM to extract a concise final answer from the agent's response."""
        prompt = (
            "You are an answer extractor. Given a question and an agent's response, "
            "extract ONLY the final answer as a short, concise value — just the word, "
            "number, name, or short phrase that directly answers the question. "
            "Do NOT include explanations, sentences, or reasoning. "
            "If the response contains no clear answer, reply with exactly: NO_ANSWER\n\n"
            f"Question: {question}\n\n"
            f"Agent's response:\n{text[:3000]}\n\n"
            "Final answer (short value only):"
        )
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30) as http:
                resp = await http.post(
                    "http://localhost:8000/v1/chat/completions",
                    json={
                        "model": "QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 50,
                        "temperature": 0,
                    },
                )
                data = resp.json()
                answer = data["choices"][0]["message"]["content"].strip()
                if answer == "NO_ANSWER":
                    return None
                return answer
        except Exception:
            # Fallback to simple regex if LLM call fails
            import re
            for pattern in [
                r"(?:final answer|the answer)(?:\s+is)?[:\s]+(.+?)(?:\.|$)",
                r"(?:FINAL ANSWER)[:\s]+(.+?)(?:\.|$)",
            ]:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
            return lines[-1][:200] if lines else text[:200]

    async def cleanup(self):
        await self.client.close()
