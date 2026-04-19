"""OWL (CAMEL-AI) runner for GAIA benchmark with trajectory capture."""
from __future__ import annotations

import json
import re
from typing import Any

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.toolkits import SearchToolkit
from camel.types import ModelPlatformType

from runners.base_runner import BaseRunner
from trajectory.writer import TrajectoryWriter


OWL_AGENT_DEFINITIONS = [
    {
        "name": "Assistant",
        "system_prompt": (
            "A helpful AI assistant solving benchmark tasks with access to web search tools. "
            "Uses search_tavily, search_duckduckgo, or search_wiki to look up facts."
        ),
        "tools": ["search_tavily", "search_duckduckgo", "search_wiki"],
    },
]


class OWLRunner(BaseRunner):
    FRAMEWORK = "owl"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_backend = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=self.model,
            url="http://localhost:8000/v1",
            api_key="unused",
            model_config_dict={"max_tokens": 4096, "temperature": 0},
        )

    def _get_agent_definitions(self) -> list[dict]:
        return OWL_AGENT_DEFINITIONS

    def _get_search_tools(self):
        """Get search tools - Tavily + DuckDuckGo + Wikipedia."""
        toolkit = SearchToolkit()
        tools = []
        for tool in toolkit.get_tools():
            name = tool.get_function_name()
            if name in ("search_tavily", "search_duckduckgo", "search_wiki"):
                tools.append(tool)
        return tools

    async def run_single_task(self, task: dict, writer: TrajectoryWriter) -> str | None:
        prompt = self._build_prompt(task)

        system_msg = (
            "You are a helpful AI assistant solving benchmark tasks. "
            "You have access to web search tools to find information online. "
            "Use search_tavily, search_duckduckgo, or search_wiki to look up facts you need. "
            "Think step by step. When you have the final answer, state it clearly as: "
            "FINAL ANSWER: <answer>\n"
            "The answer should be concise - just the value, name, number, or short phrase requested."
        )

        search_tools = self._get_search_tools()
        agent = ChatAgent(
            system_message=system_msg,
            model=self.model_backend,
            tools=search_tools,
        )

        writer.add_step(
            agent="user",
            content=prompt[:2000],
        )

        # Run the agent with the task
        user_msg = BaseMessage.make_user_message(role_name="user", content=prompt)
        response = agent.step(user_msg)

        # Process response
        content = ""
        def _record_response(resp):
            """Extract tool calls/results from a CAMEL response and emit steps."""
            tool_calls_data = []
            tool_results_data = []
            for tc in resp.info.get("tool_calls", []):
                if hasattr(tc, "tool_name"):
                    tool_calls_data.append({
                        "tool": tc.tool_name,
                        "arguments": dict(tc.args) if tc.args else {},
                    })
                    tool_results_data.append({
                        "tool": tc.tool_name,
                        "output": str(tc.result) if tc.result is not None else "",
                        "is_error": False,
                    })
                elif hasattr(tc, "function"):
                    func = tc.function
                    tool_calls_data.append({
                        "tool": getattr(func, "name", "unknown"),
                        "arguments": _parse_args(getattr(func, "arguments", "")),
                    })
                elif isinstance(tc, dict):
                    tool_calls_data.append({
                        "tool": tc.get("function", {}).get("name", "unknown"),
                        "arguments": tc.get("function", {}).get("arguments", {}),
                    })
            # Emit each tool call + its result as separate steps so the
            # trajectory reflects the agent's actual tool activity.
            for call, result in zip(tool_calls_data, tool_results_data):
                writer.add_step(
                    agent="Assistant",
                    content=f"{call['tool']}({json.dumps(call['arguments'])})",
                    tool_calls=[call],
                )
                writer.add_step(
                    agent="ComputerTerminal",
                    content=f"[{result['tool']}] {result['output'][:2000]}",
                    tool_results=[result],
                )
            # Final assistant message (reasoning / answer)
            msg_content = resp.msgs[0].content if resp.msgs else ""
            writer.add_step(
                agent="Assistant",
                content=msg_content[:2000],
            )

        if response.msgs:
            content = response.msgs[0].content
            _record_response(response)

            # Multi-turn: continue if no final answer yet
            max_turns = 8
            turn = 0
            while turn < max_turns and not self._has_final_answer(content):
                turn += 1
                follow_up = BaseMessage.make_user_message(
                    role_name="user",
                    content="Please continue your analysis. Use search tools if you need more information. "
                    "When done, state: FINAL ANSWER: <answer>",
                )
                writer.add_step(
                    agent="user",
                    content="Continue - use search tools if needed, then provide final answer",
                )

                response = agent.step(follow_up)
                if response.msgs:
                    content = response.msgs[0].content
                    _record_response(response)

            final_answer = self._extract_answer(content)
            return final_answer

        return None

    def _has_final_answer(self, text: str) -> bool:
        return bool(re.search(r"FINAL ANSWER:", text, re.IGNORECASE))

    def _extract_answer(self, text: str) -> str:
        """Extract final answer from agent response."""
        patterns = [
            r"FINAL ANSWER[:\s]+(.+?)(?:\n|$)",
            r"(?:final answer|the answer)(?:\s+is)?[:\s]+(.+?)(?:\.|$)",
            r"(?:Answer)[:\s]+(.+?)(?:\.|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if lines:
            return lines[-1][:200]
        return text[:200]


def _parse_args(args: Any) -> dict:
    """Parse function arguments that may be a JSON string or dict."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        import json
        try:
            return json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return {"raw": args}
    return {}
