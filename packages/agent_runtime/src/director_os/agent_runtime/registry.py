"""Typed tool registry — the native function-calling engine (PLAN §3.1, §3.3).

M1 ships the registry + validation core. M2 wires it into Pydantic AI agents
(Perceiver / Director / Assembler / Critic) with checkpoint-before-execute
timeline mutations. Every tool's input schema is Pydantic; dispatch validates
*before* the function runs, so a hallucinated argument never reaches ffmpeg.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError, create_model


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    input_model: type[BaseModel]


class ToolDispatchError(RuntimeError):
    pass


class ToolRegistry:
    """Registry of strongly-typed tools callable by agents or humans."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(self, name: str, description: str = "") -> Callable:
        """Decorator registering a function; input model derived from type hints."""

        def decorator(fn: Callable[..., Any]) -> Callable:
            sig = inspect.signature(fn)
            fields: dict[str, Any] = {}
            for pname, param in sig.parameters.items():
                ann = param.annotation if param.annotation is not inspect.Parameter.empty else Any
                default = param.default if param.default is not inspect.Parameter.empty else ...
                fields[pname] = (ann, default)
            model = create_model(f"{name}_Input", **fields)  # type: ignore[arg-type]
            self._tools[name] = ToolSpec(
                name=name, description=description, fn=fn, input_model=model
            )
            return fn

        return decorator

    def register(
        self, name: str, description: str, input_model: type[BaseModel], fn: Callable[..., Any]
    ) -> None:
        """Explicit registration with a hand-written Pydantic input model."""
        self._tools[name] = ToolSpec(
            name=name, description=description, fn=fn, input_model=input_model
        )

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolDispatchError(f"unknown tool: '{name}' (known: {self.names()})")
        return self._tools[name]

    def dispatch(
        self, name: str, args: dict[str, Any], _log: list[dict[str, Any]] | None = None
    ) -> Any:
        spec = self.get(name)
        try:
            validated = spec.input_model.model_validate(args)
        except ValidationError as e:
            raise ToolDispatchError(f"tool '{name}' rejected args: {e}") from e
        started = time.monotonic()
        result = spec.fn(**validated.model_dump())
        if _log is not None:
            _log.append(
                {
                    "tool": name,
                    "args": validated.model_dump(mode="json"),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                }
            )
        return result

    def openai_schema(self, name: str) -> dict[str, Any]:
        """OpenAI/Anthropic-style function-calling spec for one tool."""
        spec = self.get(name)
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_model.model_json_schema(),
            },
        }

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [self.openai_schema(n) for n in self.names()]


#: default registry shared by CLI/worker; agents get fresh registries in M2
default_registry = ToolRegistry()


class ToolCall(BaseModel):
    """A logged tool invocation (mirrors `agent_tool_logs` PLAN §5.1)."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = 0
    ok: bool = True
    error: str | None = None
