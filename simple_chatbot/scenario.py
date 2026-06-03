# simple_chatbot/scenario.py
"""In-memory model + code-first builder API for fixture scenarios.

A scenario declares agents, each with a scoped toolset, optional routing
targets, and a deterministic `script`. Tools are defined as typed Python
functions via `@tool`: the signature yields the JSON Schema, the docstring the
description, and the body is the deterministic stub.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable

from pydantic import create_model

ROUTE_TOOL_NAME = "route"


@dataclass
class ScenarioTool:
    """A tool: JSON-Schema parameters + an async-free stub callable."""

    name: str
    description: str
    parameters: dict
    func: Callable[..., dict | str]


def tool(func: Callable[..., dict | str]) -> ScenarioTool:
    """Decorator turning a typed function into a `ScenarioTool`.

    The parameter JSON Schema is generated from the signature via pydantic;
    `title` keys pydantic adds are stripped to keep the schema close to what a
    hand-written OpenAI function schema looks like.
    """
    sig = inspect.signature(func)
    fields: dict = {}
    for pname, p in sig.parameters.items():
        if pname == "self":
            continue
        annotation = p.annotation if p.annotation is not inspect.Parameter.empty else str
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[pname] = (annotation, default)

    model = create_model(f"{func.__name__}_Args", **fields)
    schema = model.model_json_schema()

    properties = {}
    for key, prop in (schema.get("properties") or {}).items():
        properties[key] = {k: v for k, v in prop.items() if k != "title"}

    parameters: dict = {"type": "object", "properties": properties}
    if schema.get("required"):
        parameters["required"] = schema["required"]

    return ScenarioTool(
        name=func.__name__,
        description=(func.__doc__ or "").strip(),
        parameters=parameters,
        func=func,
    )


@dataclass
class Call:
    """A well-formed tool call. Empty `args` (omitting a required field) is the
    deterministic required-arg-violation knob."""

    tool: ScenarioTool
    args: dict = field(default_factory=dict)
    irrelevant: bool = False  # tag for relevance-dimension test assertions


@dataclass
class Route:
    """A handoff to another agent (serialized as a `route` tool call)."""

    target: str


@dataclass
class Final:
    """End the turn with this assistant message."""

    text: str
    ignore_retrieval: bool = False  # tag for utilization/faithfulness assertions


@dataclass
class MalformedCall:
    """Emit a call whose arguments are invalid JSON (validity knob → score 2)."""

    tool: ScenarioTool


@dataclass
class UnknownToolCall:
    """Emit a call to a tool absent from the catalog (validity knob → score 1)."""

    name: str
    args: dict = field(default_factory=dict)


Step = Call | Route | Final | MalformedCall | UnknownToolCall
