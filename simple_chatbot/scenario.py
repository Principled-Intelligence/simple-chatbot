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
