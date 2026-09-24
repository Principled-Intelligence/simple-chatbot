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
from typing import Callable, Literal

from pydantic import create_model

ROUTE_TOOL_NAME = "route"


@dataclass
class ScenarioTool:
    """A tool: JSON-Schema parameters + an async-free stub callable."""

    name: str
    description: str
    parameters: dict
    func: Callable[..., dict | str]
    wants_state: bool = False


def tool(func: Callable[..., dict | str]) -> ScenarioTool:
    """Decorator turning a typed function into a `ScenarioTool`.

    The parameter JSON Schema is generated from the signature via pydantic;
    `title` keys pydantic adds are stripped to keep the schema close to what a
    hand-written OpenAI function schema looks like.
    """
    sig = inspect.signature(func)
    fields: dict = {}
    wants_state = False
    for pname, p in sig.parameters.items():
        if pname == "self":
            continue
        if pname == "state":
            # Injected at execution from the conversation's ConvState; never
            # part of the advertised JSON schema.
            wants_state = True
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
        wants_state=wants_state,
    )


@dataclass
class Call:
    """A well-formed tool call. Empty `args` (omitting a required field) is the
    deterministic required-arg-violation knob."""

    tool: ScenarioTool
    args: dict = field(default_factory=dict)
    # The tags below are descriptive metadata for eval graders, NOT enforced
    # behavior: the authored `args` already carry the misbehavior (e.g. a
    # wrong-type value), so the orchestrator plays the call verbatim. The tags
    # let graders key on the authoring intent without re-deriving it.
    irrelevant: bool = False  # tag for relevance-dimension test assertions
    redundant: bool = False  # tag: duplicate/unnecessary call → call-necessity dim
    wrong_value: bool = False  # tag: schema-valid call, semantically wrong arg value
    type_mismatch: bool = False  # tag: arg value violates the schema's declared type


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


@dataclass
class Parallel:
    """A fan-out round: emit several tool calls in ONE assistant message.

    May contain only call-family steps (`Call`, `MalformedCall`,
    `UnknownToolCall`) — never `Route`, `Final`, or a nested `Parallel`
    (a fan-out batch cannot hand off, finish, or nest). Enforced in
    `Scenario.validate()`."""

    steps: list[Call | MalformedCall | UnknownToolCall] = field(default_factory=list)


Step = Call | Route | Final | MalformedCall | UnknownToolCall | Parallel


@dataclass
class Agent:
    name: str
    system_prompt: str = ""
    tools: list[ScenarioTool] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)
    script: list[Step] = field(default_factory=list)
    terminal: bool = False
    escalation_message: str = ""


@dataclass
class Scenario:
    id: str
    entry: str
    agents: list[Agent]
    description: str = ""
    mode: Literal["deterministic", "live"] = "deterministic"

    def __post_init__(self) -> None:
        self.validate()

    def agent(self, name: str) -> Agent:
        for a in self.agents:
            if a.name == name:
                return a
        raise KeyError(f"no agent named {name!r}")

    def all_tools(self) -> list[ScenarioTool]:
        by_name: dict[str, ScenarioTool] = {}
        for a in self.agents:
            for t in a.tools:
                by_name.setdefault(t.name, t)
        return list(by_name.values())

    def route_targets(self) -> list[str]:
        targets: set[str] = set()
        for a in self.agents:
            targets.update(a.routes)
        return sorted(targets)

    def validate(self) -> None:
        names = [a.name for a in self.agents]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate agent names in scenario {self.id!r}")
        name_set = set(names)
        if self.entry not in name_set:
            raise ValueError(f"entry agent {self.entry!r} not found in scenario {self.id!r}")
        for a in self.agents:
            for target in a.routes:
                if target not in name_set:
                    raise ValueError(
                        f"agent {a.name!r} routes to unknown agent {target!r}"
                    )
        for a in self.agents:
            agent_tool_names = {t.name for t in a.tools}
            # A user tool named `route` would shadow the generated handoff tool.
            if ROUTE_TOOL_NAME in agent_tool_names:
                raise ValueError(
                    f"agent {a.name!r} declares a tool named {ROUTE_TOOL_NAME!r}, "
                    f"which collides with the generated handoff tool"
                )
            for step in a.script:
                if isinstance(step, Parallel):
                    if not step.steps:
                        raise ValueError(
                            f"empty Parallel step in agent {a.name!r}"
                        )
                    for inner in step.steps:
                        if not isinstance(inner, (Call, MalformedCall, UnknownToolCall)):
                            raise ValueError(
                                f"Parallel step in agent {a.name!r} may contain only "
                                f"Call/MalformedCall/UnknownToolCall, got "
                                f"{type(inner).__name__}"
                            )
                # Every Call/MalformedCall must reference a tool the agent owns.
                # (UnknownToolCall is the deliberate absent-tool knob — skipped.)
                call_steps = step.steps if isinstance(step, Parallel) else [step]
                for cs in call_steps:
                    if isinstance(cs, (Call, MalformedCall)) and cs.tool.name not in agent_tool_names:
                        raise ValueError(
                            f"agent {a.name!r} calls tool {cs.tool.name!r} "
                            f"not in its own tools"
                        )
