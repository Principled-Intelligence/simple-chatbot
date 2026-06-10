import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import litellm
from loguru import logger

from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.guard import ScopeGuardGate
from simple_chatbot.indexer import Indexer
from simple_chatbot.loader import Document
from simple_chatbot.scenario_catalog import build_responses_tools_from_defs
from simple_chatbot.scripted_indexer import ScriptedIndexer
from simple_chatbot.tools import (
    ToolDef,
    make_search_tool,
    SEARCH_TOOL_SCHEMA,
)

# Backward-compat re-export: tests import SEARCH_TOOL from this module
SEARCH_TOOL = SEARCH_TOOL_SCHEMA

EMPTY_KB_RESPONSE = (
    "I don't have any indexed documents to search yet. Add documents to the knowledge base "
    "and reindex it before asking document-based questions."
)

FINAL_ANSWER_INSTRUCTION = (
    "You have exhausted the available tool calls. Based on the search results already "
    "in the conversation, write your final answer now in plain text. Do not request any more tools."
)


def _sanitize_assistant_dump(dump: dict) -> dict:
    """Reduce a LiteLLM `Message.model_dump()` to the keys that are safe to
    feed back to providers on the next turn. Some providers reject extra or
    provider-specific keys (e.g. `reasoning_content`, `provider_specific_fields`,
    `audio`, `thinking_blocks`). Keep only role, content, tool_calls (themselves
    trimmed to `id`/`type`/`function:{name,arguments}`), and `name` if present.

    Full reasoning text is intentionally omitted from working messages: providers
    differ (OpenAI/Gemini hide it; Anthropic can round-trip `thinking_blocks`).
    Gemini 3+ multi-turn tool continuity is preserved via thought signatures
    embedded in tool-call `id` values by LiteLLM, which this sanitizer keeps.

    Tool-call `arguments` are forced to valid JSON: strict providers
    (Vertex/Gemini) `json.loads` every tool call in the message history when
    serialising it, so a non-JSON argument string (e.g. the `malformed_search`
    misbehavior knob) would crash the *next* request before it is sent. The raw
    string is preserved in the `tool_messages` trace, so the malformed call still
    surfaces in the Responses output; only the replayed copy is normalised.
    """
    sanitized: dict = {
        "role": dump.get("role", "assistant"),
        "content": dump.get("content"),
    }
    name = dump.get("name")
    if name is not None:
        sanitized["name"] = name
    tool_calls = dump.get("tool_calls") or []
    if tool_calls:
        sanitized["tool_calls"] = [
            {
                "id": tc.get("id"),
                "type": tc.get("type", "function"),
                "function": {
                    "name": (tc.get("function") or {}).get("name"),
                    "arguments": _provider_safe_args(
                        (tc.get("function") or {}).get("arguments", "")
                    ),
                },
            }
            for tc in tool_calls
        ]
    return sanitized


def _provider_safe_args(arguments: Any) -> str:
    """Coerce a tool call's `arguments` into a JSON-parseable string for replay.

    Returns the original string when it already parses as JSON. Otherwise (an
    unparseable string such as the `malformed_search` knob) returns ``"{}"`` so
    strict providers can serialise the history. The raw value lives on in the
    `tool_messages` trace; this only touches the copy fed back to the provider.
    """
    if not isinstance(arguments, str):
        arguments = "" if arguments is None else str(arguments)
    try:
        json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return "{}"
    return arguments


def sampling_kwargs(config: SimpleChatbotConfig) -> dict:
    """Return litellm completion kwargs the user set explicitly (skip None).

    Shared by the single-agent `Agent` and the scenario `LiveProvider` so both
    forward identical wire kwargs to litellm. Includes sampling params and
    `reasoning_effort` when configured.
    """
    # attr name on SimpleChatbotConfig  ->  key litellm/OpenAI expects on the wire
    field_map = {
        "temperature": "temperature",
        "top_p": "top_p",
        "gen_top_k": "top_k",
        "min_p": "min_p",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "repetition_penalty": "repetition_penalty",
        "reasoning_effort": "reasoning_effort",
    }
    return {
        wire: getattr(config, attr)
        for attr, wire in field_map.items()
        if getattr(config, attr) is not None
    }


@dataclass
class ChatResult:
    content: str
    retrieved_chunks: list[Document]
    blocked_by_guard: bool = False
    tools: list[dict] = field(default_factory=list)
    tool_messages: list[dict] = field(default_factory=list)
    usage: dict = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    final_messages: list[dict] = field(default_factory=list)
    final_reasoning_content: str | None = None
    responses_tools: list[dict] = field(default_factory=list)
    active_agent: str | None = None


class Agent:
    def __init__(
        self,
        config: SimpleChatbotConfig,
        indexer: Indexer | ScriptedIndexer,
        gate: ScopeGuardGate | None = None,
        acompletion: Callable[..., Awaitable[Any]] | None = None,
        tools: list[ToolDef] | None = None,
    ) -> None:
        self.config = config
        self.indexer = indexer
        self.gate = gate
        self._acompletion = acompletion
        self.tools = tools if tools is not None else [make_search_tool(indexer)]
        self._tool_by_name = {t.name: t for t in self.tools}
        logger.bind(
            model=config.chat_model,
            max_tool_rounds=config.max_tool_rounds,
            tool_names=list(self._tool_by_name),
        ).info("Agent initialised")

    def _sampling_kwargs(self) -> dict:
        """Return only the sampling params the user set explicitly (skip None)."""
        return sampling_kwargs(self.config)

    async def chat(self, messages: list[dict]) -> ChatResult:
        logger.bind(
            message_count=len(messages),
            last_role=messages[-1].get("role") if messages else "none",
        ).info("Chat started")

        usage_totals: dict = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # Flat Responses-API projection of the tool catalog this agent offers the
        # model, surfaced on every ChatResult so `Response.tools` echoes the tools
        # in effect (matching the scenario path and the standard Responses API).
        responses_tools = build_responses_tools_from_defs(self.tools)

        if self.gate is not None:
            decision = await self.gate.check(messages)
            logger.bind(
                scope_class=decision.scope_class,
                allowed=decision.allowed,
                evidence_count=len(decision.evidences),
            ).info("ScopeGuard decision")
            logger.bind(evidences=decision.evidences).debug("ScopeGuard evidences")
            if not decision.allowed:
                return ChatResult(
                    content=self.gate.refusal_text(decision),
                    retrieved_chunks=[],
                    blocked_by_guard=True,
                    tools=[t.schema for t in self.tools],
                    responses_tools=responses_tools,
                    usage=usage_totals,
                    final_messages=list(messages),
                )

        if self.indexer.document_count() == 0:
            logger.bind(document_count=0).warning("No indexed documents available")
            return ChatResult(
                content=EMPTY_KB_RESPONSE,
                retrieved_chunks=[],
                tools=[t.schema for t in self.tools],
                responses_tools=responses_tools,
                usage=usage_totals,
                final_messages=list(messages),
            )

        all_chunks: list[Document] = []
        tool_messages: list[dict] = []
        working: list[dict] = []

        already_has_system = bool(messages) and messages[0].get("role") == "system"
        if self.config.system_prompt and not already_has_system:
            logger.bind(system_prompt_chars=len(self.config.system_prompt)).info(
                "Prepending system prompt"
            )
            working.append({"role": "system", "content": self.config.system_prompt})

        working.extend(messages)

        last_content = ""

        for round_num in range(1, self.config.max_tool_rounds + 1):
            round_log = logger.bind(
                round=round_num, max_rounds=self.config.max_tool_rounds
            )
            round_log.info("Agentic loop round started")

            kwargs: dict = {
                "model": self.config.chat_model,
                "messages": working,
                "tools": [t.schema for t in self.tools],
                "tool_choice": "auto",
            }
            if self.config.chat_api_base:
                kwargs["api_base"] = self.config.chat_api_base
                logger.bind(api_base=self.config.chat_api_base).debug(
                    "Using custom chat API base"
                )
            kwargs.update(self._sampling_kwargs())

            logger.bind(
                model=self.config.chat_model,
                messages_in_context=len(working),
            ).info("Calling LLM")
            response = await (self._acompletion or litellm.acompletion)(**kwargs)
            choice = response.choices[0]
            assistant_msg = choice.message

            usage = response.usage
            if usage:
                logger.bind(
                    prompt=usage.prompt_tokens,
                    completion=usage.completion_tokens,
                    total=usage.total_tokens,
                ).info("Token usage")
                usage_totals["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                usage_totals["completion_tokens"] += (
                    getattr(usage, "completion_tokens", 0) or 0
                )
                usage_totals["total_tokens"] += getattr(usage, "total_tokens", 0) or 0

            last_content = assistant_msg.content or ""
            finish_reason = choice.finish_reason
            logger.bind(finish_reason=finish_reason).debug("LLM finish reason")

            if finish_reason == "tool_calls" and assistant_msg.tool_calls:
                round_log.bind(tool_call_count=len(assistant_msg.tool_calls)).info(
                    "Model requested tool calls"
                )
                assistant_dump = assistant_msg.model_dump()
                # `working` is sent back to the LLM on subsequent rounds (and
                # re-fed on chained Responses turns via `final_messages`), so
                # strip provider-specific extras down to the keys providers accept.
                working.append(_sanitize_assistant_dump(assistant_dump))

                # `tool_messages` is the trace and keeps the full dump plus
                # reasoning_content if present.
                reasoning = getattr(assistant_msg, "reasoning_content", None)
                trace_dump = dict(assistant_dump)
                if reasoning is not None:
                    trace_dump["reasoning_content"] = reasoning
                tool_messages.append(trace_dump)

                for tc_idx, tool_call in enumerate(assistant_msg.tool_calls, start=1):
                    fn = getattr(tool_call, "function", None)
                    fn_name = getattr(fn, "name", None)
                    raw_args = getattr(fn, "arguments", "{}")
                    tool_call_id = getattr(tool_call, "id", f"tool_call_{tc_idx}")
                    tool_log = logger.bind(
                        tool_call_id=tool_call_id,
                        tool_index=tc_idx,
                        tool_count=len(assistant_msg.tool_calls),
                        tool_name=fn_name,
                    )
                    tool_log.info("Tool call received")
                    tool_log.bind(tool_arguments=raw_args).debug("Tool call arguments")

                    tool = self._tool_by_name.get(fn_name or "")
                    is_error = False
                    if tool is None:
                        available = ", ".join(sorted(self._tool_by_name)) or "(none)"
                        error = f"unsupported tool {fn_name!r}; available tools: {available}"
                        tool_log.bind(error=error).warning("Unknown tool name")
                        context = f"Tool error: {error}"
                        is_error = True
                    else:
                        try:
                            result = await tool.executor(raw_args)
                            context = result.text
                            is_error = result.is_error
                            all_chunks.extend(result.chunks)
                            tool_log.bind(
                                result_chars=len(context),
                                chunk_count=len(result.chunks),
                                is_error=is_error,
                            ).info("Tool executed")
                        except Exception as exc:
                            tool_log.bind(error=str(exc)).exception(
                                "Tool executor raised"
                            )
                            context = f"Tool error: {exc}"
                            is_error = True

                    # `working` is replayed to the LLM, so it carries only the
                    # provider-accepted message keys. `tool_messages` is the
                    # trace the Responses adapter maps to output items, so it
                    # also carries `is_error` to drive the output status.
                    tool_result_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": fn_name,
                        "content": context,
                    }
                    working.append(tool_result_msg)
                    tool_messages.append({**tool_result_msg, "is_error": is_error})

                continue

            round_log.bind(response_chars=len(last_content)).info(
                "Final answer produced"
            )
            logger.bind(answer_preview=last_content[:120]).debug("Answer preview")
            final_reasoning = getattr(assistant_msg, "reasoning_content", None)
            final_messages = list(working) + [
                {"role": "assistant", "content": last_content}
            ]
            return ChatResult(
                content=last_content,
                retrieved_chunks=all_chunks,
                tools=[t.schema for t in self.tools],
                responses_tools=responses_tools,
                tool_messages=tool_messages,
                usage=usage_totals,
                final_messages=final_messages,
                final_reasoning_content=final_reasoning,
            )

        logger.bind(max_tool_rounds=self.config.max_tool_rounds).warning(
            "Reached max tool rounds without final answer"
        )

        context_block = "\n\n".join(
            f"[{d.metadata.get('source', 'unknown')}]\n{d.text}" for d in all_chunks
        )

        system_content = (
            (self.config.system_prompt or "") + "\n\n" + FINAL_ANSWER_INSTRUCTION
        )
        if context_block:
            system_content += "\n\nRelevant search results:\n" + context_block

        forced_messages: list[dict] = [{"role": "system", "content": system_content}]
        forced_messages.extend(messages)

        forced_response_kwargs: dict = {
            "model": self.config.chat_model,
            "messages": forced_messages,
            "tool_choice": "none",
        }
        if self.config.chat_api_base:
            forced_response_kwargs["api_base"] = self.config.chat_api_base
        forced_response_kwargs.update(self._sampling_kwargs())

        final_response = ""
        final_reasoning = None
        try:
            final_response_obj = await (self._acompletion or litellm.acompletion)(
                **forced_response_kwargs
            )
            final_usage = final_response_obj.usage
            if final_usage:
                usage_totals["prompt_tokens"] += (
                    getattr(final_usage, "prompt_tokens", 0) or 0
                )
                usage_totals["completion_tokens"] += (
                    getattr(final_usage, "completion_tokens", 0) or 0
                )
                usage_totals["total_tokens"] += (
                    getattr(final_usage, "total_tokens", 0) or 0
                )
            choice = final_response_obj.choices[0]
            final_message = choice.message
            final_response = final_message.content or ""
            final_reasoning = getattr(final_message, "reasoning_content", None)
            logger.bind(
                finish_reason=choice.finish_reason,
                completion_tokens=final_usage.completion_tokens
                if final_usage
                else None,
                content_chars=len(final_response),
                reasoning_chars=len(final_reasoning) if final_reasoning else 0,
            ).info("Forced final response call returned")
        except Exception as exc:
            logger.bind(error=str(exc)).warning("Forced final response call failed")

        logger.bind(response_chars=len(final_response)).info(
            "Forced final response produced"
        )
        final_messages = list(working) + [
            {"role": "assistant", "content": final_response}
        ]
        return ChatResult(
            content=final_response,
            retrieved_chunks=all_chunks,
            tools=[t.schema for t in self.tools],
            responses_tools=responses_tools,
            tool_messages=tool_messages,
            usage=usage_totals,
            final_messages=final_messages,
            final_reasoning_content=final_reasoning,
        )
