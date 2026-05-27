from dataclasses import dataclass, field

import litellm
from loguru import logger

from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.guard import ScopeGuardGate
from simple_chatbot.indexer import Indexer
from simple_chatbot.loader import Document
from simple_chatbot.tools import (
    ToolDef,
    ToolResult,
    make_search_tool,
    SEARCH_TOOL_NAME,
    SEARCH_TOOL_SCHEMA,
)

# Backward-compat re-export: tests import SEARCH_TOOL from this module
SEARCH_TOOL = SEARCH_TOOL_SCHEMA

EMPTY_KB_RESPONSE = (
    "I don't have any indexed documents to search yet. Add documents to the knowledge base "
    "and reindex it before asking document-based questions."
)


@dataclass
class ChatResult:
    content: str
    retrieved_chunks: list[Document]
    blocked_by_guard: bool = False
    tools: list[dict] = field(default_factory=list)
    tool_messages: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    final_messages: list[dict] = field(default_factory=list)
    final_reasoning_content: str | None = None


class Agent:
    def __init__(
        self,
        config: SimpleChatbotConfig,
        indexer: Indexer,
        gate: ScopeGuardGate | None = None,
        acompletion=None,
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

    async def chat(self, messages: list[dict]) -> ChatResult:
        logger.bind(
            message_count=len(messages),
            last_role=messages[-1].get("role") if messages else "none",
        ).info("Chat started")

        usage_totals: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

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
                    usage=usage_totals,
                    final_messages=list(messages),
                )

        if self.indexer.document_count() == 0:
            logger.bind(document_count=0).warning("No indexed documents available")
            return ChatResult(
                content=EMPTY_KB_RESPONSE,
                retrieved_chunks=[],
                tools=[t.schema for t in self.tools],
                usage=usage_totals,
                final_messages=list(messages),
            )

        all_chunks: list[Document] = []
        tool_messages: list[dict] = []
        working: list[dict] = []

        already_has_system = bool(messages) and messages[0].get("role") == "system"
        if self.config.system_prompt and not already_has_system:
            logger.bind(system_prompt_chars=len(self.config.system_prompt)).info("Prepending system prompt")
            working.append({"role": "system", "content": self.config.system_prompt})

        working.extend(messages)

        last_content = ""

        for round_num in range(1, self.config.max_tool_rounds + 1):
            round_log = logger.bind(round=round_num, max_rounds=self.config.max_tool_rounds)
            round_log.info("Agentic loop round started")

            kwargs: dict = {
                "model": self.config.chat_model,
                "messages": working,
                "tools": [t.schema for t in self.tools],
                "tool_choice": "auto",
            }
            if self.config.chat_api_base:
                kwargs["api_base"] = self.config.chat_api_base
                logger.bind(api_base=self.config.chat_api_base).debug("Using custom chat API base")

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
                usage_totals["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                usage_totals["total_tokens"] += getattr(usage, "total_tokens", 0) or 0

            last_content = assistant_msg.content or ""
            finish_reason = choice.finish_reason
            logger.bind(finish_reason=finish_reason).debug("LLM finish reason")

            if finish_reason == "tool_calls" and assistant_msg.tool_calls:
                round_log.bind(tool_call_count=len(assistant_msg.tool_calls)).info("Model requested tool calls")
                assistant_dump = assistant_msg.model_dump()
                reasoning = getattr(assistant_msg, "reasoning_content", None)
                if reasoning is not None:
                    assistant_dump["reasoning_content"] = reasoning
                working.append(assistant_dump)
                tool_messages.append(assistant_dump)

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

                    tool = self._tool_by_name.get(fn_name)
                    if tool is None:
                        available = ", ".join(sorted(self._tool_by_name)) or "(none)"
                        error = f"unsupported tool {fn_name!r}; available tools: {available}"
                        tool_log.bind(error=error).warning("Unknown tool name")
                        context = f"Tool error: {error}"
                    else:
                        try:
                            result = await tool.executor(raw_args)
                            context = result.text
                            all_chunks.extend(result.chunks)
                            tool_log.bind(
                                result_chars=len(context),
                                chunk_count=len(result.chunks),
                            ).info("Tool executed")
                        except Exception as exc:
                            tool_log.bind(error=str(exc)).exception("Tool executor raised")
                            context = f"Tool error: {exc}"

                    tool_result_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": fn_name,
                        "content": context,
                    }
                    working.append(tool_result_msg)
                    tool_messages.append(tool_result_msg)

                continue

            round_log.bind(response_chars=len(last_content)).info("Final answer produced")
            logger.bind(answer_preview=last_content[:120]).debug("Answer preview")
            final_reasoning = getattr(assistant_msg, "reasoning_content", None)
            final_messages = list(working) + [{"role": "assistant", "content": last_content}]
            return ChatResult(
                content=last_content,
                retrieved_chunks=all_chunks,
                tools=[t.schema for t in self.tools],
                tool_messages=tool_messages,
                usage=usage_totals,
                final_messages=final_messages,
                final_reasoning_content=final_reasoning,
            )

        logger.bind(max_tool_rounds=self.config.max_tool_rounds).warning(
            "Reached max tool rounds without final answer"
        )
        forced_response_kwargs: dict = {
            "model": self.config.chat_model,
            "messages": working,
            "tool_choice": "none",
        }
        if self.config.chat_api_base:
            forced_response_kwargs["api_base"] = self.config.chat_api_base

        try:
            final_response_obj = await (self._acompletion or litellm.acompletion)(**forced_response_kwargs)
            final_usage = final_response_obj.usage
            if final_usage:
                usage_totals["prompt_tokens"] += getattr(final_usage, "prompt_tokens", 0) or 0
                usage_totals["completion_tokens"] += getattr(final_usage, "completion_tokens", 0) or 0
                usage_totals["total_tokens"] += getattr(final_usage, "total_tokens", 0) or 0
            final_message = final_response_obj.choices[0].message
            final_response = final_message.content or ""
            final_reasoning = getattr(final_message, "reasoning_content", None)
        except Exception as exc:
            logger.bind(error=str(exc)).warning("Forced final response call failed")
            final_response = ""
            final_reasoning = None

        logger.bind(response_chars=len(final_response)).info("Forced final response produced")
        final_messages = list(working) + [{"role": "assistant", "content": final_response}]
        return ChatResult(
            content=final_response,
            retrieved_chunks=all_chunks,
            tools=[t.schema for t in self.tools],
            tool_messages=tool_messages,
            usage=usage_totals,
            final_messages=final_messages,
            final_reasoning_content=final_reasoning,
        )
