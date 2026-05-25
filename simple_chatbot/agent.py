import json
from dataclasses import dataclass

import litellm
from loguru import logger

from simple_chatbot.config import SimpleChatbotConfig
from simple_chatbot.guard import ScopeGuardGate
from simple_chatbot.indexer import Indexer
from simple_chatbot.loader import Document

SEARCH_TOOL_NAME = "search_documents"

EMPTY_KB_RESPONSE = (
    "I don't have any indexed documents to search yet. Add documents to the knowledge base "
    "and reindex it before asking document-based questions."
)

FINAL_ANSWER_INSTRUCTION = (
    "You have exhausted the available tool calls. Based on the search results already "
    "in the conversation, write your final answer now in plain text. Do not request any more tools."
)

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": (
            "Search the document index for relevant information. "
            "Use this whenever the user asks something that may be answered by the loaded documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        },
    },
}


def _tool_error(message: str) -> str:
    return f"Tool error: {message}"


def _parse_search_tool_query(fn_name: str | None, raw_args) -> tuple[str | None, str | None]:
    if fn_name != SEARCH_TOOL_NAME:
        return None, f"unsupported tool {fn_name!r}; available tool is {SEARCH_TOOL_NAME!r}"

    if isinstance(raw_args, dict):
        args = raw_args
    elif isinstance(raw_args, str):
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON arguments for {SEARCH_TOOL_NAME!r}: {exc.msg}"
    else:
        return None, f"arguments for {SEARCH_TOOL_NAME!r} must be a JSON object"

    if not isinstance(args, dict):
        return None, f"arguments for {SEARCH_TOOL_NAME!r} must be a JSON object"

    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return None, f"{SEARCH_TOOL_NAME!r} requires a non-empty string 'query' argument"

    return query, None


@dataclass
class ChatResult:
    content: str
    retrieved_chunks: list[Document]
    blocked_by_guard: bool = False


class Agent:
    def __init__(
        self,
        config: SimpleChatbotConfig,
        indexer: Indexer,
        gate: ScopeGuardGate | None = None,
    ) -> None:
        self.config = config
        self.indexer = indexer
        self.gate = gate
        logger.bind(
            model=config.chat_model,
            max_tool_rounds=config.max_tool_rounds,
        ).info("Agent initialised")

    def _sampling_kwargs(self) -> dict:
        """Return only the sampling params the user set explicitly (skip None)."""
        # attr name on SimpleChatbotConfig  ->  key litellm/OpenAI expects on the wire
        field_map = {
            "temperature": "temperature",
            "top_p": "top_p",
            "gen_top_k": "top_k",
            "min_p": "min_p",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
            "repetition_penalty": "repetition_penalty",
        }
        return {
            wire: getattr(self.config, attr)
            for attr, wire in field_map.items()
            if getattr(self.config, attr) is not None
        }

    async def chat(self, messages: list[dict]) -> ChatResult:
        logger.bind(
            message_count=len(messages),
            last_role=messages[-1].get("role") if messages else "none",
        ).info("Chat started")

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
                )

        if self.indexer.document_count() == 0:
            logger.bind(document_count=0).warning("No indexed documents available")
            return ChatResult(content=EMPTY_KB_RESPONSE, retrieved_chunks=[])

        all_chunks: list[Document] = []
        working: list[dict] = []

        if self.config.system_prompt:
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
                "tools": [SEARCH_TOOL],
                "tool_choice": "auto",
            }
            if self.config.chat_api_base:
                kwargs["api_base"] = self.config.chat_api_base
                logger.bind(api_base=self.config.chat_api_base).debug("Using custom chat API base")
            kwargs.update(self._sampling_kwargs())

            logger.bind(
                model=self.config.chat_model,
                messages_in_context=len(working),
            ).info("Calling LLM")
            response = await litellm.acompletion(**kwargs)
            choice = response.choices[0]
            assistant_msg = choice.message

            usage = response.usage
            if usage:
                logger.bind(
                    prompt=usage.prompt_tokens,
                    completion=usage.completion_tokens,
                    total=usage.total_tokens,
                ).info("Token usage")

            last_content = assistant_msg.content or ""
            finish_reason = choice.finish_reason
            logger.bind(finish_reason=finish_reason).debug("LLM finish reason")

            if finish_reason == "tool_calls" and assistant_msg.tool_calls:
                round_log.bind(tool_call_count=len(assistant_msg.tool_calls)).info("Model requested tool calls")
                working.append(assistant_msg.model_dump())

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

                    query, error = _parse_search_tool_query(fn_name, raw_args)
                    if error:
                        tool_log.bind(error=error).warning("Invalid tool call")
                        context = _tool_error(error)
                    else:
                        tool_log.bind(query=query).debug("Searching index")
                        results = await self.indexer.search(query)
                        all_chunks.extend(results)

                        if results:
                            tool_log.bind(result_count=len(results)).info("Search returned results")
                            context = "\n\n".join(
                                f"[{d.metadata.get('source', 'unknown')}]\n{d.text}"
                                for d in results
                            )
                        else:
                            tool_log.warning("Search returned no results")
                            context = "No relevant documents found."

                    tool_log.bind(result_chars=len(context)).debug("Tool result prepared")
                    working.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": context,
                        }
                    )

                continue

            round_log.bind(response_chars=len(last_content)).info("Final answer produced")
            logger.bind(answer_preview=last_content[:120]).debug("Answer preview")
            return ChatResult(content=last_content, retrieved_chunks=all_chunks)

        logger.bind(max_tool_rounds=self.config.max_tool_rounds).warning(
            "Reached max tool rounds without final answer"
        )

        context_block = "\n\n".join(
            f"[{d.metadata.get('source', 'unknown')}]\n{d.text}" for d in all_chunks
        )

        system_content = (self.config.system_prompt or "") + "\n\n" + FINAL_ANSWER_INSTRUCTION
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
        try:
            raw = await litellm.acompletion(**forced_response_kwargs)
            choice = raw.choices[0]
            message = choice.message
            final_response = message.content or ""
            reasoning = getattr(message, "reasoning_content", None)
            logger.bind(
                finish_reason=choice.finish_reason,
                completion_tokens=raw.usage.completion_tokens if raw.usage else None,
                content_chars=len(final_response),
                reasoning_chars=len(reasoning) if reasoning else 0,
            ).info("Forced final response call returned")
        except Exception as exc:
            logger.bind(error=str(exc)).warning("Forced final response call failed")

        logger.bind(response_chars=len(final_response)).info("Forced final response produced")
        return ChatResult(content=final_response, retrieved_chunks=all_chunks)
