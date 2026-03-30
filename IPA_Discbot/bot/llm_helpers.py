import asyncio
import json
import os
import threading

import discord
import llm

from .config import MODEL
from .storage import get_provider_key, get_user_model


def _build_transcript(context_messages: list[dict]) -> str:
    lines: list[str] = []
    for m in context_messages:
        role = m.get("role", "").strip().lower()
        content = (m.get("content") or "").strip()

        if role == "assistant":
            prefix = "Assistant"
        elif role == "system":
            prefix = "System"
        else:
            prefix = "User"

        lines.append(f"{prefix}: {content}")

    lines.append("Assistant:")

    return "\n".join(lines)


def _conversation_system_prompt() -> str:
    return (
        "You are a helpful planning assistant inside Discord. "
        "Keep answers concise. "
        "You are given persisted conversation history for this user, and that history can include previous bot sessions. "
        "Use facts, variables, preferences, and prior commitments from the provided history when answering. "
        "If the needed information appears in the provided history, do not say you cannot access previous sessions or that memory was lost after a restart. "
        "Only say information is unavailable when it truly does not appear in the supplied conversation history."
    )


def _parse_llm_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(text[start : end + 1])


def _llm_reply_sync(model_id: str, context_messages: list[dict]) -> str:
    transcript = _build_transcript(context_messages)
    model = llm.get_model(model_id)
    response = model.prompt(transcript, system=_conversation_system_prompt())
    return response.text().strip()


LLM_ENV_LOCK = threading.Lock()

PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "gemini": "LLM_GEMINI_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
}


def _provider_from_model_id(model_id: str) -> str | None:
    mid = model_id.lower()
    if mid.startswith("ollama:") or mid.startswith("ollama/"):
        return "ollama"
    if "claude" in mid or mid.startswith("anthropic"):
        return "anthropic"
    if "gemini" in mid:
        return "gemini"
    return "openai"


def _run_llm_prompt_for_user_sync(
    user_id: str, model_id: str, prompt: str, system_prompt: str
) -> str:
    provider = _provider_from_model_id(model_id)
    env_key = PROVIDER_ENV.get(provider or "")

    user_key = None
    if env_key:
        user_key = get_provider_key(user_id, provider)
        if not user_key:
            raise RuntimeError(f"No API key set for {provider}. Use /setkey {provider} <key>.")

    with LLM_ENV_LOCK:
        old = os.environ.get(env_key) if env_key else None
        try:
            if env_key and user_key:
                os.environ[env_key] = user_key

            model = llm.get_model(model_id)
            response = model.prompt(prompt, system=system_prompt)
            return response.text().strip()
        finally:
            if env_key:
                if old is None:
                    os.environ.pop(env_key, None)
                else:
                    os.environ[env_key] = old


def _run_llm_for_user_sync(
    user_id: str, model_id: str, context_messages: list[dict]
) -> str:
    provider = _provider_from_model_id(model_id)
    env_key = PROVIDER_ENV.get(provider or "")

    transcript = _build_transcript(context_messages)
    system_prompt = _conversation_system_prompt()

    user_key = None
    if env_key:
        user_key = get_provider_key(user_id, provider)
        if not user_key:
            raise RuntimeError(f"No API key set for {provider}. Use /setkey {provider} <key>.")

    with LLM_ENV_LOCK:
        old = os.environ.get(env_key) if env_key else None
        try:
            if env_key and user_key:
                os.environ[env_key] = user_key

            model = llm.get_model(model_id)
            resp = model.prompt(transcript, system=system_prompt)
            return resp.text().strip()
        finally:
            if env_key:
                if old is None:
                    os.environ.pop(env_key, None)
                else:
                    os.environ[env_key] = old


async def llm_reply(
    model_id: str, context_messages: list[dict], user_id: str | None = None
) -> str:
    print("========== MODEL DEBUG ==========")
    print("User ID:", user_id)
    print("DB model:", get_user_model(user_id) if user_id else None)
    print("Passed model_id:", model_id)
    print("Fallback MODEL constant:", MODEL)
    print("=================================")

    if user_id is None:
        return await asyncio.to_thread(_llm_reply_sync, model_id, context_messages)
    return await asyncio.to_thread(
        _run_llm_for_user_sync, user_id, model_id, context_messages
    )


async def _llm_classify_confirmation_reply(message: discord.Message) -> str:
    selected_model = get_user_model(str(message.author.id)) or MODEL
    prompt = (
        "Classify the user's reply to a confirmation question.\n"
        "Return only JSON with this schema:\n"
        '{"reply_type":"confirm|reject|cancel|other"}\n'
        f"User reply: {json.dumps(message.content or '')}"
    )
    system_prompt = (
        "You classify short confirmation replies for a Discord bot. "
        "Infer intent from meaning, not fixed keywords. "
        "Return only valid JSON."
    )

    raw = await asyncio.to_thread(
        _run_llm_prompt_for_user_sync,
        str(message.author.id),
        selected_model,
        prompt,
        system_prompt,
    )
    data = _parse_llm_json_object(raw)
    reply_type = str(data.get("reply_type", "other")).strip().lower()
    if reply_type not in {"confirm", "reject", "cancel", "other"}:
        return "other"
    return reply_type


async def _llm_classify_member_request(message: discord.Message) -> dict:
    selected_model = get_user_model(str(message.author.id)) or MODEL
    mentioned_users = [
        {
            "id": user.id,
            "name": getattr(user, "display_name", None) or user.name,
        }
        for user in message.mentions
    ]
    prompt = (
        "Decide whether this Discord message is asking the bot to mention a server member or add mentioned users to a thread.\n"
        "Return only JSON with this schema:\n"
        '{"intent":"member_lookup|thread_add_mentions|none","requested_name":"string","should_add_to_thread":true}\n'
        f"Is thread: {json.dumps(isinstance(message.channel, discord.Thread))}\n"
        f"Mentioned users: {json.dumps(mentioned_users)}\n"
        f"Message: {json.dumps(message.content or '')}"
    )
    system_prompt = (
        "You classify Discord bot control requests. "
        "Infer intent from the user's wording, even when synonyms or unusual phrasing are used. "
        "Use thread_add_mentions only when the message is asking to add already-mentioned users to the current thread. "
        "Use member_lookup when the user refers to someone by name and the bot should resolve and mention them. "
        "Set should_add_to_thread true when the user wants the matched person brought into the current thread. "
        "Return only valid JSON."
    )

    raw = await asyncio.to_thread(
        _run_llm_prompt_for_user_sync,
        str(message.author.id),
        selected_model,
        prompt,
        system_prompt,
    )
    data = _parse_llm_json_object(raw)
    intent = str(data.get("intent", "none")).strip().lower()
    if intent not in {"member_lookup", "thread_add_mentions", "none"}:
        intent = "none"

    requested_name = str(data.get("requested_name", "") or "").strip()
    should_add_to_thread = bool(data.get("should_add_to_thread", False))
    return {
        "intent": intent,
        "requested_name": requested_name,
        "should_add_to_thread": should_add_to_thread,
    }


def _all_llm_model_ids() -> list[str]:
    return sorted([m.model_id for m in llm.get_models()])
