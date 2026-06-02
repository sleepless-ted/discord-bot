import asyncio
import logging
import os
import re
from dataclasses import dataclass

import aiohttp
import discord
from discord.errors import LoginFailure
from dotenv import load_dotenv

from babouin_bot import (
    DISCORD_REPLY_LIMIT,
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_REPEAT_PENALTY,
    OLLAMA_TEMPERATURE,
    OLLAMA_THINK,
    OLLAMA_TOP_K,
    OLLAMA_TOP_P,
    OLLAMA_URL,
    OPENAI_MODEL,
    PROMPT_CACHE_KEY,
    PROMPT_CACHE_RETENTION,
    read_optional_text_file,
    recover_answer_from_thinking,
    replace_custom_emoji_names,
    split_discord_message,
)


load_dotenv(".env")


def clean_discord_token(value: str | None) -> str:
    if not value:
        return ""

    token = value.strip()
    if token.lower().startswith("bot "):
        token = token[4:].strip()
    return token


DISCORD_SUMMARY_TOKEN = clean_discord_token(os.getenv("DISCORD_SUMMARY_TOKEN"))
SUMMARY_COMMAND_PREFIX = os.getenv("SUMMARY_COMMAND_PREFIX", "!resume").strip()
SUMMARY_CONTEXT_MESSAGE_LIMIT = int(os.getenv("SUMMARY_CONTEXT_MESSAGE_LIMIT", "80"))
SUMMARY_MAX_HISTORY_LIMIT = int(os.getenv("SUMMARY_MAX_HISTORY_LIMIT", "200"))
SUMMARY_MAX_OUTPUT_TOKENS = int(os.getenv("SUMMARY_MAX_OUTPUT_TOKENS", "900"))
SUMMARY_INCLUDE_BOTS = os.getenv("SUMMARY_INCLUDE_BOTS", "true").lower() in {"1", "true", "yes", "on"}
SUMMARY_SYSTEM_PROMPT_FILE = os.getenv("SUMMARY_SYSTEM_PROMPT_FILE", "")

BASE_SUMMARY_SYSTEM_PROMPT = """Tu es un bot Discord specialise dans les resumes de conversation.
Tu fais une synthese fidele, neutre et utile.
Tu es cru, honnete et direct. Tu observes la scene avec un certain recul et détachement, ce qui la rends absurde.
Tu n'hésites pas à attaquer, rabaisser, ridiculiser ou faire des blagues sur les personnages, mais toujours de manière pertinente et en lien avec ce qui a été dit.
Tu n'inventes jamais de messages, d'intentions ou de positions absentes du contexte.
Cites des messages pour appuyer les propos ou l'avis de cetaines personnes
Objectif:
- resumer les idees principales;
- attribuer les points de vue aux bonnes personnes;
- distinguer faits, opinions, blagues et incertitudes;
- relever les accords, desaccords et questions ouvertes;
- rester clair, court et lisible sur Discord.

Format de reponse:
**Resume**
2 a 5 phrases maximum.

**Points de vue**
- Nom: position ou reaction principale.

**Accords / desaccords**
Liste courte

**A suivre**
Questions ouvertes, decisions, actions possibles
"""


@dataclass(frozen=True)
class ConversationLine:
    author: str
    content: str


def build_summary_system_prompt() -> str:
    extra_prompt = read_optional_text_file(SUMMARY_SYSTEM_PROMPT_FILE)
    if not extra_prompt:
        return BASE_SUMMARY_SYSTEM_PROMPT

    return f"""{BASE_SUMMARY_SYSTEM_PROMPT}

Instructions supplementaires:
{extra_prompt}"""


SUMMARY_SYSTEM_PROMPT = build_summary_system_prompt()


def require_env() -> None:
    missing = []
    if not DISCORD_SUMMARY_TOKEN:
        missing.append("DISCORD_SUMMARY_TOKEN")
    if LLM_PROVIDER == "openai" and not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if LLM_PROVIDER not in {"ollama", "openai"}:
        missing.append("LLM_PROVIDER=ollama ou LLM_PROVIDER=openai")

    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Variables d'environnement manquantes: {names}")


def describe_history_message(message: discord.Message) -> ConversationLine:
    author = message.author.display_name
    if message.author.bot:
        author = f"{author} (bot)"

    parts = [message.clean_content.strip()]
    if message.attachments:
        attachment_names = ", ".join(attachment.filename for attachment in message.attachments)
        parts.append(f"[pieces jointes: {attachment_names}]")

    content = " ".join(part for part in parts if part).strip()
    return ConversationLine(author=author, content=content or "[message sans texte]")


def clean_summary_request(message: discord.Message) -> str:
    content = message.content.strip()

    prefix_pattern = re.compile(rf"^\s*{re.escape(SUMMARY_COMMAND_PREFIX)}\b", re.IGNORECASE)
    content = prefix_pattern.sub("", content).strip()

    return content or "Fais le resume de la conversation recente."


def requested_message_limit(request: str) -> int:
    match = re.search(r"\b(\d+)\s*(?:derniers?\s*)?messages?\b", request, re.IGNORECASE)
    if not match:
        match = re.search(r"^\s*(\d+)\b", request)
    if not match:
        logging.info("Aucun nombre de messages demande, utilisation de %s messages", SUMMARY_CONTEXT_MESSAGE_LIMIT)
        return SUMMARY_CONTEXT_MESSAGE_LIMIT

    requested = int(match.group(1))
    limit = max(5, min(requested, SUMMARY_MAX_HISTORY_LIMIT))
    logging.info("Utilisateur a demande un resume de %s messages, limite appliquee: %s", requested, limit)
    return limit


async def get_conversation_messages(
    message: discord.Message,
    bot_user: discord.ClientUser,
    limit: int,
) -> list[ConversationLine]:
    history = []
    async for history_message in message.channel.history(
        limit=limit * 3,
        before=message,
        oldest_first=False,
    ):
        if history_message.author.id == bot_user.id:
            continue
        if history_message.author.bot and not SUMMARY_INCLUDE_BOTS:
            continue

        history.append(describe_history_message(history_message))
        if len(history) >= limit:
            break

    history.reverse()
    return history


def build_summary_input(conversation: list[ConversationLine], request: str) -> str:
    if conversation:
        conversation_block = "\n".join(f"{line.author}: {line.content}" for line in conversation)
    else:
        conversation_block = "Aucun message precedent disponible."

    return f"""Conversation Discord a resumer, du plus ancien au plus recent:
{conversation_block}

Demande de l'utilisateur:
{request}

Resume uniquement ce qui apparait dans la conversation.
Quand plusieurs personnes parlent, preserve leurs points de vue respectifs.
Si le contexte est trop mince, dis-le simplement."""


async def ask_ollama_summary(conversation: list[ConversationLine], request: str) -> str:
    endpoint = f"{OLLAMA_URL.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": OLLAMA_THINK,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": build_summary_input(conversation, request)},
        ],
        "options": {
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": SUMMARY_MAX_OUTPUT_TOKENS,
            "temperature": min(OLLAMA_TEMPERATURE, 0.4),
            "top_p": OLLAMA_TOP_P,
            "top_k": OLLAMA_TOP_K,
            "repeat_penalty": OLLAMA_REPEAT_PENALTY,
            "repeat_last_n": 256,
        },
    }

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(endpoint, json=payload) as response:
            response.raise_for_status()
            data = await response.json()

    message = data.get("message", {})
    content = message.get("content", "").strip()
    if not content:
        content = recover_answer_from_thinking(message.get("thinking", ""))
    if not content:
        logging.warning("Ollama a renvoye une reponse vide: %s", data)
    return content


async def ask_openai_summary(conversation: list[ConversationLine], request: str) -> str:
    from openai import AsyncOpenAI

    extra_body = {}
    if PROMPT_CACHE_KEY:
        extra_body["prompt_cache_key"] = f"{PROMPT_CACHE_KEY}-summary"
    if PROMPT_CACHE_RETENTION:
        extra_body["prompt_cache_retention"] = PROMPT_CACHE_RETENTION

    client = AsyncOpenAI()
    response = await client.responses.create(
        model=OPENAI_MODEL,
        instructions=SUMMARY_SYSTEM_PROMPT,
        input=build_summary_input(conversation, request),
        max_output_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
        extra_body=extra_body or None,
    )
    await client.close()
    return response.output_text.strip()


async def ask_summary_llm(conversation: list[ConversationLine], request: str) -> str:
    if LLM_PROVIDER == "openai":
        return await ask_openai_summary(conversation, request)

    return await ask_ollama_summary(conversation, request)


async def send_summary(message: discord.Message, answer: str) -> None:
    answer = replace_custom_emoji_names(answer, message.guild)
    chunks = split_discord_message(answer, DISCORD_REPLY_LIMIT)
    first_chunk, *other_chunks = chunks

    await message.reply(first_chunk, mention_author=False)
    for chunk in other_chunks:
        await message.channel.send(chunk)


def is_summary_request(message: discord.Message) -> bool:
    if not SUMMARY_COMMAND_PREFIX:
        return False

    prefix_pattern = re.compile(rf"^\s*{re.escape(SUMMARY_COMMAND_PREFIX)}\b", re.IGNORECASE)
    return bool(prefix_pattern.search(message.content))


def create_summary_bot() -> discord.Client:
    intents = discord.Intents.default()
    intents.message_content = True

    bot = discord.Client(intents=intents)

    @bot.event
    async def on_ready() -> None:
        logging.info("Bot resume connecte en tant que %s (%s)", bot.user, bot.user.id if bot.user else "id inconnu")
        logging.info(
            "Commande resume: prefixe %s uniquement, contexte par defaut=%s messages",
            SUMMARY_COMMAND_PREFIX,
            SUMMARY_CONTEXT_MESSAGE_LIMIT,
        )

    @bot.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot or not bot.user:
            return

        if not is_summary_request(message):
            return

        request = clean_summary_request(message)
        history_limit = requested_message_limit(request)

        async with message.channel.typing():
            try:
                conversation = await get_conversation_messages(message, bot.user, history_limit)
                answer = await ask_summary_llm(conversation, request)
                if not answer:
                    answer = "Je n'ai pas reussi a produire un resume pour le moment."
            except Exception:
                logging.exception("Erreur pendant la generation du resume")
                await message.reply(
                    "Je n'arrive pas a generer le resume pour le moment. Regarde les logs du bot pour le detail.",
                    mention_author=False,
                )
                return

        await send_summary(message, answer)

    return bot


async def main() -> None:
    require_env()

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if SUMMARY_SYSTEM_PROMPT_FILE and read_optional_text_file(SUMMARY_SYSTEM_PROMPT_FILE):
        logging.info("Prompt supplementaire resume charge depuis %s", SUMMARY_SYSTEM_PROMPT_FILE)
    if LLM_PROVIDER == "ollama":
        logging.info(
            "LLM local Ollama pour resume: %s (%s, num_ctx=%s, sortie=%s tokens)",
            OLLAMA_MODEL,
            OLLAMA_URL,
            OLLAMA_NUM_CTX,
            SUMMARY_MAX_OUTPUT_TOKENS,
        )
    else:
        logging.info("LLM OpenAI pour resume: %s", OPENAI_MODEL)

    bot = create_summary_bot()
    try:
        await bot.start(DISCORD_SUMMARY_TOKEN)
    except LoginFailure as exc:
        raise RuntimeError(
            "Discord refuse DISCORD_SUMMARY_TOKEN. Copie le token du deuxieme bot "
            "dans le Developer Portal > Bot > Token, sans guillemets, sans 'Bot ' "
            "devant, et sans espaces autour."
        ) from exc


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
