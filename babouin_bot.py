import asyncio
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import discord
from dotenv import load_dotenv

from llm_backend import chat_gemini, chat_ollama, normalize_provider


load_dotenv(".env")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
LLM_PROVIDER = normalize_provider(
    os.getenv("BABOUIN_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "ollama"))
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")#gemini-3.1-flash-lite
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "120"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:26b")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "700"))
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false").lower() in {"1", "true", "yes", "on"}
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.95"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.9"))
OLLAMA_TOP_K = int(os.getenv("OLLAMA_TOP_K", "80"))
OLLAMA_REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.18"))
CONTEXT_MESSAGE_LIMIT = int(os.getenv("CONTEXT_MESSAGE_LIMIT", "10"))
DISCORD_REPLY_LIMIT = 1900
STYLE_PROMPT_FILE = os.getenv("STYLE_PROMPT_FILE", "lipa_style_system_prompt.txt")
CUSTOM_EMOJI_RE = re.compile(r"(?<!<):([A-Za-z0-9_~]+):(?!\d*>|//)")

BASE_SYSTEM_PROMPT = """Tu es un assistant Discord utile, naturel et concis.
Tu reponds en francais par defaut.
Tu dois tenir compte du contexte recent du salon, sans inventer ce qui n'y est pas.
"""


@dataclass(frozen=True)
class ContextMessage:
    author: str
    content: str


@dataclass(frozen=True)
class RecentMessages:
    context: list[ContextMessage]
    bot_replies: list[str]


def read_optional_text_file(path_value: str) -> str:
    if not path_value:
        return ""

    path = Path(path_value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    if not path.exists():
        return ""

    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue

    return path.read_text().strip()


def build_system_prompt() -> str:
    style_prompt = read_optional_text_file(STYLE_PROMPT_FILE)
    if not style_prompt:
        return BASE_SYSTEM_PROMPT

    return f"""{BASE_SYSTEM_PROMPT}

Guide de style a suivre:
{style_prompt}"""


SYSTEM_PROMPT = build_system_prompt()


def require_env() -> None:
    missing = []
    if not DISCORD_TOKEN:
        missing.append("DISCORD_TOKEN")
    if LLM_PROVIDER == "openai" and not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY ou GOOGLE_API_KEY")
    if LLM_PROVIDER not in {"ollama", "openai", "gemini"}:
        missing.append("BABOUIN_LLM_PROVIDER=ollama, openai ou gemini")

    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Variables d'environnement manquantes: {names}")


def clean_user_prompt(message: discord.Message, bot_user: discord.ClientUser) -> str:
    content = message.content
    content = content.replace(bot_user.mention, "")
    content = content.replace(f"<@!{bot_user.id}>", "")
    content = content.strip()
    return content or "Reponds au contexte de la conversation."


def describe_message(message: discord.Message) -> ContextMessage:
    parts = [message.clean_content.strip()]

    if message.attachments:
        attachment_names = ", ".join(attachment.filename for attachment in message.attachments)
        parts.append(f"[pieces jointes: {attachment_names}]")

    content = " ".join(part for part in parts if part).strip()
    return ContextMessage(
        author=message.author.display_name,
        content=content or "[message sans texte]",
    )


async def get_recent_messages(message: discord.Message, bot_user: discord.ClientUser) -> RecentMessages:
    history = []
    bot_replies = []
    async for history_message in message.channel.history(
        limit=CONTEXT_MESSAGE_LIMIT * 4,
        before=message,
        oldest_first=False,
    ):
        if history_message.author.id == bot_user.id:
            if len(bot_replies) < 5:
                bot_replies.append(history_message.clean_content.strip())
            continue

        history.append(describe_message(history_message))
        if len(history) >= CONTEXT_MESSAGE_LIMIT:
            break

    history.reverse()
    bot_replies.reverse()
    return RecentMessages(context=history, bot_replies=bot_replies)


def build_llm_input(recent_messages: RecentMessages, current_prompt: str) -> str:
    if recent_messages.context:
        context_block = "\n".join(
            f"{item.author}: {item.content}"
            for item in recent_messages.context
        )
    else:
        context_block = "Aucun message precedent disponible."

    if recent_messages.bot_replies:
        bot_replies_block = "\n".join(f"- {reply}" for reply in recent_messages.bot_replies if reply)
    else:
        bot_replies_block = "Aucune."

    return f"""Contexte recent du salon Discord, du plus ancien au plus recent:
{context_block}

Tes reponses recentes a ne pas repeter:
{bot_replies_block}

Message qui te mentionne:
{current_prompt}

Reponds d'abord directement au message qui te mentionne, puis ajoute une vanne courte si ca colle.
Ne commence par "oui" ou "non" que si la question est vraiment fermee.
Une question ouverte commence souvent par "quel", "quelle", "qui", "quoi", "ou", "quand", "comment", "pourquoi", "combien", ou demande une preference.
Pour une question ouverte, donne directement un nom, un choix, une explication courte ou une esquive drole, mais pas "oui" ou "non".
Si le guide de style donne une preference connue, utilise-la au lieu d'esquiver.
Exemples: "ta chanteuse preferee ?" -> "Dua Lipa"; "tu vas voter quel president ?" -> "probablement le camp Zemmour/Knafo"; "ton jeu prefere ?" -> "Pokemon HeartGold".
Ne repete pas une phrase que tu as deja utilisee dans tes reponses recentes."""


def split_discord_message(text: str, limit: int = DISCORD_REPLY_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at == -1:
            split_at = limit

        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def recover_answer_from_thinking(thinking: str) -> str:
    if not thinking:
        return ""

    quoted_lines = re.findall(r'"([^"\n]{2,180})"', thinking)
    candidates = [
        line.strip()
        for line in quoted_lines
        if not line.lower().startswith(("option ", "draft ", "let's", "self-correction"))
    ]
    if candidates:
        return "\n".join(candidates[-2:])

    lines = [
        line.strip(" \t-*")
        for line in thinking.splitlines()
        if line.strip()
    ]
    short_lines = [
        line
        for line in lines
        if 2 <= len(line) <= 180
        and not line.lower().startswith(("the ", "he ", "if ", "or ", "wait", "actually", "maybe", "let's", "looking", "avoid"))
    ]
    return "\n".join(short_lines[-2:]).strip()


async def ask_ollama(recent_messages: RecentMessages, current_prompt: str) -> str:
    response = await chat_ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        think=OLLAMA_THINK,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_llm_input(recent_messages, current_prompt)},
        ],
        options={
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "top_k": OLLAMA_TOP_K,
            "repeat_penalty": OLLAMA_REPEAT_PENALTY,
            "repeat_last_n": 256,
        },
    )

    content = response.text
    if not content:
        message = response.raw.get("message", {})
        content = recover_answer_from_thinking(message.get("thinking", ""))
    if not content:
        logging.warning("Ollama a renvoye une reponse vide: %s", response.raw)
    return content


async def ask_gemini(recent_messages: RecentMessages, current_prompt: str) -> str:
    response = await chat_gemini(
        model=GEMINI_MODEL,
        api_key=GEMINI_API_KEY,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_llm_input(recent_messages, current_prompt)},
        ],
        temperature=TEMPERATURE,
        max_tokens=NUM_PREDICT,
        timeout=GEMINI_TIMEOUT,
    )
    if not response.text:
        logging.warning("Gemini a renvoye une reponse vide")
    return response.text


async def ask_openai(recent_messages: RecentMessages, current_prompt: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    response = await client.responses.create(
        model=OPENAI_MODEL,
        instructions=SYSTEM_PROMPT,
        input=build_llm_input(recent_messages, current_prompt),
        max_output_tokens=700,
    )
    await client.close()
    return response.output_text.strip()


async def ask_llm(recent_messages: RecentMessages, current_prompt: str) -> str:
    if LLM_PROVIDER == "openai":
        return await ask_openai(recent_messages, current_prompt)
    if LLM_PROVIDER == "gemini":
        return await ask_gemini(recent_messages, current_prompt)

    return await ask_ollama(recent_messages, current_prompt)


async def send_reply(message: discord.Message, answer: str) -> None:
    answer = replace_custom_emoji_names(answer, message.guild)
    chunks = split_discord_message(answer)
    first_chunk, *other_chunks = chunks

    await message.reply(first_chunk, mention_author=False)
    for chunk in other_chunks:
        await message.channel.send(chunk)


def build_custom_emoji_map(guild: discord.Guild | None) -> dict[str, str]:
    if not guild:
        return {}

    return {
        emoji.name.lower(): str(emoji)
        for emoji in guild.emojis
        if emoji.is_usable()
    }


def replace_custom_emoji_names(text: str, guild: discord.Guild | None) -> str:
    emoji_map = build_custom_emoji_map(guild)
    if not emoji_map:
        return text

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return emoji_map.get(name.lower(), match.group(0))

    return CUSTOM_EMOJI_RE.sub(replace, text)


def create_bot() -> discord.Client:
    intents = discord.Intents.default()
    intents.message_content = True

    bot = discord.Client(intents=intents)

    @bot.event
    async def on_ready() -> None:
        logging.info("Connecte en tant que %s (%s)", bot.user, bot.user.id if bot.user else "id inconnu")
        emoji_count = sum(len(guild.emojis) for guild in bot.guilds)
        logging.info("Emojis custom visibles: %s", emoji_count)

    @bot.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot or not bot.user:
            return

        if bot.user not in message.mentions:
            return

        current_prompt = clean_user_prompt(message, bot.user)

        async with message.channel.typing():
            try:
                recent_messages = await get_recent_messages(message, bot.user)
                answer = await ask_llm(recent_messages, current_prompt)
                if not answer:
                    answer = "j'ai rien reussi a sortir la"
            except Exception:
                logging.exception("Erreur pendant la generation de la reponse")
                await message.reply(
                    "Je n'arrive pas a generer une reponse pour le moment. Regarde les logs du bot pour le detail.",
                    mention_author=False,
                )
                return

        await send_reply(message, answer)

    return bot


async def main() -> None:
    require_env()

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if read_optional_text_file(STYLE_PROMPT_FILE):
        logging.info("Guide de style charge depuis %s", STYLE_PROMPT_FILE)
    else:
        logging.info("Aucun guide de style charge")
    if LLM_PROVIDER == "ollama":
        logging.info(
            "LLM local Ollama: %s (%s, num_ctx=%s, think=%s, temperature=%s, repeat_penalty=%s)",
            OLLAMA_MODEL,
            OLLAMA_URL,
            OLLAMA_NUM_CTX,
            OLLAMA_THINK,
            TEMPERATURE,
            OLLAMA_REPEAT_PENALTY,
        )
    elif LLM_PROVIDER == "openai":
        logging.info("LLM OpenAI: %s", OPENAI_MODEL)
    else:
        logging.info("LLM Gemini via google-genai: %s", GEMINI_MODEL)

    bot = create_bot()
    await bot.start(DISCORD_TOKEN)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
