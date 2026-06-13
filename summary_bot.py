import asyncio
import base64
import logging
import mimetypes
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import aiohttp
import discord
from discord.errors import LoginFailure
from dotenv import load_dotenv

from llm_backend import LLMBackendError, chat_gemini, normalize_provider

from babouin_bot import (
    DISCORD_REPLY_LIMIT,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_TIMEOUT,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_REPEAT_PENALTY,
    OLLAMA_THINK,
    OLLAMA_TOP_K,
    OLLAMA_TOP_P,
    OLLAMA_URL,
    OPENAI_MODEL,
    TEMPERATURE,
    read_optional_text_file,
    recover_answer_from_thinking,
    replace_custom_emoji_names,
    split_discord_message,
)
from llm_backend import chat_gemini, normalize_provider


load_dotenv(".env")

SUMMARY_LLM_PROVIDER = normalize_provider(
    os.getenv("SUMMARY_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "ollama"))
)


def clean_discord_token(value: str | None) -> str:
    if not value:
        return ""

    token = value.strip()
    if token.lower().startswith("bot "):
        token = token[4:].strip()
    return token


DISCORD_SUMMARY_TOKEN = clean_discord_token(os.getenv("DISCORD_SUMMARY_TOKEN"))
SUMMARY_COMMAND_PREFIX = os.getenv("SUMMARY_COMMAND_PREFIX", "!resumix").strip()
SUMMARY_CONTEXT_MESSAGE_DEFAULT = int(
    os.getenv("SUMMARY_CONTEXT_MESSAGE_DEFAULT", os.getenv("SUMMARY_CONTEXT_MESSAGE_LIMIT", "80"))
)
SUMMARY_CONTEXT_MESSAGE_MAX = int(
    os.getenv("SUMMARY_CONTEXT_MESSAGE_MAX", os.getenv("SUMMARY_MAX_HISTORY_LIMIT", "200"))
)
SUMMARY_MAX_OUTPUT_TOKENS = int(os.getenv("SUMMARY_MAX_OUTPUT_TOKENS", "900"))
SUMMARY_INCLUDE_BOTS = os.getenv("SUMMARY_INCLUDE_BOTS", "true").strip().lower() in {"1", "true", "yes", "on"}
SUMMARY_SYSTEM_PROMPT_FILE = os.getenv("SUMMARY_SYSTEM_PROMPT_FILE", "")
SUMMARY_READ_AUDIO_ATTACHMENTS = os.getenv("SUMMARY_READ_AUDIO_ATTACHMENTS", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SUMMARY_MAX_AUDIO_ATTACHMENTS = int(os.getenv("SUMMARY_MAX_AUDIO_ATTACHMENTS", "4"))
SUMMARY_MAX_AUDIO_BYTES = int(os.getenv("SUMMARY_MAX_AUDIO_BYTES", "10000000"))
SUMMARY_FALLBACK_WITHOUT_AUDIO = os.getenv("SUMMARY_FALLBACK_WITHOUT_AUDIO", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SUMMARY_CONVERT_AUDIO_TO_WAV = os.getenv("SUMMARY_CONVERT_AUDIO_TO_WAV", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SUMMARY_TRANSCRIBE_AUDIO_FIRST = os.getenv("SUMMARY_TRANSCRIBE_AUDIO_FIRST", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SUMMARY_AUDIO_TRANSCRIPTION_TOKENS = int(os.getenv("SUMMARY_AUDIO_TRANSCRIPTION_TOKENS", "500"))
SUMMARY_SILENT_MODE = os.getenv("SUMMARY_SILENT_MODE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SUMMARY_PRINT_AUDIO_TRANSCRIPTS = os.getenv("SUMMARY_PRINT_AUDIO_TRANSCRIPTS", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SUMMARY_PRINT_AUDIO_TRANSCRIPT_LIMIT = int(os.getenv("SUMMARY_PRINT_AUDIO_TRANSCRIPT_LIMIT", "4000"))
SUMMARY_PRINT_FINAL_PROMPT = os.getenv("SUMMARY_PRINT_FINAL_PROMPT", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SUMMARY_READ_IMAGE_ATTACHMENTS = os.getenv("SUMMARY_READ_IMAGE_ATTACHMENTS", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SUMMARY_DESCRIBE_IMAGES_FIRST = os.getenv("SUMMARY_DESCRIBE_IMAGES_FIRST", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SUMMARY_MAX_IMAGE_ATTACHMENTS = int(os.getenv("SUMMARY_MAX_IMAGE_ATTACHMENTS", "6"))
SUMMARY_MAX_IMAGE_BYTES = int(os.getenv("SUMMARY_MAX_IMAGE_BYTES", "8000000"))
SUMMARY_IMAGE_DESCRIPTION_TOKENS = int(os.getenv("SUMMARY_IMAGE_DESCRIPTION_TOKENS", "350"))
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg").strip() or "ffmpeg"

AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}

IMAGE_EXTENSIONS = {
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
}

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
class AudioAttachment:
    filename: str
    data_base64: str


@dataclass(frozen=True)
class ImageAttachment:
    filename: str
    data_base64: str
    media_type: str


@dataclass(frozen=True)
class ConversationLine:
    author: str
    content: str
    audio_attachments: tuple[AudioAttachment, ...] = ()
    image_attachments: tuple[ImageAttachment, ...] = ()


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
    if SUMMARY_LLM_PROVIDER == "openai" and not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if SUMMARY_LLM_PROVIDER not in {"ollama", "openai", "gemini"}:
        missing.append("SUMMARY_LLM_PROVIDER=ollama ou openai")

    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Variables d'environnement manquantes: {names}")


def is_audio_attachment(attachment: discord.Attachment) -> bool:
    content_type = (attachment.content_type or "").lower()
    if content_type.startswith("audio/"):
        return True

    filename = attachment.filename.lower()
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def is_image_attachment(attachment: discord.Attachment) -> bool:
    content_type = (attachment.content_type or "").lower()
    if content_type.startswith("image/"):
        return True

    filename = attachment.filename.lower()
    return any(filename.endswith(extension) for extension in IMAGE_EXTENSIONS)


async def read_audio_attachment(attachment: discord.Attachment) -> AudioAttachment | None:
    if attachment.size and attachment.size > SUMMARY_MAX_AUDIO_BYTES:
        logging.info(
            "Audio ignore car trop gros: %s (%s octets, max=%s)",
            attachment.filename,
            attachment.size,
            SUMMARY_MAX_AUDIO_BYTES,
        )
        return None

    try:
        data = await attachment.read(use_cached=True)
    except Exception:
        logging.exception("Impossible de telecharger la piece jointe audio %s", attachment.filename)
        return None

    if len(data) > SUMMARY_MAX_AUDIO_BYTES:
        logging.info(
            "Audio ignore apres telechargement car trop gros: %s (%s octets, max=%s)",
            attachment.filename,
            len(data),
            SUMMARY_MAX_AUDIO_BYTES,
        )
        return None

    data = await prepare_audio_for_ollama(attachment.filename, data)
    if not data:
        return None

    return AudioAttachment(
        filename=attachment.filename,
        data_base64=base64.b64encode(data).decode("ascii"),
    )


async def read_image_attachment(attachment: discord.Attachment) -> ImageAttachment | None:
    if attachment.size and attachment.size > SUMMARY_MAX_IMAGE_BYTES:
        logging.info(
            "Image ignoree car trop grosse: %s (%s octets, max=%s)",
            attachment.filename,
            attachment.size,
            SUMMARY_MAX_IMAGE_BYTES,
        )
        return None

    try:
        data = await attachment.read(use_cached=True)
    except Exception:
        logging.exception("Impossible de telecharger la piece jointe image %s", attachment.filename)
        return None

    if len(data) > SUMMARY_MAX_IMAGE_BYTES:
        logging.info(
            "Image ignoree apres telechargement car trop grosse: %s (%s octets, max=%s)",
            attachment.filename,
            len(data),
            SUMMARY_MAX_IMAGE_BYTES,
        )
        return None

    media_type = attachment.content_type or mimetypes.guess_type(attachment.filename)[0] or "image/png"
    return ImageAttachment(
        filename=attachment.filename,
        data_base64=base64.b64encode(data).decode("ascii"),
        media_type=media_type,
    )


def is_wav_file(filename: str) -> bool:
    return filename.lower().endswith(".wav")


async def prepare_audio_for_ollama(filename: str, data: bytes) -> bytes | None:
    if not SUMMARY_CONVERT_AUDIO_TO_WAV:
        return data

    if is_wav_file(filename):
        return data

    ffmpeg_path = shutil.which(FFMPEG_PATH) or FFMPEG_PATH
    if not shutil.which(ffmpeg_path) and not Path(ffmpeg_path).exists():
        logging.warning(
            "Audio %s ignore: Ollama/Gemma lit surtout le WAV ici, mais ffmpeg est introuvable. "
            "Installe ffmpeg ou configure FFMPEG_PATH.",
            filename,
        )
        return None

    suffix = Path(filename).suffix or ".audio"
    with tempfile.TemporaryDirectory(prefix="summary-audio-") as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        output_path = Path(temp_dir) / "output.wav"
        input_path.write_bytes(data)

        process = await asyncio.create_subprocess_exec(
            ffmpeg_path,
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(output_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            logging.warning(
                "Conversion audio impossible pour %s avec ffmpeg: %s",
                filename,
                (stderr or stdout).decode("utf-8", errors="replace")[-1000:],
            )
            return None

        wav_data = output_path.read_bytes()
        if len(wav_data) > SUMMARY_MAX_AUDIO_BYTES:
            logging.info(
                "Audio converti ignore car trop gros: %s (%s octets, max=%s)",
                filename,
                len(wav_data),
                SUMMARY_MAX_AUDIO_BYTES,
            )
            return None

        logging.info("Audio converti en WAV pour Ollama: %s", filename)
        return wav_data


async def describe_history_message(
    message: discord.Message,
    remaining_audio_slots: int,
    remaining_image_slots: int,
) -> ConversationLine:
    author = message.author.display_name
    if message.author.bot:
        author = f"{author} (bot)"

    parts = [message.clean_content.strip()]
    audio_attachments = []
    image_attachments = []
    if message.attachments:
        attachment_names = ", ".join(attachment.filename for attachment in message.attachments)
        parts.append(f"[pieces jointes: {attachment_names}]")
        if SUMMARY_READ_AUDIO_ATTACHMENTS and remaining_audio_slots > 0:
            for attachment in message.attachments:
                if len(audio_attachments) >= remaining_audio_slots:
                    break
                if not is_audio_attachment(attachment):
                    continue

                audio_attachment = await read_audio_attachment(attachment)
                if not audio_attachment:
                    continue

                audio_attachments.append(audio_attachment)
                parts.append(f"[audio envoye au modele: {attachment.filename}]")
        if SUMMARY_READ_IMAGE_ATTACHMENTS and remaining_image_slots > 0:
            for attachment in message.attachments:
                if len(image_attachments) >= remaining_image_slots:
                    break
                if not is_image_attachment(attachment):
                    continue

                image_attachment = await read_image_attachment(attachment)
                if not image_attachment:
                    continue

                image_attachments.append(image_attachment)
                parts.append(f"[image envoyee au modele: {attachment.filename}]")

    content = " ".join(part for part in parts if part).strip()
    return ConversationLine(
        author=author,
        content=content or "[message sans texte]",
        audio_attachments=tuple(audio_attachments),
        image_attachments=tuple(image_attachments),
    )


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
        logging.info("Aucun nombre de messages demande, utilisation de %s messages", SUMMARY_CONTEXT_MESSAGE_DEFAULT)
        return SUMMARY_CONTEXT_MESSAGE_DEFAULT

    requested = int(match.group(1))
    limit = max(5, min(requested, SUMMARY_CONTEXT_MESSAGE_MAX))
    logging.info("Utilisateur a demande un resume de %s messages, limite appliquee: %s", requested, limit)
    return limit


async def get_conversation_messages(
    message: discord.Message,
    bot_user: discord.ClientUser,
    limit: int,
) -> list[ConversationLine]:
    history = []
    audio_count = 0
    image_count = 0
    async for history_message in message.channel.history(
        limit=limit * 3,
        before=message,
        oldest_first=False,
    ):
        if history_message.author.id == bot_user.id:
            continue
        if history_message.author.bot and not SUMMARY_INCLUDE_BOTS:
            continue

        remaining_audio_slots = max(0, SUMMARY_MAX_AUDIO_ATTACHMENTS - audio_count)
        remaining_image_slots = max(0, SUMMARY_MAX_IMAGE_ATTACHMENTS - image_count)
        line = await describe_history_message(history_message, remaining_audio_slots, remaining_image_slots)
        audio_count += len(line.audio_attachments)
        image_count += len(line.image_attachments)
        history.append(line)
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


def print_final_summary_prompt(summary_input: str) -> None:
    if SUMMARY_SILENT_MODE or not SUMMARY_PRINT_FINAL_PROMPT:
        return

    print(
        "\n--- Prompt final envoye au LLM de resume ---\n"
        f"{summary_input}\n"
        "--- Fin prompt final envoye au LLM de resume ---\n",
        flush=True,
    )


def collect_audio_attachments(conversation: list[ConversationLine]) -> list[str]:
    audio_payloads = []
    for line in conversation:
        for attachment in line.audio_attachments:
            audio_payloads.append(attachment.data_base64)
    return audio_payloads


def collect_image_attachments(conversation: list[ConversationLine]) -> list[ImageAttachment]:
    image_payloads = []
    for line in conversation:
        image_payloads.extend(line.image_attachments)
    return image_payloads


def build_ollama_summary_payload(
    conversation: list[ConversationLine],
    request: str,
    audio_payloads: list[str] | None = None,
    image_payloads: list[ImageAttachment] | None = None,
    summary_input: str | None = None,
) -> dict:
    user_message = {"role": "user", "content": summary_input or build_summary_input(conversation, request)}
    multimodal_payloads = []
    if image_payloads:
        multimodal_payloads.extend(attachment.data_base64 for attachment in image_payloads)
        logging.info("Envoi de %s image(s) a Ollama", len(image_payloads))
    if audio_payloads:
        multimodal_payloads.extend(audio_payloads)
        logging.info("Envoi de %s piece(s) jointe(s) audio a Ollama", len(audio_payloads))
    if multimodal_payloads:
        user_message["images"] = multimodal_payloads

    return {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": OLLAMA_THINK,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            user_message,
        ],
        "options": {
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": SUMMARY_MAX_OUTPUT_TOKENS,
            "temperature": min(TEMPERATURE, 0.4),
            "top_p": OLLAMA_TOP_P,
            "top_k": OLLAMA_TOP_K,
            "repeat_penalty": OLLAMA_REPEAT_PENALTY,
            "repeat_last_n": 256,
        },
    }


async def post_ollama_payload(endpoint: str, payload: dict) -> dict:
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(endpoint, json=payload) as response:
            if response.status >= 400:
                error_body = await response.text()
                logging.error(
                    "Erreur Ollama HTTP %s pour %s: %s",
                    response.status,
                    payload.get("model"),
                    error_body[:1000],
                )
                response.raise_for_status()
            data = await response.json()
    return data


async def transcribe_audio_attachment(endpoint: str, attachment: AudioAttachment) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Transcris cet audio en francais si possible. "
                    "Retourne uniquement ce qui est dit, sans resume et sans commentaire. "
                    "Si l'audio est incomprehensible, reponds: [audio incomprehensible]."
                ),
                "images": [attachment.data_base64],
            }
        ],
        "options": {
            "num_ctx": min(OLLAMA_NUM_CTX, 8192),
            "num_predict": SUMMARY_AUDIO_TRANSCRIPTION_TOKENS,
            "temperature": 0.0,
            "top_p": 0.9,
        },
    }
    data = await post_ollama_payload(endpoint, payload)
    message = data.get("message", {})
    transcript = message.get("content", "").strip()
    if not transcript:
        transcript = recover_answer_from_thinking(message.get("thinking", ""))
    return transcript.strip()


async def add_audio_transcripts(endpoint: str, conversation: list[ConversationLine]) -> list[ConversationLine]:
    updated_conversation = []
    for line in conversation:
        transcripts = []
        for attachment in line.audio_attachments:
            try:
                transcript = await transcribe_audio_attachment(endpoint, attachment)
            except aiohttp.ClientResponseError:
                logging.exception("Transcription audio refusee par Ollama pour %s", attachment.filename)
                continue
            except Exception:
                logging.exception("Transcription audio impossible pour %s", attachment.filename)
                continue

            if not transcript:
                transcript = "[audio incomprehensible]"
            logging.info("Audio transcrit par Ollama: %s", attachment.filename)
            if SUMMARY_PRINT_AUDIO_TRANSCRIPTS and not SUMMARY_SILENT_MODE:
                visible_transcript = transcript
                if len(visible_transcript) > SUMMARY_PRINT_AUDIO_TRANSCRIPT_LIMIT:
                    visible_transcript = (
                        visible_transcript[:SUMMARY_PRINT_AUDIO_TRANSCRIPT_LIMIT]
                        + "\n[transcription tronquee]"
                    )
                print(
                    f"\n--- Transcription audio: {attachment.filename} ---\n"
                    f"{visible_transcript}\n"
                    f"--- Fin transcription audio ---\n",
                    flush=True,
                )
            transcripts.append(f"[transcription audio {attachment.filename}: {transcript}]")

        if transcripts:
            updated_conversation.append(replace(line, content=f"{line.content} {' '.join(transcripts)}"))
        else:
            updated_conversation.append(line)

    return updated_conversation


async def describe_image_attachment_ollama(endpoint: str, attachment: ImageAttachment) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Decris cette image pour aider a resumer une conversation Discord. "
                    "Reste factuel et concis. Mentionne le texte visible, les personnes/objets, "
                    "l'action, le ton apparent, et ce qui pourrait expliquer la reaction des gens. "
                    "N'invente pas ce qui n'est pas visible."
                ),
                "images": [attachment.data_base64],
            }
        ],
        "options": {
            "num_ctx": min(OLLAMA_NUM_CTX, 8192),
            "num_predict": SUMMARY_IMAGE_DESCRIPTION_TOKENS,
            "temperature": 0.0,
            "top_p": 0.9,
        },
    }
    data = await post_ollama_payload(endpoint, payload)
    message = data.get("message", {})
    description = message.get("content", "").strip()
    if not description:
        description = recover_answer_from_thinking(message.get("thinking", ""))
    return description.strip()


async def add_image_descriptions_ollama(endpoint: str, conversation: list[ConversationLine]) -> list[ConversationLine]:
    updated_conversation = []
    for line in conversation:
        descriptions = []
        for attachment in line.image_attachments:
            try:
                description = await describe_image_attachment_ollama(endpoint, attachment)
            except aiohttp.ClientResponseError:
                logging.exception("Description image refusee par Ollama pour %s", attachment.filename)
                continue
            except Exception:
                logging.exception("Description image impossible pour %s", attachment.filename)
                continue

            if not description:
                description = "[image incomprehensible]"
            logging.info("Image decrite par Ollama: %s", attachment.filename)
            descriptions.append(f"[description image {attachment.filename}: {description}]")

        if descriptions:
            updated_conversation.append(replace(line, content=f"{line.content} {' '.join(descriptions)}"))
        else:
            updated_conversation.append(line)

    return updated_conversation


async def describe_image_attachment_openai(client, attachment: ImageAttachment) -> str:
    response = await client.responses.create(
        model=OPENAI_MODEL,
        instructions=(
            "Tu decris une image pour aider a resumer une conversation Discord. "
            "Reste factuel, concis, et n'invente rien hors de l'image."
        ),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Decris cette image. Mentionne le texte visible, les personnes/objets, "
                            "l'action, le ton apparent, et ce qui peut expliquer la reaction des gens."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{attachment.media_type};base64,{attachment.data_base64}",
                    },
                ],
            }
        ],
        max_output_tokens=SUMMARY_IMAGE_DESCRIPTION_TOKENS,
    )
    return response.output_text.strip()


async def add_image_descriptions_openai(client, conversation: list[ConversationLine]) -> list[ConversationLine]:
    updated_conversation = []
    for line in conversation:
        descriptions = []
        for attachment in line.image_attachments:
            try:
                description = await describe_image_attachment_openai(client, attachment)
            except Exception:
                logging.exception("Description image impossible avec OpenAI pour %s", attachment.filename)
                continue

            if not description:
                description = "[image incomprehensible]"
            logging.info("Image decrite par OpenAI: %s", attachment.filename)
            descriptions.append(f"[description image {attachment.filename}: {description}]")

        if descriptions:
            updated_conversation.append(replace(line, content=f"{line.content} {' '.join(descriptions)}"))
        else:
            updated_conversation.append(line)

    return updated_conversation


async def ask_ollama_summary(conversation: list[ConversationLine], request: str) -> str:
    endpoint = f"{OLLAMA_URL.rstrip('/')}/api/chat"
    audio_payloads = collect_audio_attachments(conversation)
    image_payloads = collect_image_attachments(conversation)
    if audio_payloads and SUMMARY_TRANSCRIBE_AUDIO_FIRST:
        conversation = await add_audio_transcripts(endpoint, conversation)
        audio_payloads = []
    if image_payloads and SUMMARY_DESCRIBE_IMAGES_FIRST:
        conversation = await add_image_descriptions_ollama(endpoint, conversation)
        image_payloads = []

    summary_input = build_summary_input(conversation, request)
    print_final_summary_prompt(summary_input)
    payload = build_ollama_summary_payload(conversation, request, audio_payloads, image_payloads, summary_input)

    try:
        data = await post_ollama_payload(endpoint, payload)
    except aiohttp.ClientResponseError as exc:
        if exc.status < 500 or (not audio_payloads and not image_payloads) or not SUMMARY_FALLBACK_WITHOUT_AUDIO:
            raise

        logging.warning(
            "Ollama a echoue avec les pieces jointes multimedia. Nouvel essai sans elles pour produire un resume texte."
        )
        text_only_payload = build_ollama_summary_payload(
            conversation,
            request,
            audio_payloads=None,
            image_payloads=None,
            summary_input=summary_input,
        )
        data = await post_ollama_payload(endpoint, text_only_payload)

    message = data.get("message", {})
    content = message.get("content", "").strip()
    if not content:
        content = recover_answer_from_thinking(message.get("thinking", ""))
    if not content:
        logging.warning("Ollama a renvoye une reponse vide: %s", data)
    return content


async def ask_openai_summary(conversation: list[ConversationLine], request: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    try:
        image_payloads = collect_image_attachments(conversation)
        if image_payloads and SUMMARY_DESCRIBE_IMAGES_FIRST:
            conversation = await add_image_descriptions_openai(client, conversation)
            image_payloads = []

        summary_input = build_summary_input(conversation, request)
        print_final_summary_prompt(summary_input)
        if image_payloads:
            content = [{"type": "input_text", "text": summary_input}]
            for attachment in image_payloads:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{attachment.media_type};base64,{attachment.data_base64}",
                    }
                )
            input_payload = [{"role": "user", "content": content}]
            logging.info("Envoi de %s image(s) a OpenAI", len(image_payloads))
        else:
            input_payload = summary_input

        response = await client.responses.create(
            model=OPENAI_MODEL,
            instructions=SUMMARY_SYSTEM_PROMPT,
            input=input_payload,
            max_output_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
        )
        return response.output_text.strip()
    finally:
        await client.close()


async def describe_image_attachment_gemini(attachment: ImageAttachment) -> str:
    try:
        response = await chat_gemini(
            model=GEMINI_MODEL,
            api_key=GEMINI_API_KEY,
            messages=[{
                "role": "user",
                "content": (
                    "Decris cette image. Mentionne le texte visible, les personnes/objets, "
                    "l'action, le ton apparent, et ce qui peut expliquer la reaction des gens. "
                    f"Image: data:{attachment.media_type};base64,{attachment.data_base64}"
                )
            }],
            temperature=0.3,
            max_tokens=SUMMARY_IMAGE_DESCRIPTION_TOKENS,
            timeout=GEMINI_TIMEOUT,
        )
        return response.text.strip()
    except LLMBackendError as exc:
        # Handle quota errors gracefully
        error_msg = str(exc)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            logging.warning("Gemini quota depassee pour description d'image %s", attachment.filename)
            return "[image non decrite: quota API depassee]"
        raise


async def add_image_descriptions_gemini(conversation: list[ConversationLine]) -> list[ConversationLine]:
    updated_conversation = []
    for line in conversation:
        descriptions = []
        for attachment in line.image_attachments:
            try:
                description = await describe_image_attachment_gemini(attachment)
            except Exception:
                logging.exception("Description image impossible avec Gemini pour %s", attachment.filename)
                continue

            if not description:
                description = "[image incomprehensible]"
            logging.info("Image decrite par Gemini: %s", attachment.filename)
            descriptions.append(f"[description image {attachment.filename}: {description}]")

        if descriptions:
            updated_conversation.append(replace(line, content=f"{line.content} {' '.join(descriptions)}"))
        else:
            updated_conversation.append(line)

    return updated_conversation


async def ask_gemini_summary(conversation: list[ConversationLine], request: str) -> str:
    image_payloads = collect_image_attachments(conversation)
    if image_payloads and SUMMARY_DESCRIBE_IMAGES_FIRST:
        conversation = await add_image_descriptions_gemini(conversation)
        image_payloads = []

    summary_input = build_summary_input(conversation, request)
    print_final_summary_prompt(summary_input)

    messages: list[dict[str, Any]] = []
    if image_payloads:
        for attachment in image_payloads:
            messages.append({
                "role": "user",
                "content": f"Image: data:{attachment.media_type};base64,{attachment.data_base64}\n{summary_input}"
            })
        logging.info("Envoi de %s image(s) a Gemini", len(image_payloads))
    else:
        messages.append({
            "role": "user",
            "content": summary_input
        })

    response = await chat_gemini(
        model=GEMINI_MODEL,
        api_key=GEMINI_API_KEY,
        messages=[{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}] + messages,
        temperature=TEMPERATURE,
        max_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
        timeout=GEMINI_TIMEOUT,
    )
    if not response.text:
        logging.warning("Gemini a renvoye une reponse vide")
    return response.text


async def ask_summary_llm(conversation: list[ConversationLine], request: str) -> str:
    if SUMMARY_LLM_PROVIDER == "openai":
        return await ask_openai_summary(conversation, request)
    if SUMMARY_LLM_PROVIDER == "gemini":
        return await ask_gemini_summary(conversation, request)

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
            SUMMARY_CONTEXT_MESSAGE_DEFAULT,
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
    if SUMMARY_LLM_PROVIDER == "ollama":
        logging.info(
            "LLM local Ollama pour resume: %s (%s, num_ctx=%s, sortie=%s tokens)",
            OLLAMA_MODEL,
            OLLAMA_URL,
            OLLAMA_NUM_CTX,
            SUMMARY_MAX_OUTPUT_TOKENS,
        )
    elif SUMMARY_LLM_PROVIDER == "gemini":
        logging.info("LLM Gemini pour resume: %s", GEMINI_MODEL)
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
