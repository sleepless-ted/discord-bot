# Chatbot LLM Discord

Bot Discord Python qui repond quand on le mentionne dans un salon. A chaque mention, il lit les messages precedents du canal, ajoute le message actuel, puis interroge Gemma 4 via Ollama avec un system prompt de style.

Le projet contient maintenant deux bots independants:

- `babouin_bot.py`: le bot de conversation actuel, avec son style venant de `babouin_system_prompt.txt`.
- `summary_bot.py`: un deuxieme bot qui resume la conversation, les reponses et les points de vue des participants.

## Installation

1. Cree un environnement virtuel:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Installe les dependances:

   ```powershell
   pip install -r requirements.txt
   ```

3. Renseigne les variables dans `.env`:

   ```env
   DISCORD_TOKEN=ton_token_discord
   LLM_PROVIDER=ollama
   # Surcharges facultatives si les deux bots utilisent des providers differents
   BABOUIN_LLM_PROVIDER=ollama
   SUMMARY_LLM_PROVIDER=ollama
   OLLAMA_URL=http://localhost:11434
   OLLAMA_MODEL=gemma4:26b
   OLLAMA_NUM_CTX=32768
   NUM_PREDICT=220
   OLLAMA_THINK=false
   TEMPERATURE=0.95
   OLLAMA_TOP_P=0.9
   OLLAMA_TOP_K=80
   OLLAMA_REPEAT_PENALTY=1.25
   CONTEXT_MESSAGE_LIMIT=50
   STYLE_PROMPT_FILE=babouin_system_prompt.txt
   LOG_LEVEL=INFO

   # Deuxieme bot de resume
   DISCORD_SUMMARY_TOKEN=ton_token_du_deuxieme_bot
   SUMMARY_COMMAND_PREFIX=!resume
   SUMMARY_CONTEXT_MESSAGE_DEFAULT=80
   SUMMARY_CONTEXT_MESSAGE_MAX=200
   SUMMARY_MAX_OUTPUT_TOKENS=900
   SUMMARY_INCLUDE_BOTS=true
   SUMMARY_SYSTEM_PROMPT_FILE=
   SUMMARY_READ_AUDIO_ATTACHMENTS=true
   SUMMARY_MAX_AUDIO_ATTACHMENTS=4
   SUMMARY_MAX_AUDIO_BYTES=10000000
   SUMMARY_FALLBACK_WITHOUT_AUDIO=true
   SUMMARY_CONVERT_AUDIO_TO_WAV=true
   SUMMARY_TRANSCRIBE_AUDIO_FIRST=true
   SUMMARY_AUDIO_TRANSCRIPTION_TOKENS=500
   SUMMARY_SILENT_MODE=false
   SUMMARY_PRINT_AUDIO_TRANSCRIPTS=true
   SUMMARY_PRINT_AUDIO_TRANSCRIPT_LIMIT=4000
   FFMPEG_PATH=ffmpeg
   ```

4. Dans le Discord Developer Portal, active l'intent **Message Content Intent** pour chaque bot.

5. Invite le bot sur ton serveur avec au minimum:

   - View Channels
   - Send Messages
   - Read Message History

6. Lance le premier bot:

   ```powershell
   python babouin_bot.py
   ```

7. Lance le deuxieme bot dans un autre terminal:

   ```powershell
   python summary_bot.py
   ```

## Ollama

Le bot utilise Ollama par defaut. Avant de lancer `python babouin_bot.py`, verifie qu'Ollama tourne et que Gemma 4 est disponible:

```powershell
ollama list
ollama run gemma4:26b
```

Si ton modele n'a pas exactement le nom `gemma4:26b`, mets le nom affiche par `ollama list` dans `.env`:

```env
OLLAMA_MODEL=nom_du_modele
```

`NUM_PREDICT` et `TEMPERATURE` s'appliquent a Ollama comme a Gemini.
`OLLAMA_NUM_CTX` controle la taille du contexte cote Ollama. `CONTEXT_MESSAGE_LIMIT` controle combien de messages Discord le bot lit avant de repondre.

Plus ces valeurs sont hautes, plus le bot a de contexte, mais plus la generation est lente et gourmande en RAM/VRAM.

## Style

Le style est fourni par `babouin_system_prompt.txt`.

Au demarrage, `babouin_bot.py` lit ce fichier et l'ajoute au system prompt. C'est la bonne place pour decrire:

- le ton du bot;
- les phrases a eviter;
- la longueur des reponses;
- la facon de repondre aux questions;
- les limites a ne pas franchir.

Le fichier `lipa_reponses_nettoyees.txt` peut rester comme archive/corpus source, mais il n'est pas envoye au modele a chaque message. Le fichier vraiment utilise en production est `babouin_system_prompt.txt`.

## Gemini avec google-genai

Le bot Babouin peut utiliser le backend Gemini issu de `llm-slave` via le SDK officiel `google-genai`:

```powershell
pixi install
```

```env
BABOUIN_LLM_PROVIDER=gemini
GEMINI_API_KEY=ta_cle_google_ai_studio
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TIMEOUT=120
```

Les alias `genai`, `google` et `google-genai` sont aussi acceptes. Le contenu de
`babouin_system_prompt.txt` est transmis comme `system_instruction` natif de Gemini.
Le bot de resume reste sur le provider defini par `SUMMARY_LLM_PROVIDER`.

## Fichiers principaux

`babouin_bot.py` contient toute la logique du bot Discord principal: lecture du contexte, appel Ollama/OpenAI, decoupage des messages longs et remplacement des emojis custom Discord.

`llm_backend.py` contient les backends texte asynchrones communs pour Ollama et Gemini.

`summary_bot.py` contient le deuxieme bot. Il utilise son propre token Discord (`DISCORD_SUMMARY_TOKEN`) et les memes reglages LLM que le premier bot.

Pour que le bot de resume analyse les pieces jointes audio, utilise un modele compatible audio comme:

```env
OLLAMA_MODEL=gemma4:e4b
```

Les audios Discord sont telecharges puis envoyes a Ollama en base64. Les limites se reglent avec `SUMMARY_MAX_AUDIO_ATTACHMENTS` et `SUMMARY_MAX_AUDIO_BYTES`. Les vocaux Discord sont souvent en OGG/Opus; `SUMMARY_CONVERT_AUDIO_TO_WAV=true` les convertit en WAV avec `ffmpeg` avant envoi. `SUMMARY_TRANSCRIBE_AUDIO_FIRST=true` demande d'abord a Gemma de transcrire chaque audio, puis ajoute ces transcriptions au contexte du resume. `SUMMARY_PRINT_AUDIO_TRANSCRIPTS=true` affiche chaque transcription dans la console. `SUMMARY_SILENT_MODE=true` coupe les impressions de contenu texte comme les transcriptions et le prompt final. Si Ollama renvoie une erreur serveur sur l'audio, `SUMMARY_FALLBACK_WITHOUT_AUDIO=true` permet au bot de refaire un essai sans audio au lieu d'echouer completement.

Si `ffmpeg` n'est pas installe, installe-le dans l'environnement conda:

```powershell
conda install -c conda-forge ffmpeg
```

`requirements.txt` liste les dependances runtime du bot.

`.env` contient la configuration active.

`.env.gemma4` est un profil Gemma 4 de base que tu peux recopier vers `.env` si besoin:

```powershell
Copy-Item .env.gemma4 .env -Force
```

`babouin_system_prompt.txt` est le guide de style charge a chaque reponse.

`lipa_reponses_nettoyees.txt` est le corpus source nettoye. Il sert seulement de reference manuelle si tu veux reecrire le prompt de style.

## Utilisation

Dans un canal ou le bot a acces a l'historique, mentionne-le:

```text
@NomDuBot tu peux resumer ce qu'on vient de dire ?
```

Le bot recupere les messages precedents, ajoute le message qui le mentionne, puis repond dans le canal.

## Utilisation du bot de resume

Le bot de resume se declenche quand on le mentionne, ou avec le prefixe configure dans `.env`:

```text
@NomDuBotResume resume la conversation
!resume
!resume les 50 derniers messages
```

Sa reponse suit ce format:

- `Resume`: synthese courte;
- `Points de vue`: position ou reaction principale de chaque personne;
- `Accords / desaccords`: zones de consensus ou de conflit;
- `A suivre`: questions ouvertes, decisions ou actions possibles.

Pour eviter de melanger les deux bots, cree une deuxieme application dans le Discord Developer Portal, invite-la sur le serveur, puis mets son token dans `DISCORD_SUMMARY_TOKEN`. Copie bien le token depuis `Bot > Token`, sans guillemets, sans `Bot ` devant, et sans espaces autour.

## Reglages utiles

Si le bot repond trop long:

```env
NUM_PREDICT=120
```

Si le bot manque de contexte:

```env
CONTEXT_MESSAGE_LIMIT=80
OLLAMA_NUM_CTX=32768
```

Pour que le bot de resume lise aussi les images jointes:

```env
SUMMARY_READ_IMAGE_ATTACHMENTS=true
SUMMARY_DESCRIBE_IMAGES_FIRST=true
SUMMARY_MAX_IMAGE_ATTACHMENTS=6
SUMMARY_MAX_IMAGE_BYTES=8000000
SUMMARY_IMAGE_DESCRIPTION_TOKENS=350
SUMMARY_PRINT_FINAL_PROMPT=true
SUMMARY_SILENT_MODE=false
```

Avec `SUMMARY_DESCRIBE_IMAGES_FIRST=true`, chaque image est d'abord decrite separement, puis la description est replacee dans le message Discord correspondant avant le resume final. Le modele configure doit accepter les images. Avec Ollama, utilise un modele multimodal; avec OpenAI, utilise un modele compatible vision.

Si le bot repete trop souvent les memes phrases:

```env
TEMPERATURE=1
OLLAMA_REPEAT_PENALTY=1.3
```

Pour revenir a OpenAI plus tard, mets `BABOUIN_LLM_PROVIDER=openai`, renseigne `OPENAI_API_KEY`, puis configure `OPENAI_MODEL`.
