# Voice Library

The voice library lets you save voice profiles and reuse them across requests
using the `clone:ProfileName` voice prefix.  The server loads the reference
audio from disk, computes the speaker embedding, and caches it — saving roughly
0.7 s per repeated request compared to uploading the raw audio each time.

## Directory layout

```
$VOICE_LIBRARY_DIR/            # default: ./voice_library (override with env var)
└── profiles/
    ├── alice/
    │   ├── meta.json
    │   └── reference.wav
    └── bob/
        ├── meta.json
        └── reference.wav
```

Set the `VOICE_LIBRARY_DIR` environment variable to point anywhere on the filesystem.

## meta.json format

```json
{
    "name": "Alice",
    "profile_id": "alice",
    "ref_audio_filename": "reference.wav",
    "ref_text": "Optional verbatim transcript of the reference audio.",
    "x_vector_only_mode": false,
    "language": "English"
}
```

| Field                | Required | Description |
|----------------------|----------|-------------|
| `name`               | ✓        | Display name and lookup key (case-insensitive). |
| `profile_id`         | —        | Optional alternate lookup key (exact match). |
| `ref_audio_filename` | ✓        | Filename of the reference WAV/MP3 inside the profile directory. |
| `ref_text`           | —        | Transcript of the reference audio.  **Required** for ICL mode (`x_vector_only_mode: false`). |
| `x_vector_only_mode` | —        | `true` → use speaker-embedding only (no transcript needed). `false` → ICL mode (higher quality, requires `ref_text`). Default: `false`. |
| `language`           | —        | Language hint used when the request does not specify one (e.g. `"English"`, `"German"`). Default: `"Auto"`. |

## Using a profile

Pass the profile name with the `clone:` prefix as the `voice` field in a
standard `/v1/audio/speech` request:

```bash
# Non-streaming — returns MP3
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!", "voice": "clone:Alice", "model": "tts-1"}' \
  --output speech.mp3

# Real-time streaming — returns raw PCM (requires optimized backend)
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "input": "Hello!",
        "voice": "clone:Alice",
        "model": "tts-1",
        "stream": true,
        "response_format": "pcm"
      }' \
  --output speech.pcm
```

Or with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")

response = client.audio.speech.create(
    model="tts-1",
    voice="clone:Alice",
    input="This is Alice speaking.",
)
response.stream_to_file("alice.mp3")
```

## Listing profiles via the API

Saved profiles appear in the `/v1/voices` (or `/v1/audio/voices`) endpoint
response under the `voices` array, with their id prefixed by `clone:`:

```json
{
  "voices": [
    {"id": "clone:Alice", "name": "clone:Alice", "description": "Voice library profile: Alice"},
    ...
  ]
}
```

## Caching behaviour

- The reference audio file is read and decoded once per server process and
  then kept in memory (`_ref_audio_cache`).
- The speaker embedding (voice prompt) is computed on first use and then
  cached for the lifetime of the loaded model (`_voice_prompt_cache` inside
  the backend).  If the model is swapped out (e.g. CustomVoice → Base), the
  voice prompt cache is cleared and rebuilt on the next request.

## Notes

- Voice library profiles require the **Base** model.
  - When using the **optimized backend** (`TTS_BACKEND=optimized`), the server
    automatically switches to the Base model when it receives a `clone:` request.
  - For other backends, `clone:` profiles only work if a Base model is already
    loaded/selected; no automatic model switch is performed.
- If `x_vector_only_mode` is `false` (ICL mode) and `ref_text` is empty, the
  request will fail with a `missing_ref_text` error.
- The `VOICE_LIBRARY_DIR` env var must be set consistently between the API
  server and any tools (e.g. Gradio Voice Studio) that write profiles.
