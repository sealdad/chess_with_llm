"""
CosyVoice 2 TTS Microservice.

Provides HTTP API for text-to-speech synthesis using CosyVoice2-0.5B.
Designed to run on a separate GPU machine (e.g. RTX 4090) and be called
over the network by the agent service.
"""

import os
import io
import time
import subprocess
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B-2512")
REFERENCE_DIR = Path(os.getenv("REFERENCE_DIR", "reference_voices"))
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

# Default speaker for SFT mode (built-in CosyVoice speakers)
DEFAULT_SPEAKER = os.getenv("DEFAULT_SPEAKER", "")

# ---------------------------------------------------------------------------
# Global model
# ---------------------------------------------------------------------------
cosyvoice_model = None


def load_model():
    """Load CosyVoice2 model."""
    global cosyvoice_model
    print(f"[TTS] Loading CosyVoice2 model from {MODEL_DIR}...")
    t0 = time.time()

    try:
        from cosyvoice.cli.cosyvoice import AutoModel
    except ImportError:
        from cosyvoice.cli.model import AutoModel

    cosyvoice_model = AutoModel(model_dir=MODEL_DIR, fp16=True)

    dt = time.time() - t0
    print(f"[TTS] Model loaded in {dt:.1f}s")

    # List available SFT speakers
    if hasattr(cosyvoice_model, 'list_available_spks'):
        spks = cosyvoice_model.list_available_spks()
        print(f"[TTS] Available speakers: {spks}")


# ---------------------------------------------------------------------------
# Audio conversion helpers
# ---------------------------------------------------------------------------
def pcm_tensor_to_wav_bytes(speech_tensor: torch.Tensor, sample_rate: int) -> bytes:
    """Convert a speech tensor to WAV bytes."""
    buf = io.BytesIO()
    # Ensure 2D: (1, N)
    if speech_tensor.dim() == 1:
        speech_tensor = speech_tensor.unsqueeze(0)
    torchaudio.save(buf, speech_tensor.cpu(), sample_rate, format="wav")
    buf.seek(0)
    return buf.read()


def wav_bytes_to_mp3(wav_bytes: bytes) -> bytes:
    """Convert WAV bytes to MP3 using ffmpeg."""
    proc = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", "pipe:0",
            "-f", "mp3",
            "-b:a", "128k",
            "-loglevel", "error",
            "pipe:1",
        ],
        input=wav_bytes,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {proc.stderr.decode()}")
    return proc.stdout


def speech_tensor_to_mp3(speech_tensor: torch.Tensor, sample_rate: int) -> bytes:
    """Convert speech tensor directly to MP3 bytes."""
    wav = pcm_tensor_to_wav_bytes(speech_tensor, sample_rate)
    return wav_bytes_to_mp3(wav)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="CosyVoice TTS Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str = "default"
    language: str = "zh"  # "zh", "en", "zh-TW"
    instruct_text: str = ""  # raw CosyVoice instruct (advanced)
    emotion: str = ""  # preset emotion name: happy, sad, angry, etc.
    output_format: str = "mp3"  # "mp3" or "wav"
    stream: bool = False


# Emotion presets → CosyVoice instruct_text (language-dependent)
EMOTION_PRESETS_ZH = {
    "happy":       "You are a helpful assistant. 请用开心愉悦的语气说话。<|endofprompt|>",
    "sad":         "You are a helpful assistant. 请用悲伤低落的语气说话。<|endofprompt|>",
    "angry":       "You are a helpful assistant. 请用生气愤怒的语气说话。<|endofprompt|>",
    "surprised":   "You are a helpful assistant. 请用惊讶的语气说话。<|endofprompt|>",
    "fearful":     "You are a helpful assistant. 请用恐惧害怕的语气说话。<|endofprompt|>",
    "disgusted":   "You are a helpful assistant. 请用厌恶不满的语气说话。<|endofprompt|>",
    "calm":        "You are a helpful assistant. 请用冷静平和的语气说话。<|endofprompt|>",
    "serious":     "You are a helpful assistant. 请用严肃认真的语气说话。<|endofprompt|>",
    "gentle":      "You are a helpful assistant. 请用温柔亲切的语气说话。<|endofprompt|>",
    "encouraging": "You are a helpful assistant. 请用热情鼓励的语气说话。<|endofprompt|>",
}
EMOTION_PRESETS_EN = {
    "happy":       "You are a helpful assistant. Please speak in a happy and cheerful tone.<|endofprompt|>",
    "sad":         "You are a helpful assistant. Please speak in a sad and melancholic tone.<|endofprompt|>",
    "angry":       "You are a helpful assistant. Please speak in an angry and frustrated tone.<|endofprompt|>",
    "surprised":   "You are a helpful assistant. Please speak in a surprised and astonished tone.<|endofprompt|>",
    "fearful":     "You are a helpful assistant. Please speak in a fearful and anxious tone.<|endofprompt|>",
    "disgusted":   "You are a helpful assistant. Please speak in a disgusted and disapproving tone.<|endofprompt|>",
    "calm":        "You are a helpful assistant. Please speak in a calm and composed tone.<|endofprompt|>",
    "serious":     "You are a helpful assistant. Please speak in a serious and stern tone.<|endofprompt|>",
    "gentle":      "You are a helpful assistant. Please speak in a gentle and warm tone.<|endofprompt|>",
    "encouraging": "You are a helpful assistant. Please speak in an encouraging and enthusiastic tone.<|endofprompt|>",
}
# Legacy single dict (fallback)
EMOTION_PRESETS = EMOTION_PRESETS_ZH


def _resolve_instruct(req: SynthesizeRequest) -> str:
    """Get the final instruct_text: explicit instruct_text wins, then emotion preset (language-aware)."""
    if req.instruct_text:
        return req.instruct_text
    if req.emotion:
        # Detect language from text to pick matching emotion preset
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', req.text))
        is_zh = chinese_chars > len(req.text) * 0.1
        presets = EMOTION_PRESETS_ZH if is_zh else EMOTION_PRESETS_EN
        return presets.get(req.emotion, "")
    return ""


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: str = ""
    speakers: list = []


class VoiceInfo(BaseModel):
    voice_id: str
    type: str  # "builtin" or "cloned"
    has_reference: bool = False


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------
def _get_reference_path(voice_id: str) -> Optional[Path]:
    """Get the reference audio path for a cloned voice."""
    wav_path = REFERENCE_DIR / f"{voice_id}.wav"
    if wav_path.exists():
        return wav_path
    return None


def _get_reference_text(voice_id: str) -> str:
    """Get the prompt text for a cloned voice."""
    txt_path = REFERENCE_DIR / f"{voice_id}.txt"
    if txt_path.exists():
        return txt_path.read_text().strip()
    return ""


def _synthesize_full(req: SynthesizeRequest) -> tuple:
    """Run synthesis, return (speech_tensor, sample_rate)."""
    model = cosyvoice_model
    if model is None:
        raise RuntimeError("Model not loaded")

    ref_path = _get_reference_path(req.voice_id)
    instruct = _resolve_instruct(req)
    all_speech = []

    if instruct and ref_path:
        # Instruction-based synthesis with voice reference (emotion/style)
        for chunk in model.inference_instruct2(
            req.text,
            instruct,
            str(ref_path),
            stream=False,
        ):
            all_speech.append(chunk["tts_speech"])

    elif ref_path:
        # Zero-shot voice cloning
        prompt_text = _get_reference_text(req.voice_id)
        # CosyVoice3 requires <|endofprompt|> token in prompt_text
        if "<|endofprompt|>" not in prompt_text:
            prompt_text = prompt_text + "<|endofprompt|>"
        for chunk in model.inference_zero_shot(
            req.text,
            prompt_text,
            str(ref_path),
            stream=False,
        ):
            all_speech.append(chunk["tts_speech"])

    else:
        # SFT mode — use built-in speaker
        spk_id = DEFAULT_SPEAKER
        if hasattr(model, 'list_available_spks'):
            available = model.list_available_spks()
            if available and spk_id not in available:
                spk_id = available[0]

        if spk_id:
            for chunk in model.inference_sft(
                req.text,
                spk_id,
                stream=False,
            ):
                all_speech.append(chunk["tts_speech"])
        else:
            # Fallback: cross-lingual with a default reference if available
            default_ref = REFERENCE_DIR / "default.wav"
            if default_ref.exists():
                for chunk in model.inference_cross_lingual(
                    req.text,
                    str(default_ref),
                    stream=False,
                ):
                    all_speech.append(chunk["tts_speech"])
            else:
                raise RuntimeError(
                    "No built-in speakers and no reference voice. "
                    "Upload a reference voice or use a model with SFT speakers."
                )

    if not all_speech:
        raise RuntimeError("Synthesis produced no audio")

    speech = torch.cat(all_speech, dim=-1)
    return speech, model.sample_rate


def _synthesize_streaming(req: SynthesizeRequest):
    """Generator yielding (speech_tensor, sample_rate) chunks."""
    model = cosyvoice_model
    if model is None:
        raise RuntimeError("Model not loaded")

    ref_path = _get_reference_path(req.voice_id)
    instruct = _resolve_instruct(req)

    if instruct and ref_path:
        gen = model.inference_instruct2(
            req.text, instruct, str(ref_path), stream=True,
        )
    elif ref_path:
        prompt_text = _get_reference_text(req.voice_id)
        if "<|endofprompt|>" not in prompt_text:
            prompt_text = prompt_text + "<|endofprompt|>"
        gen = model.inference_zero_shot(
            req.text, prompt_text, str(ref_path), stream=True,
        )
    else:
        spk_id = DEFAULT_SPEAKER
        if hasattr(model, 'list_available_spks'):
            available = model.list_available_spks()
            if available and spk_id not in available:
                spk_id = available[0]

        if spk_id:
            gen = model.inference_sft(req.text, spk_id, stream=True)
        else:
            default_ref = REFERENCE_DIR / "default.wav"
            if default_ref.exists():
                gen = model.inference_cross_lingual(
                    req.text, str(default_ref), stream=True,
                )
            else:
                raise RuntimeError("No speakers available")

    for chunk in gen:
        yield chunk["tts_speech"], model.sample_rate


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/emotions")
async def list_emotions():
    """List available emotion presets."""
    return {
        "emotions": list(EMOTION_PRESETS.keys()),
        "presets": {k: v.replace("<|endofprompt|>", "") for k, v in EMOTION_PRESETS.items()},
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else ""
    speakers = []
    if cosyvoice_model and hasattr(cosyvoice_model, 'list_available_spks'):
        speakers = cosyvoice_model.list_available_spks() or []

    return HealthResponse(
        status="ok" if cosyvoice_model else "loading",
        model_loaded=cosyvoice_model is not None,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        speakers=speakers,
    )


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Full (non-streaming) synthesis. Returns complete audio file."""
    try:
        speech, sr = await asyncio.to_thread(_synthesize_full, req)

        if req.output_format == "wav":
            audio_bytes = pcm_tensor_to_wav_bytes(speech, sr)
            media_type = "audio/wav"
        else:
            audio_bytes = await asyncio.to_thread(speech_tensor_to_mp3, speech, sr)
            media_type = "audio/mpeg"

        # Free GPU memory after synthesis
        torch.cuda.empty_cache()

        return Response(content=audio_bytes, media_type=media_type)

    except Exception as e:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize/stream")
async def synthesize_stream(req: SynthesizeRequest):
    """Streaming synthesis. Returns chunked audio."""
    async def generate():
        try:
            for speech_chunk, sr in await asyncio.to_thread(
                lambda: list(_synthesize_streaming(req))
            ):
                if req.output_format == "wav":
                    chunk_bytes = pcm_tensor_to_wav_bytes(speech_chunk, sr)
                else:
                    chunk_bytes = speech_tensor_to_mp3(speech_chunk, sr)
                yield chunk_bytes
        except Exception as e:
            print(f"[TTS] Streaming error: {e}")
        finally:
            torch.cuda.empty_cache()

    media_type = "audio/wav" if req.output_format == "wav" else "audio/mpeg"
    return StreamingResponse(generate(), media_type=media_type)


@app.get("/voices")
async def list_voices():
    """List available voices (built-in + cloned)."""
    voices = []

    # Built-in SFT speakers
    if cosyvoice_model and hasattr(cosyvoice_model, 'list_available_spks'):
        for spk in (cosyvoice_model.list_available_spks() or []):
            voices.append(VoiceInfo(voice_id=spk, type="builtin"))

    # Cloned voices (from reference_voices directory)
    for wav_file in REFERENCE_DIR.glob("*.wav"):
        voice_id = wav_file.stem
        voices.append(VoiceInfo(
            voice_id=voice_id,
            type="cloned",
            has_reference=True,
        ))

    return {"voices": [v.dict() for v in voices]}


@app.post("/voices/reference")
async def upload_reference(
    voice_id: str = Form(...),
    prompt_text: str = Form(""),
    audio_file: UploadFile = File(...),
):
    """Upload reference audio for voice cloning."""
    # Save audio file
    wav_path = REFERENCE_DIR / f"{voice_id}.wav"
    content = await audio_file.read()

    # Convert to WAV if needed (ffmpeg handles any input format)
    # Keep original sample rate — CosyVoice handles resampling internally
    if not audio_file.filename.endswith(".wav"):
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-i", "pipe:0",
                "-ac", "1",
                "-f", "wav", "pipe:1",
            ],
            input=content,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise HTTPException(500, f"Audio conversion failed: {proc.stderr.decode()}")
        content = proc.stdout

    wav_path.write_bytes(content)

    # Save prompt text
    if prompt_text:
        txt_path = REFERENCE_DIR / f"{voice_id}.txt"
        txt_path.write_text(prompt_text)

    return {"success": True, "voice_id": voice_id, "message": f"Voice '{voice_id}' saved"}


@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a cloned voice."""
    wav_path = REFERENCE_DIR / f"{voice_id}.wav"
    txt_path = REFERENCE_DIR / f"{voice_id}.txt"

    if not wav_path.exists():
        raise HTTPException(404, f"Voice '{voice_id}' not found")

    wav_path.unlink()
    if txt_path.exists():
        txt_path.unlink()

    return {"success": True, "message": f"Voice '{voice_id}' deleted"}
