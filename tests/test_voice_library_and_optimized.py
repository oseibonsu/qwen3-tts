# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the voice library and optimized-backend features added in the
dingausmwald fork port.

These tests do NOT require PyTorch or CUDA — they test the routing and
helper functions at the Python level, using a temporary filesystem for
voice library profiles.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_profile_dir(root: Path, profile_id: str, meta: dict, wav_content: bytes = b"RIFF") -> Path:
    """Create a voice library profile directory with meta.json and a dummy WAV."""
    profile_dir = root / "profiles" / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    ref_filename = meta.get("ref_audio_filename", "reference.wav")
    (profile_dir / ref_filename).write_bytes(wav_content)
    return profile_dir


# ---------------------------------------------------------------------------
# _load_voice_profile
# ---------------------------------------------------------------------------


class TestLoadVoiceProfile:
    """Unit tests for the _load_voice_profile() helper."""

    def test_load_by_name(self, tmp_path):
        """Profile can be found by name (case-insensitive)."""
        from api.routers import openai_compatible as oc

        _make_profile_dir(
            tmp_path,
            "alice",
            {
                "name": "Alice",
                "profile_id": "alice",
                "ref_audio_filename": "reference.wav",
                "ref_text": "Hello world.",
                "x_vector_only_mode": False,
                "language": "English",
            },
        )

        with patch.object(oc, "VOICE_LIBRARY_DIR", tmp_path):
            result = oc._load_voice_profile("Alice")

        assert result["name"] == "Alice"
        assert result["ref_text"] == "Hello world."
        assert result["x_vector_only_mode"] is False
        assert result["language"] == "English"
        assert Path(result["ref_audio_path"]).name == "reference.wav"

    def test_load_by_name_case_insensitive(self, tmp_path):
        """Name lookup is case-insensitive."""
        from api.routers import openai_compatible as oc

        _make_profile_dir(
            tmp_path,
            "bob",
            {
                "name": "Bob",
                "profile_id": "bob",
                "ref_audio_filename": "ref.wav",
                "x_vector_only_mode": True,
            },
        )

        with patch.object(oc, "VOICE_LIBRARY_DIR", tmp_path):
            result = oc._load_voice_profile("bob")  # lower-case lookup

        assert result["name"] == "Bob"

    def test_load_by_profile_id(self, tmp_path):
        """Profile can be found by exact profile_id match."""
        from api.routers import openai_compatible as oc

        _make_profile_dir(
            tmp_path,
            "carol_v2",
            {
                "name": "Carol",
                "profile_id": "carol_v2",
                "ref_audio_filename": "ref.wav",
            },
        )

        with patch.object(oc, "VOICE_LIBRARY_DIR", tmp_path):
            result = oc._load_voice_profile("carol_v2")

        assert result["name"] == "Carol"

    def test_not_found_raises_value_error(self, tmp_path):
        """Missing profile raises ValueError."""
        from api.routers import openai_compatible as oc

        (tmp_path / "profiles").mkdir(parents=True, exist_ok=True)

        with patch.object(oc, "VOICE_LIBRARY_DIR", tmp_path):
            with pytest.raises(ValueError, match="not found"):
                oc._load_voice_profile("nobody")

    def test_missing_library_raises_value_error(self, tmp_path):
        """Missing voice library directory raises ValueError."""
        from api.routers import openai_compatible as oc

        missing_dir = tmp_path / "does_not_exist"
        with patch.object(oc, "VOICE_LIBRARY_DIR", missing_dir):
            with pytest.raises(ValueError, match="Voice library not found"):
                oc._load_voice_profile("anything")

    def test_missing_ref_audio_raises_value_error(self, tmp_path):
        """Profile with missing ref_audio_filename raises ValueError."""
        from api.routers import openai_compatible as oc

        profile_dir = tmp_path / "profiles" / "empty"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "meta.json").write_text(
            json.dumps({"name": "Empty", "profile_id": "empty"}),
            encoding="utf-8",
        )

        with patch.object(oc, "VOICE_LIBRARY_DIR", tmp_path):
            with pytest.raises(ValueError, match="no reference audio"):
                oc._load_voice_profile("Empty")

    def test_defaults_for_optional_fields(self, tmp_path):
        """Optional meta.json fields fall back to sensible defaults."""
        from api.routers import openai_compatible as oc

        _make_profile_dir(
            tmp_path,
            "minimal",
            {
                "name": "Minimal",
                "ref_audio_filename": "ref.wav",
                # no ref_text, x_vector_only_mode, language
            },
        )

        with patch.object(oc, "VOICE_LIBRARY_DIR", tmp_path):
            result = oc._load_voice_profile("minimal")

        assert result["ref_text"] == ""
        assert result["x_vector_only_mode"] is False
        assert result["language"] == "Auto"


# ---------------------------------------------------------------------------
# list_voices — voice library profiles appear in response
# ---------------------------------------------------------------------------


class TestListVoicesVoiceLibrary:
    """The /v1/voices endpoint must include voice library profiles."""

    @pytest.mark.asyncio
    async def test_clone_voices_included_in_listing(self, tmp_path):
        """Saved profiles appear as 'clone:Name' entries in the voices list."""
        from api.routers import openai_compatible as oc

        _make_profile_dir(
            tmp_path,
            "dave",
            {
                "name": "Dave",
                "ref_audio_filename": "reference.wav",
            },
        )

        # Patch VOICE_LIBRARY_DIR and a minimal backend
        mock_backend = MagicMock()
        mock_backend.is_ready.return_value = True
        mock_backend.get_supported_voices.return_value = []
        mock_backend.get_supported_languages.return_value = ["English"]
        mock_backend.get_model_type.return_value = "customvoice"
        mock_backend.is_custom_voice.return_value = False

        with patch.object(oc, "VOICE_LIBRARY_DIR", tmp_path), \
             patch("api.routers.openai_compatible.get_tts_backend", return_value=mock_backend):
            result = await oc.list_voices()

        voice_ids = [v["id"] for v in result["voices"]]
        assert "clone:Dave" in voice_ids


# ---------------------------------------------------------------------------
# Factory — optimized backend selection
# ---------------------------------------------------------------------------


class TestOptimizedBackendSelection:
    """Factory must return OptimizedQwen3TTSBackend when TTS_BACKEND=optimized."""

    def teardown_method(self):
        try:
            from api.backends.factory import reset_backend
            reset_backend()
        except Exception:
            pass

    def test_optimized_backend_selected(self, monkeypatch):
        """TTS_BACKEND=optimized returns OptimizedQwen3TTSBackend."""
        pytest.importorskip("torch")
        monkeypatch.setenv("TTS_BACKEND", "optimized")

        from api.backends.factory import get_backend, reset_backend
        reset_backend()

        from api.backends.optimized_backend import OptimizedQwen3TTSBackend
        backend = get_backend()
        assert isinstance(backend, OptimizedQwen3TTSBackend)
        assert backend.get_backend_name() == "optimized"

    def test_optimized_backend_implements_interface(self):
        """OptimizedQwen3TTSBackend implements the TTSBackend interface."""
        pytest.importorskip("torch")
        from api.backends.optimized_backend import OptimizedQwen3TTSBackend
        from api.backends.base import TTSBackend

        backend = OptimizedQwen3TTSBackend()
        assert isinstance(backend, TTSBackend)
        assert hasattr(backend, "initialize")
        assert hasattr(backend, "generate_speech")
        assert hasattr(backend, "generate_speech_streaming")
        assert hasattr(backend, "generate_voice_clone")
        assert hasattr(backend, "generate_voice_clone_streaming")
        assert hasattr(backend, "supports_voice_cloning")
        assert hasattr(backend, "get_backend_name")
        assert hasattr(backend, "get_model_id")
        assert hasattr(backend, "get_supported_voices")
        assert hasattr(backend, "get_supported_languages")
        assert hasattr(backend, "is_ready")
        assert hasattr(backend, "get_device_info")

    def test_optimized_backend_supports_voice_cloning(self):
        """OptimizedQwen3TTSBackend reports voice cloning as supported."""
        pytest.importorskip("torch")
        from api.backends.optimized_backend import OptimizedQwen3TTSBackend

        backend = OptimizedQwen3TTSBackend()
        assert backend.supports_voice_cloning() is True

    def test_optimized_backend_not_ready_initially(self):
        """OptimizedQwen3TTSBackend is not ready before initialize() is called."""
        pytest.importorskip("torch")
        from api.backends.optimized_backend import OptimizedQwen3TTSBackend

        backend = OptimizedQwen3TTSBackend()
        assert backend.is_ready() is False

    def test_optimized_backend_loads_config_yaml(self, tmp_path, monkeypatch):
        """OptimizedQwen3TTSBackend reads config.yaml when it exists."""
        pytest.importorskip("torch")
        import yaml

        config = {
            "default_model": "my-model",
            "models": {
                "my-model": {"hf_id": "test/model", "type": "customvoice"}
            },
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config))

        monkeypatch.setenv("TTS_CONFIG", str(config_file))

        # Re-import to get a fresh instance
        from api.backends.optimized_backend import OptimizedQwen3TTSBackend

        backend = OptimizedQwen3TTSBackend()
        assert backend._default_model_key() == "my-model"


# ---------------------------------------------------------------------------
# stream_generate_custom_voice — method signature check
# ---------------------------------------------------------------------------


class TestStreamGenerateCustomVoice:
    """stream_generate_custom_voice must exist on Qwen3TTSModel."""

    def test_method_exists(self):
        """Qwen3TTSModel has a stream_generate_custom_voice method."""
        pytest.importorskip("torch")
        pytest.importorskip("librosa")
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        assert hasattr(Qwen3TTSModel, "stream_generate_custom_voice")
        assert callable(getattr(Qwen3TTSModel, "stream_generate_custom_voice"))

    def test_method_raises_on_wrong_model_type(self):
        """Calling stream_generate_custom_voice on a non-customvoice model raises ValueError."""
        pytest.importorskip("torch")
        pytest.importorskip("librosa")
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        # Build a minimal mock model with wrong type
        mock_model = MagicMock()
        mock_model.tts_model_type = "base"  # NOT custom_voice
        mock_processor = MagicMock()

        wrapper = Qwen3TTSModel(model=mock_model, processor=mock_processor)

        with pytest.raises(ValueError, match="custom_voice"):
            list(wrapper.stream_generate_custom_voice(text="hello", speaker="Vivian"))

    def test_method_raises_on_batch_input(self):
        """Passing a list as text raises ValueError (no batching supported)."""
        pytest.importorskip("torch")
        pytest.importorskip("librosa")
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        mock_model = MagicMock()
        mock_model.tts_model_type = "custom_voice"
        mock_processor = MagicMock()

        wrapper = Qwen3TTSModel(model=mock_model, processor=mock_processor)

        with pytest.raises(ValueError, match="single text"):
            list(wrapper.stream_generate_custom_voice(text=["a", "b"], speaker="Vivian"))
