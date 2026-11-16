"""ONNX Runtime session creation with the configured execution providers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import onnxruntime as ort

from alpr_id.config import RuntimeConfig
from alpr_id.logging_setup import get_logger

log = get_logger(__name__)

_dlls_preloaded = False


def _expose_nvidia_dll_dirs() -> None:
    """Put the DLL folders of the pip-installed NVIDIA packages on the search path.

    `ort.preload_dlls()` loads the libraries ONNX Runtime links against, but cuDNN looks up
    `nvrtc64_120_0.dll` by name later on and only finds it through PATH.
    """
    try:
        import nvidia
    except ImportError:
        return
    for base in getattr(nvidia, "__path__", []):
        for bin_dir in Path(base).glob("*/bin"):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def _preload_gpu_dlls() -> None:
    """Load the CUDA and cuDNN DLLs that came from pip, once per process."""
    global _dlls_preloaded
    if _dlls_preloaded or not hasattr(ort, "preload_dlls"):
        return
    _dlls_preloaded = True
    _expose_nvidia_dll_dirs()
    try:
        ort.preload_dlls()
    except Exception as exc:  # noqa: BLE001 - a failed preload only means the CPU fallback is used
        log.warning("gpu_dll_preload_failed", error=str(exc))


def resolve_providers(runtime: RuntimeConfig) -> list[str | tuple[str, dict[str, Any]]]:
    """Requested providers that this ONNX Runtime build offers, with their options."""
    available = set(ort.get_available_providers())
    resolved: list[str | tuple[str, dict[str, Any]]] = []
    for name in runtime.providers:
        if name not in available:
            log.warning("provider_unavailable", provider=name, available=sorted(available))
            continue
        if name == "CUDAExecutionProvider":
            _preload_gpu_dlls()
            options: dict[str, Any] = {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
            }
            if runtime.gpu_mem_limit_mb:
                options["gpu_mem_limit"] = runtime.gpu_mem_limit_mb * 1024 * 1024
            resolved.append((name, options))
        else:
            resolved.append(name)
    if not resolved:
        resolved.append("CPUExecutionProvider")
    return resolved


def create_session(model_path: str | Path, runtime: RuntimeConfig) -> ort.InferenceSession:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}. Run scripts/fetch_models.py or see README.")
    session = ort.InferenceSession(str(path), providers=resolve_providers(runtime))
    log.info("model_loaded", model=path.name, providers=session.get_providers())
    return session
