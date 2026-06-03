import pyarrow  # must load before datasets/f5_tts to prevent Windows DLL conflict
import os
import torch
import numpy as np
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_AUDIO_PATH = os.path.join(PROJECT_ROOT, "tests", "audio_samples", "ref_cropped.wav")
REF_TEXT = "தாமிரபரணி ஆற்றின் கரையுரங்களில் வசிக்கும்."

# Model is loaded lazily on first call — never at import time
_f5_tts = None
_infer_process = None


def _load_model():
    global _f5_tts, _infer_process

    if _f5_tts is not None:
        return  # already loaded

    print("\nLoading AI4Bharat IndicF5 Model...")
    from transformers import AutoModel
    from f5_tts.infer.utils_infer import infer_process as _ip

    _f5_tts = AutoModel.from_pretrained(
        "ai4bharat/IndicF5",
        trust_remote_code=True,
        remove_sil=False
    ).to(DEVICE)
    _f5_tts.config.remove_sil = False
    _infer_process = _ip

    print("IndicF5 Loaded Successfully!")
    try:
        print(
            "IndicF5 Device:",
            next(_f5_tts.ema_model.parameters()).device
        )
    except Exception:
        print("IndicF5 Device:", DEVICE)


def generate_tamil_speech(target_text: str) -> tuple[np.ndarray, int]:
    """
    Generates Tamil speech from text using IndicF5.
    Model is loaded on first call (lazy loading).
    Returns: (audio_array, sample_rate)
    """
    _load_model()

    print(f"Generating Tamil speech for: '{target_text}'...")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    audio_arr, final_sample_rate, _ = _infer_process(
        REF_AUDIO_PATH,
        REF_TEXT,
        target_text,
        _f5_tts.ema_model,
        _f5_tts.vocoder,
        mel_spec_type="vocos",
        speed=0.85,
        device=_f5_tts.device,
        fix_duration=None
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time

    print(
        f"[RAW INDICF5] "
        f"Generation Time = {elapsed_time:.3f}s"
    )

    if hasattr(audio_arr, "cpu"):
        audio_arr = audio_arr.cpu().numpy()
    else:
        audio_arr = np.array(audio_arr)

    audio_duration = (
        len(audio_arr) / final_sample_rate
        if final_sample_rate > 0
        else 0
    )

    print(
        f"[RAW INDICF5] Samples = {len(audio_arr)}"
    )

    print(
        f"[RAW INDICF5] "
        f"Audio Duration = {audio_duration:.3f}s"
    )

    if audio_duration > 0:
        print(
            f"[RAW INDICF5] "
            f"RTF = {elapsed_time / audio_duration:.3f}"
        )

    return audio_arr, final_sample_rate