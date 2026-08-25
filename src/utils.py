import logging
import os
import sys
import torch
import torchaudio
import numpy as np



def setup_logger(name: str, level=logging.INFO):
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger


def load_audio(path, target_sr: int = None):
    """Read an audio file as mono float32, optionally resampled.

    Deliberately soundfile rather than `torchaudio.load`: from torchaudio 2.9 the
    built-in decoding backends were removed and `load` delegates to TorchCodec,
    an extra dependency that needs a matching FFmpeg. soundfile is already
    required here, ships its own libsndfile, and reads wav, flac, ogg and mp3 -
    which also lets the corpora keep their original formats.

    Returns (tensor of shape [1, samples], sample_rate).
    """
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # to mono

    wav = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)

    if target_sr is not None and sample_rate != target_sr:
        wav = torchaudio.functional.resample(wav, sample_rate, target_sr)
        sample_rate = target_sr

    return wav, sample_rate


def trim_onset_artifact(
    audio: np.ndarray,
    sample_rate: int,
    *,
    max_scan_seconds: float = 0.35,
    fade_seconds: float = 0.01,
) -> np.ndarray:
    """Drop the silence a clip opens with, and fade in so it does not click.

    This also used to cut what it read as a decoder artifact: a stretch of
    quiet, high-zero-crossing sound before the speech proper. That reading was
    wrong. Measured against the test sentence, the frames it was keying on -
    110 to 180 ms, zero-crossing rate 0.70 to 0.79, energy 6% of the speech
    level - are the /s/ of "سلام". Sibilants are noisy and quiet by nature, so
    the rule removed the first phoneme of any clip that began with one, which
    a listener heard as "سلام" arriving unfinished. Before it, frames 0 to 100
    measure exactly zero: there was no burst in that clip to remove.

    Energy alone decides now, at a threshold 40 dB below the speech level -
    below the quietest fricative, above digital silence - so no phoneme can
    fall under it. The fade-in stays, because a hard cut into speech clicks.
    """
    if audio.size == 0:
        return audio

    frame = max(1, int(0.01 * sample_rate))
    usable = audio[: len(audio) // frame * frame]
    if usable.size < frame * 2:
        return audio

    frames = usable.reshape(-1, frame)
    energies = np.sqrt((frames ** 2).mean(axis=1))

    speech_level = float(np.percentile(energies, 90))
    if speech_level <= 0:
        return audio

    audible = np.flatnonzero(energies > 0.01 * speech_level)
    if audible.size == 0:
        return audio

    start_frame = int(audible[0])
    scan_limit = int(max_scan_seconds * sample_rate / frame)
    if start_frame == 0 or start_frame > scan_limit:
        # Speech starts immediately, or the quiet stretch is longer than any
        # onset silence and is therefore worth keeping.
        return audio

    trimmed = audio[start_frame * frame:]

    fade = min(int(fade_seconds * sample_rate), trimmed.size)
    if fade > 1:
        trimmed = trimmed.copy()
        trimmed[:fade] *= np.linspace(0.0, 1.0, fade, dtype=trimmed.dtype)
    return trimmed


def normalise_peak(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
    """Scale to a fixed peak, so nothing clips and levels match between clips."""
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak <= 0:
        return audio
    return audio * (target / peak)


_VAD_MODEL = None
_GET_SPEECH_TIMESTAMPS = None

def load_vad_model():
    """Lazy loads the Silero VAD model."""
    
    global _VAD_MODEL, _GET_SPEECH_TIMESTAMPS
    
    if _VAD_MODEL is not None:
        return _VAD_MODEL, _GET_SPEECH_TIMESTAMPS
    
    try:
        
        #print("Loading Silero VAD model...")
        
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        
        _GET_SPEECH_TIMESTAMPS = utils[0]
        _VAD_MODEL = model
        
        #print("Silero VAD loaded.")
        
        return _VAD_MODEL, _GET_SPEECH_TIMESTAMPS
    
    except Exception as e:
        print(f"Error loading VAD: {e}")
        return None, None


def trim_silence_with_vad(audio_waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Trims silence/noise from the end of the audio using Silero VAD.
    """
    
    vad_model, get_timestamps = load_vad_model()
    if vad_model is None:
        return audio_waveform

    VAD_SR = 16000
    # Convert numpy to tensor
    audio_tensor = torch.from_numpy(audio_waveform).float()

    # Resample for VAD if necessary
    if sample_rate != VAD_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=VAD_SR)
        vad_input = resampler(audio_tensor)
        
    else:
        vad_input = audio_tensor

    try:
        # Get speech timestamps
        speech_timestamps = get_timestamps(vad_input, vad_model, sampling_rate=VAD_SR)
        
        if not speech_timestamps:
            return audio_waveform

        # Get the end of the last speech chunk
        last_speech_end_vad = speech_timestamps[-1]['end']

        # Scale back to original sample rate
        scale_factor = sample_rate / VAD_SR
        cut_point = int(last_speech_end_vad * scale_factor)

        trimmed_wav = audio_waveform[:cut_point]
        
        return trimmed_wav


    except Exception as e:
        print(f"VAD trimming failed: {e}")
        return audio_waveform
    
    
    
def check_pretrained_models(model_dir="pretrained_models", mode="chatterbox"):
    """Checks for the existence of the necessary model files. """

    if mode == "chatterbox_turbo":
        required_files = [
            "ve.safetensors",
            "t3_turbo_v1.safetensors",
            "s3gen_meanflow.safetensors",
            "conds.pt",
            "vocab.json",
            "added_tokens.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "merges.txt",
            "grapheme_mtl_merged_expanded_v1.json"
        ]

    else:

        required_files = [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "conds.pt",
            "tokenizer.json"
        ]


    missing_files = []


    if not os.path.exists(model_dir):
        print(f"\nERROR: '{model_dir}' folder doesn't exist!")
        missing_files = required_files
        
    else:

        for filename in required_files:
            file_path = os.path.join(model_dir, filename)
            if not os.path.exists(file_path):
                missing_files.append(filename)


    if missing_files:
        print("\n" + "!" * 60)
        print("ATTENTION: The following model files could not be found:")
        for f in missing_files:
            print(f"   - {f}")
        
        print("\nPlease run the following command to download the models:")
        print(f" python setup.py")
        print("!" * 60 + "\n")
        return False
    
    print(f"All necessary models are available under '{model_dir}'.")
    return True