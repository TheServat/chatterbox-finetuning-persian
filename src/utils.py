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
    """Cut the noise burst the decoder emits before speech actually starts.

    S3Gen's flow matching needs a few frames to settle, and what it emits in the
    meantime is a short hiss rather than speech - measured on real output as
    ~80 ms at zero-crossing rates of 0.5-0.8 against 0.14 for the speech that
    follows. It is brief, but it is the first thing a listener hears, and it
    reads as a robotic click at the start of every clip.

    This is a property of the decoder, not of an undertrained model, so more
    training will not remove it.

    Detection uses both energy and zero-crossing rate, because neither alone is
    enough: the burst is quiet *and* noisy, while a soft speech onset is quiet
    but not noisy. Only the first `max_scan_seconds` are considered, so a clip
    that simply begins with a pause is left alone, and a fade-in replaces the
    click that a hard cut would create.
    """
    if audio.size == 0:
        return audio

    frame = max(1, int(0.01 * sample_rate))
    usable = audio[: len(audio) // frame * frame]
    if usable.size < frame * 2:
        return audio

    frames = usable.reshape(-1, frame)
    energies = np.sqrt((frames ** 2).mean(axis=1))
    crossings = (np.diff(np.sign(frames), axis=1) != 0).mean(axis=1)

    speech_level = float(np.percentile(energies, 90))
    if speech_level <= 0:
        return audio

    loud_enough = energies > 0.08 * speech_level
    tonal_enough = crossings < 0.35
    speech_frames = np.flatnonzero(loud_enough & tonal_enough)
    if speech_frames.size == 0:
        return audio

    start_frame = int(speech_frames[0])
    scan_limit = int(max_scan_seconds * sample_rate / frame)
    if start_frame == 0 or start_frame > scan_limit:
        # Speech starts immediately, or the quiet stretch is longer than any
        # decoder artifact and is therefore real silence worth keeping.
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