import os
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.utils import setup_logger


logger = setup_logger(__name__)



class ChatterboxDataset(Dataset):
    
    def __init__(self, config):
        self.cfg = config
        self.preprocessed_dir = config.preprocessed_dir
        
        if not os.path.exists(self.preprocessed_dir):
            raise FileNotFoundError(f"Preprocessing folder not found: {self.preprocessed_dir}.")
            
        self.files = [f for f in os.listdir(self.preprocessed_dir) if f.endswith(".pt")]
        
        if len(self.files) == 0:
            raise RuntimeError(f"There are no .pt files in the folder: {self.preprocessed_dir}")
            
        logger.info(f"Dataset loaded. Total sample: {len(self.files)}")

        self.sot_token = config.start_text_token 
        self.eot_token = config.stop_text_token

        self._warn_about_truncation()


    def _warn_about_truncation(self, sample_size: int = 2000):
        """Report clips whose text or audio would be cut by the length limits.

        Truncation is silent and its damage is indirect: the audio still holds
        the words that were cut from the text, so the model learns to say
        things its input never asked for. A random sample is enough to notice.
        """
        import random

        sample = random.Random(0).sample(
            self.files, min(sample_size, len(self.files))
        )
        text_over = speech_over = 0
        for filename in sample:
            try:
                data = torch.load(
                    os.path.join(self.preprocessed_dir, filename), weights_only=True
                )
            except Exception:
                continue
            if data["text_tokens"].size(0) > self.cfg.max_text_len - 2:
                text_over += 1
            if data["speech_tokens"].size(0) > self.cfg.max_speech_len:
                speech_over += 1

        for count, limit_name, limit in (
            (text_over, "max_text_len", self.cfg.max_text_len),
            (speech_over, "max_speech_len", self.cfg.max_speech_len),
        ):
            if count:
                logger.warning(
                    f"{count}/{len(sample)} sampled clips exceed {limit_name}"
                    f"={limit} and will be truncated. Raise it in src/config.py, "
                    f"or rebuild with a tighter filter in tools/build_dataset.py."
                )


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        try:
            
            filename = self.files[idx]
            
            pt_path = os.path.join(self.preprocessed_dir, filename)
            
            data = torch.load(pt_path, weights_only=True)
            
            
            text_tokens = data["text_tokens"]
            if text_tokens.size(0) > self.cfg.max_text_len - 2:
                text_tokens = text_tokens[:self.cfg.max_text_len - 2]
                
            sot = torch.tensor([self.sot_token], dtype=torch.long)
            eot = torch.tensor([self.eot_token], dtype=torch.long)
            text_tokens = torch.cat([sot, text_tokens, eot])

            speech_tokens = data["speech_tokens"]
            if speech_tokens.size(0) > self.cfg.max_speech_len:
                speech_tokens = speech_tokens[:self.cfg.max_speech_len]

            speaker_emb = data["speaker_emb"]
            prompt_tokens = data["prompt_tokens"]

            # Voice-conditioning dropout. Without it the model learns to copy
            # the reference clip instead of reading the text, and cannot speak
            # at all when no prompt is supplied.
            if random.random() < getattr(self.cfg, "cond_dropout", 0.20):
                speaker_emb = torch.zeros_like(speaker_emb)
                prompt_tokens = torch.zeros(1, dtype=torch.long)


            return {
                "text_tokens": text_tokens,
                "speech_tokens": speech_tokens,
                "speaker_emb": speaker_emb,
                "prompt_tokens": prompt_tokens
            }


        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None


def data_collator_standart(batch):

    batch = [item for item in batch if item is not None]
    if not batch: 
        return {}

    # Padding
    text_tokens = pad_sequence([x["text_tokens"] for x in batch], batch_first=True, padding_value=0)
    speech_tokens = pad_sequence([x["speech_tokens"] for x in batch], batch_first=True, padding_value=0)
    prompt_tokens = pad_sequence([x["prompt_tokens"] for x in batch], batch_first=True, padding_value=0)

    speaker_embs = torch.stack([x["speaker_emb"] for x in batch])

    # Lengths
    text_lens = torch.tensor([len(x["text_tokens"]) for x in batch], dtype=torch.long)
    speech_lens = torch.tensor([len(x["speech_tokens"]) for x in batch], dtype=torch.long)


    return {
        "text_tokens": text_tokens,
        "text_token_lens": text_lens,
        "speech_tokens": speech_tokens,
        "speech_token_lens": speech_lens,
        "speaker_emb": speaker_embs,
        "prompt_tokens": prompt_tokens
    }
    
    


def data_collator_turbo(batch):

    batch = [item for item in batch if item is not None]
    if not batch: 
        return {}

    # 1. Text Tokens Padding
    text_tokens = pad_sequence([x["text_tokens"] for x in batch], batch_first=True, padding_value=0)
    text_lens = torch.tensor([len(x["text_tokens"]) for x in batch], dtype=torch.long)

    # 2. Speech Tokens Padding
    speech_tokens = pad_sequence([x["speech_tokens"] for x in batch], batch_first=True, padding_value=0)
    speech_lens = torch.tensor([len(x["speech_tokens"]) for x in batch], dtype=torch.long)

    # 3. Prompt Tokens Padding
    prompt_tokens = pad_sequence([x["prompt_tokens"] for x in batch], batch_first=True, padding_value=0)
    prompt_lens = torch.tensor([x["prompt_tokens"].shape[0] for x in batch], dtype=torch.long)

    # 4. Speaker Embedding
    speaker_embs = torch.stack([x["speaker_emb"] for x in batch])

    return {
        "text_tokens": text_tokens,
        "text_token_lens": text_lens,
        "speech_tokens": speech_tokens,
        "speech_token_lens": speech_lens,
        "speaker_emb": speaker_embs,
        "prompt_tokens": prompt_tokens,
        "prompt_lens": prompt_lens
    }