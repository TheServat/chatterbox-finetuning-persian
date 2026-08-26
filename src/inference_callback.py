import os
import torch
import soundfile as sf
from transformers import TrainerCallback
from safetensors.torch import load_file

from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3
from src.model import resize_and_load_t3_weights
from src.utils import (
    noise_floor,
    normalise_peak,
    setup_logger,
    trim_onset_artifact,
    trim_silence_with_vad,
)


logger = setup_logger("InferenceCallback")


class InferenceCallback(TrainerCallback):

    def __init__(self, config, engine=None):
        """`engine` lets the Persian path sample from the model being trained.

        The original approach rebuilds a whole second engine per checkpoint and
        moves it to the GPU, which doubles VRAM - fine on a large card, an
        immediate OOM on the 6 GB one this was developed against. Passing the
        live engine also makes the samples honest: they come from the exact
        weights at that step rather than from a checkpoint reloaded off disk.
        """
        self.config = config
        self.engine = engine
        self.inference_dir = os.path.join(config.output_dir, "inference_samples")
        os.makedirs(self.inference_dir, exist_ok=True)

        if not hasattr(config, 'inference_prompt_path') or not config.inference_prompt_path:
            logger.warning("The inference prompt path is not specified; sampling will be skipped.")
            self.skip_inference = True

        elif not hasattr(config, 'inference_test_text') or not config.inference_test_text:
            logger.warning("The inference test text is not specified; the sample will be skipped.")
            self.skip_inference = True

        else:
            self.skip_inference = False
            logger.info(f"Inference Callback is ready. Examples will be saved here: {self.inference_dir}")

    def on_step_end(self, args, state, control, **kwargs):
        """Last hook before a checkpoint is written.

        transformers saves and *then* calls on_save, so a cache left by the
        previous sample is still attached when the next save serialises the
        model. Clearing it here closes that window even if a sample failed
        part-way through its own cleanup.
        """
        if control.should_save and self.engine is not None:
            from src import compat

            compat.drop_inference_cache(self.engine.t3)
        return control

    def on_save(self, args, state, control, **kwargs):

        if self.skip_inference:
            return

        step = state.global_step

        # Frequent saves are what survives a power cut; frequent samples are
        # what makes them expensive. Only every nth save draws audio.
        every = max(1, getattr(self.config, "inference_every_n_saves", 1))
        self._saves_seen = getattr(self, "_saves_seen", 0) + 1
        if self._saves_seen % every:
            logger.info(
                f"checkpoint-{step} saved; next audio in "
                f"{every - self._saves_seen % every} more save(s)"
            )
            return

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
        is_lora = getattr(self.config, "is_lora", False)

        if getattr(self.config, "is_persian", False) and self.engine is not None:
            output_path = os.path.join(self.inference_dir, f"checkpoint-{step}.wav")
            try:
                self._generate_sample_persian(output_path)
            except Exception as exc:
                logger.error(f"Persian sample failed at step {step}: {exc}", exc_info=True)
            return

        if is_lora:
            if not os.path.exists(checkpoint_dir):
                logger.warning(f"Checkpoint directory could not be found: {checkpoint_dir}")
                return

            logger.info(f"Initializing inference for checkpoint-{step} (LoRA)...")

            try:
                logger.info(f"Saving PEFT adapters explicitly to {checkpoint_dir}...")
                model_wrapper = kwargs.get('model')

                peft_model_to_save = None
                if hasattr(model_wrapper, 'model') and isinstance(model_wrapper.model, torch.nn.Module):
                    peft_model_to_save = model_wrapper.model
                elif hasattr(model_wrapper, 't3'):
                    peft_model_to_save = model_wrapper.t3
                else:
                    peft_model_to_save = model_wrapper

                if hasattr(peft_model_to_save, 'save_pretrained'):
                    peft_model_to_save.save_pretrained(checkpoint_dir)
                    logger.info("Adapter config and weights saved successfully.")
                else:
                    logger.warning("Could not find a save_pretrained method on the model.")

            except Exception as e:
                logger.error(f"Failed to force save PEFT adapters: {e}")

            try:
                output_path = os.path.join(self.inference_dir, f"checkpoint-{step}.wav")
                self._generate_sample_lora(checkpoint_dir, output_path)
            except Exception as e:
                logger.error(f"An error occurred during LoRA inference (Step: {step}): {e}", exc_info=True)

        else:

            weights_path = os.path.join(checkpoint_dir, "model.safetensors")
            if not os.path.exists(weights_path):
                weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

            if not os.path.exists(weights_path):
                logger.warning(f"Checkpoint weights could not be found: {checkpoint_dir}")
                return

            logger.info(f"Initializing inference for checkpoint-{step} (Full Fine-Tune)...")

            try:
                output_path = os.path.join(self.inference_dir, f"checkpoint-{step}.wav")
                self._generate_sample_full(weights_path, output_path)
            except Exception as e:
                logger.error(f"An error occurred during inference (Step: {step}): {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # Persian: sample from the model currently being trained
    # -------------------------------------------------------------------------
    def _generate_sample_persian(self, output_path: str):
        from src import compat

        engine = self.engine
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        t3 = engine.t3
        was_training = t3.training
        # Training keeps sdpa for speed, but the alignment analyzer reads
        # attention weights that sdpa never produces - and T3 builds that
        # analyzer for any multilingual vocabulary, so generating under sdpa
        # would stack a list of None. Switch for the sample, switch back after.
        previous_attn = getattr(
            getattr(t3, "tfmr", None), "config", None
        ) and t3.tfmr.config._attn_implementation

        # Seeding makes one checkpoint comparable with the next, but the
        # training stream must not notice, so its state is put back below.
        rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )

        # S3Gen and the voice encoder live on the CPU during training; they are
        # ~1 GB and are only needed for these few seconds.
        try:
            t3.eval()
            compat.use_eager_attention(t3)
            engine.s3gen.to(device).eval()
            engine.ve.to(device).eval()
            engine.device = device

            base_seed = getattr(self.config, "inference_seed", 1234)
            draws = max(1, getattr(self.config, "inference_draws", 3))
            stem, extension = os.path.splitext(output_path)
            floors = []

            for draw in range(draws):
                torch.manual_seed(base_seed + draw)

                with torch.no_grad():
                    wav = engine.generate(
                        self.config.inference_test_text,
                        language_id=self.config.language_id,
                        audio_prompt_path=self.config.inference_prompt_path,
                        temperature=0.8,
                        cfg_weight=0.5,
                        exaggeration=0.5,
                        repetition_penalty=1.2,
                    )

                audio = wav.squeeze().cpu().numpy()
                # Same treatment the real inference path applies, so a training
                # sample sounds like what the model will actually produce.
                audio = normalise_peak(trim_onset_artifact(audio, engine.sr))
                path = output_path if draw == 0 else f"{stem}_{draw + 1}{extension}"
                sf.write(path, audio, engine.sr)
                floors.append(noise_floor(audio, engine.sr))
                logger.info(
                    f"Sample saved: {path} ({len(audio) / engine.sr:.1f} s, "
                    f"floor {floors[-1]:.5f})"
                )

                # The cache is rebuilt per generation; dropping it between draws
                # keeps only one copy of the layers alive at a time.
                compat.drop_inference_cache(t3)

            # The median is what to read across checkpoints; a single draw is
            # too noisy to compare, and the median of three is not.
            floors.sort()
            logger.info(
                f"checkpoint-{os.path.basename(stem).split('-')[-1]}: "
                f"noise floor median {floors[len(floors) // 2]:.5f} "
                f"over {len(floors)} draw(s), range {floors[0]:.5f}-{floors[-1]:.5f}"
            )

        finally:
            # T3.inference caches a `patched_model` wrapping the same layers the
            # trainer owns, which makes the next checkpoint save fail on shared
            # tensors. compat walks the module tree to find it: `t3` here is a
            # PeftModel and the cache lives on the T3 inside it, so deleting the
            # name off this object does nothing. Costs one rebuild per sample.
            dropped = compat.drop_inference_cache(t3)
            if dropped:
                logger.debug(f"dropped inference cache: {', '.join(dropped)}")

            # Hand the training stream back exactly the randomness it had, so
            # seeding the sample cannot shift the data order or the dropout.
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

            engine.s3gen.to("cpu")
            engine.ve.to("cpu")
            if previous_attn and hasattr(t3.tfmr, "set_attn_implementation"):
                t3.tfmr.set_attn_implementation(previous_attn)
            if was_training:
                t3.train()
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # LoRA inference
    # -------------------------------------------------------------------------
    def _generate_sample_lora(self, checkpoint_dir: str, output_path: str):

        from peft import PeftModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_turbo = getattr(self.config, "is_turbo", False)
        EngineClass = ChatterboxTurboTTS if is_turbo else ChatterboxTTS

        inference_engine = None
        new_t3 = None

        try:
            # Rebuild the base T3 with resized vocab
            temp_original = EngineClass.from_local(self.config.model_dir, device="cpu")
            pretrained_state = temp_original.t3.state_dict()
            original_config = temp_original.t3.hp

            new_config = original_config
            new_config.text_tokens_dict_size = self.config.new_vocab_size
            if hasattr(new_config, "use_cache"):
                new_config.use_cache = False

            new_t3 = T3(hp=new_config)
            new_t3 = resize_and_load_t3_weights(new_t3, pretrained_state)

            if is_turbo and hasattr(new_t3.tfmr, "wte"):
                del new_t3.tfmr.wte

            del temp_original
            del pretrained_state

            inference_engine = EngineClass.from_local(self.config.model_dir, device="cpu")
            inference_engine.t3 = new_t3

            logger.info(f"Loading LoRA adapters from: {checkpoint_dir}")
            inference_engine.t3 = PeftModel.from_pretrained(
                inference_engine.t3,
                checkpoint_dir,
                is_trainable=False,
            )

            inference_engine.t3.to(device).eval()
            inference_engine.s3gen.to(device).eval()
            inference_engine.ve.to(device).eval()
            inference_engine.device = device

            params = {"temperature": 0.8, "repetition_penalty": 1.2}
            if not is_turbo:
                params["cfg_weight"] = 0.5
                params["exaggeration"] = 0.5

            with torch.no_grad():
                wav = inference_engine.generate(
                    text=self.config.inference_test_text,
                    audio_prompt_path=self.config.inference_prompt_path,
                    **params,
                )

            if isinstance(wav, tuple):
                wav = wav[0]

            wav_np = wav.squeeze().cpu().numpy()
            trimmed_wav = trim_silence_with_vad(wav_np, inference_engine.sr)
            sf.write(output_path, trimmed_wav, inference_engine.sr)
            logger.info(f"Example saved: {output_path}")

        except Exception as e:
            logger.error(f"LoRA inference callback failed: {e}", exc_info=True)

        finally:
            if inference_engine:
                del inference_engine
            if new_t3:
                del new_t3
            torch.cuda.empty_cache()
            logger.info("LoRA inference cleanup done.")

    # -------------------------------------------------------------------------
    # Full fine-tune inference
    # -------------------------------------------------------------------------
    def _generate_sample_full(self, checkpoint_path: str, output_path: str):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_turbo = getattr(self.config, "is_turbo", False)
        EngineClass = ChatterboxTurboTTS if is_turbo else ChatterboxTTS

        tts_engine = EngineClass.from_local(self.config.model_dir, device="cpu")

        t3_config = tts_engine.t3.hp
        if hasattr(self.config, 'new_vocab_size'):
            t3_config.text_tokens_dict_size = self.config.new_vocab_size

        new_t3 = T3(hp=t3_config)

        if is_turbo and hasattr(new_t3.tfmr, "wte"):
            del new_t3.tfmr.wte

        if checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        clean_state_dict = {}
        for k, v in state_dict.items():
            k_clean = k.replace("module.", "").replace("model.", "").replace("t3.", "")
            if k_clean.startswith("t3."):
                clean_state_dict[k_clean.replace("t3.", "")] = v
            elif not any(x in k_clean for x in ["s3gen", "ve.", "tokenizer"]):
                clean_state_dict[k_clean] = v

        missing_keys, unexpected_keys = new_t3.load_state_dict(clean_state_dict, strict=False)

        critical_missing = [k for k in missing_keys if "tfmr.layers" in k]
        if len(critical_missing) > 0:
            logger.error("[CRITICAL ERROR] Model weights COULD NOT BE LOADED!")
            logger.error(f"Number of missing keys: {len(missing_keys)}")
            logger.error(f"Examples of missing keys: {critical_missing[:3]}")
            logger.error("The sound produced will be 100% NOISE. Check your checkpoint saving method.")
        elif len(missing_keys) > 0:
            non_wte_missing = [k for k in missing_keys if "wte" not in k]
            if non_wte_missing:
                logger.warning(f"Some weights are missing ({len(non_wte_missing)} keys): {non_wte_missing[:3]}...")
            else:
                logger.info("Weights loaded successfully (WTE missing is normal for Turbo).")
        else:
            logger.info("All weights loaded completely and successfully.")

        tts_engine.t3 = new_t3
        tts_engine.t3.to(device).eval()
        tts_engine.s3gen.to(device).eval()
        tts_engine.ve.to(device).eval()
        tts_engine.device = device

        params = {"temperature": 0.8, "repetition_penalty": 1.2}
        if not is_turbo:
            params["cfg_weight"] = 0.2
            params["exaggeration"] = 1.2

        with torch.no_grad():
            wav = tts_engine.generate(
                text=self.config.inference_test_text,
                audio_prompt_path=self.config.inference_prompt_path,
                **params,
            )

        if isinstance(wav, tuple):
            wav = wav[0]

        wav_np = wav.squeeze().cpu().numpy()
        trimmed_wav = trim_silence_with_vad(wav_np, tts_engine.sr)
        sf.write(output_path, trimmed_wav, tts_engine.sr)
        logger.info(f"Example saved: {output_path}")

        del tts_engine
        del new_t3
        del state_dict
        del clean_state_dict
        torch.cuda.empty_cache()