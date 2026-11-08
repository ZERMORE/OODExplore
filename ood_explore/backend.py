"""Use native HF generation for multimodal positions/cache; force replay tokens only.

Geometry always reads RAW LM-head output, before any generation processors.
Only the last position is retained. No all-layer hidden states or full-vocabulary
generation traces are stored. The validated GeometryScorer is imported unchanged.
"""
import importlib.util
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image

from .data import prompt


def load_scorer(directory):
    path = Path(directory) / "ood_repro" / "scoring.py"
    if not path.is_file():
        raise FileNotFoundError(f"Existing scorer not found: {path}; set oodmethod_dir")
    spec = importlib.util.spec_from_file_location("ood_explore_reused_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GeometryScorer


@contextmanager
def capture_head(model):
    state = {}

    def before(module, args):
        state["z"] = args[0][0, -1].detach().clone()

    def after(module, args, output):
        state["logits"] = output[0, -1].detach().float().clone()

    head = model.get_output_embeddings()
    handles = [head.register_forward_pre_hook(before), head.register_forward_hook(after)]
    try:
        yield state
    finally:
        for handle in handles:
            handle.remove()


class TraceProcessor:
    """HF-compatible logits processor. Works without subclassing Transformers."""
    def __init__(self, state, scorer, stop_ids, forced=None):
        self.state, self.scorer, self.stop_ids = state, scorer, stop_ids
        self.forced, self.trace, self.step = forced, [], 0

    def __call__(self, input_ids, scores):
        if scores.shape[0] != 1:
            raise ValueError("Only batch=1, greedy generation is supported")
        raw, z = self.state.pop("logits"), self.state.pop("z")
        token = int(scores[0].argmax()) if self.forced is None else self.forced[self.step]
        if token not in self.stop_ids:
            result = self.scorer.score(z, raw, token)
            self.trace.append({"position": self.step, "token_id": token,
                               "scores": {k: float(v) for k, v in result.items()}})
        self.step += 1
        if self.forced is not None:
            scores = torch.full_like(scores, -torch.inf)
            scores[0, token] = 0
        return scores


class QwenBackend:
    def __init__(self, config):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.config = config
        torch.manual_seed(config["seed"])
        self.processor = AutoProcessor.from_pretrained(config["model"], revision=config["revision"],
            min_pixels=config["min_pixels"], max_pixels=config["max_pixels"])
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config["model"],
            revision=config["revision"], torch_dtype=getattr(torch, config["dtype"]),
            device_map=config["device_map"], attn_implementation=config["attention"]).eval()
        head = self.model.get_output_embeddings()
        self.scorer = load_scorer(config["oodmethod_dir"])(head.weight, head.bias,
            ks=(config["k"],), metrics={"nci", "mean_logprob", "negative_entropy"})
        self.device = self.model.get_input_embeddings().weight.device

    def inputs(self, row, image):
        messages = [{"role": "user", "content": [{"type": "image"},
                     {"type": "text", "text": prompt(row)}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def run(self, inputs, forced=None):
        from transformers import GenerationConfig, LogitsProcessorList

        # Native generate owns multimodal rotary positions and creates a fresh cache per call.
        cfg = GenerationConfig.from_model_config(self.model.config)
        for name in ("eos_token_id", "pad_token_id", "bos_token_id"):
            value = getattr(self.model.generation_config, name, None)
            if value is not None:
                setattr(cfg, name, value)
        cfg.do_sample, cfg.num_beams, cfg.use_cache = False, 1, True
        cfg.max_new_tokens = len(forced) if forced is not None else self.config["max_new_tokens"]
        cfg.repetition_penalty, cfg.no_repeat_ngram_size = 1.0, 0
        cfg.forced_bos_token_id = cfg.forced_eos_token_id = None
        cfg.suppress_tokens = cfg.begin_suppress_tokens = None
        stops = cfg.eos_token_id
        stops = set(stops if isinstance(stops, (tuple, list)) else [stops]) - {None}
        if forced is not None and (not forced or any(t in stops for t in forced[:-1])):
            raise ValueError("Forced sequence must be nonempty with EOS only at the end")
        cuda_devices = list(range(torch.cuda.device_count()))
        for device in cuda_devices:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with capture_head(self.model) as state:
            proc = TraceProcessor(state, self.scorer, stops, forced)
            result = self.model.generate(**inputs, generation_config=cfg,
                logits_processor=LogitsProcessorList([proc]), return_dict_in_generate=False)
        for device in cuda_devices:
            torch.cuda.synchronize(device)
        ids = result[0, inputs["input_ids"].shape[1]:].tolist()
        if forced is not None and ids != forced:
            raise RuntimeError("Replay did not reproduce the exact forced token sequence")
        if len(ids) != proc.step:
            raise RuntimeError("Generation callbacks and output tokens are misaligned")
        return {"token_ids": ids, "trace": proc.trace,
                "text": self.processor.tokenizer.decode(ids, skip_special_tokens=True),
                "stop_reason": "eos" if ids and ids[-1] in stops else "max_new_tokens",
                "elapsed_seconds": time.perf_counter()-start,
                "peak_cuda_bytes_by_device": {str(d): torch.cuda.max_memory_allocated(d) for d in cuda_devices}}

    def pair_inputs(self, row, path):
        with Image.open(path) as source:
            image = source.convert("RGB")
        original = self.inputs(row, image)
        gray = self.inputs(row, Image.new("RGB", image.size, (127, 127, 127)))
        for name in ("input_ids", "attention_mask", "image_grid_thw"):
            if name not in original or not torch.equal(original[name], gray[name]):
                raise RuntimeError(f"Original/gray prompt or image layout mismatch: {name}")
        return original, gray
