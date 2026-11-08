import importlib.metadata
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .io import append_jsonl, digest, read_json, read_jsonl, write_json


def parse_answer(text, choice_count):
    matches = re.findall(r"Final\s+answer\s*:\s*([A-Z])\b", text, flags=re.I)
    if len(matches) != 1:
        return None
    answer = ord(matches[0].upper()) - 65
    return answer if 0 <= answer < choice_count else None


def paired_features(original, control, k):
    a, b = original["trace"], control["trace"]
    if [(t["position"], t["token_id"]) for t in a] != [(t["position"], t["token_id"]) for t in b]:
        raise ValueError("Paired traces do not score identical tokens")
    if not a:
        return {}, []
    records = []
    for x, y in zip(a, b):
        values = {}
        for name, key in (("nci", "nci"), ("fdbd", f"fdbd_k{k}"), ("logprob", "mean_logprob")):
            values[name] = x["scores"][key]
            values[name + "_gain"] = x["scores"][key] - y["scores"][key]
        values["negative_entropy"] = x["scores"]["negative_entropy"]
        records.append({"position": x["position"], "token_id": x["token_id"], "features": values})
    means = {key: float(np.mean([t["features"][key] for t in records])) for key in records[0]["features"]}
    if not all(np.isfinite(v) for v in means.values()):
        raise ValueError("Nonfinite paired features")
    return means, records


def check_replay(original, replay, atol, rtol):
    if original["token_ids"] != replay["token_ids"]:
        raise RuntimeError("Original replay token mismatch")
    a, b = original["trace"], replay["trace"]
    if len(a) != len(b):
        raise RuntimeError("Original replay trace length mismatch")
    error = 0.0
    for x, y in zip(a, b):
        if (x["position"], x["token_id"]) != (y["position"], y["token_id"]):
            raise RuntimeError("Original replay position mismatch")
        for key in x["scores"]:
            u, v = x["scores"][key], y["scores"][key]
            error = max(error, abs(u-v))
            if not np.isclose(u, v, atol=atol, rtol=rtol):
                raise RuntimeError(f"Original replay score mismatch at {x['position']} / {key}: {u} vs {v}")
    return {"max_absolute_score_difference": error, "atol": atol, "rtol": rtol,
            "checked": "per-token NCI, fDBD, actual-token logprob, negative entropy"}


def collect(config, data_path, output, *, limit=None, resume=False):
    from .backend import QwenBackend

    data_path, output = Path(data_path).resolve(), Path(output).resolve()
    rows = read_jsonl(data_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Empty data or duplicate sample IDs")
    versions = {}
    for package in ("torch", "transformers", "datasets", "numpy", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    scorer_path = Path(config["oodmethod_dir"]) / "ood_repro/scoring.py"
    image_hash = hashlib.sha256()
    for row in rows:
        image_hash.update(digest((data_path.parent / row["image"]).resolve()).encode())
    code_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        code_hash.update(path.name.encode() + path.read_bytes())
    contract = {"config": config, "data_sha256": digest(data_path), "scorer_sha256": digest(scorer_path),
                "images_sha256": image_hash.hexdigest(), "code_sha256": code_hash.hexdigest(),
                "limit": limit, "versions": versions}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, predictions = output / "manifest.json", output / "predictions.jsonl"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if not resume or previous["contract"] != contract:
            raise ValueError("Output exists: use --resume with identical config/data/limit/dependencies, or a new directory")
    elif any(output.iterdir()):
        raise ValueError("Output must be empty or contain a matching run manifest")
    else:
        write_json(manifest_path, {"contract": contract, "data_path": str(data_path),
                   "created": datetime.now(timezone.utc).isoformat(), "status": "running"})
    done = read_jsonl(predictions) if predictions.exists() else []
    if [r["sample_id"] for r in done] != [r["sample_id"] for r in rows[:len(done)]]:
        raise ValueError("Existing results are not an exact prefix of selected data")
    if len(done) > len(rows):
        raise ValueError("Existing result count exceeds selected data")
    if len(done) == len(rows):
        manifest = read_json(manifest_path)
        manifest.update(status="complete", n=len(rows))
        write_json(manifest_path, manifest)
        return
    backend = QwenBackend(config)
    import torch
    manifest = read_json(manifest_path)
    model_commit = getattr(backend.model.config, "_commit_hash", None)
    if manifest.get("resolved_model_commit") not in (None, model_commit):
        raise ValueError("Model revision resolved to a different commit; pin revision or use a new directory")
    manifest.update(resolved_model_commit=model_commit, status="running",
                    runtime={"cuda": torch.version.cuda,
                             "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    write_json(manifest_path, manifest)
    from tqdm import tqdm
    for index in tqdm(range(len(done), len(rows)), desc=data_path.stem):
        row = rows[index]
        image_path = (data_path.parent / row["image"]).resolve()
        original_inputs, gray_inputs = backend.pair_inputs(row, image_path)
        original = backend.run(original_inputs)
        verification = None
        replay = None
        if index < config["verify_replay_samples"]:
            replay = backend.run(original_inputs, forced=original["token_ids"])
            verification = check_replay(original, replay, config["verify_atol"], config["verify_rtol"])
        gray = backend.run(gray_inputs, forced=original["token_ids"])
        means, tokens = paired_features(original, gray, config["k"])
        parsed = parse_answer(original["text"], len(row["choices"]))
        # Use cumulative decoding for token-to-display-line mapping; no guessed token strings.
        tokenizer = backend.processor.tokenizer
        for token in tokens:
            prefix = tokenizer.decode(original["token_ids"][:token["position"]], skip_special_tokens=True)
            token["line"] = prefix.count("\n")
        record = {**row, "image_path": str(image_path), "image_sha256": digest(image_path),
                  "text": original["text"], "token_ids": original["token_ids"],
                  "parsed_answer": parsed, "is_correct": parsed == row["answer"],
                  "stop_reason": original["stop_reason"], "features": means, "tokens": tokens,
                  "original_trace": original["trace"], "gray_trace": gray["trace"],
                  "verification": verification,
                  "timing": {"generation": original["elapsed_seconds"], "gray_replay": gray["elapsed_seconds"],
                             "verification_replay": replay["elapsed_seconds"] if replay else 0.0},
                  "peak_cuda_bytes_by_device": {d: max(result["peak_cuda_bytes_by_device"].get(d, 0)
                       for result in [original, gray] + ([replay] if replay else []))
                       for d in original["peak_cuda_bytes_by_device"]},
                  "eligible": parsed is not None and original["stop_reason"] != "max_new_tokens" and bool(means)}
        append_jsonl(predictions, record)
    manifest = read_json(manifest_path)
    manifest.update(status="complete", n=len(rows))
    write_json(manifest_path, manifest)
