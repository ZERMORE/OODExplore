"""Materialize image-bearing ScienceQA examples, never exposing solutions in prompts."""
import hashlib
import random
from pathlib import Path

from PIL import Image

from .io import write_json, append_jsonl

INSTRUCTION = ("Answer the multiple-choice question using the image.\n"
               "Give a brief explanation in 2–4 numbered lines.\n"
               "End with exactly: Final answer: <option letter>.")


def prompt(row):
    choices = "\n".join(f"{chr(65+i)}. {value}" for i, value in enumerate(row["choices"]))
    return f"{INSTRUCTION}\n\nQuestion: {row['question']}\n{choices}"


def image_key(image):
    image = image.convert("RGB")
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def prepare(output, *, dataset="derek-thomas/ScienceQA", revision="main", dev_count=100,
            test_count=500, seed=42):
    from .storage import configure_storage
    root = configure_storage()
    from datasets import load_dataset

    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preparation output must be empty; use another directory")
    if dev_count < 1 or test_count < 0:
        raise ValueError("dev_count >= 1; test_count >= 0 (0 means all)")
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir()
    data = load_dataset(dataset, revision=revision, cache_dir=str(root / "huggingface/datasets"))
    rng = random.Random(seed)
    used, counts, fingerprints = set(), {}, {}
    for split, source_split, limit in (("dev", "validation", dev_count), ("test", "test", test_count)):
        source = data[source_split]
        fingerprints[source_split] = source._fingerprint
        ids = list(range(len(source)))
        rng.shuffle(ids)
        written = 0
        for index in ids:
            row = source[index]
            image = row.get("image")
            if image is None:
                continue
            if not isinstance(image, Image.Image):
                raise TypeError("Expected datasets image decoding to return PIL images")
            key = image_key(image)
            # Unique images in both splits, including repeated figures/questions.
            if key in used:
                continue
            choices, answer = row["choices"], int(row["answer"])
            if not 2 <= len(choices) <= 26 or not 0 <= answer < len(choices):
                raise ValueError(f"Invalid choices/answer at {source_split}:{index}")
            used.add(key)
            relative = f"images/{key}.png"
            image.convert("RGB").save(output / relative)
            record = {"sample_id": f"{source_split}:{index}", "source_split": source_split,
                      "source_index": index, "image_id": key, "image": relative,
                      "question": row["question"], "choices": choices, "answer": answer}
            append_jsonl(output / f"{split}.jsonl", record)
            written += 1
            if limit and written == limit:
                break
        if not written or (limit and written < limit):
            raise ValueError(f"Only {written} unique images in {source_split}, requested {limit}")
        counts[split] = written
    write_json(output / "manifest.json", {"dataset": dataset, "revision": revision, "seed": seed,
               "counts": counts, "fingerprints": fingerprints, "deduplication": "RGB pixel hash; dev first",
               "sampling": "seeded shuffled source indices; no selection by model outcomes"})
