import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_config(path):
    config = read_json(path)
    root = Path(path).resolve().parent.parent
    location = Path(config.get("oodmethod_dir", "../OODmethod")).expanduser()
    config["oodmethod_dir"] = str((root / location).resolve() if not location.is_absolute() else location.resolve())
    if config["k"] < 1 or config["max_new_tokens"] < 1:
        raise ValueError("k and max_new_tokens must be positive")
    return config
