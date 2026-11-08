"""Frozen development ECDF fusion, paired cluster bootstrap and compact artifacts."""
import csv
import os
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .io import read_json, read_jsonl, write_json, digest

CONFIDENCES = ("nci", "nci_gain", "fdbd", "fdbd_gain", "logprob", "logprob_gain", "negative_entropy")
PAIRS = {"joint_fdbd": ("fdbd", "fdbd_gain"), "joint_nci": ("nci", "nci_gain"),
         "joint_probability": ("logprob", "logprob_gain")}


def fit_reference(rows):
    rows = [r for r in rows if r["eligible"]]
    if not rows:
        raise ValueError("No eligible development responses")
    return {key: sorted(r["features"][key] for r in rows) for key in CONFIDENCES}


def risks(features, reference):
    result = {key: -float(features[key]) for key in CONFIDENCES}
    for name, pair in PAIRS.items():
        result[name] = float(np.mean([1 - np.searchsorted(reference[k], features[k], side="right") / len(reference[k])
                                     for k in pair]))
    return result


def cluster_indices(rows, rng):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["image_id"], []).append(i)
    values = list(groups.values())
    return np.concatenate([values[i] for i in rng.integers(0, len(values), len(values))])


def evaluate(rows, reference, bootstrap=1000, seed=42):
    valid = [r for r in rows if r["eligible"]]
    report = {"n": len(rows), "n_eligible": len(valid),
              "accuracy_all": float(np.mean([r["is_correct"] for r in rows])) if rows else None,
              "parse_failures": sum(r["parsed_answer"] is None for r in rows),
              "truncated": sum(r["stop_reason"] == "max_new_tokens" for r in rows),
              "positive_class": "incorrect final option; not necessarily a visual hallucination",
              "orientation": "all analysis scores are risks (higher = more likely error)"}
    if not valid:
        return {**report, "undefined_reason": "No eligible responses"}
    y = np.array([not r["is_correct"] for r in valid], dtype=int)
    report["error_prevalence"] = float(y.mean())
    report["n_errors"] = int(y.sum())
    report["low_error_count"] = int(y.sum()) < 50
    if len(set(y)) < 2:
        return {**report, "undefined_reason": "AUROC requires correct and incorrect responses"}
    values = [risks(r["features"], reference) for r in valid]
    arrays = {key: np.array([r[key] for r in values]) for key in values[0]}
    report["metrics"] = {key: {"auroc": float(roc_auc_score(y, a)),
                               "auprc_average_precision": float(average_precision_score(y, a))}
                         for key, a in arrays.items()}
    rng = np.random.default_rng(seed)
    aucs = {key: [] for key in arrays}
    for _ in range(bootstrap):
        ids = cluster_indices(valid, rng)
        if len(set(y[ids])) < 2:
            continue
        for key, a in arrays.items():
            aucs[key].append(roc_auc_score(y[ids], a[ids]))
    accepted = len(aucs["joint_fdbd"])
    report["bootstrap"] = {"requested": bootstrap, "accepted": accepted, "unit": "image cluster", "seed": seed}
    if accepted:
        for key, a in aucs.items():
            report["metrics"][key]["auroc_ci95"] = np.quantile(a, [.025, .975]).tolist()
        report["paired_delta_auroc"] = {}
        for baseline in ("fdbd", "joint_probability"):
            delta = np.array(aucs["joint_fdbd"]) - np.array(aucs[baseline])
            report["paired_delta_auroc"][f"joint_fdbd_minus_{baseline}"] = {
                "delta": report["metrics"]["joint_fdbd"]["auroc"] - report["metrics"][baseline]["auroc"],
                "ci95": np.quantile(delta, [.025, .975]).tolist()}
    return report


def analyze(dev_path, test_path, output, *, bootstrap=1000, seed=42, cases=5):
    dev, test = read_jsonl(dev_path), read_jsonl(test_path)
    if bootstrap < 0 or cases < 0:
        raise ValueError("bootstrap and cases must be nonnegative")
    for name, rows in (("dev", dev), ("test", test)):
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate IDs in {name}")
        expected = "validation" if name == "dev" else "test"
        if any(r["source_split"] != expected for r in rows):
            raise ValueError(f"{name} must contain only official {expected} rows")
    manifests = [read_json(Path(p).parent / "manifest.json") for p in (dev_path, test_path)]
    for key in ("config", "scorer_sha256", "code_sha256"):
        if manifests[0]["contract"][key] != manifests[1]["contract"][key]:
            raise ValueError(f"Development and test run differ in {key}")
    if manifests[0].get("resolved_model_commit") != manifests[1].get("resolved_model_commit"):
        raise ValueError("Development/test model revisions differ")
    if any(m["status"] != "complete" for m in manifests):
        raise ValueError("Analyze complete runs only")
    if {r["image_id"] for r in dev} & {r["image_id"] for r in test}:
        raise ValueError("Development/test image leakage")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Analysis output must be empty; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    reference = fit_reference(dev)
    write_json(output / "reference.json", reference)
    write_json(output / "provenance.json", {"dev_sha256": digest(dev_path), "test_sha256": digest(test_path),
               "run_manifest": manifests[0], "bootstrap": bootstrap, "seed": seed, "cases": cases})
    report = evaluate(test, reference, bootstrap, seed)
    report["resources"] = {
        "total_seconds_by_phase": {key: sum(r.get("timing", {}).get(key, 0) for r in test)
                                   for key in ("generation", "gray_replay", "verification_replay")},
        "max_allocated_cuda_bytes_by_device": {device: max(r.get("peak_cuda_bytes_by_device", {}).get(device, 0)
                                                           for r in test)
            for device in sorted({d for r in test for d in r.get("peak_cuda_bytes_by_device", {})})},
        "note": "Allocated PyTorch memory, includes model; excludes reserved memory. Generation and replay are separate passes."
    }
    write_json(output / "metrics.json", report)
    valid = [r for r in test if r["eligible"]]
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["sample_id", "is_correct", *CONFIDENCES, *PAIRS]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in valid:
            writer.writerow({"sample_id": row["sample_id"], "is_correct": row["is_correct"],
                             **risks(row["features"], reference)})
    table = ["# Visual-conditioned geometry experiment", "",
             f"Total: {len(test)}; eligible: {len(valid)}. Positive class: final-answer error.", "",
             "| Method (higher risk = more error) | AUROC | AP |", "|---|---:|---:|"]
    for name, value in report.get("metrics", {}).items():
        table.append(f"| {name} | {value['auroc']:.4f} | {value['auprc_average_precision']:.4f} |")
    if report.get("undefined_reason"):
        table.extend(["", report["undefined_reason"]])
    if report.get("low_error_count"):
        table.extend(["", "Fewer than 50 eligible errors: consider the predeclared full image-test extension."])
    table.extend(["", "logprob = NLL risk; negative_entropy = entropy risk. AP is average precision.",
                  "", "Full paired confidence intervals, prevalence and exclusions: metrics.json.",
                  "", "Visual sensitivity is not equivalent to correct visual grounding."])
    (output / "report.md").write_text("\n".join(table)+"\n", encoding="utf-8")
    if valid:
        make_figures(valid, reference, output, seed, cases)
    (output / "COMPLETE").write_text("Analysis completed.\n", encoding="utf-8")


def make_figures(rows, reference, output, seed, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c0, g0 = np.median(reference["fdbd"]), np.median(reference["fdbd_gain"])
    groups = {}
    fig, ax = plt.subplots(figsize=(7, 5))
    for correct, color in ((True, "#257a73"), (False, "#be4a3b")):
        selected = [r for r in rows if r["is_correct"] == correct]
        ax.scatter([r["features"]["fdbd"] for r in selected], [r["features"]["fdbd_gain"] for r in selected],
                   s=14, alpha=.55, color=color, label="Correct" if correct else "Incorrect")
    ax.axvline(c0, color="gray", ls="--")
    ax.axhline(g0, color="gray", ls="--")
    ax.set(xlabel="Original-image fDBD confidence C", ylabel="Original minus gray gain G")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "quadrants.png", dpi=160)
    plt.close(fig)
    for row in rows:
        f = row["features"]
        key = ("high_C" if f["fdbd"] >= c0 else "low_C") + "/" + ("high_G" if f["fdbd_gain"] >= g0 else "low_G")
        groups.setdefault(key, []).append(row)
    write_json(output / "quadrants.json", {"thresholds_from_dev": {"C": float(c0), "G": float(g0)},
               "groups": {key: {"n": len(group), "errors": sum(not r["is_correct"] for r in group),
                                  "error_rate": float(np.mean([not r["is_correct"] for r in group]))}
                          for key, group in groups.items()}})
    pool = groups.get("high_C/low_G", [])
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(pool), min(20, len(pool)), replace=False).tolist())
    with (output / "manual_review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "is_correct", "question", "text", "image_path", "error_type", "visual_necessity", "notes"])
        for i in chosen:
            r = pool[i]
            writer.writerow([r["sample_id"], r["is_correct"], r["question"], r["text"], r["image_path"], "", "", ""])
    selected = []
    for correct in (True, False):
        selected.extend(sorted([r for r in rows if r["is_correct"] == correct], key=lambda r: r["sample_id"])[:cases])
    lines = ["# Process cases", "", "Deterministic first IDs per outcome, not selected for curve shape.",
             "Physical output lines are display units, not verified internal reasoning stages.", ""]
    case_dir = output / "cases"
    case_dir.mkdir(exist_ok=True)
    for i, row in enumerate(selected):
        grouped = {}
        for token in row["tokens"]:
            grouped.setdefault(token["line"], []).append(token["features"])
        positions = sorted(grouped)
        fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        for ax, key in zip(axes, ("fdbd", "fdbd_gain")):
            ax.plot(positions, [np.mean([t[key] for t in grouped[j]]) for j in positions], marker="o")
            ax.set_ylabel("C" if key == "fdbd" else "G")
        axes[-1].set_xlabel("Output line (zero-based)")
        fig.suptitle(f"{row['sample_id']} — {'correct' if row['is_correct'] else 'incorrect'}")
        fig.tight_layout()
        fig.savefig(case_dir / f"case-{i:02d}.png", dpi=160)
        plt.close(fig)
        image = os.path.relpath(row["image_path"], output).replace("\\", "/")
        # Keep verbatim model text in a code fence; it is data, not document instructions.
        fence = "`" * max(3, max((len(part) for part in re.findall(r'`+', row['text'])), default=0)+1)
        lines.extend([f"## {row['sample_id']}", "", row["question"], "",
                      f"![input](<{image}>)", "", f"![trajectory](cases/case-{i:02d}.png)", "",
                      fence+"text", row["text"], fence, ""])
    (output / "cases.md").write_text("\n".join(lines), encoding="utf-8")
