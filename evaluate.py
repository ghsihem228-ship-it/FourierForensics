"""Evaluate fixed detector scores; no model fitting or threshold optimization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from fourierforensics.io import load_image
from fourierforensics.scoring import analyze

NEGATIVE = {"au", "auth", "authentic", "original", "4cam_auth"}
POSITIVE = {"tp", "tampered", "spliced", "forged", "4cam_splc"}
EXCLUDED = {"mask", "masks", "groundtruth", "ground_truth", "gt"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def discover(root: Path, manifest: Path | None = None) -> list[dict]:
    root = root.resolve()
    manifest = manifest or (root / "manifest.json" if (root / "manifest.json").exists() else None)
    if manifest:
        if manifest.suffix.lower() == ".csv":
            with manifest.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
        else:
            payload = json.loads(manifest.read_text())
            rows = payload["samples"] if isinstance(payload, dict) else payload
    else:
        rows = []
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in EXTENSIONS or not path.is_file():
                continue
            folders = {p.lower() for p in path.relative_to(root).parts[:-1]}
            if folders & EXCLUDED:
                continue
            labels = ({0} if folders & NEGATIVE else set()) | ({1} if folders & POSITIVE else set())
            if len(labels) != 1:
                raise ValueError(f"Ambiguous or unlabelled image: {path}; supply a manifest")
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "label": labels.pop(),
                    "group": "",
                    "operation": "unknown",
                }
            )
    seen = set()
    for row in rows:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or path in seen or not path.is_file():
            raise ValueError(f"Invalid or duplicate manifest path: {row['path']}")
        if str(row["label"]) not in ("0", "1"):
            raise ValueError("Labels must be 0 or 1")
        row["label"] = int(row["label"])
        seen.add(path)
    if not rows:
        raise ValueError("No labelled images found")
    return rows


def metrics(labels: list[int], scores: list[float], threshold: float) -> dict:
    predicted = np.asarray(scores) >= threshold
    p, r, f, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    return {
        "images": len(labels),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else None,
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def evaluate(
    root: Path,
    output: Path,
    manifest: Path | None = None,
    threshold: float = 0.5,
    skip_errors: bool = False,
) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0,1]")
    samples = discover(root, manifest)
    predictions, errors = [], []
    for row in samples:
        try:
            data = (root / row["path"]).read_bytes()
            loaded = load_image(data)
            result = analyze(loaded.pixels, data)
            prediction = row | {
                "score": result.score,
                "predicted": int(result.score >= threshold),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            prediction.update({d.name: d.score if d.available else None for d in result.detectors})
            predictions.append(prediction)
        except (OSError, ValueError) as exc:
            if not skip_errors:
                raise ValueError(f"{row['path']}: {exc}") from exc
            errors.append({"path": row["path"], "error": str(exc)})
    if not predictions:
        raise ValueError("No images were successfully analyzed")
    labels = [p["label"] for p in predictions]
    scores = [p["score"] for p in predictions]
    report = {
        "combined": metrics(labels, scores, threshold),
        "threshold": threshold,
        "python": platform.python_version(),
        "errors": errors,
        "detectors": {},
        "operations": {},
    }
    for name in ("resampling", "noise", "jpeg"):
        cohort = [p for p in predictions if p[name] is not None]
        report["detectors"][name] = (
            metrics([p["label"] for p in cohort], [p[name] for p in cohort], threshold)
            if cohort
            else None
        )
    for operation in sorted({p.get("operation", "unknown") for p in predictions}):
        cohort = [p for p in predictions if p.get("operation", "unknown") == operation]
        report["operations"][operation] = {
            "images": len(cohort),
            "flagged": sum(p["predicted"] for p in cohort),
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    fields = [
        "path",
        "label",
        "group",
        "operation",
        "score",
        "predicted",
        "sha256",
        "resampling",
        "noise",
        "jpeg",
    ]
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predictions)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    if len(set(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, scores)
        axes[0, 0].plot(fpr, tpr, label=f"AUC = {report['combined']['roc_auc']:.3f}")
        axes[0, 0].legend()
        precision, recall, _ = precision_recall_curve(labels, scores)
        axes[0, 1].plot(recall, precision)
    axes[0, 0].set(
        title="ROC",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        xlim=(0, 1),
        ylim=(0, 1.05),
    )
    axes[0, 1].set(
        title="Precision–recall", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.05)
    )
    ConfusionMatrixDisplay(
        np.array(report["combined"]["confusion_matrix"]), display_labels=["Control", "Processed"]
    ).plot(ax=axes[1, 0], colorbar=False)
    for label, title in [(0, "Controls"), (1, "Processed")]:
        axes[1, 1].hist(
            [s for y, s in zip(labels, scores) if y == label],
            bins=np.linspace(0, 1, 16),
            alpha=0.6,
            label=title,
        )
    axes[1, 1].axvline(threshold, color="black", linestyle="--")
    axes[1, 1].set(title="Score distribution", xlabel="Artifact suspicion", ylabel="Images")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(output / "evaluation.png", dpi=140)
    plt.close(fig)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/custom"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip-errors", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.dataset, args.output, args.manifest, args.threshold, args.skip_errors),
            indent=2,
        )
    )
