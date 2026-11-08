import json

import pytest

from evaluate import discover, evaluate, metrics
from scripts.generate_synthetic import generate


def test_dataset_and_evaluation(tmp_path):
    root = tmp_path / "data"
    generate(root, count=1, seed=1234)
    assert len(discover(root)) == 6
    output = tmp_path / "results"
    report = evaluate(root, output)
    assert report["combined"]["images"] == 6
    assert (output / "evaluation.png").stat().st_size > 1000
    assert len((output / "predictions.csv").read_text().splitlines()) == 7
    assert json.loads((output / "metrics.json").read_text()) == report
    with pytest.raises(FileExistsError):
        generate(root)


def test_folder_labels_and_path_safety(tmp_path):
    root = tmp_path / "data"
    generate(root, count=1)
    (root / "manifest.json").unlink()
    assert len(discover(root)) == 6
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps([{"path": "../outside.png", "label": 1}]))
    with pytest.raises(ValueError):
        discover(root)


def test_single_class_and_threshold():
    assert metrics([0, 0], [0.1, 0.2], 0.5)["roc_auc"] is None
    assert metrics([0, 1], [0.4, 0.5], 0.5)["recall"] == 1
