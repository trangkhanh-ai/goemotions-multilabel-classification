"""Phần B: BART-MNLI zero-shot, không fine-tune; lưu điểm sau từng batch.

Chạy từ gốc repo: python -m scripts.zero_shot.run_zero_shot --smoke --device cpu
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from src.datasets.goemotions import REVISION, load_goemotions, multi_hot, sha256
from src.evaluation.metrics import evaluate_multilabel
from src.models.zero_shot import (CHECKPOINT, HYPOTHESIS_TEMPLATE, predict_in_batches,
                           validate_test_protocol, write_json, write_scores)


from src.paths import ROOT


def build_pipeline(checkpoint, revision, device, dtype="float32"):
    """Chỉ tải dependency nặng khi thật sự chạy, không tải khi --help hay test."""
    import torch
    from transformers import pipeline

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Bạn chọn CUDA nhưng PyTorch chưa thấy GPU CUDA")
    pipeline_device = 0 if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else -1
    if dtype == "float16" and pipeline_device < 0:
        raise ValueError("float16 chỉ dùng trên CUDA; CPU dùng float32")
    torch.set_num_threads(4)
    classifier = pipeline("zero-shot-classification", model=checkpoint, tokenizer=checkpoint,
                          revision=revision, device=pipeline_device, framework="pt",
                          trust_remote_code=False, dtype=getattr(torch, dtype))
    if classifier.entailment_id < 0:
        raise ValueError("Checkpoint không khai báo entailment; không thể tính điểm NLI đúng")
    classifier.model.eval()
    return classifier, "cuda" if pipeline_device == 0 else "cpu"


def resolve_model_revision(checkpoint, revision):
    """Lưu commit SHA chính xác, không ghi 'main' như một phiên bản cố định."""
    from huggingface_hub import HfApi

    resolved = HfApi().model_info(checkpoint, revision=revision).sha
    if not resolved or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("Không xác định được commit SHA 40 ký tự của checkpoint")
    return resolved


def environment():
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "pyarrow", "scikit-learn", "torch", "transformers", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def run(args):
    labels = json.loads((ROOT / "data/labels.json").read_text(encoding="utf-8"))
    # 28 nhãn chuẩn là điều kiện của đồ án; unit tests của module dùng ít nhãn.
    if len(labels) != 28 or len(set(labels)) != 28:
        raise ValueError("data/labels.json cần đúng 28 nhãn khác nhau")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("batch-size và limit phải dương")
    smoke = args.smoke or args.limit is not None
    output = ROOT / "data/processed/zero_shot" / ("smoke" if smoke else "full")
    protocol = None
    if args.split == "test":
        if smoke or not args.protocol:
            raise ValueError("Test cần --protocol từ full validation; không cho phép smoke/limit trên test")
        protocol, validation_metadata = validate_test_protocol(args.protocol, output, labels)
        if args.dtype != validation_metadata.get("dtype", "float32"):
            raise ValueError("Test phải giữ dtype đã dùng trên validation")
        checkpoint = protocol["checkpoint"]
        revision = protocol["model_revision"]
        template = protocol["hypothesis_template"]
    else:
        if args.protocol:
            raise ValueError("--protocol chỉ dùng cho test")
        checkpoint = CHECKPOINT
        revision = resolve_model_revision(checkpoint, args.revision)
        template = HYPOTHESIS_TEMPLATE

    # Validation không mở train/test. Test chỉ mở sau validator khóa cấu hình.
    frames, names, data_manifest = load_goemotions(ROOT, write_metadata=False, splits=(args.split,))
    if names != labels:
        raise ValueError("Nhãn Parquet không khớp data/labels.json")
    frame = frames[args.split]
    if smoke:
        frame = frame.iloc[:args.limit if args.limit is not None else 8]
    classifier, resolved_device = build_pipeline(checkpoint, revision, args.device, args.dtype)
    config = {"method": "zero_shot", "checkpoint": checkpoint, "model_revision": revision,
              "hypothesis_template": template, "multi_label": True, "split": args.split,
              "data_revision": REVISION,
              "data_sha256": {item["split"]: item["sha256"] for item in data_manifest["files"]},
              "smoke": smoke, "device": resolved_device, "dtype": args.dtype, "environment": environment()}
    if protocol is not None:
        config["protocol_sha256"] = sha256(args.protocol)
    started = perf_counter()
    # batch_size ở HF là số cặp NLI trên GPU; mỗi câu có 28 giả thuyết nhãn.
    def predict_batch(texts):
        return classifier(texts, candidate_labels=labels, hypothesis_template=template,
                          multi_label=True, batch_size=args.batch_size)

    scores, _ = predict_in_batches(
        predict_batch, frame["text"].tolist(), frame["id"].tolist(), labels,
        output / f"{args.split}_checkpoints", config, batch_size=args.batch_size,
    )
    scores_path = output / f"{args.split}_scores.npz"
    if scores_path.exists():
        with np.load(scores_path, allow_pickle=False) as saved:
            same = (saved["ids"].tolist() == frame["id"].tolist()
                    and saved["label_names"].tolist() == labels
                    and np.array_equal(saved["scores"], scores))
        if not same:
            raise ValueError("File điểm đã hoàn tất khác với checkpoint; không ghi đè")
    else:
        write_scores(scores_path, frame["id"].tolist(), scores, labels)
    truth = multi_hot(frame["labels"].tolist(), len(labels))
    threshold = 0.5 if protocol is None else protocol["thresholds"]
    metrics = evaluate_multilabel(truth, scores, labels, threshold=threshold)
    metadata = dict(config, artifact_version=1, status="complete", seed=None,
                    created_at_utc=datetime.now(timezone.utc).isoformat(), sample_count=len(frame),
                    label_names=labels, split_evaluated=args.split,
                    inference_seconds_this_invocation=round(perf_counter() - started, 2),
                    artifact_sha256={scores_path.name: sha256(scores_path)},
                    checkpoint_manifest_sha256=sha256(output / f"{args.split}_checkpoints/checkpoint_manifest.json"),
                    metrics=metrics)
    if args.split == "validation":
        metadata["n_validation"] = len(frame)
    else:
        metadata["threshold_mode"] = protocol["threshold_mode"]
        metadata["validation_run_metadata_sha256"] = protocol["run_metadata_sha256"]
        configurations = protocol.get("configurations", [{"threshold_mode": protocol["threshold_mode"],
                                                           "thresholds": protocol["thresholds"]}])
        metadata["configurations"] = [
            {"threshold_mode": item["threshold_mode"], "thresholds": item["thresholds"],
             "metrics": evaluate_multilabel(truth, scores, labels, item["thresholds"])}
            for item in configurations
        ]
    metadata_path = output / ("run_metadata.json" if args.split == "validation" else "test_run_metadata.json")
    # Khi resume một run đã hoàn tất, giữ metadata cũ để không làm hỏng protocol.
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        identity = ("checkpoint", "model_revision", "hypothesis_template", "data_revision", "data_sha256",
                    "label_names", "smoke", "split", "artifact_sha256")
        if any(previous.get(key) != metadata.get(key) for key in identity):
            raise ValueError("Metadata hoàn tất cũ không khớp lần chạy; không ghi đè")
        metadata = previous
    else:
        write_json(metadata_path, metadata)
    write_json(output / f"{args.split}_metrics.json", metadata["metrics"])
    print(f"{'SMOKE — không dùng làm kết quả đầy đủ' if smoke else 'FULL'} | {args.split}: {len(frame):,} câu")
    print(f"Macro-F1={metrics['macro_f1']:.4f}; Micro-F1={metrics['micro_f1']:.4f}; Hamming Loss={metrics['hamming_loss']:.4f}")
    print(f"Đã lưu: {scores_path}")
    return metadata


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="GoEmotions B: BART-MNLI zero-shot, không fine-tune")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32",
                        help="float16 cho GPU 8 GB; lưu rõ dtype, không đổi khi resume")
    parser.add_argument("--revision", default="main", help="HF commit/tag cho validation; lưu SHA chính xác")
    parser.add_argument("--smoke", action="store_true", help="Mặc định 8 câu validation")
    parser.add_argument("--limit", type=int, help="Số câu validation; luôn đánh dấu smoke kể cả nếu chọn hết")
    parser.add_argument("--protocol", type=Path, help="JSON đã khóa trên full validation trước khi dùng test")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
