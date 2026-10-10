"""Dữ liệu, checkpoint và suy luận C3; app và CLI dùng cùng một hàm dự đoán."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

from src.paths import ROOT
# Cache lớn ở ổ chứa repo, không chiếm ổ hệ thống. Tôn trọng giá trị do người chạy đặt.
os.environ.setdefault("HF_HOME", str(ROOT / "data/cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from src.datasets.goemotions import load_goemotions, multi_hot, sha256
from src.evaluation.metrics import validate_thresholds


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def validate_config(config):
    if config["mode"] not in ("pilot", "full"):
        raise ValueError("mode phải là pilot/full")
    if config["model_id"] != "distilbert/distilbert-base-uncased":
        raise ValueError("Runner C3 này dành cho checkpoint DistilBERT đã thống nhất")
    if len(config["revision"]) != 40:
        raise ValueError("Ghim model revision bằng commit đầy đủ")
    for key in ("epochs", "max_length", "batch_size", "gradient_accumulation_steps", "eval_batch_size"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} phải là số nguyên dương")
    if config["max_length"] > 512 or config["threshold"] != 0.5:
        raise ValueError("C3 dùng max_length <= 512; bảng chính/chọn checkpoint tại ngưỡng 0.5")
    if config["learning_rate"] <= 0 or not 0 <= config["warmup_ratio"] <= 1:
        raise ValueError("Sai learning_rate hoặc warmup_ratio")
    if config["weight_decay"] < 0 or config["max_grad_norm"] <= 0:
        raise ValueError("Sai weight_decay hoặc max_grad_norm")
    limits = [config["train_limit"], config["validation_limit"]]
    if config["mode"] == "full" and any(x is not None for x in limits):
        raise ValueError("Không được ghi full cho dữ liệu bị lấy mẫu")
    if config["mode"] == "pilot" and any(not isinstance(x, int) or x < 1 for x in limits):
        raise ValueError("Pilot phải khai báo số mẫu train/validation")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Một số backend không có thuật toán deterministic; log cảnh báo và ghi môi trường.
    torch.use_deterministic_algorithms(True, warn_only=True)


def prepare_frames(config):
    validate_config(config)
    frames, names, manifest = load_goemotions(
        ROOT, write_metadata=False, splits=("train", "validation")
    )
    if names != read_json(ROOT / "data/labels.json"):
        raise ValueError("Mapping không khớp data/labels.json")
    for split in ("train", "validation"):
        limit = config[f"{split}_limit"]
        df = frames[split]
        if limit is not None:
            if limit > len(df):
                raise ValueError(f"{split}_limit lớn hơn số mẫu gốc")
            # Seed lấy mẫu cố định, độc lập seed huấn luyện; không chọn mẫu theo nhãn.
            df = df.sample(n=limit, random_state=config["subset_seed"]).sort_index()
        df = df.reset_index(drop=True)
        if df.id.duplicated().any() or df.text.isna().any() or df.text.str.strip().eq("").any():
            raise ValueError("Có ID trùng hoặc văn bản thiếu/rỗng")
        frames[split] = df
    if set(frames["train"].id) & set(frames["validation"].id):
        raise ValueError("ID giao nhau giữa train và validation")
    return frames, names, manifest


class EmotionDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length, n_labels=28):
        self.encodings = tokenizer(frame.text.tolist(), truncation=True, max_length=max_length, padding=False)
        self.targets = multi_hot(frame.labels.tolist(), n_labels).astype(np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.targets[index].tolist()
        return item


def make_loader(dataset, tokenizer, batch_size, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed),
                      collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"))


@torch.inference_mode()
def score_loader(model, loader, device):
    model.eval()
    scores = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        scores.append(model(**inputs).logits.float().sigmoid().cpu().numpy())
    return np.concatenate(scores)


def load_bundle(run_dir, device="cpu"):
    """Chỉ nạp checkpoint cục bộ hoàn thành; kiểm hash/mapping trước suy luận."""
    run_dir = Path(run_dir).resolve()
    metadata = read_json(run_dir / "run.json")
    if metadata.get("status") != "complete":
        raise ValueError("Run chưa hoàn thành")
    for filename, expected in metadata["artifact_sha256"].items():
        if sha256(run_dir / filename) != expected:
            raise ValueError(f"Artifact bị thay đổi: {filename}")
    names = metadata["label_names"]
    if names != read_json(ROOT / "data/labels.json"):
        raise ValueError("Run không theo thứ tự 28 nhãn chuẩn")
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "best", local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(run_dir / "best", local_files_only=True)
    if model.config.num_labels != len(names) or [model.config.id2label[i] for i in range(len(names))] != names:
        raise ValueError("Mapping của checkpoint khác metadata")
    if model.config.problem_type != "multi_label_classification":
        raise ValueError("Checkpoint không phải multi-label")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Môi trường này chưa có CUDA khả dụng")
    model.to(device).eval()
    return {"model": model, "tokenizer": tokenizer, "metadata": metadata, "device": device}


@torch.inference_mode()
def predict_texts(bundle, texts):
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Hãy nhập ít nhất một văn bản không rỗng")
    if any(len(t) > 20000 for t in texts):
        raise ValueError("Mỗi văn bản tối đa 20.000 ký tự")
    tok, model, metadata = bundle["tokenizer"], bundle["model"], bundle["metadata"]
    model.eval()
    lengths = [len(x) for x in tok(texts, truncation=False, padding=False)["input_ids"]]
    max_length = metadata["config"]["max_length"]
    inputs = tok(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    scores = model(**{k: v.to(bundle["device"]) for k, v in inputs.items()}).logits.float().sigmoid().cpu().numpy()
    return format_predictions(texts, scores, metadata["label_names"], lengths, max_length,
                              metadata["config"]["threshold"])


def format_predictions(texts, scores, names, lengths, max_length, threshold):
    limits = validate_thresholds(threshold, len(names))
    scores = np.asarray(scores)
    if scores.shape != (len(texts), len(names)) or len(lengths) != len(texts):
        raise ValueError("Kích thước scores/độ dài không khớp")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores không hợp lệ")
    return [{"text": text, "labels": [name for name, take in zip(names, row >= limits) if take],
             "scores": {name: float(value) for name, value in zip(names, row)},
             "original_tokens": int(length), "truncated": bool(length > max_length),
             "max_length": max_length}
            for text, row, length in zip(texts, scores, lengths)]
