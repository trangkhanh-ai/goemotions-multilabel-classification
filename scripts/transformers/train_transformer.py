"""C1/C2/C3: fine-tune một kiến trúc, một seed, chỉ mở train/validation.

Ví dụ: python -m scripts.transformers.train_transformer --architecture bert --seed 42
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from src.datasets.goemotions import REVISION, load_goemotions, multi_hot
from src.evaluation.metrics import evaluate_multilabel
from src.models.transformer import (ARCHITECTURES, artifact_hashes, load_transformer_run,
                        multilabel_loss, positive_weights, resolve_device, run_folder,
                        seed_everything, training_config, write_json, configure_console)

from src.paths import ROOT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--revision", default="main", help="Resolve thành commit SHA và lưu lại")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--weighted", action="store_true", help="BCE pos_weight từ train, folder riêng")
    parser.add_argument("--smoke", action="store_true", help="64 train/32 validation, 1 epoch; không báo cáo cuối")
    parser.add_argument("--resume", action="store_true", help="Bỏ qua run hoàn tất đúng cấu hình/hash")
    parser.add_argument("--overwrite", action="store_true", help="Cho phép chạy lại riêng folder run này")
    return parser.parse_args()


def encoded_dataset(frame, tokenizer, labels, max_length):
    """Tokenize một lần trước epochs; nhãn float để BCE hỗ trợ đa nhãn."""
    import torch
    tokenizer.padding_side = "right"  # Collate bên dưới chỉ loại padding ở cuối câu.
    tokens = tokenizer(frame["text"].tolist(), truncation=True, padding="max_length",
                       max_length=max_length, return_tensors="pt")
    targets = torch.tensor(multi_hot(frame["labels"].tolist(), len(labels)), dtype=torch.float32)

    class TextDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(targets)

        def __getitem__(self, index):
            return {**{name: values[index] for name, values in tokens.items()},
                    "labels": targets[index]}

    return TextDataset(), targets.numpy()


def trim_padding_collate(items):
    """Stack batch rồi bỏ padding dư ở cuối, không cắt token thật hay 28 nhãn.

    Dataset vẫn tokenize/cache tối đa 128 token để code dễ hiểu. Trong từng
    batch, encoder chỉ tính tới câu dài nhất thực tế thay vì luôn đủ 128.
    """
    import torch
    batch = torch.utils.data.default_collate(items)
    mask = batch["attention_mask"]
    if mask.ndim != 2 or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("attention_mask phải là ma trận nhị phân N × token")
    longest = int(mask.sum(dim=1).max().item())
    if longest < 1 or torch.any(mask[:, longest:] != 0):
        raise ValueError("Dynamic trim cần tokenizer padding bên phải và ít nhất một token")
    for name, values in batch.items():
        if name != "labels" and values.ndim == 2 and values.shape == mask.shape:
            batch[name] = values[:, :longest].contiguous()
    return batch


# Tên mô tả được dùng trong phiên bản hướng dẫn đầu; cùng một hàm, không hai luồng.
dynamic_padding_collate = trim_padding_collate


def predict_scores(model, loader, device):
    """model.eval + no_grad; sigmoid độc lập cho 28 cảm xúc, không softmax."""
    import torch
    model.eval()
    batches = []
    with torch.no_grad():
        for batch in loader:
            inputs = {name: value.to(device) for name, value in batch.items() if name != "labels"}
            context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with context:
                logits = model(**inputs).logits
            if logits.ndim != 2 or logits.shape[1] != 28:
                raise ValueError("Head Transformer phải trả N × 28 logits")
            batches.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(batches)


def main():
    configure_console()
    args = parse_args()
    config = training_config(args.architecture, smoke=args.smoke, weighted=args.weighted,
                             epochs=args.epochs, learning_rate=args.learning_rate,
                             batch_size=args.batch_size, gradient_accumulation=args.gradient_accumulation,
                             max_length=args.max_length, device=args.device, revision=args.revision)
    output = run_folder(ROOT, args.architecture, args.seed, smoke=args.smoke, weighted=args.weighted)
    metadata_path = output / "run_metadata.json"
    if metadata_path.exists() and args.resume:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("completed"):
            previous = load_transformer_run(output, require_full=not args.smoke, expected_config=config)
            if previous["architecture"] != args.architecture or previous["seed"] != args.seed:
                raise ValueError("Run có seed/kiến trúc khác")
            print(f"Đã hoàn tất, hash/config đúng; bỏ qua {output}", flush=True)
            return
        if not args.overwrite:
            raise RuntimeError("Run này bị ngắt trước khi hoàn tất. Chạy lại đúng seed với "
                               "--overwrite; có thể kết hợp --resume để bỏ qua các seed đã xong.")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Run đã có dữ liệu: {output}. Dùng --resume hoặc --overwrite.")

    # Imports nặng nằm sau phần argparse/resume: --help không cần GPU/checkpoint.
    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    seed_everything(args.seed)
    torch.set_num_threads(4)  # Tránh quá nhiều CPU threads trên Windows 16 GB RAM.
    device = resolve_device(args.device)
    amp_enabled = device.type == "cuda"
    # Khóa cùng SHA cho tokenizer và encoder; không bật remote custom code.
    revision = HfApi().model_info(config["checkpoint"], revision=args.revision).sha
    if not revision:
        raise RuntimeError("Không resolve được commit SHA của checkpoint")
    started = time.perf_counter()
    frames, labels, manifest = load_goemotions(ROOT, write_metadata=False,
                                              splits=("train", "validation"))
    if labels != json.loads((ROOT / "data/labels.json").read_text(encoding="utf-8")):
        raise ValueError("Mapping dữ liệu không khớp data/labels.json")
    train = frames["train"].iloc[:64] if args.smoke else frames["train"]
    validation = frames["validation"].iloc[:32] if args.smoke else frames["validation"]
    if (train["id"].duplicated().any() or validation["id"].duplicated().any() or
            set(train["id"]) & set(validation["id"])):
        raise ValueError("ID trùng trong/between train và validation")
    if any(frame["text"].isna().any() or frame["text"].str.strip().eq("").any()
           for frame in (train, validation)):
        raise ValueError("Train/validation có câu rỗng")
    tokenizer = AutoTokenizer.from_pretrained(config["checkpoint"], revision=revision,
                                               trust_remote_code=False)
    train_data, y_train = encoded_dataset(train, tokenizer, labels, args.max_length)
    validation_data, y_validation = encoded_dataset(validation, tokenizer, labels, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               shuffle=True, generator=generator, num_workers=0,
                                               pin_memory=amp_enabled, collate_fn=trim_padding_collate)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size,
                                                    shuffle=False, num_workers=0,
                                                    pin_memory=amp_enabled, collate_fn=trim_padding_collate)
    model = AutoModelForSequenceClassification.from_pretrained(
        config["checkpoint"], revision=revision, num_labels=len(labels),
        id2label={i: name for i, name in enumerate(labels)},
        label2id={name: i for i, name in enumerate(labels)},
        problem_type="multi_label_classification", trust_remote_code=False,
    ).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    weights = torch.tensor(positive_weights(y_train), device=device) if args.weighted else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_steps = steps_per_epoch * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * config["warmup_ratio"]),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output.mkdir(parents=True, exist_ok=True)
    mapping = {"label_names": labels, "label2id": {name: i for i, name in enumerate(labels)},
               "id2label": {str(i): name for i, name in enumerate(labels)}}
    write_json(output / "label_mapping.json", mapping)
    metadata = {
        "artifact_version": 1, "method": "C", "completed": False,
        "architecture": args.architecture, "owner": ARCHITECTURES[args.architecture]["owner"],
        "seed": args.seed, "smoke": args.smoke, "variant": "weighted" if args.weighted else "standard",
        "checkpoint": config["checkpoint"], "checkpoint_revision": revision, "model_revision": revision,
        "data_revision": REVISION, "data_files": manifest["files"], "label_names": labels,
        "sizes": {"train": len(train), "validation": len(validation)},
        "config": config, "parameter_count": parameter_count,
        "pos_weight": weights.cpu().tolist() if weights is not None else None,
        "train_label_support": y_train.sum(axis=0).astype(int).tolist(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "device": str(device),
                        "gpu": torch.cuda.get_device_name(device) if amp_enabled else None,
                        "cuda_runtime": torch.version.cuda, "amp_fp16": amp_enabled,
                        "deterministic_algorithms": "warn_only; same-machine reproducibility goal"},
        "history": [], "selected_epoch": None,
    }
    write_json(metadata_path, metadata)
    best_f1 = -1.0
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, seen = 0.0, 0
        for index, batch in enumerate(train_loader):
            targets = batch["labels"].to(device)
            inputs = {name: value.to(device) for name, value in batch.items() if name != "labels"}
            context = torch.autocast("cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with context:
                logits = model(**inputs).logits
                loss = multilabel_loss(logits, targets, weights)
            # Mỗi group tích lũy tính mean theo số mẫu thực, kể cả group cuối thiếu batch.
            group_start = (index // args.gradient_accumulation) * args.gradient_accumulation
            group_end = min(group_start + args.gradient_accumulation, len(train_loader))
            first_sample = group_start * args.batch_size
            group_samples = min(group_end * args.batch_size, len(train_data)) - first_sample
            scaled_loss = loss * (len(targets) / group_samples)
            scaler.scale(scaled_loss).backward()
            total_loss += float(loss.detach()) * len(targets)
            seen += len(targets)
            if (index + 1) % args.gradient_accumulation == 0 or index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:  # Không advance LR khi AMP bỏ step vì overflow.
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (index + 1) % 100 == 0:
                print(f"{args.architecture} seed={args.seed} epoch={epoch} "
                      f"batch={index + 1}/{len(train_loader)} loss={total_loss / seen:.5f}", flush=True)
        prediction_started = time.perf_counter()
        scores = predict_scores(model, validation_loader, device)
        validation_seconds = time.perf_counter() - prediction_started
        metrics = evaluate_multilabel(y_validation, scores, labels, threshold=0.5)
        metadata["history"].append({"epoch": epoch, "train_loss": total_loss / seen,
                                    "macro_f1": metrics["macro_f1"], "micro_f1": metrics["micro_f1"],
                                    "epoch_seconds": time.perf_counter() - epoch_start,
                                    "validation_seconds": validation_seconds})
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            model.save_pretrained(output / "checkpoint", safe_serialization=True)
            tokenizer.save_pretrained(output / "checkpoint")
            np.savez_compressed(output / "validation_scores.npz", ids=validation["id"].to_numpy(dtype=str),
                                scores=scores, label_names=np.asarray(labels, dtype=str))
            write_json(output / "validation_metrics.json", metrics)
            metadata["selected_epoch"] = epoch
        write_json(metadata_path, metadata)
        print(f"{args.architecture} seed={args.seed} epoch={epoch}: "
              f"Macro-F1={metrics['macro_f1']:.4f}; Micro-F1={metrics['micro_f1']:.4f}", flush=True)
    metadata["completed"] = True
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["elapsed_seconds"] = time.perf_counter() - started
    metadata["artifact_sha256"] = artifact_hashes(output)
    write_json(metadata_path, metadata)
    # Lần kiểm cuối phát hiện artifact thiếu/sai trước khi báo thành công.
    load_transformer_run(output, require_full=not args.smoke, expected_config=config)
    print(f"Đã lưu C {'SMOKE' if args.smoke else 'FULL'}: {output}; best epoch={metadata['selected_epoch']}", flush=True)


if __name__ == "__main__":
    main()
