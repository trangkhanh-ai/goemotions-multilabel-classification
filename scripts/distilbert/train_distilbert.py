"""C3 pilot/full, train + validation only. python -m scripts.distilbert.train_distilbert --help"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import subprocess
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from src.models.distilbert_study import (ROOT, read_json, write_json, validate_config, seed_everything,
                    prepare_frames, EmotionDataset, make_loader, score_loader, load_bundle, predict_texts)
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from src.datasets.goemotions import REVISION, sha256
from src.evaluation.metrics import evaluate_multilabel


def optimizer_update(model, optimizer, scheduler, scaler, max_grad_norm):
    """AMP overflow: GradScaler bỏ bước, giảm scale; scheduler chỉ đi khi weights cập nhật."""
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm,
                                  error_if_nonfinite=not scaler.is_enabled())
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= old_scale
    if updated:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return updated


def run_training(config, output):
    validate_config(config)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Không ghi đè run có sẵn: {output}. Chọn --output mới.")
    device = config["device"]
    if device not in ("cpu", "cuda"):
        raise ValueError("device phải là cpu hoặc cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Chưa có CUDA. Dùng môi trường .venv-huy đã cài torch CUDA.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    if "cpu_threads" in config:
        torch.set_num_threads(config["cpu_threads"])
    sources = ["src/models/distilbert_study.py", "src/datasets/goemotions.py", "src/evaluation/metrics.py", "scripts/distilbert/train_distilbert.py"]
    source_hashes = {p: sha256(ROOT/p) for p in sources}
    for name in sources:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    seed_everything(config["seed"])
    started = perf_counter()
    frames, names, manifest = prepare_frames(config)
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"])
    datasets = {s: EmotionDataset(df, tokenizer, config["max_length"]) for s, df in frames.items()}
    train_loader = make_loader(datasets["train"], tokenizer, config["batch_size"], True, config["seed"])
    val_loader = make_loader(datasets["validation"], tokenizer, config["eval_batch_size"])
    for split, df in frames.items():
        df[["id"]].to_csv(output / f"{split}_ids.csv", index=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_id"], revision=config["revision"], num_labels=28,
        problem_type="multi_label_classification", id2label=dict(enumerate(names)),
        label2id={name: i for i, name in enumerate(names)}, use_safetensors=True,
    ).to(device)
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Không decay bias/LayerNorm; ghi rõ để các run dùng cùng optimizer.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim == 1 or name.endswith(".bias") else decay).append(p)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": config["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0}], lr=config["learning_rate"])
    accumulation = config["gradient_accumulation_steps"]
    n_steps = math.ceil(len(train_loader) / accumulation) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, math.floor(n_steps * config["warmup_ratio"]), n_steps)
    amp = config["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, best_score, best_epoch, optimizer_steps = [], -1.0, None, 0
    skipped_amp_steps = []
    fit_started = perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        epoch_started = perf_counter()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["labels"] = batch["labels"].float()
            batch_n = len(batch["labels"])
            # Chuẩn hóa theo số mẫu thực trong cửa sổ; xử lý đúng cửa sổ cuối chưa đủ batch.
            window_start = (step // accumulation) * accumulation * config["batch_size"]
            window_n = min(accumulation * config["batch_size"], len(datasets["train"]) - window_start)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Loss không hữu hạn")
            loss_sum += loss.item() * batch_n
            scaler.scale(loss * batch_n / window_n).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                updated = optimizer_update(model, optimizer, scheduler, scaler, config["max_grad_norm"])
                optimizer_steps += int(updated)
                if not updated:
                    skipped_amp_steps.append({"epoch":epoch,"batch":step+1,"new_scale":scaler.get_scale()})
                    print(f"AMP overflow: skipped update epoch={epoch} batch={step+1}; new scale={scaler.get_scale()}",flush=True)
            if (step + 1) % 256 == 0 or step + 1 == len(train_loader):
                print(f"epoch {epoch}/{config['epochs']} batch {step+1}/{len(train_loader)} loss={loss.item():.5f}", flush=True)
        scores = score_loader(model, val_loader, device)
        metric = evaluate_multilabel(datasets["validation"].targets, scores, names, threshold=0.5)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(datasets["train"]),
                        "validation_macro_f1": metric["macro_f1"], "validation_micro_f1": metric["micro_f1"],
                        "epoch_train_and_eval_seconds": perf_counter() - epoch_started})
        # Hòa Macro-F1 giữ epoch sớm nhất, không chọn lại bằng micro-F1/test.
        if metric["macro_f1"] > best_score:
            best_score, best_epoch = metric["macro_f1"], epoch
            model.save_pretrained(output / "best", safe_serialization=True)
            tokenizer.save_pretrained(output / "best")
            np.savez_compressed(output / "validation_scores.npz", ids=frames["validation"].id.to_numpy(dtype=str),
                                scores=scores, label_names=np.asarray(names, dtype=str))
            write_json(output / "validation_metrics.json", metric)
            pd.DataFrame(metric["per_label"]).to_csv(output / "per_label_validation.csv", index=False, encoding="utf-8-sig")
        write_json(output / "history.json", history)
        print(f"epoch={epoch}, validation Macro-F1@0.5={metric['macro_f1']:.5f}", flush=True)
    fit_seconds = perf_counter() - fit_started
    allocated = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
    reserved = torch.cuda.max_memory_reserved() / 2**20 if device == "cuda" else None
    # Cất trạng thái cuối để kiểm tra/hỗ trợ khôi phục ở epoch boundary trong tương lai.
    model.save_pretrained(output / "last", safe_serialization=True)
    torch.save({"epoch": config["epochs"], "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state()}, output / "training_state.pt")
    del optimizer, model, scheduler, scaler
    if device == "cuda":
        torch.cuda.empty_cache()
    restored = AutoModelForSequenceClassification.from_pretrained(output / "best", local_files_only=True).to(device)
    replay = score_loader(restored, val_loader, device)
    with np.load(output / "validation_scores.npz", allow_pickle=False) as saved:
        reference = saved["scores"]
        np.testing.assert_allclose(replay, reference, rtol=0, atol=1e-6)
    sample_text = frames["validation"].text.iloc[0]
    probe = {"model": restored, "tokenizer": tokenizer, "device": device,
             "metadata": {"label_names": names, "config": config}}
    probe_result = predict_texts(probe, [sample_text])[0]
    write_json(output / "inference_example.json", probe_result)
    packages = ["torch", "transformers", "tokenizers", "huggingface-hub", "numpy", "pandas", "pyarrow", "scikit-learn"]
    metadata = {
        "status": "complete", "artifact_version": 1, "method": "C3 DistilBERT", "mode": config["mode"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "config_sha256": sha256(output / "config.json"), "data_revision": REVISION,
        "data_sha256": {f["split"]: f["sha256"] for f in manifest["files"]},
        "label_names": names, "n_train": len(frames["train"]), "n_validation": len(frames["validation"]),
        "test_used": False, "best_epoch": best_epoch, "selection": "max validation macro_f1 @0.5; tie earliest epoch",
        "effective_batch_size": config["batch_size"] * accumulation, "trainable_parameters": trainable,
        "optimizer_steps": optimizer_steps, "fit_and_validation_seconds": fit_seconds,
        "planned_optimizer_steps": n_steps, "skipped_amp_steps": skipped_amp_steps,
        "total_seconds": perf_counter() - started, "peak_cuda_allocated_mib": allocated,
        "peak_cuda_reserved_mib": reserved, "reload_max_abs_score_diff": float(np.max(np.abs(replay-reference))),
        "environment": {"python": sys.version.split()[0], "packages": {p: importlib.metadata.version(p) for p in packages},
                        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                        "cuda_build": torch.version.cuda},
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "source_sha256": source_hashes,
    }
    files = [p for p in (output/"best").iterdir() if p.is_file()]
    files += [output/name for name in ["config.json", "validation_scores.npz", "train_ids.csv", "validation_ids.csv",
                                      "validation_metrics.json", "history.json", "per_label_validation.csv", "inference_example.json"]]
    files += [output/"source"/name for name in sources]
    metadata["artifact_sha256"] = {p.relative_to(output).as_posix(): sha256(p) for p in files}
    write_json(output / "run.json", metadata)
    del restored, probe
    if device == "cuda":
        torch.cuda.empty_cache()
    # Kiểm loader chung dùng cho CLI/app cũng nhận đúng bundle/hash.
    bundle = load_bundle(output, device)
    after_load = predict_texts(bundle, [sample_text])[0]
    np.testing.assert_allclose(list(after_load["scores"].values()), list(probe_result["scores"].values()), atol=1e-6, rtol=0)
    print(f"PASS: {config['mode']} complete, best epoch={best_epoch}, reload diff={metadata['reload_max_abs_score_diff']:.2g}", flush=True)
    print(output, flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distilbert_pilot.json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", help="New output directory; existing run never overwritten")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    output = args.output or ROOT / "data/processed/c3_distilbert" / config["mode"] / f"seed_{config['seed']}"
    run_training(config, output)


if __name__ == "__main__":
    main()
