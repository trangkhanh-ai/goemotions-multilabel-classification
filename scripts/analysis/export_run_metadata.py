"""Xuất hồ sơ JSON nhỏ để đưa Git; không copy weights/scores/văn bản thô.

python -m scripts.analysis.export_run_metadata
JSON gốc được giữ nguyên byte, kèm source path + SHA-256 trong manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from src.models.baseline import load_run_metadata, load_thresholds
from src.datasets.goemotions import EXPECTED_ROWS, EXPECTED_SHA256, REVISION, sha256
from src.evaluation.protocols import read_json, save_json, validate_protocol
from src.models.transformer import (ARCHITECTURES, configure_console, load_demo_selection,
                        load_transformer_run, run_folder)
from src.models.zero_shot import CHECKPOINT, validate_test_protocol


from src.paths import ROOT
MAX_JSON_BYTES = 1024 * 1024
# Chỉ file JSON đã có trong whitelist được copy. Không tìm/copy toàn bộ thư mục.
C_FILES = ("run_metadata.json", "label_mapping.json", "validation_metrics.json",
           "checkpoint/config.json", "checkpoint/tokenizer_config.json", "checkpoint/special_tokens_map.json",
           "final_protocol.json", "test_results.json")
B_FILES = ("run_metadata.json", "validation_metrics.json", "final_protocol.json",
           "test_run_metadata.json", "test_metrics.json")
A_FILES = ("validation_metrics.json", "analysis_validation.json", "thresholds_validation.json")
FORBIDDEN_KEYS = {"text", "texts", "raw_text", "examples", "sequence", "sequences",
                  "password", "secret", "token", "hf_token", "huggingface_token",
                  "access_token", "api_key", "authorization"}


def assert_metadata_only(value):
    """Phòng việc một file được mở rộng thêm raw text hoặc credential về sau."""
    if isinstance(value, dict):
        if any(str(key).lower() in FORBIDDEN_KEYS for key in value):
            raise ValueError("JSON có trường raw text/credential; không export")
        for child in value.values():
            assert_metadata_only(child)
    elif isinstance(value, list):
        for child in value:
            assert_metadata_only(child)


def copy_json(root, output, relative_path, group, *, permitted=True, reason="", max_bytes=MAX_JSON_BYTES):
    """Copy đúng byte nguồn; file thiếu/sai chỉ thành record, không tạo JSON giả."""
    root, output = Path(root).resolve(), Path(output).resolve()
    source = (root / relative_path).resolve()
    target = (output / "artifacts" / relative_path).resolve()
    if not source.is_relative_to(root) or not target.is_relative_to(output / "artifacts"):
        raise ValueError("Source/target phải nằm trong repo hoặc thư mục export được chỉ định")
    record = {"group": group, "source_path": Path(relative_path).as_posix(), "export_path": None,
              "source_sha256": None, "export_sha256": None, "bytes": None,
              "status": "missing", "reason": "File nguồn chưa có"}
    if not source.is_file():
        return record
    record["bytes"] = source.stat().st_size
    if source.suffix.lower() != ".json" or record["bytes"] > max_bytes:
        record.update(status="invalid", reason="Chỉ export JSON nhỏ trong giới hạn bytes")
        return record
    try:
        payload = source.read_bytes()
        value = json.loads(payload)
        assert_metadata_only(value)
        record["source_sha256"] = hashlib.sha256(payload).hexdigest()
        if not permitted:
            record.update(status="incomplete", reason=reason)
            return record
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.part")
        temporary.write_bytes(payload)
        temporary.replace(target)
        record.update(status="exported", reason="", export_path=target.relative_to(output).as_posix(),
                      export_sha256=sha256(target))
    except (OSError, ValueError) as error:
        record.update(status="invalid", reason=str(error))
    return record


def clear_previous_generated_files(output):
    """Gỡ các bản copy do chính exporter tạo; giữ file đã được người khác sửa.

    Chỉ unlink từng JSON có đường dẫn trong artifacts và hash trùng manifest
    cũ. Không xóa đệ quy và không tác động file nguồn data/processed.
    """
    output = Path(output).resolve()
    manifest = output / "manifest.json"
    if not manifest.is_file():
        return []
    previous = read_json(manifest)
    conflicts = []
    for record in previous.get("artifacts", []):
        if record.get("status") not in ("exported", "conflict") or not record.get("export_path"):
            continue
        target = (output / record["export_path"]).resolve()
        if not target.is_relative_to(output / "artifacts") or target.suffix.lower() != ".json":
            raise ValueError("Đường dẫn copy cũ vượt thư mục artifacts")
        if target.is_file():
            if record.get("status") == "conflict":
                conflicts.append(target.relative_to(output).as_posix())
                continue
            if sha256(target) == record.get("export_sha256"):
                target.unlink()
            else:
                conflicts.append(target.relative_to(output).as_posix())
    return conflicts


def validate_b_full(folder, labels):
    folder = Path(folder)
    metadata = read_json(folder / "run_metadata.json")
    if (metadata.get("method") != "zero_shot" or metadata.get("status") != "complete"
            or metadata.get("smoke") is not False or metadata.get("split") != "validation"
            or metadata.get("data_revision") != REVISION or metadata.get("label_names") != labels
            or metadata.get("sample_count") != EXPECTED_ROWS["validation"]
            or metadata.get("checkpoint") != CHECKPOINT or metadata.get("multi_label") is not True
            or not re.fullmatch(r"[0-9a-f]{40}", str(metadata.get("model_revision")))
            or metadata.get("data_sha256", {}).get("validation") != EXPECTED_SHA256["validation"]):
        raise ValueError("B chưa phải full validation hoàn tất với identity đúng")
    scores = folder / "validation_scores.npz"
    if metadata.get("artifact_sha256", {}).get("validation_scores.npz") != sha256(scores):
        raise ValueError("B validation scores không khớp metadata")
    with np.load(scores, allow_pickle=False) as stored:
        if (stored["scores"].shape != (EXPECTED_ROWS["validation"], len(labels))
                or stored["ids"].shape != (EXPECTED_ROWS["validation"],)
                or len(set(stored["ids"].tolist())) != EXPECTED_ROWS["validation"]
                or stored["label_names"].tolist() != labels
                or not np.isfinite(stored["scores"]).all()
                or np.any((stored["scores"] < 0) | (stored["scores"] > 1))):
            raise ValueError("B validation scores thiếu mẫu/nhãn hoặc điểm không hợp lệ")
    return metadata


def code_provenance(root):
    """Ghi trạng thái source lúc export; không suy ra commit đã training."""
    files = ("src/datasets/goemotions.py", "src/evaluation/metrics.py", "src/models/baseline.py", "src/models/zero_shot.py", "src/models/transformer.py",
             "src/evaluation/protocols.py", "scripts/baseline/run_baseline.py", "scripts/transformers/train_transformer.py",
             "scripts/zero_shot/run_zero_shot.py", "scripts/baseline/freeze_baseline.py", "scripts/pipeline/freeze_experiment.py",
             "scripts/baseline/evaluate_baseline_test.py", "scripts/transformers/evaluate_transformer_test.py")
    record = {"scope": "repository_at_export_time_not_training_commit", "git_head": None, "working_tree_dirty": None,
              "source_file_sha256": {name: sha256(Path(root) / name) for name in files if (Path(root) / name).is_file()}}
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, encoding="utf-8", errors="replace", capture_output=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, encoding="utf-8", errors="replace", capture_output=True, check=True)
        record.update(git_head=head.stdout.strip(), working_tree_dirty=bool(status.stdout.strip()))
    except (OSError, subprocess.CalledProcessError):
        pass
    return record


def export_metadata(root, seeds=(42, 123, 2026), output=None, *, max_bytes=MAX_JSON_BYTES):
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / "reports/reproducibility"
    labels = read_json(root / "data/labels.json")
    if len(labels) != 28 or len(set(labels)) != 28:
        raise ValueError("Cần mapping 28 nhãn chuẩn")
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or max_bytes < 1:
        raise ValueError("Cần >=3 seed khác nhau và max_bytes dương")
    output.mkdir(parents=True, exist_ok=True)
    conflicts = set(clear_previous_generated_files(output))
    artifacts, groups, baseline_metadata, c_metadata = [], {}, {}, {}

    def add(relative, group, permitted=True, reason=""):
        target_relative = (Path("artifacts") / relative).as_posix()
        if target_relative in conflicts:
            artifacts.append({"group": group, "source_path": Path(relative).as_posix(), "export_path": target_relative,
                              "status": "conflict", "reason": "Bản copy được sửa sau lần export trước; giữ nguyên",
                              "source_sha256": None, "export_sha256": None, "bytes": None})
        else:
            artifacts.append(copy_json(root, output, relative, group, permitted=permitted, reason=reason, max_bytes=max_bytes))

    for relative in ("data/labels.json", "data/manifest.json"):
        add(relative, "shared_data")

    for variant, folder in (("standard", root / "data/processed/baseline/full"),
                            ("balanced", root / "data/processed/baseline/balanced/full")):
        group = f"A/{variant}"
        try:
            metadata = load_run_metadata(folder, variant, labels)
            if metadata.get("n_train") != EXPECTED_ROWS["train"] or metadata.get("n_validation") != EXPECTED_ROWS["validation"]:
                raise ValueError("A chưa đủ số mẫu full")
            baseline_metadata[variant] = metadata
            groups[group] = {"status": "complete", "seed": metadata.get("config", {}).get("random_state"),
                             "data_revision": metadata["data_revision"]}
        except (OSError, ValueError, KeyError) as error:
            groups[group] = {"status": "incomplete", "reason": str(error)}
        for name in A_FILES:
            permitted = groups[group]["status"] == "complete"
            if permitted and name == "thresholds_validation.json" and (folder / name).is_file():
                try:
                    load_thresholds(folder, baseline_metadata[variant], "tuned")
                except (OSError, ValueError, KeyError):
                    permitted = False
            if permitted and name == "analysis_validation.json" and (folder / name).is_file():
                try:
                    analysis = read_json(folder / name)
                    permitted = (analysis.get("variant") == variant
                                 and analysis.get("artifact_sha256") == baseline_metadata[variant]["artifact_sha256"])
                except (OSError, ValueError, KeyError):
                    permitted = False
            add((folder / name).relative_to(root), group, permitted, "A chưa hoàn tất/metadata hoặc ngưỡng không hợp lệ")

    base = root / "data/processed/baseline"
    a_protocol_ok = False
    if (base / "final_protocol.json").is_file() and len(baseline_metadata) == 2:
        try:
            protocol = read_json(base / "final_protocol.json")
            a_protocol_ok = (protocol.get("schema_version") == 1 and protocol.get("label_names") == labels
                             and protocol.get("data_revision") == REVISION
                             and set(protocol.get("runs", {})) == {"standard", "balanced"})
            for variant, run in protocol["runs"].items():
                folder = (root / run["folder"]).resolve()
                if not folder.is_relative_to(root):
                    raise ValueError("A protocol trỏ ngoài repo")
                a_protocol_ok = (a_protocol_ok and run.get("artifact_sha256") == baseline_metadata[variant]["artifact_sha256"]
                                 and run.get("threshold_file_sha256") == sha256(folder / "thresholds_validation.json"))
        except (OSError, ValueError, KeyError):
            a_protocol_ok = False
    add("data/processed/baseline/final_protocol.json", "A/frozen", a_protocol_ok, "Chưa có A protocol hợp lệ")
    a_test_ok = False
    if a_protocol_ok and (base / "final/final_results.json").is_file():
        summary = read_json(base / "final/final_results.json")
        a_test_ok = summary.get("protocol_sha256") == sha256(base / "final_protocol.json")
    add("data/processed/baseline/final/final_results.json", "A/test", a_test_ok, "A test chưa có protocol/results hợp lệ")
    for variant in ("standard", "balanced"):
        for mode in ("fixed", "global", "tuned"):
            add(f"data/processed/baseline/final/{variant}_{mode}_test_metrics.json", "A/test", a_test_ok,
                "A test chưa có protocol/results hợp lệ")

    b_folder = root / "data/processed/zero_shot/full"
    try:
        metadata = validate_b_full(b_folder, labels)
        groups["B"] = {"status": "complete", "checkpoint": metadata["checkpoint"],
                       "model_revision": metadata["model_revision"], "seed": None, "data_revision": REVISION}
    except (OSError, ValueError, KeyError) as error:
        groups["B"] = {"status": "incomplete", "reason": str(error)}
    b_protocol_ok, b_test_ok = False, False
    if groups["B"]["status"] == "complete" and (b_folder / "final_protocol.json").is_file():
        try:
            validate_test_protocol(b_folder / "final_protocol.json", b_folder, labels)
            b_protocol_ok = True
            if (b_folder / "test_run_metadata.json").is_file():
                test = read_json(b_folder / "test_run_metadata.json")
                b_test_ok = (test.get("protocol_sha256") == sha256(b_folder / "final_protocol.json")
                             and test.get("status") == "complete" and test.get("split") == "test"
                             and test.get("artifact_sha256", {}).get("test_scores.npz") == sha256(b_folder / "test_scores.npz"))
        except (OSError, ValueError, KeyError):
            b_protocol_ok, b_test_ok = False, False
    for name in B_FILES:
        permitted = (b_test_ok if name.startswith("test_") else b_protocol_ok if name == "final_protocol.json"
                     else groups["B"]["status"] == "complete")
        add((b_folder / name).relative_to(root), "B", permitted, "B full/protocol/test chưa hoàn tất hoặc hash không khớp")

    for architecture in ARCHITECTURES:
        for seed in seeds:
            folder = run_folder(root, architecture, seed)
            group = f"C/{architecture}/seed_{seed}"
            try:
                metadata = load_transformer_run(folder, require_full=True)
                if metadata["architecture"] != architecture or metadata["seed"] != seed or metadata["label_names"] != labels:
                    raise ValueError("C identity không khớp folder/mapping")
                groups[group] = {"status": "complete", "checkpoint": metadata["checkpoint"],
                                 "model_revision": metadata["model_revision"], "seed": seed, "data_revision": REVISION}
                c_metadata[(architecture, seed)] = metadata
            except (OSError, ValueError, KeyError) as error:
                groups[group] = {"status": "incomplete", "reason": str(error)}
            complete = groups[group]["status"] == "complete"
            protocol_ok, test_ok = False, False
            if complete and (folder / "final_protocol.json").is_file():
                try:
                    protocol = read_json(folder / "final_protocol.json")
                    validate_protocol(folder, protocol, labels)
                    if (protocol.get("architecture") != architecture or protocol.get("seed") != seed
                            or protocol.get("checkpoint") != metadata["checkpoint"]
                            or protocol.get("model_revision") != metadata["model_revision"]):
                        raise ValueError("C protocol identity không khớp metadata hiện tại")
                    protocol_ok = True
                    if (folder / "test_results.json").is_file():
                        test = read_json(folder / "test_results.json")
                        test_ok = (test.get("protocol_sha256") == sha256(folder / "final_protocol.json")
                                   and test.get("scores_sha256") == sha256(folder / "test_scores.npz"))
                except (OSError, ValueError, KeyError):
                    protocol_ok, test_ok = False, False
            for name in C_FILES:
                permitted = test_ok if name == "test_results.json" else protocol_ok if name == "final_protocol.json" else complete
                add((folder / name).relative_to(root), group, permitted, "C full/protocol/test chưa hoàn tất hoặc artifact không hợp lệ")

    all_c_complete = all(group["status"] == "complete" for name, group in groups.items() if name.startswith("C/"))
    for name in ("selected_model.json", "seed_summary.json"):
        permitted = all_c_complete
        path = root / f"data/processed/transformers/{name}"
        if permitted and path.is_file():
            try:
                selection = read_json(path)
                if name == "selected_model.json":
                    load_demo_selection(root, path)
                    sources = selection.get("run_sources", [])
                    if len(sources) != len(c_metadata):
                        raise ValueError("Selection thiếu run sources")
                    for item in sources:
                        if (item["architecture"], item["seed"]) not in c_metadata:
                            raise ValueError("Selection trỏ run ngoài tập seeds hiện tại")
                        folder = (root / item["run_dir"]).resolve()
                        if not folder.is_relative_to(root) or item["run_metadata_sha256"] != sha256(folder / "run_metadata.json"):
                            raise ValueError("Selection dùng metadata cũ")
                else:
                    if selection.get("ddof") != 1 or len(selection.get("runs", [])) != len(c_metadata):
                        raise ValueError("Seed summary không đủ runs/ddof đúng")
                    seen = set()
                    for item in selection["runs"]:
                        key = (item["architecture"], item["seed"])
                        if key in seen or key not in c_metadata or item["metrics"] != c_metadata[key]["metrics"]:
                            raise ValueError("Seed summary cũ, trùng run hoặc metric không khớp")
                        seen.add(key)
            except (OSError, ValueError, KeyError):
                permitted = False
        add(f"data/processed/transformers/{name}", "C/selection", permitted,
            "Chưa đủ ba kiến trúc với toàn bộ seeds để xác nhận selection/summary")
    manifest = {"schema_version": 1, "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                "data_revision": REVISION, "label_names": labels, "seeds": list(seeds),
                "max_json_bytes": max_bytes, "groups": groups, "artifacts": artifacts,
                "code_provenance": code_provenance(root),
                "n_exported": sum(row["status"] == "exported" for row in artifacts),
                "n_not_exported": sum(row["status"] != "exported" for row in artifacts),
                "c_full_completed": sum(group["status"] == "complete" for name, group in groups.items() if name.startswith("C/")),
                "c_full_expected": len(ARCHITECTURES) * len(seeds),
                "notes": ["JSON bytes are copied from existing verified artifacts; configs/history are not invented.",
                          "Repository .gitattributes marks artifact mirrors -text to preserve source bytes in Git blobs and clones.",
                          "Git HEAD and source hashes describe the repository at export time, not the unrecorded training commit.",
                          "Weights, prediction NPZ, raw text, datasets, tokenizer vocabulary and virtual environments are omitted.",
                          "Missing/incomplete/invalid/conflict entries are not results. Read only status=exported entries."]}
    save_json(output / "manifest.json", manifest)
    lines = ["# Hồ sơ tái hiện A/B/C", "",
             f"Đã xuất **{manifest['n_exported']} JSON**; **{manifest['c_full_completed']}/{manifest['c_full_expected']}** run C full hoàn tất được kiểm.",
             "", "Mỗi JSON là bản copy nguyên byte; đối chiếu source/export SHA-256 trong `manifest.json`. "
             "Config, seed, môi trường, labels, data/model revision và history nằm trong metadata gốc.", "",
             "Git HEAD/source hashes chỉ mô tả repo tại lúc export; trainer chưa ghi training commit nên không suy ra commit đã huấn luyện.", "",
             "| Nhóm | Trạng thái | Checkpoint | Model revision | Seed |", "|---|---|---|---|---|"]
    for name, group in groups.items():
        lines.append("| " + " | ".join([name, group["status"], str(group.get("checkpoint", "—")),
                                         str(group.get("model_revision", "—")), str(group.get("seed", "—"))]) + " |")
    lines += ["", "Các file có status khác exported được ghi rõ trong manifest; không tạo số liệu thay thế. "
              "Model/scores dùng để suy luận cần bản đầy đủ khớp hash trong metadata. "
              "Hướng dẫn: `docs/REPRODUCIBILITY.md` ở gốc repo."]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = export_metadata(ROOT, args.seeds, args.output)
    print(f"Đã xuất {result['n_exported']} JSON nhỏ; C full {result['c_full_completed']}/{result['c_full_expected']}; "
          f"{result['n_not_exported']} file thiếu/chưa hợp lệ được ghi trong manifest.")


if __name__ == "__main__":
    main()
