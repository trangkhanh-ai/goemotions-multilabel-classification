"""Phần B: đổi kết quả NLI thành ma trận đa nhãn, lưu và tiếp tục từng batch.

Module này không tải mô hình và không huấn luyện. Pipeline thật được truyền vào
như một hàm, nên các bước ghép nhãn/lưu điểm có thể kiểm tra bằng dữ liệu nhỏ.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from src.datasets.goemotions import EXPECTED_ROWS, EXPECTED_SHA256, REVISION, sha256
from src.evaluation.metrics import validate_thresholds


CHECKPOINT = "facebook/bart-large-mnli"
HYPOTHESIS_TEMPLATE = "This text expresses {}."


def write_json(path, value):
    """Ghi file tạm rồi thay thế, tránh JSON dở dang khi chương trình bị dừng."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_scores(path, ids, scores, label_names):
    """Lưu NPZ không dùng pickle; chỉ lưu các hàng đã dự đoán đầy đủ."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, ids=np.asarray(ids, dtype=str), scores=np.asarray(scores),
                            label_names=np.asarray(label_names, dtype=str))
    temporary.replace(path)


def input_fingerprint(ids, texts):
    """Định danh thứ tự ID và văn bản để resume không ghép nhầm dữ liệu."""
    payload = json.dumps(list(zip(ids, texts)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def map_pipeline_scores(results, label_names):
    """HF trả nhãn theo điểm giảm dần; đưa lại đúng thứ tự data/labels.json.

    Không lấy trực tiếp results['scores'] làm một hàng: thứ tự nhãn HF trả về
    có thể khác nhau giữa các câu. Mỗi câu phải có đúng một điểm cho mỗi nhãn.
    """
    if not label_names or len(set(label_names)) != len(label_names):
        raise ValueError("Danh sách nhãn phải không rỗng và không trùng")
    if isinstance(results, dict):
        results = [results]
    if not results:
        raise ValueError("Pipeline không trả kết quả")
    rows = []
    for result in results:
        names, values = result.get("labels", []), result.get("scores", [])
        if len(names) != len(values) or len(names) != len(label_names):
            raise ValueError("Pipeline thiếu nhãn hoặc số điểm không khớp số nhãn")
        if len(set(names)) != len(names) or set(names) != set(label_names):
            raise ValueError("Pipeline có nhãn trùng, nhãn lạ hoặc thiếu nhãn")
        by_label = dict(zip(names, values))
        row = np.asarray([by_label[name] for name in label_names], dtype=float)
        if not np.isfinite(row).all() or np.any((row < 0) | (row > 1)):
            raise ValueError("Điểm phải hữu hạn và nằm trong [0, 1]")
        rows.append(row)
    return np.asarray(rows)


def predict_in_batches(predictor, texts, ids, label_names, output, config, batch_size=8):
    """Dự đoán theo batch, tự tiếp tục batch hoàn tất nếu chạy lại cùng cấu hình.

    Mỗi chunk là file riêng. Manifest chỉ tham chiếu chunk đã ghi xong và có
    SHA-256. Nếu dừng sau khi ghi chunk nhưng trước manifest, chunk đó được
    tính lại; các batch đã được manifest xác nhận vẫn được giữ.
    """
    output = Path(output)
    texts, ids, label_names = list(texts), list(ids), list(label_names)
    if batch_size < 1 or len(ids) != len(texts) or not ids:
        raise ValueError("batch_size phải dương; ID và văn bản phải cùng số hàng và không rỗng")
    if len(set(ids)) != len(ids):
        raise ValueError("ID không được trùng")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Văn bản không được rỗng")
    if not label_names or len(set(label_names)) != len(label_names):
        raise ValueError("Nhãn không được rỗng hoặc trùng")
    output.mkdir(parents=True, exist_ok=True)
    expected = dict(config, input_sha256=input_fingerprint(ids, texts), sample_count=len(ids),
                    label_names=label_names, batch_size=batch_size)
    manifest_path = output / "checkpoint_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != expected or manifest.get("schema_version") != 1:
            raise ValueError("Resume bị từ chối: cấu hình, phiên bản hoặc dữ liệu đã thay đổi")
    else:
        manifest = {"schema_version": 1, "config": expected, "status": "running", "chunks": []}
        write_json(manifest_path, manifest)

    chunks, completed = [], 0
    for record in manifest["chunks"]:
        path = output / record["file"]
        if record["file"] != f"chunk_{record['start']:06d}_{record['stop']:06d}.npz":
            raise ValueError("Tên chunk trong manifest không hợp lệ")
        if record["start"] != completed or not completed < record["stop"] <= len(ids):
            raise ValueError("Manifest có khoảng trống hoặc thứ tự chunk sai")
        if not path.exists() or sha256(path) != record["sha256"]:
            raise ValueError("Chunk bị thiếu hoặc SHA-256 đã thay đổi")
        with np.load(path, allow_pickle=False) as saved:
            points = saved["scores"].copy()
            if saved["ids"].tolist() != ids[completed:record["stop"]] or saved["label_names"].tolist() != label_names:
                raise ValueError("Chunk không khớp ID hoặc thứ tự nhãn")
        if points.shape != (record["stop"] - completed, len(label_names)):
            raise ValueError("Chunk có kích thước điểm sai")
        if not np.isfinite(points).all() or np.any((points < 0) | (points > 1)):
            raise ValueError("Chunk có điểm không hợp lệ")
        chunks.append(points)
        completed = record["stop"]

    for start in range(completed, len(ids), batch_size):
        stop = min(start + batch_size, len(ids))
        results = predictor(texts[start:stop])
        points = map_pipeline_scores(results, label_names)
        if points.shape != (stop - start, len(label_names)):
            raise ValueError("Pipeline không trả đủ số câu trong batch")
        filename = f"chunk_{start:06d}_{stop:06d}.npz"
        write_scores(output / filename, ids[start:stop], points, label_names)
        manifest["chunks"].append({"start": start, "stop": stop, "file": filename,
                                   "sha256": sha256(output / filename)})
        write_json(manifest_path, manifest)
        chunks.append(points)
        print(f"Saved {stop:,}/{len(ids):,} texts", flush=True)

    scores = np.concatenate(chunks, axis=0)
    manifest["status"] = "complete"
    manifest["completed_count"] = len(ids)
    write_json(manifest_path, manifest)
    return scores, manifest


def validate_test_protocol(protocol_path, run_dir, label_names):
    """Kiểm khóa cấu hình từ validation TRƯỚC khi mở split test.

    Đọc metadata và hash file validation; không đọc nhãn validation/test để
    chọn lại ngưỡng. Protocol được tạo bằng scripts.pipeline.freeze_experiment.
    """
    run_dir = Path(run_dir)
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != 1 or protocol.get("method") != "zero_shot":
        raise ValueError("Protocol không đúng phiên bản hoặc phương pháp zero_shot")
    if protocol.get("smoke") is not False or metadata.get("smoke") is not False:
        raise ValueError("Không đánh giá test chính thức từ một lần chạy smoke")
    if metadata.get("split") != "validation" or metadata.get("status") != "complete":
        raise ValueError("Cần metadata của validation đã hoàn tất")
    if metadata.get("method") != "zero_shot" or metadata.get("multi_label") is not True:
        raise ValueError("Metadata không mô tả pipeline zero-shot đa nhãn")
    if metadata.get("sample_count") != EXPECTED_ROWS["validation"]:
        raise ValueError("Protocol chỉ được khóa sau khi dự đoán đủ validation chính thức")
    if metadata.get("data_sha256", {}).get("validation") != EXPECTED_SHA256["validation"]:
        raise ValueError("Metadata không khớp SHA-256 validation chính thức")
    for key in ("checkpoint", "model_revision", "hypothesis_template", "data_revision", "label_names"):
        if protocol.get(key) != metadata.get(key):
            raise ValueError(f"Protocol không khớp metadata: {key}")
    if protocol.get("label_names") != list(label_names) or protocol.get("data_revision") != REVISION:
        raise ValueError("Protocol không khớp mapping nhãn hoặc snapshot dữ liệu hiện tại")
    if protocol.get("checkpoint") != CHECKPOINT or not re.fullmatch(r"[0-9a-f]{40}", str(protocol.get("model_revision"))):
        raise ValueError("Protocol phải khóa checkpoint BART-MNLI bằng commit SHA chính xác")
    scores_path = run_dir / "validation_scores.npz"
    if protocol.get("validation_scores_sha256") != sha256(scores_path):
        raise ValueError("Điểm validation đã thay đổi sau khi khóa")
    if metadata.get("artifact_sha256", {}).get("validation_scores.npz") != sha256(scores_path):
        raise ValueError("Hash điểm không khớp metadata validation")
    with np.load(scores_path, allow_pickle=False) as saved:
        points, ids = saved["scores"], saved["ids"]
        if (points.shape != (EXPECTED_ROWS["validation"], len(label_names))
                or ids.shape != (EXPECTED_ROWS["validation"],)
                or len(set(ids.tolist())) != len(ids)
                or saved["label_names"].tolist() != list(label_names)):
            raise ValueError("File điểm chưa đủ validation hoặc không khớp ID/nhãn")
        if not np.isfinite(points).all() or np.any((points < 0) | (points > 1)):
            raise ValueError("File validation có điểm không hợp lệ")
    if protocol.get("run_metadata_sha256") != sha256(metadata_path):
        raise ValueError("Metadata validation đã thay đổi sau khi khóa")
    if not protocol.get("frozen_at_utc") or protocol.get("threshold_mode") not in ("fixed", "global", "tuned"):
        raise ValueError("Protocol thiếu thời điểm khóa hoặc chế độ ngưỡng hợp lệ")
    validate_thresholds(protocol.get("thresholds"), len(label_names))
    configurations = protocol.get("configurations", [])
    if configurations:
        modes = [item.get("threshold_mode") for item in configurations]
        if sorted(modes) != ["fixed", "global", "tuned"]:
            raise ValueError("Các cấu hình đã khóa cần đúng fixed/global/tuned")
        for item in configurations:
            validate_thresholds(item.get("thresholds"), len(label_names))
        selected = next(item for item in configurations if item["threshold_mode"] == protocol["threshold_mode"])
        if selected["thresholds"] != protocol["thresholds"]:
            raise ValueError("Ngưỡng chọn chính không khớp cấu hình đã khóa")
    return protocol, metadata
