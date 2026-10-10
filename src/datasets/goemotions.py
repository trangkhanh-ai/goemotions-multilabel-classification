"""Tải đúng snapshot GoEmotions, kiểm SHA-256 và đọc mapping từ metadata."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import urllib.request

import numpy as np
import pyarrow.parquet as pq

REPO = "google-research-datasets/go_emotions"
REVISION = "add492243ff905527e67aeb8b80c082af02207c3"
SPLITS = ("train", "validation", "test")
EXPECTED_ROWS = dict(zip(SPLITS, (43410, 5426, 5427)))
EXPECTED_SHA256 = {
    "train": "b7d74279616ae7c9b8374ab62ea9f9d6504d36a577bb17f745d720dc2b0d4e76",
    "validation": "d46ad5633c4fa41829d22d549743a7bf858d94129536a02af7280c281db5e63a",
    "test": "fd0953e535ba2569edc6a1daaa1133f8e4b9071691d540c9bab812fda132bc26",
}


def sha256(path):
    """Đọc từng phần để kiểm cả checkpoint lớn mà không nạp hết vào RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path, expected_hash=None):
    """Cache cục bộ, timeout, retry; chỉ thay file đích khi tải/kiểm hash xong."""
    path = Path(path)
    if path.exists():
        if expected_hash and sha256(path) != expected_hash:
            raise ValueError(f"Sai SHA-256 tại {path}; kiểm tra hoặc xóa riêng file này để tải lại.")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "GoEmotions-course-EDA/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if expected_hash and hashlib.sha256(payload).hexdigest() != expected_hash:
                raise ValueError(f"Dữ liệu tải về không khớp SHA-256: {url}")
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(path)
            return path
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def load_goemotions(root, *, write_metadata=True, splits=SPLITS):
    """Trả frames, label_names, manifest; giữ nguyên thứ tự dòng/nhãn gốc."""
    root = Path(root)
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or any(s not in SPLITS for s in splits):
        raise ValueError("splits phải là danh sách không lặp gồm train/validation/test")
    if write_metadata and splits != SPLITS:
        raise ValueError("Chỉ ghi manifest khi đọc đủ ba split chính thức")
    frames, records, label_names = {}, [], None
    for split in splits:
        filename = f"{split}-00000-of-00001.parquet"
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/simplified/{filename}"
        path = download(url, root / "data/raw" / filename, EXPECTED_SHA256[split])
        table = pq.read_table(path)
        metadata = json.loads(table.schema.metadata[b"huggingface"])
        names = metadata["info"]["features"]["labels"]["feature"]["names"]
        if label_names is None:
            label_names = names
        if names != label_names or len(names) != 28:
            raise ValueError(f"Mapping nhãn không thống nhất ở {split}")
        frame = table.to_pandas()
        if list(frame.columns) != ["text", "labels", "id"] or len(frame) != EXPECTED_ROWS[split]:
            raise ValueError(f"Sai schema hoặc số dòng ở {split}")
        frame["labels"] = frame["labels"].map(lambda values: [int(x) for x in values])
        frames[split] = frame
        records.append({"split": split, "url": url, "path": path.relative_to(root).as_posix(),
                        "sha256": sha256(path), "bytes": path.stat().st_size, "rows": len(frame)})
    manifest = {"repository": REPO, "configuration": "simplified", "revision": REVISION,
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "columns": ["text", "labels", "id"], "label_names": label_names, "files": records}
    if write_metadata:
        (root / "data").mkdir(exist_ok=True)
        (root / "data/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (root / "data/labels.json").write_text(json.dumps(label_names, ensure_ascii=False, indent=2), encoding="utf-8")
    return frames, label_names, manifest


def multi_hot(label_lists, n_labels=28):
    """Y[i,j]=1 khi mẫu i mang nhãn j; từ chối nhãn rỗng, trùng hoặc ngoài miền."""
    y = np.zeros((len(label_lists), n_labels), dtype=np.uint8)
    for i, labels in enumerate(label_lists):
        if not labels or len(labels) != len(set(labels)):
            raise ValueError(f"Mẫu {i} có nhãn rỗng hoặc lặp")
        if any(not isinstance(j, (int, np.integer)) or not 0 <= j < n_labels for j in labels):
            raise ValueError(f"Mẫu {i} có nhãn ngoài miền")
        y[i, labels] = 1
    return y
