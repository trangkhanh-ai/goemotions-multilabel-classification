"""Kiểm bản Git chỉ chứa source, notebook sạch và hướng dẫn.

Chạy từ gốc repo: python tools/check_repository.py
Chỉ đọc tệp/Git index; không tải dataset hoặc model.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".docx", ".doc", ".pdf", ".tex", ".html", ".pptx", ".xlsx",
                      ".joblib", ".safetensors", ".pt", ".pth"}
GENERATED_PREFIXES = ("reports/", "data/raw/", "data/processed/", "data/cache/",
                      "data/tokenizers/", "artifacts/", "outputs/")
LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")
SECRETS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def tracked_paths():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                            check=True, stdout=subprocess.PIPE)
    return [path for path in result.stdout.decode("utf-8").split("\0") if path]


def check_source(paths):
    errors = []
    counts = {"files": len(paths), "python": 0, "notebooks": 0, "local_links": 0}
    for name in paths:
        path = ROOT / name
        if not path.is_file():
            errors.append(f"{name}: tệp đã đổi chỗ nhưng chưa git add")
            continue
        if name.startswith(GENERATED_PREFIXES) or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"{name}: tệp ngoài phạm vi source")
        if path.suffix == ".py":
            counts["python"] += 1
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=name)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        module = node.module
                        if module.split(".")[0] in {"src", "scripts", "examples"}:
                            target = ROOT.joinpath(*module.split("."))
                            if not target.with_suffix(".py").is_file() and not (target/"__init__.py").is_file():
                                errors.append(f"{name}: import local không tồn tại: {module}")
            except (SyntaxError, UnicodeError) as error:
                errors.append(f"{name}: {error}")
        if path.suffix == ".ipynb":
            counts["notebooks"] += 1
            notebook = json.loads(path.read_text(encoding="utf-8"))
            if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
                errors.append(f"{name}: sai định dạng notebook")
            for index, cell in enumerate(notebook.get("cells", []), 1):
                if cell.get("cell_type") == "code" and (
                    cell.get("outputs") or cell.get("execution_count") is not None
                ):
                    errors.append(f"{name}: cell {index} có output/execution count")
        if path.suffix == ".md":
            text = path.read_text(encoding="utf-8")
            # Chỉ kiểm link Markdown; đường dẫn output trong inline code không phải link.
            for match in LINK.finditer(text):
                target = match.group(1).strip().strip("<>")
                if target.startswith(("#", "http:", "https:", "mailto:", "data:")):
                    continue
                target = unquote(target.split("#", 1)[0])
                if not target:
                    continue
                counts["local_links"] += 1
                if not (path.parent / target).exists():
                    errors.append(f"{name}: link local thiếu: {target}")
        if path.suffix in {".py", ".md", ".toml", ".yml", ".txt", ".json"}:
            text = path.read_text(encoding="utf-8-sig")
            if any(pattern.search(text) for pattern in SECRETS):
                errors.append(f"{name}: chuỗi giống secret cần kiểm")
    return counts, errors


def main():
    counts, errors = check_source(tracked_paths())
    print(json.dumps({"passed": not errors, **counts, "errors": errors},
                     ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
