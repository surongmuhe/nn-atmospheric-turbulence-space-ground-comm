"""Build a clean GitHub publishing snapshot.

The working project contains many historical thesis drafts, intermediate
prediction CSVs, checkpoints, and render artifacts. This script creates a
curated snapshot with the current source code, datasets, final thesis, and
key result summaries suitable for a private GitHub repository.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_github_publish"

ROOT_FILES = [
    "README.md",
    "requirements.txt",
    ".gitignore",
]

DATA_FILES = [
    "data/Sklavounos and Cohn, Spring. 2022.xlsx",
    "data/dataWfon0U_ml_ready.csv",
    "data/dataWfon0U_ml_ready.summary.json",
]

DOC_FILES = [
    "docs/graduation_thesis_v34_final_review_fixed.docx",
    "docs/graduation_thesis_v34_final_review_fixed.md",
    "docs/reproducibility_checklist_20260426.md",
    "docs/project_cleanup_manifest.md",
]

OUTPUT_DIRS = [
    "outputs/thesis_lstm_final_best_seq72/formal_20260424",
    "outputs/current_route_main_site/formal_20260425_delta",
    "outputs/strong_baselines_seq72_20260425",
    "outputs/final_ablation",
    "outputs/external_site_four_model_route_20260426_validated",
    "outputs/cross_site_fewshot_adaptation/formal_20260426",
    "outputs/fewshot_strong_transfer_controls_20260427",
    "outputs/review_response_improvements_20260426",
    "outputs/ber_mcs_link_enhancement_20260427",
]

EXTRA_OUTPUT_FILES = [
    "outputs/graduation_thesis_v34_final_review_fixed_quality.json",
]

SKIP_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".pkl",
    ".npy",
    ".npz",
    ".pyc",
    ".log",
}

SKIP_NAME_PARTS = [
    "prediction",
    "predictions",
    "timeseries",
    "residual",
    "rolling_error",
    "rolling_errors",
]

KEEP_NAME_PARTS = [
    "summary",
    "metric",
    "metrics",
    "ranking",
    "rank",
    "bootstrap",
    "sensitivity",
    "assumption",
    "config",
    "quality",
    "manifest",
    "best_by",
    "sweep",
    "ablation",
    "policy_summary",
]


def safe_reset_out() -> None:
    resolved = OUT.resolve()
    root = ROOT.resolve()
    if not str(resolved).startswith(str(root)):
        raise RuntimeError(f"Refusing to remove path outside project: {resolved}")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)


def copy_file(rel: str, copied: list[dict]) -> None:
    src = ROOT / rel
    if not src.exists() or not src.is_file():
        return
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append({"path": rel.replace("\\", "/"), "bytes": src.stat().st_size})


def copy_tree(rel_dir: str, copied: list[dict], suffixes: set[str] | None = None) -> None:
    base = ROOT / rel_dir
    if not base.exists():
        return
    for src in base.rglob("*"):
        if not src.is_file():
            continue
        if "__pycache__" in src.parts:
            continue
        if src.suffix.lower() in SKIP_SUFFIXES:
            continue
        if suffixes is not None and src.suffix.lower() not in suffixes:
            continue
        rel = src.relative_to(ROOT).as_posix()
        copy_file(rel, copied)


def should_keep_output(src: Path) -> bool:
    if src.suffix.lower() in SKIP_SUFFIXES:
        return False
    name = src.name.lower()
    if any(part in name for part in SKIP_NAME_PARTS):
        return False
    if src.stat().st_size > 2 * 1024 * 1024:
        return False
    if src.suffix.lower() in {".json", ".yaml", ".yml", ".md"}:
        return True
    if src.suffix.lower() == ".csv" and any(part in name for part in KEEP_NAME_PARTS):
        return True
    return False


def copy_outputs(copied: list[dict]) -> None:
    for rel_dir in OUTPUT_DIRS:
        base = ROOT / rel_dir
        if not base.exists():
            continue
        for src in base.rglob("*"):
            if not src.is_file():
                continue
            if should_keep_output(src):
                copy_file(src.relative_to(ROOT).as_posix(), copied)
    for rel in EXTRA_OUTPUT_FILES:
        copy_file(rel, copied)


def write_snapshot_notes(copied: list[dict]) -> None:
    notes = OUT / "outputs" / "README.md"
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text(
        "# 输出结果说明\n\n"
        "本目录保留论文相关的关键结果汇总、指标表、配置快照和质量检查文件。\n\n"
        "为控制 GitHub 仓库体积，以下文件没有进入发布快照：模型权重、逐样本 predictions、"
        "逐时刻 link timeseries、residual/rolling-error 中间文件以及旧版临时输出。\n\n"
        "如需完整逐样本结果，可运行 `scripts/` 中对应实验脚本重新生成。\n",
        encoding="utf-8",
    )
    copied.append({"path": "outputs/README.md", "bytes": notes.stat().st_size})


def main() -> None:
    safe_reset_out()
    copied: list[dict] = []

    for rel in ROOT_FILES + DATA_FILES + DOC_FILES:
        copy_file(rel, copied)

    copy_tree("configs", copied, suffixes={".yaml", ".yml"})
    copy_tree("src", copied, suffixes={".py"})
    copy_tree("scripts", copied, suffixes={".py"})
    copy_outputs(copied)
    write_snapshot_notes(copied)

    total_bytes = sum(item["bytes"] for item in copied)
    manifest = {
        "snapshot_dir": str(OUT),
        "file_count": len(copied),
        "total_mb": round(total_bytes / 1024 / 1024, 2),
        "largest_files": sorted(copied, key=lambda item: item["bytes"], reverse=True)[:20],
        "files": sorted(copied, key=lambda item: item["path"]),
    }
    (OUT / "UPLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "UPLOAD_MANIFEST.md").write_text(
        "# GitHub 发布快照清单\n\n"
        f"- 文件数：{manifest['file_count']}\n"
        f"- 总大小：{manifest['total_mb']} MB\n"
        "- 发布策略：保留源码、配置、数据、最终论文和关键结果汇总；排除模型权重、缓存、旧论文草稿和大体积逐样本中间文件。\n\n"
        "## 最大文件\n\n"
        + "\n".join(
            f"- `{item['path']}`：{round(item['bytes'] / 1024 / 1024, 2)} MB"
            for item in manifest["largest_files"]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: manifest[k] for k in ["snapshot_dir", "file_count", "total_mb"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
