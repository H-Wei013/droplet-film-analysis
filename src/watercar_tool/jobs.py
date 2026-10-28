"""Independent sequence jobs, restartable only against identical inputs."""
import argparse
import csv
import hashlib
import html
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from droplet_analysis.core import Settings, frame_number, read_gray
from .review import write_combined_review
from . import ALGORITHM_VERSION, TOOL_VERSION
from .io import atomic_json, load_json
from .provenance import digest


def preflight(input_path, settings=None):
    root = Path(input_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("请选择存在的图像文件夹。")
    cfg = Settings.from_dict(settings or {})
    directories = sorted({p.parent for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".bmp"})
    if not directories:
        raise ValueError("没有找到 BMP 图像序列。")
    entries, errors = [], []
    for directory in directories:
        label = directory.relative_to(root).as_posix() if directory != root else root.name
        try:
            files = sorted([p for p in directory.iterdir() if p.suffix.lower() == ".bmp"], key=frame_number)
            numbers = [frame_number(p) for p in files]
            if len(files) < cfg.reference_frames:
                raise ValueError(f"只有 {len(files)} 帧，少于 {cfg.reference_frames} 个参考帧。")
            if len(set(numbers)) != len(numbers):
                raise ValueError("存在重复帧号。")
            image_size, records = None, []
            for p, number in zip(files, numbers):
                image = read_gray(p)
                if image_size is None:
                    image_size = list(image.shape[::-1])
                if list(image.shape[::-1]) != image_size:
                    raise ValueError("同组图像尺寸不一致。")
                records.append(dict(frame=number, name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            metadata = directory / "video_param.json"
            if metadata.exists():
                value = load_json(metadata)
                rate = value.get("frame_rate")
                if rate is not None and (not isinstance(rate, (int, float)) or rate <= 0):
                    raise ValueError("video_param.json 中的帧率无效。")
            fingerprint = hashlib.sha256(json.dumps(dict(files=records, metadata=hashlib.sha256(metadata.read_bytes()).hexdigest() if metadata.exists() else None), sort_keys=True).encode()).hexdigest()
            ident = hashlib.sha256(str(directory).encode()).hexdigest()[:12]
            entries.append(dict(id=ident, label=label, directory=str(directory), frames=len(files), image_size=image_size,
                                input_sha256=fingerprint, first_frame=numbers[0], last_frame=numbers[-1],
                                has_frame_gaps=any(b != a+1 for a, b in zip(numbers, numbers[1:]))))
        except (ValueError, OSError, json.JSONDecodeError) as error:
            errors.append(dict(sequence=label, reason=str(error)))
    if not entries:
        raise ValueError("没有可处理的序列：\n" + "\n".join(e["sequence"] + ": " + e["reason"] for e in errors))
    return dict(input=str(root), sequences=entries, rejected=errors)


def create_run(input_path, runs_root, settings, workers=4):
    input_path, runs_root = Path(input_path).resolve(), Path(runs_root).resolve()
    if runs_root == input_path or runs_root.is_relative_to(input_path) or input_path.is_relative_to(runs_root):
        raise ValueError("图像目录与工具的 runs 输出目录不能重叠。请选择独立的原始图像目录。")
    scanned = preflight(input_path, settings)
    folder = runs_root / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6])
    folder.mkdir(parents=True)
    sha, files = digest()
    state = dict(tool_version=TOOL_VERSION, algorithm_version=ALGORITHM_VERSION, algorithm_sha256=sha,
                 source_files_sha256=files, settings=settings, workers=max(1, min(4, workers)), units="px",
                 **scanned, status="ready", completed=[], failed={}, created_at=datetime.now().isoformat())
    atomic_json(folder / "run.json", state)
    publish_index(folder, state)
    return folder


def publish_index(folder, state):
    cards = []
    for entry in state["sequences"]:
        ident, label = entry["id"], html.escape(entry["label"])
        if ident in state["completed"]:
            cards.append(f'<li><a href="{ident}/review.html">{label} · 逐帧复核</a> · <a href="{ident}/measurements.csv">自动正式 CSV</a></li>')
        else:
            cards.append(f'<li>{label} · {html.escape(state["failed"].get(ident, "尚未完成"))}</li>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>液膜分析任务</title><style>body{font:16px/1.8 system-ui;max-width:1100px;margin:35px auto;padding:20px}a{color:#08738b}</style><h1>液膜分析工具 0.1.0 · 识别算法 5.3.1</h1><p>仅像素单位；新数据识别效果需人工核对。自动数值保持原样，人工修正保存在 reviews，修正后导出保存在 exports。</p>'
    page += f'<p>状态：{html.escape(state["status"])}；完成 {len(state["completed"])}/{len(state["sequences"])} 组。</p>'
    if state["completed"]:
        page += '<p><a href="review.html">统一复核页</a> · <a href="parameters_all.csv">自动正式范围总表</a></p>'
    page += '<ul>' + ''.join(cards) + '</ul>'
    if state.get("rejected"):
        page += '<h2>未处理的目录</h2><pre>' + html.escape(json.dumps(state["rejected"], ensure_ascii=False, indent=2)) + '</pre>'
    (folder / "index.html").write_text(page + '</html>', encoding="utf-8")


def worker(entry, folder, settings, cancel):
    from .engine import analyze_sequence
    folder = Path(folder)
    if Path(cancel).exists():
        raise InterruptedError("Task stopped")
    temporary = folder / (".partial_" + entry["id"] + "_" + uuid4().hex[:8])
    summary = analyze_sequence(entry["directory"], temporary, settings, sequence_name=entry["label"], cancel_path=cancel)
    fresh = preflight(entry["directory"], settings)
    if len(fresh["sequences"]) != 1 or fresh["sequences"][0]["input_sha256"] != entry["input_sha256"]:
        raise ValueError("处理期间原图发生变化，请创建新任务。")
    atomic_json(temporary / "complete.json", dict(input_sha256=entry["input_sha256"], algorithm_sha256=digest()[0]))
    # Promotion occurs only after a complete sequence; interrupted work is never resumed as complete.
    destination = folder / entry["id"]
    if destination.exists():
        raise ValueError("Completed output already exists; refusing replacement.")
    temporary.rename(destination)
    return summary


def run(folder):
    folder = Path(folder).resolve()
    state = load_json(folder / "run.json")
    if state["algorithm_sha256"] != digest()[0]:
        raise ValueError("算法源码已变化，不能继续旧任务。请创建新任务。")
    check = preflight(state["input"], state["settings"])
    if {e["id"]: e["input_sha256"] for e in check["sequences"]} != {e["id"]: e["input_sha256"] for e in state["sequences"]}:
        raise ValueError("输入图像或相机元数据已变化，不能继续旧任务。")
    cancel = folder / "STOP"
    cancel.unlink(missing_ok=True)
    state["status"] = "running"
    state["failed"] = {}
    for entry in state["sequences"]:
        destination = folder / entry["id"]
        if not destination.exists():
            if entry["id"] in state["completed"]:
                raise ValueError("已完成组的结果目录缺失。")
            continue
        marker = load_json(destination / "complete.json")
        m = load_json(destination / "measurements.json")
        if marker != dict(input_sha256=entry["input_sha256"], algorithm_sha256=state["algorithm_sha256"]) or m["algorithm_sha256"] != state["algorithm_sha256"]:
            raise ValueError("已完成组的输入或算法标识不一致。")
        # Recover a completed folder if the parent process exited before updating run.json.
        if entry["id"] not in state["completed"]:
            state["completed"].append(entry["id"])
    pending = [e for e in state["sequences"] if e["id"] not in state["completed"]]
    atomic_json(folder / "run.json", state)
    with ProcessPoolExecutor(max_workers=state["workers"]) as pool:
        futures = {pool.submit(worker, e, str(folder), state["settings"], str(cancel)): e for e in pending}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                future.result()
                state["completed"].append(entry["id"])
            except Exception as error:
                state["failed"][entry["id"]] = str(error)
            atomic_json(folder / "run.json", state)
            publish_index(folder, state)
            print(f'已完成 {len(state["completed"])}/{len(state["sequences"])} 组；{entry["label"]}', flush=True)
    state["status"] = "stopped" if cancel.exists() else "complete_with_errors" if state["failed"] else "complete"
    if state["completed"]:
        paths = [folder / e["id"] / "measurements.json" for e in state["sequences"] if e["id"] in state["completed"]]
        write_combined_review(folder / "review.html", paths)
        rows = []
        for p in paths:
            with (p.parent / "measurements.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows.extend(csv.DictReader(stream))
        # Also create a header-only combined table when no impact was detected.
        with (paths[0].parent / "measurements.csv").open(encoding="utf-8-sig", newline="") as stream:
            fields = next(csv.reader(stream))
        with (folder / "parameters_all.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    atomic_json(folder / "run.json", state)
    publish_index(folder, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.run)
    except Exception as error:
        print(str(error), file=sys.stderr, flush=True)
        if (args.run / "run.json").exists():
            state = load_json(args.run / "run.json"); state["status"] = "failed"; state["error"] = str(error)
            atomic_json(args.run / "run.json", state)
        sys.exit(1)
