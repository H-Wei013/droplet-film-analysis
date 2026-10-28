"""Local desktop launcher. Numerical jobs execute in a separate Python process."""
import argparse
import importlib
import importlib.metadata
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import webbrowser

from . import TOOL_VERSION, ALGORITHM_VERSION
from .io import load_json
from .server import ReviewService

ROOT = Path(__file__).resolve().parents[2]


def check_environment():
    from .provenance import digest
    result = dict(tool_version=TOOL_VERSION, algorithm_version=ALGORITHM_VERSION, python=sys.version.split()[0], units="px", engine_sha256=digest()[0])
    for module in ["numpy", "scipy", "cv2", "matplotlib", "yaml"]:
        importlib.import_module(module)
    result["dependencies"] = {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "opencv-python", "matplotlib", "PyYAML"]}
    import tkinter
    result["tkinter"] = tkinter.TkVersion
    return result


def main():
    parser = argparse.ArgumentParser(description="BMP droplet/film tool; numerical engine 5.3.1; pixel units only")
    parser.add_argument("--check", action="store_true", help="Check imports/environment without opening a window or loading images")
    parser.add_argument("--input", type=Path, help="Headless batch input folder")
    parser.add_argument("--resume", type=Path, help="Resume a previous run folder")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.check:
        import json
        print(json.dumps(check_environment(), ensure_ascii=False, indent=2)); return
    from .jobs import create_run, run
    config_path = Path(__file__).resolve().parents[1] / "droplet_analysis" / "config.json"
    settings = load_json(config_path)["defaults"]
    if args.input or args.resume:
        folder = args.resume or create_run(args.input, ROOT / "runs", settings, args.workers)
        print(str(folder), flush=True); run(folder); return
    gui(settings)


def gui(settings):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from .jobs import create_run
    window = tk.Tk()
    window.title("液滴 / 液膜识别工具 0.1.0 · 算法 5.3.1 · px")
    window.geometry("850x560"); window.minsize(700, 460)
    outer = ttk.Frame(window, padding=16); outer.pack(fill="both", expand=True)
    ttk.Label(outer, text="液滴 / 液膜识别工具", font=("Microsoft YaHei", 18)).pack(anchor="w")
    ttk.Label(outer, text="工具 0.1.0 · 识别算法 5.3.1 · 仅像素单位 · 尚待新数据测试").pack(anchor="w", pady=(3, 12))
    ttk.Label(outer, text="选择单组 BMP 文件夹，或包含多组序列的上级目录：").pack(anchor="w")
    row = ttk.Frame(outer); row.pack(fill="x", pady=6)
    path = tk.StringVar(); ttk.Entry(row, textvariable=path).pack(side="left", fill="x", expand=True)
    ttk.Button(row, text="选择文件夹", command=lambda: choose_input()).pack(side="left", padx=(6, 0))
    bar = ttk.Frame(outer); bar.pack(fill="x", pady=8)
    status = tk.StringVar(value="请提供图像路径。前 32 帧用于参考背景；壁面需位于右侧。")
    ttk.Label(outer, textvariable=status, wraplength=800).pack(anchor="w", pady=(3, 8))
    log = tk.Text(outer, height=15, wrap="word", state="disabled"); log.pack(fill="both", expand=True)
    service = ReviewService(); messages = queue.Queue()
    current = {"process": None, "folder": None, "preparing": False, "closing": False}

    def append(value):
        log.configure(state="normal"); log.insert("end", value+"\n"); log.see("end"); log.configure(state="disabled")

    def choose_input():
        selected = filedialog.askdirectory(title="选择 BMP 图像目录")
        if selected: path.set(selected)

    def is_running():
        return current["preparing"] or (current["process"] is not None and current["process"].poll() is None)

    def spawn(folder):
        current["folder"] = Path(folder)
        env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
        executable = Path(sys.executable)
        if executable.name.lower() == "pythonw.exe": executable = executable.with_name("python.exe")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen([str(executable), "-m", "watercar_tool.jobs", "--run", str(folder)], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=flags)
        current["process"] = process
        def reader():
            for line in process.stdout: messages.put(("line", line.rstrip()))
            messages.put(("finished", process.wait()))
        threading.Thread(target=reader, daemon=True).start()
        status.set("正在识别；结果保存到："+str(folder)); append("结果目录："+str(folder))

    def start():
        if is_running(): return
        requested = path.get().strip().strip('"')
        if not requested: messagebox.showinfo("图像目录", "请先提供图像文件夹路径。"); return
        current["preparing"] = True; status.set("正在检查图片、帧号和尺寸…")
        def prepare():
            try: messages.put(("prepared", create_run(requested, ROOT / "runs", settings)))
            except Exception as error: messages.put(("error", str(error)))
        threading.Thread(target=prepare, daemon=True).start()

    def stop():
        if current["preparing"]: status.set("正在完成输入检查，检查完成后可停止识别。")
        if current["folder"] and current["process"] and current["process"].poll() is None:
            (current["folder"] / "STOP").touch(); status.set("已请求停止；等待当前帧结束，已完成组会保留。")

    def resume():
        if is_running(): return
        selected = filedialog.askdirectory(title="选择包含 run.json 的历史任务目录", initialdir=ROOT / "runs")
        if selected:
            if not (Path(selected) / "run.json").exists(): messagebox.showerror("无法继续", "目录内没有 run.json。"); return
            spawn(selected)

    def open_result():
        selected = current["folder"]
        if selected is None:
            value = filedialog.askdirectory(title="选择结果目录", initialdir=ROOT / "runs")
            if not value: return
            selected = Path(value)
        try: webbrowser.open(service.open_project(selected))
        except Exception as error: messagebox.showerror("打开结果", str(error))

    def open_history():
        value = filedialog.askdirectory(title="选择已有 V5 结果目录", initialdir=ROOT / "runs")
        if value:
            try: webbrowser.open(service.open_project(value))
            except Exception as error: messagebox.showerror("打开历史结果", str(error))

    def open_folder():
        folder = current["folder"] or ROOT / "runs"; folder.mkdir(parents=True, exist_ok=True)
        if os.name == "nt": os.startfile(folder)
        else: webbrowser.open(folder.as_uri())

    for title, command in [("开始识别", start), ("停止", stop), ("继续任务", resume), ("打开本次结果", open_result), ("打开历史结果", open_history), ("结果目录", open_folder)]:
        ttk.Button(bar, text=title, command=command).pack(side="left", padx=(0, 5))

    def closing():
        current["closing"] = True; stop()
        if not is_running(): service.close(); window.destroy()
        else: status.set("正在安全停止任务，完成后关闭窗口…")

    def poll():
        try:
            while True:
                kind, value = messages.get_nowait()
                if kind == "line": append(value); status.set(value)
                elif kind == "prepared":
                    current["preparing"] = False
                    if not current["closing"]: spawn(value)
                elif kind == "error":
                    current["preparing"] = False; status.set("输入检查失败："+value); append(value)
                elif kind == "finished":
                    current["process"] = None
                    state = load_json(current["folder"] / "run.json")
                    status.set("任务结束："+state["status"]+"；完成 "+str(len(state["completed"]))+"/"+str(len(state["sequences"]))+" 组")
                    if state["completed"] and not current["closing"]: open_result()
        except queue.Empty: pass
        if current["closing"] and not is_running(): service.close(); window.destroy(); return
        window.after(200, poll)

    window.protocol("WM_DELETE_WINDOW", closing); window.after(200, poll); window.mainloop()


if __name__ == "__main__":
    main()
