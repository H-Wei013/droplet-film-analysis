"""Loopback-only, session-scoped access to explicitly opened result folders."""
import copy
import hashlib
import json
import math
import mimetypes
import os
import secrets
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from .io import atomic_json, load_json, pixel_csv, series_plots
from .review import render_review, write_review


class Conflict(ValueError):
    pass


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_edit(edit, manifest):
    known = {str(f["frame"]) for f in manifest["frames"]}
    frames = edit.get("frames", {})
    if edit.get("analysis_signature") != manifest["analysis_signature"] or edit.get("coordinate_system") != "aligned_reference_px":
        raise ValueError("复核文件与结果签名或坐标系不一致。")
    if not isinstance(frames, dict) or set(frames)-known:
        raise ValueError("复核文件含未知帧号。")
    impact = edit.get("impact_frame")
    if impact is not None and (not isinstance(impact, int) or str(impact) not in known):
        raise ValueError("撞击帧不存在。")
    width, height = manifest["image_size"]
    lo, hi = manifest["wall"]["s_px"][0], manifest["wall"]["s_px"][-1]
    for item in frames.values():
        if not isinstance(item, dict) or len(str(item.get("note", ""))) > 100000:
            raise ValueError("复核记录格式无效。")
        for a, b in item.get("contact_intervals", []):
            if not finite(a) or not finite(b) or not lo <= a < b <= hi:
                raise ValueError("接触段超出有效壁面。")
        for poly in item.get("contours", []):
            if len(poly) < 3 or len(poly) > 100000:
                raise ValueError("轮廓点数无效。")
            for point in poly:
                if len(point) != 2 or not all(finite(x) for x in point) or not (0 <= point[0] < width and 0 <= point[1] < height):
                    raise ValueError("轮廓超出图像。")
        thickness = item.get("manual_thickness")
        if thickness is not None:
            s, point = thickness.get("wall_s_px"), thickness.get("outer_point")
            if not finite(s) or not lo <= s <= hi or not isinstance(point, list) or len(point) != 2 or not all(finite(x) for x in point) or not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError("人工厚度选点无效。")


class Project:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.groups = {}
        paths = sorted(self.folder.glob("*/measurements.json"))
        if (self.folder / "measurements.json").exists():
            paths = [self.folder / "measurements.json"]
        for p in paths:
            m = load_json(p)
            if m.get("schema_version") != 5:
                raise ValueError("仅支持 V5 测量结果。")
            key = m["analysis_signature"]
            if key in self.groups:
                raise ValueError("存在重复分析签名。")
            self.groups[key] = dict(manifest=m, path=p)
        if not self.groups:
            raise ValueError("此目录还没有完整的测量结果。")
        self.assets = {}
        for group in self.groups.values():
            m = copy.deepcopy(group["manifest"])
            m["units"] = "px"
            m["physical_calibration"] = dict(status="not_configured", mm_per_px=None)
            if m.get("impact_point"):
                m["impact_point"]["s_mm"] = None
            for f in m["frames"]:
                for key in ["image", "mask"]:
                    asset = (group["path"].parent / f[key]).resolve()
                    if asset.suffix.lower() not in [".png", ".bmp"]:
                        raise ValueError("结果图像格式不支持。")
                    ident = hashlib.sha256(str(asset).encode()).hexdigest()[:24]
                    self.assets[ident] = asset
                    f[key] = "@asset/" + ident
            group["view"] = m
        self.lock = threading.Lock()

    @property
    def state_path(self):
        return self.folder / "reviews" / "state.json"

    def read_state(self):
        if not self.state_path.exists():
            return dict(revision=0, sequences={})
        state = load_json(self.state_path)
        for key, edit in state["sequences"].items():
            if key not in self.groups:
                raise ValueError("磁盘复核记录包含未知分析签名。")
            validate_edit(edit, self.groups[key]["manifest"])
        return state

    def save_state(self, revision, changes):
        if not isinstance(changes, dict) or set(changes)-self.groups.keys():
            raise ValueError("复核目标不存在。")
        for key, edit in changes.items():
            validate_edit(edit, self.groups[key]["manifest"])
        with self.lock:
            state = self.read_state()
            if revision != state["revision"]:
                raise Conflict("另一页面已保存复核。请先导出当前未保存记录，再刷新页面后导入，避免覆盖。")
            state["sequences"].update(changes)
            state["revision"] += 1
            state["saved_at"] = datetime.now().isoformat(timespec="seconds")
            atomic_json(self.state_path, state)
            return state

    def render(self, prefix, selected=None):
        groups = list(self.groups.values())
        if selected is not None:
            if selected not in self.groups:
                raise ValueError("结果不存在。")
            groups = [self.groups[selected]]
        manifests = []
        for group in groups:
            m = copy.deepcopy(group["view"])
            for f in m["frames"]:
                for k in ["image", "mask"]:
                    f[k] = prefix + "/assets/" + f[k].removeprefix("@asset/")
            manifests.append(m)
        payload = {"sequences": manifests} if selected is None else manifests[0]
        return render_review(payload, dict(api=prefix, sequence_signatures=list(self.groups)), back_link=prefix+"/index.html")

    def export(self, signature, revision, value):
        with self.lock:
            state = self.read_state()
            if revision != state["revision"]:
                raise Conflict("复核版本已变化，请刷新后重试导出。")
            base = self.groups[signature]["manifest"]
            if value.get("analysis_signature") != signature or value.get("wall") != base["wall"] or value.get("settings") != base["settings"]:
                raise ValueError("导出几何参考不一致。")
            if [r["frame"] for r in value["frames"]] != [r["frame"] for r in base["frames"]]:
                raise ValueError("导出帧号不一致。")
            if value["impact"]["first_contact_frame"] not in [None]+[r["frame"] for r in base["frames"]]:
                raise ValueError("导出撞击帧无效。")
            for frame, original in zip(value["frames"], base["frames"]):
                if frame.get("within_valid_view", True) != original.get("within_valid_view", True):
                    raise ValueError("不能改变自动有效测量范围。")
                invalid = original.get("within_valid_view") is False
                impact = value["impact"]["first_contact_frame"]
                phase = "after_valid_view" if invalid else "postimpact" if impact is not None and frame["frame"] >= impact else "preimpact"
                if frame.get("phase") != phase:
                    raise ValueError("导出阶段与有效范围不一致。")
                if not finite(frame["contact_length_px"]) or frame["contact_length_px"] < 0:
                    raise ValueError("导出长度无效。")
                seg = frame["contact_segments"]
                if len(seg) > 1 or (not seg and frame["contact_length_px"] != 0) or (seg and abs(seg[0]["s_end_px"]-seg[0]["s_start_px"]-frame["contact_length_px"]) > 1e-6):
                    raise ValueError("导出长度与接触跨度不一致。")
                thickness = frame.get("maximum_visible_thickness")
                if thickness is not None and (not finite(thickness.get("value_px")) or thickness["value_px"] < 0):
                    raise ValueError("导出厚度无效。")
            destination = self.folder / "exports" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6])
            destination.mkdir(parents=True)
            value = copy.deepcopy(value)
            value["units"] = "px"; value["physical_calibration"] = dict(status="not_configured", mm_per_px=None)
            if value.get("impact_point"):
                value["impact_point"]["s_mm"] = None
            for f, original in zip(value["frames"], base["frames"]):
                for key in ["image", "mask"]:
                    actual = (self.groups[signature]["path"].parent / original[key]).resolve()
                    f[key] = Path(os.path.relpath(actual, destination)).as_posix()
            value["review_revision"] = revision
            atomic_json(destination / "measurements.json", value)
            pixel_csv(destination / "measurements.csv", value)
            pixel_csv(destination / "all_frames_trace.csv", value, True)
            series_plots(destination, value)
            write_review(destination / "review.html", value)
            return str(destination)


class ReviewService:
    def __init__(self, port=0):
        self.projects = {}
        self.token = secrets.token_urlsafe(24)
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def route(self):
                parts = unquote(urlsplit(self.path).path).strip("/").split("/")
                if len(parts) < 4 or parts[0] != "s" or not secrets.compare_digest(parts[1], service.token) or parts[2] not in service.projects:
                    raise ValueError("未授权的结果路径。")
                return service.projects[parts[2]], parts[3:], "/"+"/".join(parts[:3])

            def respond(self, value, status=200, mime="application/json; charset=utf-8"):
                data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
                self.send_response(status); self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff"); self.end_headers(); self.wfile.write(data)

            def do_GET(self):
                try:
                    project, parts, prefix = self.route()
                    if parts == ["api", "reviews"]:
                        self.respond(project.read_state())
                    elif parts[0] == "assets" and len(parts) == 2 and parts[1] in project.assets:
                        p = project.assets[parts[1]]
                        self.respond(p.read_bytes(), mime=mimetypes.guess_type(p.name)[0] or "image/png")
                    elif parts == ["review.html"]:
                        self.respond(project.render(prefix).encode(), mime="text/html; charset=utf-8")
                    elif len(parts) == 2 and parts[1] == "review.html":
                        self.respond(project.render(prefix, parts[0]).encode(), mime="text/html; charset=utf-8")
                    elif parts == ["index.html"]:
                        import html
                        links = ''.join('<li><a href="'+prefix+'/'+key+'/review.html">'+html.escape(g["manifest"]["sequence"])+"</a></li>" for key,g in project.groups.items())
                        doc = '<!doctype html><meta charset="utf-8"><title>液膜工具</title><h1>液膜工具 · 算法 5.3.1 · px</h1><p>复核自动保存到结果目录；需要新数据测试确认适用性。</p><a href="'+prefix+'/review.html">统一复核页</a><ul>'+links+'</ul>'
                        self.respond(doc.encode(), mime="text/html; charset=utf-8")
                    else:
                        self.respond({"error": "文件不存在"}, 404)
                except Exception as error:
                    self.respond({"error": str(error)}, 400)

            def do_POST(self):
                try:
                    host = self.headers.get("Host", "")
                    allowed = {f"127.0.0.1:{service.http.server_port}", f"localhost:{service.http.server_port}"}
                    origin = self.headers.get("Origin")
                    if host not in allowed or (origin and origin not in {"http://"+h for h in allowed}):
                        raise ValueError("不接受跨站保存请求。")
                    project, parts, _ = self.route()
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 128*1024*1024:
                        raise ValueError("请求大小无效。")
                    value = json.loads(self.rfile.read(size))
                    if parts == ["api", "reviews"]:
                        result = project.save_state(value["revision"], value["changes"])
                    elif parts == ["api", "export"]:
                        result = dict(directory=project.export(value["signature"], value["revision"], value["manifest"]))
                    else:
                        raise ValueError("未知保存操作。")
                    self.respond(result)
                except Conflict as error:
                    self.respond({"error": str(error)}, 409)
                except Exception as error:
                    self.respond({"error": str(error)}, 400)

        self.http = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()

    def open_project(self, folder):
        project = Project(folder)
        ident = hashlib.sha256(str(project.folder).encode()).hexdigest()[:16]
        self.projects[ident] = project
        return f"http://127.0.0.1:{self.http.server_port}/s/{self.token}/{ident}/review.html"

    def close(self):
        self.http.shutdown(); self.http.server_close()
