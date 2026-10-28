"""Pixel-only products and atomic JSON persistence."""
import csv
import json
import os
from pathlib import Path
from uuid import uuid4

from droplet_analysis.impact import impact_point


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def pixel_csv(path, manifest, trace=False):
    columns = ["sequence", "frame", "source_file", "phase", "relative_time_s",
               "contact_length_px", "contact_start_s_px", "contact_end_s_px",
               "maximum_visible_thickness_px", "thickness_start_x_px", "thickness_start_y_px",
               "thickness_end_x_px", "thickness_end_y_px", "impact_frame", "impact_point_x_px",
               "impact_point_y_px", "impact_point_s_px", "impact_point_raw_x_px",
               "impact_point_raw_y_px", "within_valid_view", "measurement_status", "manual_reviewed", "flags"]
    origin = impact_point(manifest)
    xy, raw = origin.get("point_px") or [None, None], origin.get("raw_point_px") or [None, None]
    rows = []
    for frame in manifest["frames"]:
        if not trace and (frame["phase"] != "postimpact" or frame.get("within_valid_view") is False):
            continue
        contact = frame["contact_segments"][0] if frame["contact_segments"] else {}
        height = frame.get("maximum_visible_thickness") or {}
        start, end = height.get("start", [None, None]), height.get("end", [None, None])
        rows.append(dict(sequence=manifest["sequence"], frame=frame["frame"], source_file=frame.get("source_file", ""),
                         phase=frame["phase"], relative_time_s=frame.get("relative_time_s"),
                         contact_length_px=frame["contact_length_px"], contact_start_s_px=contact.get("s_start_px"),
                         contact_end_s_px=contact.get("s_end_px"), maximum_visible_thickness_px=height.get("value_px"),
                         thickness_start_x_px=start[0], thickness_start_y_px=start[1], thickness_end_x_px=end[0], thickness_end_y_px=end[1],
                         impact_frame=origin.get("frame"), impact_point_x_px=xy[0], impact_point_y_px=xy[1], impact_point_s_px=origin.get("s_px"),
                         impact_point_raw_x_px=raw[0], impact_point_raw_y_px=raw[1], within_valid_view=frame.get("within_valid_view", True),
                         measurement_status=frame["measurement_status"], manual_reviewed=frame.get("manual_reviewed", False), flags=";".join(frame["flags"])))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def series_plots(folder, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = [f for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"] if f in available]
    rows = [r for r in manifest["frames"] if r["phase"] == "postimpact" and r.get("within_valid_view") is not False]
    # No valid measurement means an empty plot, never preimpact data substituted as formal.
    for stem, label, values in [
        ("thickness_vs_frame", "Maximum main-film thickness (px)", [(r["maximum_visible_thickness"] or {}).get("value_px", np.nan) for r in rows]),
        ("contact_length_vs_frame", "Contact endpoint-span arc length (px)", [r["contact_length_px"] if r["measurement_status"] != "missing_main_film" else np.nan for r in rows])]:
        fig, ax = plt.subplots(figsize=(10.8, 3.8), layout="constrained")
        ax.plot([r["frame"] for r in rows], values, color="#146b87", lw=1.4)
        if manifest["impact"]["first_contact_frame"] is not None:
            ax.axvline(manifest["impact"]["first_contact_frame"], color="#697a83", ls="--", lw=1)
        numbers = [r["frame"] for r in manifest["frames"]]
        ax.set(xlabel="Source frame number", ylabel=label, title=manifest["sequence"], xlim=(min(numbers), max(numbers) if max(numbers) > min(numbers) else min(numbers)+1))
        ax.grid(alpha=.18)
        for suffix in ["png", "svg"]:
            fig.savefig(Path(folder) / (stem + "." + suffix), dpi=160)
        plt.close(fig)
