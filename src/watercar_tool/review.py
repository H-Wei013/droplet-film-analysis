"""Renderer shared by disk review pages and the local persistence service."""
import json
import html
from pathlib import Path
from . import TOOL_VERSION
from .datasets import display_manifest,dataset_key

ASSETS = Path(__file__).resolve().parents[1] / "droplet_analysis"
ASSET_FIELDS = ("image", "mask", "observed_mask", "contact_completion_mask")


def render_review(manifest, context=None, back_link="../index.html"):
    if 'sequences' in manifest:
        manifest=dict(manifest,sequences=sorted([display_manifest(m) for m in manifest['sequences']],key=dataset_key))
    else:manifest=display_manifest(manifest)
    template = (ASSETS / "review_template.html").read_text(encoding="utf-8")
    groups = manifest.get('sequences', [manifest])
    versions = sorted({m.get('algorithm_version', 'unknown').split('-')[0] for m in groups})
    replacements = {
        "__TOOL_VERSION__": TOOL_VERSION,
        "__ALGORITHM_LABEL__": html.escape('/'.join(versions)),
        "__GEOMETRY__": (ASSETS / "review_geometry.js").read_text(encoding="utf-8") + "\n" + (ASSETS / "thickness_extension.js").read_text(encoding="utf-8"),
        "__THICKNESS_UI__": (ASSETS / "thickness_ui.js").read_text(encoding="utf-8"),
        "__BATCH_REVIEW_UI__": (ASSETS / "batch_review_ui.js").read_text(encoding="utf-8"),
        "__SERIES_UI__": (ASSETS / "series_ui.js").read_text(encoding="utf-8"),
        "__PERSISTENCE_UI__": Path(__file__).with_name("persistence.js").read_text(encoding="utf-8"),
        "__BACK_LINK__": back_link,
        "__MANIFEST__": json.dumps(manifest, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"),
        "__TOOL_CONTEXT__": "const toolContext=" + json.dumps(context).replace("<", "\\u003c") + ";",
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def write_review(path, manifest):
    Path(path).write_text(render_review(manifest), encoding="utf-8")


def write_combined_review(path, manifest_paths):
    import os
    path = Path(path)
    manifests = []
    for source in manifest_paths:
        source = Path(source)
        m = json.loads(source.read_text(encoding="utf-8"))
        for f in m["frames"]:
            for key in ASSET_FIELDS:
                if key not in f: continue
                f[key] = Path(os.path.relpath(source.parent / f[key], path.parent)).as_posix()
        manifests.append(m)
    path.write_text(render_review({"sequences": manifests}, back_link="index.html"), encoding="utf-8")
