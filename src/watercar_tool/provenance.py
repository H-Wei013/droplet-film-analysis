import hashlib
from pathlib import Path


def digest():
    source = Path(__file__).resolve().parents[1]
    paths = [source / "watercar_tool" / "engine.py"]
    paths += [source / "droplet_analysis" / name for name in
              ["core.py", "v5.py", "optical.py", "temporal.py", "silhouette.py", "render.py", "impact.py", "config.json"]]
    files = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    h = hashlib.sha256()
    for name, value in sorted(files.items()):
        h.update(name.encode()); h.update(value.encode())
    return h.hexdigest(), files
