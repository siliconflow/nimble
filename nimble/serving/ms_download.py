"""stdlib-only ModelScope snapshot downloader for the serving bootstrap.

Ported from kev/kev/evaluate.py:_ms_snapshot (same workspace, proven on the SF
GPU Function runtime): stdlib only — the API venv has no torch here and the
SGlang venv stays untouched until the merge step. The ModelScope mirror
carries the Qwen base at full size (~18 GB bf16) with no token; the HF Hub
carries only the 165 MB LoRA adapter. base_revision pins an HF commit and is
meaningless on the MS mirror (master) — see deploy/nimble-9b-4090.yaml.
"""

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request

MS_API = "https://modelscope.cn/api/v1/models"


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_download(repo, cache_root=None):
    """Download a ModelScope repo (all files) to <cache>/<namespace>/<name>.
    Files are cached by their listed Sha256 — a completed tree is never
    re-downloaded, so a restarted pod resumes instead of starting over."""
    root = os.environ.get("NIMBLE_MS_CACHE") or cache_root or "/workspace/ms-cache"
    dest = os.path.join(root, *repo.split("/"))
    os.makedirs(dest, exist_ok=True)
    url = f"{MS_API}/{repo}/repo/files?Revision=master&Recursive=true"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                files = json.load(r)["Data"]["Files"]
            break
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"modelscope: file listing failed (attempt {attempt + 1}/4): {e!r}; retrying in {wait}s", flush=True)
            if attempt == 3:
                raise
            time.sleep(wait)
    files = [f for f in files if f.get("Type") != "tree"]
    total = sum(f.get("Size", 0) for f in files)
    print(f"modelscope: {repo} -> {dest} ({len(files)} files, {total / 1e9:.2f} GB); downloading (progress below)...", flush=True)
    n = done_bytes = 0
    for f in files:
        path, sha, size = f["Path"], f.get("Sha256"), f.get("Size", 0)
        out = os.path.join(dest, path)
        os.makedirs(os.path.dirname(out) or dest, exist_ok=True)
        if os.path.exists(out) and (not sha or _file_sha256(out) == sha):
            continue
        # 8MB chunks with per-file + running-total progress; a multi-GB shard takes
        # minutes on a slow egress and silence looks like a hang in the pod log.
        # Retry per file: transient CDN resets shouldn't kill a 20-minute download;
        # .part keeps partial data but a reset mid-stream is easiest to restart from
        # byte 0 (ModelScope's CDN is fast enough that this stays rare).
        file_url = f"{MS_API}/{repo}/repo?FilePath={urllib.parse.quote(path)}&Revision=master"
        got = 0
        for attempt in range(4):
            try:
                if os.path.exists(out + ".part"):
                    os.remove(out + ".part")
                with urllib.request.urlopen(file_url, timeout=120) as r, open(out + ".part", "wb") as w:
                    got = 0
                    while True:
                        chunk = r.read(8 << 20)
                        if not chunk:
                            break
                        w.write(chunk)
                        got += len(chunk)
                        done_bytes += len(chunk)
                        if got and size and got % (64 << 20) < (8 << 20):  # log every ~64 MB, one line each (pod log panels render \r poorly)
                            print(f"  {path}: {got / 1e6:.0f}/{size / 1e6:.0f} MB (total {done_bytes / 1e9:.2f}/{total / 1e9:.2f} GB)", flush=True)
                if size and got != size:
                    raise IOError(f"incomplete read: {got} of {size} bytes")
                os.replace(out + ".part", out)  # atomic: a .part file is never mistaken for a complete download
                print(f"  {path}: {size / 1e6:.0f} MB done (total {done_bytes / 1e9:.2f}/{total / 1e9:.2f} GB)", flush=True)
                break
            except Exception as e:
                done_bytes -= got if (size and got != size and got) else 0
                wait = 15 * (attempt + 1)
                print(f"  {path}: download failed (attempt {attempt + 1}/4): {e!r}; retrying in {wait}s", flush=True)
                if attempt == 3:
                    raise
                time.sleep(wait)
        n += 1
    print(f"modelscope: {repo} -> {dest} ({n} file(s) downloaded, {len(files) - n} cached)", flush=True)
    return dest
