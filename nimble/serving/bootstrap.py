"""Prepare the merged Nimble checkpoint in-container, then hand over to the server.

Runs in the API venv (/opt/nimble-api: huggingface_hub, no torch). The merge
itself is delegated to the SGlang venv (/opt/sglang: torch + transformers +
peft) via the repo's own merge tool — nimble.scoring.merge_local_adapter,
unchanged from upstream, which already verifies the prompt-code contract and
writes READY.json.

Per-replica cold path (overlay fs is not shared across replicas):
  1. skip everything if NIMBLE_MERGED_DIR already has a valid READY.json
  2. ModelScope snapshot of the Qwen base (~18 GB bf16, no token)
  3. Hugging Face snapshot of the LoRA adapter (165 MB; via the platform's
     injected HF proxy, Xet disabled — plain HTTP falls back through the proxy)
  4. merge in the SGlang venv -> merged bf16 weights + schema_config + tokenizer
  5. delete the download caches (reclaims ~18 GB overlay), then exec the
     server with NIMBLE_MODEL_PATH pointing at the merged directory.

Any failure exits non-zero so the pod restarts and retries; the Sha256-keyed
MS cache and the .part convention make every retry resume rather than restart.
"""

import os
import subprocess
import sys
from pathlib import Path

from .ms_download import snapshot_download as ms_snapshot_download

SGLANG_PYTHON = "/opt/sglang/bin/python"

# The nimble source lives only in the API venv's site-packages (Dockerfile
# COPY); the merge subprocess runs under the SGlang venv, which needs it on
# sys.path. Derived from this module's own location — no hardcoded venv paths.
_SITE_PACKAGES = str(Path(__file__).resolve().parents[2])


def _hf_snapshot(repo):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo))


def main():
    merged = Path(os.environ.get("NIMBLE_MERGED_DIR", "/workspace/nimble-merged"))
    base_repo = os.environ.get("NIMBLE_BASE_MS_REPO", "Qwen/Qwen3.5-9B")
    adapter_repo = os.environ.get("NIMBLE_ADAPTER_REPO", "bespokelabs/Bespoke-Nimble-9B")

    if (merged / "READY.json").is_file() and (merged / "schema_config.json").is_file():
        print(f"bootstrap: {merged} already merged; skipping download+merge", flush=True)
    else:
        if merged.exists():
            import shutil
            shutil.rmtree(merged)
        print("bootstrap: [1/3] downloading base from ModelScope", flush=True)
        base = Path(ms_snapshot_download(base_repo))
        print("bootstrap: [2/3] downloading adapter from Hugging Face", flush=True)
        adapter = _hf_snapshot(adapter_repo)
        print("bootstrap: [3/3] merging adapter into base (SGlang venv, CPU, bf16)", flush=True)
        result = subprocess.run(
            [SGLANG_PYTHON, "-m", "nimble.scoring.merge_local_adapter",
             "--adapter", str(adapter), "--base", str(base), "--output", str(merged)],
            env={**os.environ, "PYTHONPATH": _SITE_PACKAGES})
        if result.returncode != 0:
            print(f"bootstrap: merge failed with exit code {result.returncode}; pod will restart and retry (MS cache resumes)", flush=True)
            sys.exit(result.returncode)
        if not (merged / "READY.json").is_file():
            print("bootstrap: merge reported success but READY.json is missing", flush=True)
            sys.exit(1)

    # The merged tree carries config/tokenizer/schema_config; the 18 GB base and
    # the adapter are dead weight now — reclaim the overlay space before the
    # SGLang heap claims the node's RAM.
    ms_root = Path(os.environ.get("NIMBLE_MS_CACHE", "/workspace/ms-cache"))
    if ms_root.exists():
        import shutil
        shutil.rmtree(ms_root, ignore_errors=True)
    import shutil
    shutil.rmtree(Path.home() / ".cache/huggingface", ignore_errors=True)

    print(f"bootstrap: handing over to nimble.serving.server (NIMBLE_MODEL_PATH={merged})", flush=True)
    os.environ["NIMBLE_MODEL_PATH"] = str(merged)
    os.execve(sys.executable, [sys.executable, "-m", "nimble.serving.server"], os.environ)


if __name__ == "__main__":
    main()
