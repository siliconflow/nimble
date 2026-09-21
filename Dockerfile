# nimble serving image for SF GPU Functions (4090 g1, Bespoke-Nimble-9B).
#
# Layout follows openjev-sglang's own deployment convention (modal_app.py):
# the SGLang image keeps its *tested* CUDA environment as the base, and a
# separate small uv venv on top owns the API side (openjev + nimble source).
# nimble.serving.server then starts the SGLang backend with the base image's
# python (/opt/sglang/bin/python) as a subprocess — exactly as upstream.
#
# Base: platform-mirrored official sglang image. NOT the vllm-openai image:
# v0.5.20 pairs with the sglang pins nimble is tested against (openjev
# default is v0.5.19-cu130; 0.5.20 is the platform-synced evolution).
FROM hub.6scloud.com/siliconflow/lmsysorg/sglang:v0.5.20

# Install uv for the API-side venv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# API side, into its own venv (does NOT touch the SGLang venv under /opt/sglang).
# openjev-sglang is pinned to the tarball revision nimble's requirements use.
RUN uv venv /opt/nimble-api \
    && VIRTUAL_ENV=/opt/nimble-api uv pip install \
      "openjev-sglang @ https://github.com/ekzhang/openjev-sglang/archive/7f84bedc169439f03379c2fa8d00ada220af2295.tar.gz"

# Merge side: the SGlang venv already ships torch + transformers; add peft/
# accelerate so bootstrap.py can run merge_local_adapter (upstream's own merge
# tool, contract-checked) in that venv. Pinned to the versions the adapter was
# trained with (schema_config.json `versions`). Not installed into the API
# venv — /opt/nimble-api stays torch-free.
RUN uv pip install --python /opt/sglang/bin/python \
      peft==0.21.0 accelerate==1.15.0

# nimble source. NOTE: the repo is NOT a packaged project (no pyproject.toml /
# setup.py — requirements/ layout, like upstream's modal deployment which
# sys.path-mounts the source). So it cannot be `pip install .`-ed; drop the
# package straight into the API venv's site-packages instead.
COPY nimble /opt/nimble-api/lib/python3.12/site-packages/nimble

# Non-root runtime (images in this family run as root by default).
# /workspace is auto-created by WORKDIR as root:root 755 — MUST be chowned
# to appuser, or the very first write in bootstrap.py (os.makedirs under
# NIMBLE_MS_CACHE/NIMBLE_MERGED_DIR, both under /workspace) dies with
# PermissionError and the pod crashloops before the log collector attaches
# ("unable to retrieve container logs", no other output).
RUN useradd --create-home --shell /bin/bash appuser || true \
    && chown appuser:appuser /workspace
USER appuser

# NOTE: no HF_ENDPOINT here — the SF GPU Function runtime auto-injects HF
# proxy env (API downloads stay on the internal network) + blobcache. A baked
# mirror URL could bypass that injection. Keep only the Xet disable: the Xet
# CAS backend (*.hf.sc4.ai) is a separate endpoint the proxy may not cover,
# and Xet disabled falls back to plain HTTP via the injected proxy.
ENV PATH="/opt/nimble-api/bin:${PATH}" \
    NIMBLE_MAX_PROMPT_TOKENS=2048 \
    NIMBLE_MERGED_DIR=/workspace/nimble-merged \
    NIMBLE_MS_CACHE=/workspace/ms-cache \
    HF_HUB_DISABLE_XET=1

EXPOSE 8000
# The SF cloud-function yaml overrides `command` (see deploy/nimble-9b-4090.yaml
# in os_jev_exp). bootstrap.py downloads base (ModelScope) + adapter (HF via the
# injected proxy), merges in the SGlang venv, then execs nimble.serving.server
# (which itself hardcodes the backend python at /opt/sglang/bin/python).
CMD ["python", "-m", "nimble.serving.bootstrap"]
