"""Opt-in image channel for nimble serving (NIMBLE_IMAGES=1). Channel open, untrained readout.

Bespoke-Nimble-9B is trained text-only: the LoRA has never seen an image, and the
serving stack goes further - the launch command passes
`--json-model-override-args '{"language_model_only": true}'`, which drops the
vision tower the Qwen3.5-9B base checkpoint carries. This module re-opens the
channel WITHOUT touching any contract-hashed file:

  * `nimble/scoring/parallel_schema.py` is byte-frozen by the published training
    contract (adapter `schema_config.json` pins `prompt_code_sha256`; both
    merge_local_adapter and server.main assert it). The text prompt pipeline is
    therefore called unmodified; images are spliced at the TOKEN level after it
    returns, as a `[vision_start] [image_pad] [vision_end]` segment right after
    the context tokens. SGLang's own multimodal processor expands the single
    image_pad placeholder to the grid size and feeds the tower - the engine, not
    this repo, owns the vision math.
  * The base checkpoint keeps its `model.visual.*` weights through the merge
    (merge_local_adapter copies the full tree), and the merged tree's config
    stays multimodal, so dropping `language_model_only` is the only launch
    change needed for the tower to load.

With NIMBLE_IMAGES unset, none of this module is imported and every request
takes the stock EvaluationService path - prompts, token budgets, logprob reads
are byte-identical.

Gate-off refusal: a state that still carries `state.images` decodes it into an
image request, which fails at the 422 admission check below only when the gate
is off... rather: this module ALWAYS refuses when loaded while the gate is off
it is simply never imported, and `state.images` then renders into the context
JSON as ordinary data (same "non-list images still renders" semantics as the
kev/openjev image layers - a list of strings is the claim; anything else is
data).
"""
import base64
import io
import re
import urllib.request

IMAGE_MIMES = ("image/jpeg", "image/png", "image/webp")
IMAGE_MAX_BYTES = 5 * 1024 * 1024    # decoded, per image
IMAGE_FETCH_LIMIT = 8 * 1024 * 1024  # https fetch cap
MAX_IMAGES = 4

_DATA_URL = re.compile(r"^data:([\w./+-]+);base64,(.*)$", re.S)


def decode_image(ref):
    """data URL | https URL -> PIL.Image (in-memory). Raises ValueError on any refusal.

    Refusal surface is shared verbatim with the kev (④) and Open-Jev (⑥) image
    layers: jpeg/png/webp, 5 MiB decoded per image, 8 MiB https fetch cap.
    The server side never decodes the pixels though - SGLang does; here the
    decode only VALIDATES the ref and re-serializes it to the form the engine
    takes ({"url": ...} | {"bytes": b64}).
    """
    from PIL import Image

    m = _DATA_URL.match(ref.strip())
    if m:
        mime, b64 = m.group(1), m.group(2)
        if mime not in IMAGE_MIMES:
            raise ValueError(f"unsupported image mime {mime} (jpeg/png/webp)")
        raw = base64.b64decode(b64, validate=True)
        if len(raw) > IMAGE_MAX_BYTES:
            raise ValueError("image exceeds 5 MiB decoded")
        im = Image.open(io.BytesIO(raw))   # format/magic sanity
        im.load()
        return {"bytes": b64}
    if ref.startswith("https://"):
        req = urllib.request.Request(ref, headers={"user-agent": "nimble-images/1"})
        with urllib.request.urlopen(req, timeout=10) as r:  # type: ignore[attr-defined]
            raw = r.read(IMAGE_FETCH_LIMIT + 1)
        if len(raw) > IMAGE_FETCH_LIMIT:
            raise ValueError("image fetch exceeded 8 MiB cap")
        if len(raw) > IMAGE_MAX_BYTES:
            raise ValueError("image exceeds 5 MiB decoded")
        im = Image.open(io.BytesIO(raw))
        im.load()
        # validated; hand the engine the URL itself so IT can re-fetch through
        # its own data loader (bytes would also work: {"bytes": <b64>})
        return {"url": ref}
    raise ValueError("images must be data URLs or https URLs")


def extract_images(state):
    """Split a dict state's `images` list off before rendering; (rest, images|None).

    Same semantics as the kev/openjev layers: only a non-empty LIST of strings
    is an image claim; dict states with any other `images` value, and non-dict
    states, pass through untouched (the value then renders as ordinary data,
    matching the stock JSON serialization).
    """
    if isinstance(state, dict) and isinstance(state.get("images"), list):
        images, rest = state["images"], {k: v for k, v in state.items() if k != "images"}
        if not images:
            return state, None
        if len(images) > MAX_IMAGES:
            raise ValueError(f"at most {MAX_IMAGES} images per request")
        for ref in images:
            if not isinstance(ref, str):
                raise ValueError("state.images entries must be strings (data/https URLs)")
        return rest, images
    return state, None
