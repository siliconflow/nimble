"""Token-level vision splice for the nimble image channel.

Inserts a `[vision_start] [image_pad] [vision_end]` triplet per image into
prepared token ids right after the context JSON tokens, so SGLang's multimodal
processor expands the placeholder and runs the tower (base_processor.
build_input_ids scans for vision_start followed by exactly one image_pad and
repeats the pad to the image's grid size - the engine, not this repo, owns the
vision math).

The splice point cannot be expressed in the text pipeline: safe_json escapes
`<`/`>`, so the vision literals can never ride through the contract-hashed
prompt compiler (parallel_schema.py is byte-frozen by the published training
contract). The point is instead located on the finished token stream: decode
the prefix token-by-token until the decoded text covers the end of the
serialized context embedded in it. Special-token boundaries never join BPE
merges, so decoding the spliced stream yields exactly the original text with
the segment inserted. `prepare_prompts` re-derives the common prefix from the
FULL ids, so the splice is applied to each branch's full_ids before the warmup
prefix - the engine then sees one expanded segment on every request.

Vision token ids are read from the tokenizer (Qwen3.5 keeps them family-wide,
but reading them removes the hardcoded-id assumption, matching the kev/Open-Jev
image layers).
"""


def vision_token_ids(tokenizer):
    """(vision_start, image_pad, vision_end) token ids for this tokenizer."""
    ids = []
    for token in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is None or tid < 0:
            raise ValueError(f"tokenizer has no {token} special token")
        ids.append(tid)
    return tuple(ids)


def segment_for(n_images, token_ids):
    """Flat [vs, pad, ve] * n_images segment."""
    vs, pad, ve = token_ids
    out = []
    for _ in range(n_images):
        out.extend((vs, pad, ve))
    return out


def locate_after_context(tokenizer, prefix_ids, context):
    """Token index in prefix_ids right after the embedded `context` value.

    `context` is the serialized state string NimbleCompiler hands to the
    prompt compiler. prepare_prompts embeds it as the value of {"context": ...}
    through safe_json, which JSON-escapes the string - quotes double up, so the
    literal `context` never appears in the rendered text. Search instead for
    the JSON-escaped form (safe_json is json.dumps plus < > escapes; the
    escaped form is computed with plain json.dumps, which never disagrees here
    because the vision splice never needs to find a context containing < or >
    exactly at its boundary - a < inside the context still escapes identically
    under both).

    Binary search over decoded prefix lengths: find the smallest k whose
    decoded length >= end-of-first-occurrence; the boundary k splits the token
    covering the context end, so the caller inserts AFTER the covering token
    and never cuts inside the context text.

    Returns len(prefix_ids) when the context cannot be located (caller falls
    back to appending after the prefix end - a valid, if less well-placed,
    splice point that the engine still expands).
    """
    import json

    text = tokenizer.decode(prefix_ids)
    # reproduce safe_json's escaping exactly (json.dumps + < > replaced)
    escaped = (json.dumps(context, ensure_ascii=False, allow_nan=False)
               .replace("<", "\\u003c").replace(">", "\\u003e"))
    i = text.find(escaped)
    if i < 0:
        i = text.find(context)        # unescaped fallback (string states)
        if i < 0:
            return len(prefix_ids)
        end = i + len(context)
    else:
        end = i + len(escaped)
    lo, hi = 0, len(prefix_ids)
    # smallest k whose decoded length >= end
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(prefix_ids[:mid])) >= end:
            hi = mid
        else:
            lo = mid + 1
    return min(lo, len(prefix_ids))


def splice(prefix_ids, full_ids, at, segment):
    """Insert `segment` at token index `at` into each id list.

    prefix_ids and every entry of full_ids share their first len(prefix_ids)
    tokens, so one insertion point applies to all of them (a branch whose
    prefix diverges earlier still gets a well-formed stream - the segment
    lands after its own divergence, inside the shared user-turn text).
    """
    out_prefix = list(prefix_ids)
    if at <= len(out_prefix):
        out_prefix = out_prefix[:at] + list(segment) + out_prefix[at:]
    else:
        out_prefix = out_prefix + list(segment)
    out_full = []
    for ids in full_ids:
        ids = list(ids)
        if at <= len(prefix_ids) and len(ids) >= at:
            ids = ids[:at] + list(segment) + ids[at:]
        else:
            pos = min(len(ids), len(prefix_ids))
            ids = ids[:pos] + list(segment) + ids[pos:]
        out_full.append(ids)
    return out_prefix, out_full
