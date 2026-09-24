"""Image-aware evaluation service for the nimble image channel (NIMBLE_IMAGES=1).

Injectable subclass seam: openjev's /v1/systemone route calls
app.state.service.evaluate(payload); server.main builds that service. This
module replaces it with a subclass that

  * delegates every request WITHOUT a claimed state.images to the stock
    EvaluationService path (byte-identical text behavior), and
  * routes image claims through an image-aware flow: strip images from the
    state, prepare the (unchanged, contract-hashed) text prompts, splice the
    vision segment into the token stream, and POST each branch to SGLang's
    /generate with `image_data` so the ENGINE runs the tower.

The token-budget admission checks are replicated from
EvaluationService._evaluate (they guard the same limits; copied rather than
inherited because the parent builds them into the text-only flow).
"""
import asyncio
import time

from openjev.backend import SGLangClient

from .image_splice import segment_for, locate_after_context, splice, vision_token_ids
from .images import decode_image, extract_images


class ImageSGLangClient(SGLangClient):
    """SGLangClient whose /generate payload carries image_data.

    Verbatim copy of upstream generate() with exactly one payload addition;
    the logprob read (token_ids_logprob) and abort handling stay identical.
    """

    async def generate_with_images(self, input_ids, label_ids, image_data):
        import httpx
        import orjson
        from uuid import uuid4

        rid = f"openjev-{uuid4().hex}"
        payload = {
            "rid": rid,
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "ignore_eos": True,
            },
            "stream": False,
            # Same mixed-batch crash workaround as the parent class.
            "return_logprob": True,
            "token_ids_logprob": label_ids or [0],
            "logprob_start_len": -1,
            "top_logprobs_num": 0,
            "return_text_in_logprobs": False,
            "image_data": image_data,
        }
        from .backend import BackendError  # openjev's BackendError
        async with self.slots:
            try:
                response = await self.client.post(
                    "/generate",
                    content=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"},
                )
            except asyncio.CancelledError:
                await self._abort(rid)
                raise
            except httpx.TimeoutException as exc:
                await self._abort(rid)
                raise BackendError("SGLang request timed out", 504) from exc
            except httpx.HTTPError as exc:
                raise BackendError("Cannot reach SGLang", 503) from exc
        if not response.is_success:
            status = response.status_code
            if status not in {429, 503, 529}:
                status = 502
            raise BackendError(f"SGLang returned HTTP {response.status_code}", status)
        from openjev.backend import parse_generation
        try:
            return parse_generation(response.json(), label_ids or [])
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise BackendError(f"Invalid SGLang logprob response: {exc}") from exc


class NimbleImageService:
    """Drop-in service for app.state.service under NIMBLE_IMAGES=1.

    Wraps (not subclasses) the stock EvaluationService: requests without an
    image claim delegate to it untouched; image claims run the image flow with
    the same tokenizer/compiler instances and the same settings.
    """

    def __init__(self, inner, tokenizer, image_client, *, max_prompt_tokens):
        self.inner = inner                    # EvaluationService (text path, unchanged)
        self.tokenizer = tokenizer
        self.image_client = image_client      # ImageSGLangClient
        self.max_prompt_tokens = max_prompt_tokens
        self.active = 0
        # reuse the inner service's admission knobs
        self.settings = inner.settings
        self.compiler = inner.compiler
        self.backend = inner.backend

    # -- API surface used by openjev.api ------------------------------------

    @property
    def backend_health(self):
        return self.inner.backend.health

    async def health(self):
        return await self.inner.backend.health()

    async def evaluate(self, request):
        state, images = extract_images(request.state)
        if images is None:
            return await self.inner.evaluate(request)
        # capacity gate, same shape as the parent's
        if self.active >= self.settings.max_concurrent_requests:
            from openjev.service import RequestError
            raise RequestError("Server is at capacity; retry shortly", 529)
        self.active += 1
        try:
            async with asyncio.timeout(self.settings.request_timeout):
                return await self._evaluate_images(request, state, images)
        except TimeoutError as exc:
            from openjev.backend import BackendError
            raise BackendError("Evaluation deadline exceeded", 504) from exc
        finally:
            self.active -= 1

    # -- image flow ----------------------------------------------------------

    async def _evaluate_images(self, request, text_state, image_refs):
        from openjev.prompts import Branch
        from openjev.scoring import answer
        from openjev.service import RequestError
        from .compiler import serialize

        started = time.perf_counter()
        # validate/normalize refs -> engine payloads; decode_image raises
        # ValueError on any refusal, which the API maps to a 422 like the
        # text path's RequestError
        engine_images = [decode_image(ref) for ref in image_refs]
        # rebuild the request with the stripped state (pydantic copy keeps
        # model validation; the only change is the state field)
        stripped = request.model_copy(update={"state": text_state})

        # text preparation first: the contract-hashed compiler, unmodified
        try:
            prepared = await asyncio.to_thread(self.compiler.prepare, stripped)
        except (ValueError, TypeError) as exc:
            raise RequestError(str(exc)) from exc

        # token budget admission (mirrors EvaluationService._evaluate)
        limit = self.settings.max_input_tokens
        # the vision segment adds 3 tokens per image MAX at the API layer; the
        # engine expands further (grid tokens), which the engine enforces
        # against ITS context length - admission here stays on the text form
        # so the text limits keep their published meaning.
        if any(len(branch.input_ids) + 1 > limit for branch in prepared.branches):
            raise RequestError(f"A question branch exceeds the {limit}-token context limit")
        total_input = len(prepared.prefix_ids) + sum(
            len(branch.input_ids) for branch in prepared.branches
        )
        if total_input > self.settings.max_total_input_tokens:
            raise RequestError("Request exceeds the total input-token budget across branches")

        # splice: locate the context inside the rendered prefix
        context = serialize(text_state)
        token_ids = vision_token_ids(self.tokenizer)
        at = locate_after_context(self.tokenizer, prepared.prefix_ids, context)
        segment = segment_for(len(engine_images), token_ids)

        def do_splice():
            new_prefix, new_branches = splice(
                prepared.prefix_ids,
                [b.input_ids for b in prepared.branches],
                at, segment)
            return new_prefix, [
                Branch(b.question_id, b.question, ids, b.label_ids, b.option_keys)
                for b, ids in zip(prepared.branches, new_branches, strict=True)
            ]

        new_prefix, image_branches = await asyncio.to_thread(do_splice)

        prepared_at = time.perf_counter()
        # warmup on the shared prefix carries the images once; branches reuse
        # the engine's prefix cache (the segment sits inside the shared prefix,
        # so every branch's cache key includes it).
        warmup = await self.image_client.generate_with_images(
            new_prefix, None, engine_images)
        warmed_at = time.perf_counter()
        tasks = [
            asyncio.create_task(self.image_client.generate_with_images(
                b.input_ids, b.label_ids, engine_images))
            for b in image_branches
        ]
        try:
            results = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        answers = {
            branch.question_id: answer(branch, result.logprobs, self.settings.temperature)
            for branch, result in zip(image_branches, results, strict=True)
        }
        from openjev.service import Evaluation
        from openjev.models import SystemOneResponse, Usage
        return Evaluation(
            response=SystemOneResponse(
                model=request.model,
                answers=answers,
                usage=Usage(
                    input_tokens=warmup.input_tokens + sum(r.input_tokens for r in results),
                    output_tokens=warmup.output_tokens + sum(r.output_tokens for r in results),
                ),
            ),
            prefix_tokens=len(new_prefix),
            cached_tokens=(
                sum(r.cached_tokens for r in results)
                if all(r.cached_tokens is not None for r in results)
                else None
            ),
            prepare_ms=(prepared_at - started) * 1000,
            prefill_ms=(warmed_at - prepared_at) * 1000,
            branches_ms=(time.perf_counter() - warmed_at) * 1000,
        )
