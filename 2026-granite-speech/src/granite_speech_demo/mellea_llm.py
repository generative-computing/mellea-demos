"""Custom pipecat LLMService using IVR Best-of-N generation with Granite Switch."""

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from mellea.backends.model_options import ModelOption
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.components.chat import Message as MelleaMessage
from mellea.stdlib.components.docs import Document
from mellea.stdlib.components.intrinsic import core
from mellea.stdlib.context import ChatContext
import mellea.stdlib.functional as mfuncs

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    InputTransportMessageFrame,
)
from pipecat.processors.frameworks.rtvi import RTVIClientMessageFrame, RTVIServerMessageFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService

from loguru import logger

LLM_URL = os.environ.get("LLM_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "granite4.1:3b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")

GRANITE_SWITCH_ENABLED = os.environ.get(
    "GRANITE_SWITCH_ENABLED", "false").lower() in ("1", "true", "yes")

DEFAULT_REQUIREMENT_THRESHOLD = 0.5


@dataclass(frozen=True)
class RequirementSpec:
    label: str
    description: str
    instruction: str
    threshold: float = DEFAULT_REQUIREMENT_THRESHOLD


IVR_REQUIREMENT_SPECS = [
    RequirementSpec(
        label="No markdown",
        description="The response contains no bullet points, no numbered lists, no headers, and no markdown formatting.",
        instruction="No bullet points. No numbered lists. No headers. No markdown formatting.",
    ),
]

IVR_REQUIREMENTS = [spec.description for spec in IVR_REQUIREMENT_SPECS]
IVR_REQUIREMENT_LABELS = [spec.label for spec in IVR_REQUIREMENT_SPECS]
IVR_REQUIREMENT_THRESHOLDS = [spec.threshold for spec in IVR_REQUIREMENT_SPECS]
IVR_REQUIREMENT_INSTRUCTIONS = [
    spec.instruction for spec in IVR_REQUIREMENT_SPECS]

BEST_OF_N = 3

_BASE_SYSTEM_INSTRUCTION = (
    "You are Granite, IBM's real-time interactive speech assistant, running live."
)

_REQUIREMENTS_BLOCK = "\n\n" + " ".join(IVR_REQUIREMENT_INSTRUCTIONS)

_DEFAULT_SYSTEM_INSTRUCTION = _BASE_SYSTEM_INSTRUCTION
_DEFAULT_SYSTEM_INSTRUCTION_WITH_REQS = _BASE_SYSTEM_INSTRUCTION + _REQUIREMENTS_BLOCK

_DEFAULT_INSTRUCT_TEMPLATE = (
    "Answer the following question as a voice assistant. "
    "Speak directly to the user using 'you'. "
    + " ".join(IVR_REQUIREMENT_INSTRUCTIONS)
    + "\n\n{question}"
)


def _load_documents() -> list[Document]:
    docs_dir = os.environ.get("DOCUMENTS_DIR", "")
    if not docs_dir:
        return []
    path = Path(docs_dir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.is_dir():
        logger.warning(
            "DOCUMENTS_DIR={!r} is not a directory, skipping", docs_dir)
        return []
    docs = []
    for i, txt_file in enumerate(sorted(path.glob("*.txt"))):
        text = txt_file.read_text().strip()
        if text:
            docs.append(
                Document(text=text, title=txt_file.stem, doc_id=str(i)))
            logger.info("Loaded document {}: {} ({} chars)",
                        i, txt_file.name, len(text))
    return docs


_documents = _load_documents()

_documents_block = ""
if _documents:
    doc_lines = "\n".join(
        json.dumps({"text": doc.text, "title": doc.title, "doc_id": doc.doc_id})
        for doc in _documents
    )
    _documents_block = (
        "You are a helpful assistant with access to the following documents. "
        "You may use one or more documents to assist with the user query.\n\n"
        "You are given a list of documents within <documents></documents> XML tags:\n"
        f"<documents>\n{doc_lines}\n</documents>\n\n"
        "Write the response to the user's input by strictly aligning with the facts "
        "in the provided documents. If the information needed to answer the question "
        "is not available in the documents, inform the user that the question cannot "
        "be answered based on the available data.\n\n"
    )

SYSTEM_INSTRUCTION = _documents_block + _DEFAULT_SYSTEM_INSTRUCTION
SYSTEM_INSTRUCTION_WITH_REQS = _documents_block + \
    _DEFAULT_SYSTEM_INSTRUCTION_WITH_REQS

INSTRUCT_TEMPLATE = _DEFAULT_INSTRUCT_TEMPLATE


def _check_one_requirement(gen_ctx, backend, req_desc, req_index, gen_index, t0, emit,
                           threshold=DEFAULT_REQUIREMENT_THRESHOLD):
    """Run a single requirement check. Executed in a thread."""
    check_started = time.monotonic()
    score = core.requirement_check(gen_ctx, backend, req_desc)
    passed = score > threshold
    if emit is not None:
        emit({
            "phase": "check",
            "gen": gen_index,
            "req_index": req_index,
            "passed": passed,
            "score": float(score),
            "threshold": float(threshold),
            "ms": int((time.monotonic() - check_started) * 1000),
            "t_ms": int((time.monotonic() - t0) * 1000),
        })
    return {"description": req_desc, "passed": passed, "score": score, "threshold": threshold}


def _single_generation(action, ctx, backend, model_options, requirements, validate,
                       gen_index=0, t0=None, emit=None):
    """Run one generation + parallel validation pass. Executed in a thread."""
    if emit is not None:
        emit({
            "phase": "sample_start",
            "gen": gen_index,
            "t_ms": int((time.monotonic() - (t0 or time.monotonic())) * 1000),
        })
    output, gen_ctx = mfuncs.act(
        action, ctx, backend, strategy=None, model_options=model_options)
    answer = str(output)
    if emit is not None:
        emit({
            "phase": "sample_text",
            "gen": gen_index,
            "text": answer,
            "t_ms": int((time.monotonic() - (t0 or time.monotonic())) * 1000),
        })
    if not validate:
        return {"answer": answer, "requirements": [], "passed": True}
    with ThreadPoolExecutor(max_workers=len(requirements)) as req_executor:
        futures = []
        for i, req in enumerate(requirements):
            if isinstance(req, RequirementSpec):
                req_desc = req.description
                threshold = req.threshold
            else:
                req_desc = req
                threshold = DEFAULT_REQUIREMENT_THRESHOLD
            futures.append(req_executor.submit(
                _check_one_requirement, gen_ctx, backend, req_desc, i, gen_index, t0, emit,
                threshold,
            ))
        req_results = [f.result() for f in futures]
    all_passed = all(r["passed"] for r in req_results)

    # Log the result of this specific generation attempt
    logger.info(f"Generation attempt: passed={all_passed}, answer={answer!r}")
    for res in req_results:
        status = "PASSED" if res["passed"] else "FAILED"
        logger.info(
            f"  - {status}: {res['description']} (score: {res['score']:.2f})")

    return {"answer": answer, "requirements": req_results, "passed": all_passed}


def _extract_text(content):
    """Flatten pipecat message content (str or list-of-parts) to a string."""
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text", "")
        return ""
    return str(content) if content else ""


class MelleaLLMService(LLMService):
    """LLM service.

    Default (IVR validation off): stream tokens straight through to TTS so the
    first sentence can be synthesised while the rest of the answer is still being
    generated.

    IVR validation on: Best-of-N parallel generation with Granite Switch
    requirement-check ALoRA validation. This path is intentionally non-streaming
    because intrinsic validation needs the full answer before scoring.
    """

    def __init__(self, ivr_validation: bool | str | None = None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(ivr_validation, str):
            requested = ivr_validation.lower() in ("1", "true", "yes")
        else:
            requested = bool(ivr_validation)

        if requested and not GRANITE_SWITCH_ENABLED:
            logger.warning(
                "IVR validation requested but GRANITE_SWITCH_ENABLED=false; "
                "running without requirement-check validation."
            )
        self._ivr_validation = requested and GRANITE_SWITCH_ENABLED

        self._backend = OpenAIBackend(
            model_id=LLM_MODEL,
            base_url=LLM_URL,
            api_key=LLM_API_KEY,
        )
        if self._ivr_validation:
            self._backend.register_embedded_adapter_model(LLM_MODEL)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            await self._process_context(frame.context)
        elif isinstance(frame, RTVIClientMessageFrame):
            msg_type = frame.type
            data = frame.data or {}
            if msg_type == "set_ivr_validation":
                self._set_ivr_validation(data.get("enabled", False))
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputTransportMessageFrame):
            msg = frame.message
            if isinstance(msg, dict):
                data = msg.get("data", {})
                if isinstance(data, dict) and data.get("t") == "set_ivr_validation":
                    self._set_ivr_validation(
                        (data.get("d") or {}).get("enabled", False))
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    def _set_ivr_validation(self, enabled: bool):
        if enabled and not GRANITE_SWITCH_ENABLED:
            logger.warning(
                "Ignoring set_ivr_validation=true: GRANITE_SWITCH_ENABLED=false."
            )
            self._ivr_validation = False
            return
        logger.info("Dynamically setting IVR validation to {}", enabled)
        self._ivr_validation = enabled
        if self._ivr_validation:
            try:
                self._backend.register_embedded_adapter_model(LLM_MODEL)
            except Exception as e:
                logger.warning("Failed to register adapter dynamically: {}", e)

    async def _push_llm_text(self, text: str):
        if text:
            await self.push_frame(LLMTextFrame(text))

    async def _process_context(self, context: LLMContext):
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()

            user_text = ""
            last_user_idx = -1
            for i in range(len(context.messages) - 1, -1, -1):
                msg = context.messages[i]
                if msg.get("role") == "user":
                    user_text = _extract_text(msg.get("content", ""))
                    last_user_idx = i
                    break

            if not user_text:
                return

            t0 = time.monotonic()
            logger.info("MelleaLLM input: {!r}", user_text)

            # Build history without the current user turn — it is carried by `action`.
            ctx = ChatContext()
            for i, msg in enumerate(context.messages):
                if i == last_user_idx:
                    continue
                role = msg.get("role")
                content = _extract_text(msg.get("content", ""))
                if role in ("user", "assistant") and content:
                    ctx = ctx.add(MelleaMessage(role, content))

            if self._ivr_validation:
                await self._run_best_of_n(user_text, ctx, t0)
            else:
                await self._run_streaming(user_text, ctx, t0)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("MelleaLLMService error")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

    async def _run_streaming(self, user_text: str, ctx: ChatContext, t0: float):
        action = MelleaMessage("user", user_text)
        model_options = {
            ModelOption.TEMPERATURE: 0.7,
            ModelOption.SYSTEM_PROMPT: SYSTEM_INSTRUCTION,
            ModelOption.STREAM: True,
        }

        output, _ = await self._backend.generate_from_chat_context(
            action, ctx, model_options=model_options
        )

        first_chunk_at: float | None = None
        total = ""
        while not output._computed:
            delta = await output.astream()
            if not delta:
                continue
            if first_chunk_at is None:
                first_chunk_at = time.monotonic()
            total += delta
            await self._push_llm_text(delta)

        elapsed = time.monotonic() - t0
        ttfb = (first_chunk_at - t0) if first_chunk_at is not None else elapsed
        logger.info(
            "MelleaLLM streamed in {:.3f}s (first chunk {:.3f}s): {!r}",
            elapsed, ttfb, total[:80],
        )

    async def _run_best_of_n(self, user_text: str, ctx: ChatContext, t0: float):
        action = MelleaMessage(
            "user", INSTRUCT_TEMPLATE.format(question=user_text))
        model_options = {
            ModelOption.TEMPERATURE: 0.7,
            ModelOption.SYSTEM_PROMPT: SYSTEM_INSTRUCTION_WITH_REQS,
        }

        loop = asyncio.get_event_loop()
        turn_id = f"t{int(t0 * 1000)}"

        async def _send(payload: dict):
            await self.push_frame(RTVIServerMessageFrame(data={
                "type": "ivr_validation",
                "turn_id": turn_id,
                **payload,
            }))

        def emit(payload: dict):
            # Called from worker threads; schedule frame push onto the loop.
            asyncio.run_coroutine_threadsafe(_send(payload), loop)

        await _send({
            "phase": "start",
            "n_samples": BEST_OF_N,
            "requirements": IVR_REQUIREMENT_LABELS,
            "t_ms": 0,
        })

        with ThreadPoolExecutor(max_workers=BEST_OF_N) as executor:
            futures = [
                loop.run_in_executor(
                    executor,
                    _single_generation,
                    action, ctx, self._backend, model_options, IVR_REQUIREMENT_SPECS, True,
                    i, t0, emit,
                )
                for i in range(BEST_OF_N)
            ]
            all_attempts = await asyncio.gather(*futures)

        elapsed = time.monotonic() - t0
        passing_indices = [i for i, a in enumerate(
            all_attempts) if a["passed"]]
        if passing_indices:
            chosen_index = passing_indices[0]
            chosen = all_attempts[chosen_index]
        else:
            chosen_index = -1
            chosen = {
                "answer": (
                    "That's a bit outside what I can get into right now."
                ),
                "passed": False,
            }

        n_passed = sum(1 for a in all_attempts if a["passed"])
        logger.info(
            "MelleaLLM IVR done in {:.3f}s — {}/{} passed, chosen: {!r}",
            elapsed, n_passed, len(all_attempts), chosen["answer"][:80],
        )
        for i, a in enumerate(all_attempts):
            reqs_ok = sum(1 for r in a["requirements"] if r["passed"])
            marker = " <- chosen" if i == chosen_index else ""
            logger.debug(
                "  gen {}: {}/{} reqs, passed={}{}",
                i + 1, reqs_ok, len(IVR_REQUIREMENTS), a["passed"], marker,
            )

        await _send({
            "phase": "done",
            "chosen_gen": chosen_index,
            "elapsed_ms": int(elapsed * 1000),
            "n_passed": n_passed,
            "n_samples": len(all_attempts),
        })

        await self._push_llm_text(chosen["answer"])
