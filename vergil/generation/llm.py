# generation/llm.py
"""Qwen3-4B inference wrapper with two interchangeable backends.

Generator model (research 2026-07-06): Qwen3-4B-Instruct-2507 — Apache-2.0,
ungated, natively NON-thinking instruct (no <think> blocks, no enable_thinking
flag needed), supported by transformers>=4.51. A strict upgrade over the previous
Qwen2.5-7B-Instruct: at/above Qwen3-8B and above Qwen2.5-7B on benchmarks while
using ~half the VRAM. Same standard apply_chat_template + .generate() path.

* ``backend="transformers"`` (default) — HF ``AutoModelForCausalLM`` in bf16 on CUDA.
  This is the reference route and mirrors the ``_QwenSummarizer`` used during the
  build (``modal_build.py``), so build-time summaries and eval-time answers come
  from the exact same generation path.
* ``backend="vllm"`` — vLLM AsyncLLMEngine (continuous batching, paged KV). ~10x
  the decode speed of HF ``.generate()`` — plain HF runs Qwen3-30B-A3B at ~5-10
  tok/s on an A100 while vLLM does 50-100+. Streaming is bridged to the same sync
  ``generate_stream`` generator API via a background asyncio loop, so callers
  don't change. Requires ``vllm`` in the image (serving images only).
* ``backend="llama_cpp"`` — the original Q4_K_M GGUF route via ``llama-cpp-python``.
  This is the Kaggle T4 16GB inference route (~4.5GB model file); it is NOT used on
  Modal and ``llama_cpp`` is intentionally not in the Modal image.

Heavy imports live inside ``__init__`` so importing this module stays free of
torch/transformers/llama_cpp (the package must import on machines without them).
"""


class QwenLLM:
    """Minimal LLM with the ``.generate(prompt, max_tokens, temperature) -> str``
    shape that the summarizer, RAG pipeline, and eval harness all expect."""

    def __init__(
        self,
        model_path_or_id: str = "Qwen/Qwen3-4B-Instruct-2507",
        backend: str = "transformers",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
    ):
        """
        Args:
            model_path_or_id: HF model id (transformers backend) or a local GGUF
                file path (llama_cpp backend).
            backend: "transformers" (Modal A100, default) or "llama_cpp"
                (Kaggle T4 GGUF route).
            n_ctx: context window — only used by the llama_cpp backend
                (transformers uses the model's native config).
            n_gpu_layers: llama_cpp only; -1 = all layers on GPU.

        llama_cpp model file (Kaggle route): pass a local GGUF path as
            model_path_or_id. A Qwen3-4B-Instruct-2507 Q4_K_M GGUF (~2.5GB, fits
            the Kaggle T4 comfortably) is the recommended file, e.g.:
            huggingface-cli download Qwen/Qwen3-4B-Instruct-2507-GGUF \
                Qwen3-4B-Instruct-2507-Q4_K_M.gguf --local-dir models/
            (verify the exact filename on the repo before downloading).
        """
        if backend not in ("transformers", "vllm", "llama_cpp"):
            raise ValueError(
                f"unknown backend {backend!r}; use 'transformers', 'vllm' or 'llama_cpp'")
        self.backend = backend

        if backend == "vllm":
            import asyncio
            import threading

            from transformers import AutoTokenizer
            from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

            self.tok = AutoTokenizer.from_pretrained(model_path_or_id)
            self._SamplingParams = SamplingParams
            # enforce_eager skips CUDA-graph capture (minutes of extra cold-start for a
            # MoE) — eager decode of the 3B-active experts is still ~10x HF .generate().
            # gpu_memory_utilization leaves headroom for the co-resident encoders
            # (bi-encoder / bge-small / ColBERT live on the same GPU in SPARDA).
            self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
                model=model_path_or_id, dtype="bfloat16",
                gpu_memory_utilization=0.85, max_model_len=16384,
                enforce_eager=True, disable_log_stats=True,
            ))
            # vLLM's generate is async-only; run a dedicated event loop in a daemon
            # thread and bridge results back through a queue so the public sync
            # generate/generate_stream API is unchanged for every caller.
            self._loop = asyncio.new_event_loop()
            threading.Thread(target=self._loop.run_forever, daemon=True).start()
        elif backend == "transformers":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._torch = torch
            self.tok = AutoTokenizer.from_pretrained(model_path_or_id)
            # `dtype=` verified supported on transformers 4.57.x (torch_dtype is the
            # deprecated alias there; 4.57 warns on it, so use dtype).
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path_or_id, dtype=torch.bfloat16, device_map="cuda"
            )
            self.model.eval()
        else:  # llama_cpp — guarded import, Kaggle-T4 route only
            try:
                from llama_cpp import Llama
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "backend='llama_cpp' requires llama-cpp-python "
                    "(pip install llama-cpp-python). On Modal use the default "
                    "backend='transformers' instead."
                ) from e
            self.llm = Llama(
                model_path=model_path_or_id,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )

    def _vllm_stream(self, prompt: str, max_tokens: int, temperature: float):
        """Yield text deltas from the background-loop AsyncLLMEngine (sync bridge)."""
        import asyncio
        import queue as _queue
        import uuid

        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        sp = self._SamplingParams(max_tokens=max_tokens, temperature=temperature)
        q: "_queue.Queue" = _queue.Queue()

        async def _run():
            prev = ""
            try:
                async for out in self.engine.generate(text, sp, str(uuid.uuid4())):
                    cur = out.outputs[0].text
                    if len(cur) > len(prev):
                        q.put(cur[len(prev):])
                        prev = cur
                q.put(None)
            except Exception as exc:  # surface engine errors in the caller thread
                q.put(exc)

        asyncio.run_coroutine_threadsafe(_run(), self._loop)
        while True:
            item = q.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Single-turn chat completion; returns the stripped assistant text."""
        if self.backend == "vllm":
            return "".join(self._vllm_stream(prompt, max_tokens, temperature)).strip()
        if self.backend == "transformers":
            torch = self._torch
            text = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = self.tok(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    do_sample=temperature > 0, temperature=max(temperature, 1e-4),
                    pad_token_id=self.tok.eos_token_id,
                )
            gen = out[0][inputs["input_ids"].shape[1]:]
            return self.tok.decode(gen, skip_special_tokens=True).strip()

        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (response["choices"][0]["message"]["content"] or "").strip()

    def generate_stream(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1):
        """Yield the assistant text incrementally as it is generated (token chunks).

        Used by streaming UIs so the SSE/websocket connection keeps receiving data during a
        long generation instead of going idle (which a proxy will drop → 'connection lost').
        """
        if self.backend == "vllm":
            yield from self._vllm_stream(prompt, max_tokens, temperature)
            return
        if self.backend == "transformers":
            import threading

            from transformers import TextIteratorStreamer

            torch = self._torch
            text = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = self.tok(text, return_tensors="pt").to("cuda")
            streamer = TextIteratorStreamer(
                self.tok, skip_prompt=True, skip_special_tokens=True,
            )
            gen_kwargs = dict(
                **inputs, max_new_tokens=max_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-4),
                pad_token_id=self.tok.eos_token_id, streamer=streamer,
            )

            def _run():
                with torch.no_grad():
                    self.model.generate(**gen_kwargs)

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            for chunk in streamer:
                if chunk:
                    yield chunk
            thread.join()
            return

        # llama_cpp streaming
        for part in self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature, stream=True,
        ):
            delta = part["choices"][0].get("delta", {}).get("content")
            if delta:
                yield delta
