# generation/llm.py
"""Qwen2.5 inference wrapper with two interchangeable backends.

* ``backend="transformers"`` (default) — HF ``AutoModelForCausalLM`` in bf16 on CUDA.
  This is the Modal A100 route and mirrors the ``_QwenSummarizer`` used during the
  build (``modal_build.py``), so build-time summaries and eval-time answers come
  from the exact same generation path.
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
        model_path_or_id: str = "Qwen/Qwen2.5-7B-Instruct",
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

        llama_cpp model file (Kaggle route):
            huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
                qwen2.5-7b-instruct-q4_k_m.gguf --local-dir models/
        """
        if backend not in ("transformers", "llama_cpp"):
            raise ValueError(f"unknown backend {backend!r}; use 'transformers' or 'llama_cpp'")
        self.backend = backend

        if backend == "transformers":
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

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Single-turn chat completion; returns the stripped assistant text."""
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
