# generation/llm.py


class QwenLLM:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        """
        Load Qwen2.5-7B-Instruct Q4_K_M via llama.cpp.

        For Kaggle T4 16GB: n_gpu_layers=-1 (all layers on GPU).
        Model file: ~4.5GB Q4_K_M GGUF.

        Download from HuggingFace:
        huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
            qwen2.5-7b-instruct-q4_k_m.gguf --local-dir models/
        """
        raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §7.1")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        raise NotImplementedError("TODO: see VERGIL_BUILD_PLAN.md §7.1")
