"""Quick reproducer for vllm 0.11.0 + Qwen3-VL multi-modal + logprobs>0 crash.

Mirrors the EXACT input shape verl's vllm_rollout_spmd.py passes to vllm:
    {"prompt_token_ids": [pre-tokenized with image_token placeholders expanded],
     "multi_modal_data": {"image": [np.ndarray]}}

Run with the same vllm + torch + transformers stack as the main training run.
Pass / exit code: 0 = bug NOT present, 1 = bug reproduced.

  python scripts/check_vllm_mm_logprobs.py

Takes ~30 s on a single H20. Adjust MODEL path / image size for your setup.
"""
import sys
import numpy as np
import torch
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

MODEL = "/preset-models"


def main() -> int:
    print(f"torch={torch.__version__}, cuda available={torch.cuda.is_available()}")
    print("Loading vllm engine...")
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.6,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
    )

    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    img = Image.new("RGB", (768, 768), color=(200, 50, 50))
    img_arr = np.array(img.convert("RGB"))

    # Same recipe verl's self_judge_client._tokenize_batch uses:
    chat = [{"role": "user", "content": "<image>\nChoose: A. red B. blue C. green"}]
    prompt_with_template = processor.tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=False)
    image_inputs = processor.image_processor([img_arr], return_tensors="pt")
    image_grid_thw = image_inputs["image_grid_thw"]
    merge_length = processor.image_processor.merge_size ** 2

    idx = 0
    while "<image>" in prompt_with_template:
        n_placeholders = image_grid_thw[idx].prod().item() // merge_length
        vision_block = "<|vision_start|>" + "<|placeholder|>" * n_placeholders + "<|vision_end|>"
        prompt_with_template = prompt_with_template.replace("<image>", vision_block, 1)
        idx += 1
    prompt_with_template = prompt_with_template.replace("<|placeholder|>", processor.image_token)
    prompt_token_ids = processor.tokenizer.encode(prompt_with_template)
    print(f"prompt_len={len(prompt_token_ids)}, image_grid_thw={image_grid_thw.tolist()}")

    vllm_input = {
        "prompt_token_ids": prompt_token_ids,
        "multi_modal_data": {"image": [img_arr]},
    }

    sp = SamplingParams(max_tokens=1, logprobs=20, temperature=0)
    out = llm.generate([vllm_input], sp, use_tqdm=False)
    print(f"OK: top tokens = {list(out[0].outputs[0].logprobs[0].keys())[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())