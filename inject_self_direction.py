import json
import torch
import modal
import numpy as np

MODEL = "Qwen/Qwen2.5-14B-Instruct"
PROMPTS_PATH = "lake_prompts.json"
OUTPUT_PATH = "lake_result.json"
INJECTION_LAYER = 20

alphas = [0.0, 4.0, 8.0, 12.0]

app = modal.App("inject-self-direction")
volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "transformer_lens", "transformers", "huggingface_hub", "tqdm",
)

def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n

def common_suffix_len(a, b):
    return common_prefix_len(a[::-1], b[::-1])

def make_targeted_steeing_hook(direction, alpha, start, end):
    direction_tensor = torch.tensor(direction, dtype=torch.bfloat16)

    def hook_function(resid, hook):
        resid[:, start:end, :] = resid[:, start:end, :] + (alpha * direction_tensor.to(resid.device))
        return resid

    return hook_function

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/root/.cache/huggingface": volume},
    timeout=3600,
)
def inject_direction(model_name: str, direction: np.ndarray, prompts: list[dict]):
    import numpy as np
    from tqdm import tqdm
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(model_name, dtype=torch.bfloat16)

    result = []

    rng = np.random.default_rng(42)
    random_direction = rng.normal(size=direction.shape)
    random_direction = random_direction / np.linalg.norm(random_direction) * np.linalg.norm(direction)

    for i, row in enumerate(tqdm(prompts)):
        prompt = row["prompt"]
        referent = row["referent"]

        messages = [{"role": "user", "content": prompt}]
        text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        referent_index = text.lower().index(referent.lower())
        prefix_ids = model.tokenizer(text[:referent_index])["input_ids"]
        suffix_ids = model.tokenizer(text[referent_index + len(referent):])["input_ids"]
        full_ids = model.tokenizer(text)["input_ids"]
        start = common_prefix_len(full_ids, prefix_ids)
        end = len(full_ids) - common_suffix_len(full_ids, suffix_ids)

        for alpha in alphas:
            if alpha == 0.0:
                output_ids = model.generate(
                    text,
                    max_new_tokens=120,
                    use_past_kv_cache=False,
                    do_sample=False,
                    return_type="tokens",
                )
                response = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                result.append({
                    "prompt": prompt,
                    "alpha": float(alpha),
                    "response": response,
                    "direction": "none",
                })
            else:
                hook_function = make_targeted_steeing_hook(direction, alpha, start, end)

                with model.hooks(fwd_hooks=[(f"blocks.{INJECTION_LAYER}.hook_resid_post", hook_function)]):
                    output_ids = model.generate(
                        text,
                        max_new_tokens=120,
                        use_past_kv_cache=False,
                        do_sample=False,
                        return_type="tokens",
                    )
                response = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                result.append({
                    "prompt": prompt,
                    "alpha": float(alpha),
                    "response": response,
                    "direction": "self",
                })

                hook_function = make_targeted_steeing_hook(random_direction, alpha, start, end)
                
                with model.hooks(fwd_hooks=[(f"blocks.{INJECTION_LAYER}.hook_resid_post", hook_function)]):
                    output_ids = model.generate(
                        text,
                        max_new_tokens=120,
                        use_past_kv_cache=False,
                        do_sample=False,
                        return_type="tokens",
                    )
                response = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                result.append({
                    "prompt": prompt,
                    "alpha": float(alpha),
                    "response": response,
                    "direction": "random",
                })

    return result


@app.local_entrypoint()
def main():
    with open(PROMPTS_PATH) as f:
        rows = json.load(f)

    direction = np.load("self_direction_layer20.npy")

    response = inject_direction.remote(MODEL, direction, rows)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(response, f, indent=2)
    
    print(f"Saved {len(response)} responses to {OUTPUT_PATH}")