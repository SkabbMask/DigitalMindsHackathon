import json
import numpy as np
import modal

MODEL = "Qwen/Qwen2.5-14B-Instruct"
PROMPTS_PATH = "self_vs_entity_prompts_with_actual.json"
OUT_PATH = "activations.npz"

app = modal.App("self-vs-entity-activations")
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

@app.function(
        image=image,
        gpu="A100-80GB",
        volumes={"/root/.cache/huggingface": volume},
        timeout=600,
)
def get_activations(model_name: str, prompts_with_meta: list[dict]):
    import numpy as np
    import torch
    from tqdm import tqdm
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(model_name, dtype=torch.bfloat16)
    n_layers = model.cfg.n_layers
    hidden_size = model.cfg.d_model

    acts = np.zeros((len(prompts_with_meta), n_layers, hidden_size), dtype=np.float32)
    labels = []
    referents = []
    prompts = []
    with torch.no_grad():
        for i, row in enumerate(tqdm(prompts_with_meta)):
            prompt = row["prompt"]
            messages = [{"role": "user", "content": prompt}]
            text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            referent = row["actual_referent"]

            referent_index = text.lower().index(referent.lower())
            prefix_ids = model.tokenizer(text[:referent_index])["input_ids"]
            suffix_ids = model.tokenizer(text[referent_index + len(referent):])["input_ids"]
            full_ids = model.tokenizer(text)["input_ids"]
            start = common_prefix_len(full_ids, prefix_ids)
            end = len(full_ids) - common_suffix_len(full_ids, suffix_ids)

            _, cache = model.run_with_cache(text, names_filter=lambda name: "resid_post" in name)
            last_token_acts = torch.stack(
                [cache["resid_post", l][0, start:end, :].mean(dim=0) for l in range(n_layers)]
            )

            acts[i] = last_token_acts.float().cpu().numpy()
            labels.append(row["label"])
            referents.append(referent)
            prompts.append(prompt)
    return {
        "activations": acts,
        "labels": labels,
        "referents": referents,
        "prompts": prompts,
    }

@app.local_entrypoint()
def main():
    with open(PROMPTS_PATH) as f:
        rows = json.load(f)

    results = get_activations.remote(MODEL, rows)
    np.savez(
        OUT_PATH,
        activations=results["activations"],
        labels=np.array(results["labels"]),
        referents=np.array(results["referents"]),
        prompts=np.array(results["prompts"]),
    )
    print(f"Saved {results['activations'].shape} activations to {OUT_PATH}")

