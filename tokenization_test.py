import json
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPTS_PATH = "self_vs_entity_prompts_with_actual.json"

def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n

def common_suffix_len(a, b):
    return common_prefix_len(a[::-1], b[::-1])

tokenizer = AutoTokenizer.from_pretrained(MODEL)

with open(PROMPTS_PATH) as f:
    rows = json.load(f)

# check a handful spread across the dataset, including known-tricky ones
sample_indices = [0, 1, 9, 22, 50, len(rows) // 2, len(rows) - 1]

for i in sample_indices:
    row = rows[i]
    prompt = row["prompt"]
    referent = row["actual_referent"]

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    referent_index = text.lower().index(referent.lower())
    prefix_ids = tokenizer(text[:referent_index])["input_ids"]
    suffix_ids = tokenizer(text[referent_index + len(referent):])["input_ids"]
    full_ids = tokenizer(text)["input_ids"]

    start = common_prefix_len(full_ids, prefix_ids)
    end = len(full_ids) - common_suffix_len(full_ids, suffix_ids)

    recovered = tokenizer.convert_ids_to_tokens(full_ids[start:end])
    print(f"[{i}] referent={referent!r:15} -> recovered tokens: {recovered}")