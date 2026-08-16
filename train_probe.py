import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

MODEL = "Qwen/Qwen2.5-14B-Instruct"

data = np.load("activations.npz")
activations = data["activations"]
labels = data["labels"]
referents = data["referents"]

test_self_referents = {"this AI", "the artificial intelligence system", "you"}
test_external_referents = {
	"the Pope", "Satoshi Nakamoto", "the Svalbard Global Seed Vault", "a small football team in rural Brazil",
}

is_test = np.array([r in test_self_referents or r in test_external_referents for r in referents])
is_train = ~is_test

print(f"Train rows: {is_train.sum()}, Test rows: {is_test.sum()}")
print(f"Train referents: {sorted(set(referents[is_train]))}")
print(f"Test referents: {sorted(set(referents[is_test]))}")

results = []

for layer in range(activations.shape[1]):
	X = activations[:, layer, :]
	y = (labels == "self").astype(int)

	X_train, X_test = X[is_train], X[is_test]
	y_train, y_test = y[is_train], y[is_test]

	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	clf = LogisticRegression(max_iter=1000, class_weight="balanced")
	clf.fit(X_train_scaled, y_train)

	train_acc = balanced_accuracy_score(y_train, clf.predict(X_train_scaled))
	test_acc = balanced_accuracy_score(y_test, clf.predict(X_test_scaled))

	results.append({"layer": layer, "train_acc": train_acc, "test_acc": test_acc})
	print(f"Layer {layer:2d}: train_acc={train_acc:.3f} test_acc={test_acc:.3f}")

with open("probe_results.json", "w") as f:
	json.dump(results, f, indent=2)

print("Saved results")

layers = [r["layer"] for r in results]
train_accs = [r["train_acc"] for r in results]
test_accs = [r["test_acc"] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(layers, train_accs, marker="o", label="train accuracy")
plt.plot(layers, test_accs, marker="o", label="test accuracy (held-out referents)")
plt.axhline(0.5, color="gray", linestyle="--", label="chance (50%)")
plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.title("Self vs. External Entity Probe Accuracy by Layer")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("probe_results.png", dpi=150)
print("Saved plot")



# Diagnostics for shuffled labels - should be ~50% accuracy
#  --- Diagnostic 1: permutation / null baseline ---
rng = np.random.default_rng(42)
n_shuffles = 20
check_layers = [1, 5, 10, 20, 30, 40, 46]  # spot-check a few layers spanning the range
 
print("\n--- Null baseline (shuffled labels) ---")
null_results = {}
for layer in check_layers:
	X = activations[:, layer, :]
	X_train, X_test = X[is_train], X[is_test]
 
	shuffled_test_accs = []
	for _ in range(n_shuffles):
		y_shuffled = rng.permutation((labels == "self").astype(int))
		y_train_shuf, y_test_shuf = y_shuffled[is_train], y_shuffled[is_test]
 
		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(X_train)
		X_test_scaled = scaler.transform(X_test)
 
		clf = LogisticRegression(max_iter=1000, class_weight="balanced")
		clf.fit(X_train_scaled, y_train_shuf)
		shuffled_test_accs.append(balanced_accuracy_score(y_test_shuf, clf.predict(X_test_scaled)))
 
	mean_acc, std_acc = np.mean(shuffled_test_accs), np.std(shuffled_test_accs)
	null_results[layer] = {"mean": mean_acc, "std": std_acc}
	print(f"Layer {layer:2d}: shuffled-label test acc = {mean_acc:.3f} +/- {std_acc:.3f}")
 
with open("probe_null_baseline.json", "w") as f:
	json.dump({str(k): v for k, v in null_results.items()}, f, indent=2)



# --- Diagnostic 2: sequence length as sole predictor ---
from transformers import AutoTokenizer
 
tokenizer = AutoTokenizer.from_pretrained(MODEL)
prompts_full = data["prompts"]
 
token_lengths = np.array([
	len(tokenizer(p)["input_ids"]) for p in prompts_full
]).reshape(-1, 1)
 
X_train_len, X_test_len = token_lengths[is_train], token_lengths[is_test]
y_train_len = (labels[is_train] == "self").astype(int)
y_test_len = (labels[is_test] == "self").astype(int)
 
scaler = StandardScaler()
X_train_len_scaled = scaler.fit_transform(X_train_len)
X_test_len_scaled = scaler.transform(X_test_len)
 
clf_len = LogisticRegression(max_iter=1000, class_weight="balanced")
clf_len.fit(X_train_len_scaled, y_train_len)
 
len_train_acc = balanced_accuracy_score(y_train_len, clf_len.predict(X_train_len_scaled))
len_test_acc = balanced_accuracy_score(y_test_len, clf_len.predict(X_test_len_scaled))
 
print("\n--- Length-only baseline ---")
print(f"Train acc: {len_train_acc:.3f}")
print(f"Test acc:  {len_test_acc:.3f}")
 
self_lens = token_lengths[labels == "self"].flatten()
ext_lens = token_lengths[labels == "external"].flatten()
print(f"\nSelf prompt lengths:     mean={self_lens.mean():.1f}, min={self_lens.min()}, max={self_lens.max()}")
print(f"External prompt lengths: mean={ext_lens.mean():.1f}, min={ext_lens.min()}, max={ext_lens.max()}")
 
with open("probe_length_baseline.json", "w") as f:
	json.dump({
		"train_acc": len_train_acc,
		"test_acc": len_test_acc,
		"self_len_mean": float(self_lens.mean()),
		"self_len_min": int(self_lens.min()),
		"self_len_max": int(self_lens.max()),
		"ext_len_mean": float(ext_lens.mean()),
		"ext_len_min": int(ext_lens.min()),
		"ext_len_max": int(ext_lens.max()),
	}, f, indent=2)
print("Saved diagnostic results")
 
#Comparison plot
plt.figure(figsize=(10, 6))

# Main probe results (reuse from earlier)
plt.plot(layers, test_accs, marker="o", label="probe test accuracy (held-out referents)", color="tab:blue")

# Null baseline: shuffled labels, with error bars showing std across shuffles
null_layers = list(null_results.keys())
null_layers = [int(l) for l in null_layers]  # keys were stringified for JSON
null_means = [null_results[l]["mean"] for l in null_layers]
null_stds = [null_results[l]["std"] for l in null_layers]
plt.errorbar(
    null_layers, null_means, yerr=null_stds,
    marker="s", linestyle="--", label="null baseline (shuffled labels)", color="tab:orange"
)

# Length-only baseline: a single number, not per-layer, so draw it as a horizontal line
plt.axhline(len_test_acc, color="tab:green", linestyle=":", label=f"length-only baseline ({len_test_acc:.3f})")

# Chance level
plt.axhline(0.5, color="gray", linestyle="--", label="chance (50%)")

plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.title("Probe Accuracy vs. Null and Length-Only Baselines")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("probe_vs_baselines.png", dpi=150)
print("Saved comparison plot")