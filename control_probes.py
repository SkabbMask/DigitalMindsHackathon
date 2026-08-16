import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

MODEL = "Qwen/Qwen2.5-7B-Instruct"

data = np.load("activations.npz")
activations = data["activations"]
labels = data["labels"]
referents = data["referents"]

group_a_referents = {"Google", "Disney", "Emmanuel Macron", "a car mechanic in Tbilisi", "the International Seabed Authority", "a mid-sized regional airline"}
group_b_referents = {"the Pope", "Satoshi Nakamoto", "the Svalbard Global Seed Vault", "Amazon", "the prime minister of Sweden", "a small football team in rural Brazil", "the Bank for International Settlements"}

is_control_row = np.array([r in group_a_referents or r in group_b_referents for r in referents])
y_control = np.array([1 if r in group_a_referents else 0 for r in referents])

control_test_referents = {"Google", "a small football team in rural Brazil"}
is_control_test = np.array([r in control_test_referents for r in referents]) & is_control_row
is_control_train = is_control_row & ~is_control_test

control_results = []
for layer in range(activations.shape[1]):
    X = activations[:, layer, :]
    X_train, X_test = X[is_control_train], X[is_control_test]
    y_train, y_test = y_control[is_control_train], y_control[is_control_test]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_scaled, y_train)

    test_acc = balanced_accuracy_score(y_test, clf.predict(X_test_scaled))
    control_results.append({"layer": layer, "test_acc": test_acc})
    print(f"Layer {layer:2d}: external-vs-external test_acc={test_acc:.3f}")

layers = [r["layer"] for r in control_results]
test_accs = [r["test_acc"] for r in control_results]

plt.figure(figsize=(10, 6))
plt.plot(layers, test_accs, marker="o", label="test accuracy (held-out referents)")
plt.axhline(0.5, color="gray", linestyle="--", label="chance (50%)")
plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.title("Different External Entities Probe Accuracy by Layer")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("control_probe_results.png", dpi=150)
print("Saved plot")