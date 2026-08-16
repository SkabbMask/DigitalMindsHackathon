import numpy as np

INJECTION_LAYER = 20

data = np.load("activations.npz")
activations = data["activations"]
labels = data["labels"]

self_mask = labels == "self"
external_mask = labels == "external"

self_acts = activations[self_mask, INJECTION_LAYER, :]
external_acts = activations[external_mask, INJECTION_LAYER, :]

self_mean = self_acts.mean(axis=0)
external_mean = external_acts.mean(axis=0)

direction = self_mean - external_mean

np.save("self_direction_layer20.npy", direction)
print("Saved direction")