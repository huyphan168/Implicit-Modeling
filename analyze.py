import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

with open("results/states.pkl", "rb") as f:
    unpickler = pkl.Unpickler(f)
    states = unpickler.load()
    states = states["states"]
B = states[0]
plt.figure(figsize=(6,6), dpi=300)
counts, bins = np.histogram(B.flatten(), bins=10000)
sparsity = (B <1e-30).sum()/len(B.flatten())
print("sparsity", sparsity)
plt.stairs(counts, bins)
plt.savefig("states.png")

