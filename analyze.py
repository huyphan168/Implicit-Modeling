import matplotlib.pyplot as plt
import pickle as pkl

with open("results/states.pkl", "rb") as f:
    unpickler = pkl.Unpickler(f)
    states = unpickler.load()
A = states[0]
plt.figure(figsize=(15,15), dpi=300)
plt.spy(A.numpy(), markersize=0.01, color='black')


