import torch
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())
import numpy as np


x = torch.linspace(-5, 5, 200)
x_np = x.numpy()
relu_np = torch.relu(x).numpy()
sigmoid_np = torch.sigmoid(x).numpy()
tanh_np = torch.tanh(x).numpy()

plt.figure(1,figsize=(6,8))
plt.subplot(221)
plt.plot(x_np,relu_np,c="red", label="relu")
plt.ylim((-1,5))
plt.legend(loc='best')
plt.show()