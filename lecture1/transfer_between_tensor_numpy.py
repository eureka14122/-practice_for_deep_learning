import numpy as np
import torch

n = np.array([1, 2, 3])
t = torch.from_numpy(n)

n = t.numpy()
