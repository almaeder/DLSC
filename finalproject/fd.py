import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
import torch
import common
import matplotlib.pyplot as plt

class FD_solver:
    def __init__(
        self: object,
        xl: float,
        xr: float,
        n: float
    ):
        self.h = (xr-xl) / n
        self.N = n-1
        self.grid = np.arange(1,self.N+1)*self.h
        assert np.allclose(self.grid[1]-self.grid[0],self.h)
        assert np.allclose(self.grid[-1]+self.h,xr)
        assert np.allclose(self.grid[0]-self.h,xl)
        self.potential = common.compute_potential(torch.from_numpy(self.grid)).cpu().numpy()
        self.H = 0
        self.eigenvalues, self.eigenfunctions = 0, 0

    def assemble(
      self: object
    ):
        val_diag = self.potential + 2/self.h**2
        val_off = -1/self.h**2 *np.ones(self.N-1)

        val = np.concatenate((val_diag, val_off, val_off))
        idx_diag = np.arange(self.N)
        idx_off = np.arange(self.N-1)
        idx_i = np.concatenate((idx_diag, idx_off+1, idx_off))
        idx_j = np.concatenate((idx_diag, idx_off, idx_off+1))

        self.H = sparse.coo_array((val,(idx_i,idx_j))).tocsr()
    def eigensolve(
      self: object,
      num_eig: int = 4
    ):
        self.eigenvalues, self.eigenfunctions = linalg.eigsh(self.H, k=num_eig, which="SM")
        print(self.eigenvalues)

    def plot(
      self: object,
      num_eig: int = 4
    ):
        plt.plot(self.grid, self.potential/self.potential.max()*self.eigenfunctions.max(), label="potential")
        for i in range(num_eig):
            plt.plot(self.grid, self.eigenfunctions[:,i], label=str(self.eigenvalues[i]))
        plt.legend(loc="upper left")
        plt.show()
