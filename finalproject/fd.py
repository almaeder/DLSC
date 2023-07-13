import numpy as np
import scipy.sparse as sparse
import common
import matlplotlib.pyplot as plt

class FD_solver:
    def __init__(
        self: object,
        xl: float,
        xr: float,
        n: float
    ):
        self.h = (xr-xl) / n
        self.N = n-2
        self.grid = np.arange(xl+self.h,xr,self.h)
        self.potential = common.potential(self.grid).cpu().numpy()
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
        idx_x = np.concatenate((idx_diag, idx_off+1, idx_off))
        idx_y = np.concatenate((idx_diag, idx_off, idx_off+1))
        idx = np.concatenate((idx_x, idx_y), dim=1)
        self.H = sparse.coo_array(idx,val).tocsr()
    def eigensolve(
      self: object,
      num_eig: int = 4
    ):
        self.eigenvalues, self.eigenfunctions = sparse.linalg.eigsh(self.H, k=num_eig, which="SM")
        print(self.eigenvalues)

    def plot(
      self: object,
      num_eig: int = 4
    ):
        for i in range(num_eig):
            plt.plot(self.grid, self.eigenfunctions[:,i], label=str(self.eigenvalues[i]))
        plt.legend(loc="upper left")