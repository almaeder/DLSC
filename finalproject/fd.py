import torch
import torch.sparse as sparse
import common

class FD_solver:
    def __init__(
        self: object,
        xl: float,
        xr: float,
        n: float,
        device: torch.device
    ):
        self.domain_extrema = torch.tensor([[xl, xr]])
        self.domain_extrema = self.domain_extrema.to(self.device)
        self.h = (xr-xl) / n
        self.grid = torch.arange(xl,xr,self.h)
        self.potential = common.potential(self.grid)

    def assemble(
      self: object      
    ):
        