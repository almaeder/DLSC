
import torch
import common

class Pinns:
    """
    Class to create pinns for the thermal storage equation of task2.
    """
    def __init__(
        self: object,
        n_: int,
        batchsize_: int,
        xl_: float,
        xr_: float,
        ub_: float,
        device: torch.device
    ):
        self.device = device
        self.n = n_
        self.batchsize = batchsize_
        self.ub = ub_
        self.domain_extrema = torch.tensor([xl_, xr_])  
        self.space_dimensions = 1
        self.alpha_norm = 1.0
        self.alpha_ortho = 1.0

        self.approximate_solution = common.NeuralNet(
            input_dimension=self.domain_extrema.shape[0],
            output_dimension=1,
            n_hidden_layers=3,
            neurons=20,
            regularization_param=0.1,
            regularization_exp=2.,
            retrain_seed=42
        ).to(device)