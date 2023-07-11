import torch
import os
import sys
import matplotlib.pyplot as plt
from torch import optim
import numpy as np

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, ".."))
sys.path.append(parent_path)

from finalproject import pinns

torch.autograd.set_detect_anomaly(False)
torch.manual_seed(128)

print(f"PyTorch version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"currently available device: {device}")
USE_GPU = torch.cuda.is_available()


if __name__ == "__main__":

    # number of the different training points
    # gpu vram limit of 16GB
    n_int = 100000
    n_sb = 1000

    # number of batches
    nb_int = 100
    nb_sb = 10
    # batch size
    batchsize_int = n_int // nb_int
    batchsize_sb = n_sb // nb_sb
    xl = 0.0
    xr = 1.0
    ub = 0.0
    pinn = pinns.Pinns(n_int, n_sb, batchsize_int, batchsize_sb, xl, xr, ub, device)


    
    n_epochs = 2
    parameters = list(pinn.approximate_solution.parameters()) + [pinn.approximate_solution.eigenvalue]
    # parameters = list(pinn.approximate_solution.parameters())
    optimizer_LBFGS = optim.LBFGS(parameters,
                                lr=float(0.5),
                                max_iter=50,
                                max_eval=50,
                                history_size=100,
                                line_search_fn="strong_wolfe",
                                tolerance_change=1.0 * np.finfo(float).eps)
    optimizer_ADAM = optim.Adam(parameters,
                                lr=float(0.1))
    # choose optimizer
    optimizer = optimizer_LBFGS

    hist = pinn.fit_multiple(num_epochs=n_epochs,
                optimizer=optimizer,
                verbose=True)
    

    # plot the loss
    fig = plt.figure(dpi=150)
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
    # plt.xscale("log")
    # plt.yscale("log")
    plt.legend()
    fig_path = os.path.join(main_path, "train_loss.png")
    fig.savefig(fig_path)

    # plot the predicted solution
    fig_path = os.path.join(main_path, "prediction")
    pinn.plotting_multiple(name=fig_path)
