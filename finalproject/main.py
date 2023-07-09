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
    n = 20000

    # number of batches
    nb = 1
    # batch size
    batchsize_int = n // nb
    xl = 0.0
    xr = 1.0
    ub = 0.0
    pinn = pinns.Pinns(n, batchsize_int, xl, xr, ub, device)


    
    n_epochs = 1
    # parameters = list(pinn.approximate_solution.parameters()) + [pinn.approximate_solution.eigenvalue]
    parameters = list(pinn.approximate_solution.parameters())
    optimizer_LBFGS = optim.LBFGS(parameters,
                                lr=float(0.5),
                                max_iter=10000,
                                max_eval=10000,
                                history_size=500,
                                line_search_fn="strong_wolfe",
                                tolerance_change=1.0 * np.finfo(float).eps)
    optimizer_ADAM = optim.Adam(parameters,
                                lr=float(0.02))
    # choose optimizer
    optimizer = optimizer_LBFGS

    hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer,
                verbose=True)
    

    # plot the loss
    fig = plt.figure(dpi=150)
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    fig_path = os.path.join(main_path, "train_loss.png")
    fig.savefig(fig_path)

    # plot the predicted solution
    fig_path = os.path.join(main_path, "prediction")
    pinn.plotting(name=fig_path)
