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
from finalproject import fd

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
    n_sb = 10000

    # number of batches
    nb_int = 100
    nb_sb = 10
    # batch size
    batchsize_int = n_int // nb_int
    batchsize_sb = n_sb // nb_sb
    xl = 0.0
    xr = 5.0
    ub = 0.0
    num_eig = 6
    load_eig = 1
    max_value = 10.0
    pinn = pinns.Pinns(n_int, n_sb, batchsize_int, batchsize_sb, xl, xr, ub, num_eig, max_value, device)

    pinn.load_eigenfunctions(main_path + "/2", load_eig)


    n_epochs = 1
    # parameters = list(pinn.approximate_solution.parameters()) + [pinn.approximate_solution.eigenvalue]
    # parameters = list(pinn.approximate_solution.parameters())
    parameters = list(pinn.approximate_solution.parameters())
    optimizer_LBFGS = optim.LBFGS(parameters,
                                lr=float(0.5),
                                max_iter=50,
                                max_eval=50,
                                history_size=100,
                                line_search_fn="strong_wolfe",
                                tolerance_change=1.0 * np.finfo(float).eps)
    optimizer_ADAM = optim.Adam(params=parameters,
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
    plt.legend()
    fig_path = os.path.join(main_path, "train_loss.png")
    fig.savefig(fig_path)

    # plot the predicted solution
    fig_path = os.path.join(main_path, "prediction")
    pinn.plotting_multiple(name=fig_path)
    pinn.save_eigenfunctions(main_path + "/2")


    solver = fd.FD_solver(xl, xr, 501)
    solver.assemble()
    solver.eigensolve(num_eig=num_eig)

    # fig_path = os.path.join(main_path, "comparison")
    # for i in range(num_eig):
    #     pinn.plotting_prob_amplitude_fd(i, solver.grid, solver.eigenfunctions[:,i],name=fig_path)
    fig_path = os.path.join(main_path, "fd")
    solver.plot(num_eig=num_eig,name=fig_path)
