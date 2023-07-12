"""
This script is used to train the PINN model for task 1
"""
import os
import sys
import matplotlib.pyplot as plt
import argparse
import torch
from torch import optim
import numpy as np
import pandas as pd

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, ".."))
sys.path.append(parent_path)

from task1 import pinns

torch.autograd.set_detect_anomaly(False)
torch.manual_seed(128)

print(f"PyTorch version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"currently available device: {device}")
USE_GPU = torch.cuda.is_available()


if __name__ == "__main__":
    model_path = os.path.join(main_path, "model.pth")
    parser = argparse.ArgumentParser(
        description="Forward solving with PINN"
    )
    parser.add_argument(
        "-plt",
        "--plot",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Plot the input training points"
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default="True",
        choices=["True", "False"],
        help="If model should be saved"
    )
    parser.add_argument(
        "-l",
        "--load",
        type=str,
        default="False",
        choices=["True", "False"],
        help="If model should be loaded first then either trained or inference"
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str,
        default="True",
        choices=["True", "False"],
        help="If model should be trained before inference"
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=model_path,
        help="Model name"
    )
    args = parser.parse_args()

    # number of the different training points
    # gpu vram limit of 16GB
    n_int = 12000 # 256
    n_sb_tb = 1200 # 64
    n_sb = n_sb_tb
    n_tb = n_sb_tb
    # batch sizes
    nb = 1
    batchsize_int = n_int // nb
    batchsize_bc = n_sb_tb // nb
    pinn = pinns.Pinns(n_int, n_sb, n_tb, batchsize_int, batchsize_bc, device)



    if args.plot == "True":
        # Add training points
        input_sb_n0_, output_sb_n0_, input_sb_nl_, output_sb_nl_, input_sb_d0_, output_sb_d0_ = pinn.add_spatial_boundary_points()
        input_tb_, output_tb_ = pinn.add_temporal_boundary_points()
        input_int_, output_int_ = pinn.add_interior_points()

        # Plot the input training points
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        axs[0].scatter(input_sb_n0_[:, 1].detach().numpy(), input_sb_n0_[:, 0].detach().numpy(), label="Left Neumann Boundary Points")
        axs[0].scatter(input_sb_nl_[:, 1].detach().numpy(), input_sb_nl_[:, 0].detach().numpy(), label="Right Neumann Boundary Points")
        axs[0].scatter(input_sb_d0_[:, 1].detach().numpy(), input_sb_d0_[:, 0].detach().numpy(), label="Left Dirichlet Boundary Points")
        axs[0].scatter(input_int_[:, 1].detach().numpy(), input_int_[:, 0].detach().numpy(), label="Interior Points")
        axs[0].scatter(input_tb_[:, 1].detach().numpy(), input_tb_[:, 0].detach().numpy(), label="Initial Points")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        axs[0].legend()
        axs[0].set_title("Random Points")
        
        # plot the dirichlet condition
        axs[1].scatter(input_sb_d0_[:, 0].detach().numpy(), output_sb_d0_.detach().numpy(), label="Dirichlet Boundary Points")
        axs[1].set_ylabel("T")
        axs[1].set_xlabel("t")
        axs[1].set_title("Dirichlet Condition")
        fig_path = os.path.join(main_path, "input_points_task1.png")
        fig.savefig(fig_path)

    
    n_epochs = 1
    optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                                lr=float(0.5),
                                max_iter=800,
                                max_eval=800,
                                history_size=100,
                                line_search_fn="strong_wolfe",
                                tolerance_change=1.0 * np.finfo(float).eps)
    optimizer_ADAM = optim.Adam(pinn.approximate_solution.parameters(),
                                lr=float(0.001))
    # choose optimizer
    optimizer = optimizer_LBFGS

    # load the model
    if args.load == "True":
        try:
            checkpoint = torch.load(args.path, map_location=device)
            pinn.approximate_solution.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            pinn.approximate_solution.eval()
            pinn.approximate_solution.to(device)
        except (IsADirectoryError, IsADirectoryError):
            print("No model found")
            print("Train a new model")
            args.train = "True"

    # train the model
    if args.train == "True":
        hist = pinn.fit(num_epochs=n_epochs,
                    optimizer=optimizer,
                    verbose=True)
    
    # save the model
    if args.save == "True":
        torch.save({"model_state_dict": pinn.approximate_solution.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                    },
                    args.path)
    
    # plot the loss and the predicted solution
    if args.plot == "True" and args.train == "True":
        # plot the loss
        fig = plt.figure(dpi=150)
        plt.grid(True, which="both", ls=":")
        plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
        plt.xscale("log")
        plt.legend()
        fig_path = os.path.join(main_path, "train_loss_task1.png")
        fig.savefig(fig_path)

    if args.plot == "True":
        # plot the predicted solution
        fig_path = os.path.join(main_path, "plot_task1.png")
        pinn.plotting(name=fig_path)

    # load given data
    data_path = os.path.join(main_path, "TestingData.txt")
    data = pd.read_csv(data_path, sep=",")
    inp = torch.tensor(data.values, dtype=torch.float32).to(device)

    # inference of the given data
    output = pinn.approximate_solution(inp).to("cpu")

    # add to dataframe
    data["tf"] = output.detach().numpy()[:,0]
    data["ts"] = output.detach().numpy()[:,1]

    # save dataframe
    save_path = os.path.join(parent_path, "Task1.txt")
    data.to_csv(save_path, sep=",", index=False)
