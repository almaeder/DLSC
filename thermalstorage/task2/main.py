"""
This script is used to train the PINN model for task 2.
"""
import torch
import os
import sys
import matplotlib.pyplot as plt
from torch import optim
import numpy as np
import argparse
import pandas as pd

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, ".."))
sys.path.append(parent_path)

from task2 import pinns

torch.autograd.set_detect_anomaly(False)
torch.manual_seed(128)

print(f"PyTorch version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"currently available device: {device}")
USE_GPU = torch.cuda.is_available()


if __name__ == "__main__":
    model_path = os.path.join(main_path, "model.pth")
    parser = argparse.ArgumentParser(
        description="Inverse solving with PINN"
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
    # gpu vram limit of 8GB
    # given amount of measurement points: 39200
    n_int = 10000 # 256
    n_sb_tb = 1000 # 64
    n_meas = 39200
    n_sb = n_sb_tb
    n_tb = n_sb_tb
    # number of batches
    nb = 1
    # batch size
    batchsize_int = n_int // nb
    batchsize_bc = n_sb_tb // nb
    batchsize_meas = n_meas // nb
    pinn = pinns.Pinns(n_int, n_sb, n_tb, batchsize_int, batchsize_bc, batchsize_meas, device)



    if args.plot == "True":
        # Add training points
        input_sb_c_0_, output_sb_c_0_, input_sb_c_l_, _ = pinn.add_sbp_charge()
        input_sb_dc_0_, _, input_sb_dc_l_, output_sb_dc_l_ = pinn.add_sbp_discharge()
        input_sb_i_0_, _, input_sb_i_l_, _ = pinn.add_sbp_idle()
        input_tb_, output_tb_ = pinn.add_temporal_boundary_points()
        input_int_, output_int_ = pinn.add_interior_points()

        # Plot the input training points
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        axs[0].scatter(input_sb_c_0_[:, 1].detach().numpy(), input_sb_c_0_[:, 0].detach().numpy(), label="Left Charge Boundary Points")
        axs[0].scatter(input_sb_c_l_[:, 1].detach().numpy(), input_sb_c_l_[:, 0].detach().numpy(), label="Right Charge Boundary Points")
        axs[0].scatter(input_sb_dc_0_[:, 1].detach().numpy(), input_sb_dc_0_[:, 0].detach().numpy(), label="Left Discharge Boundary Points")
        axs[0].scatter(input_sb_dc_l_[:, 1].detach().numpy(), input_sb_dc_l_[:, 0].detach().numpy(), label="Right Discharge Boundary Points")
        axs[0].scatter(input_sb_i_0_[:, 1].detach().numpy(), input_sb_i_0_[:, 0].detach().numpy(), label="Left Idle Boundary Points")
        axs[0].scatter(input_sb_i_l_[:, 1].detach().numpy(), input_sb_i_l_[:, 0].detach().numpy(), label="Right Idle Boundary Points")
        axs[0].scatter(input_int_[:, 1].detach().numpy(), input_int_[:, 0].detach().numpy(), label="Interior Points")
        axs[0].scatter(input_tb_[:, 1].detach().numpy(), input_tb_[:, 0].detach().numpy(), label="Initial Points")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        axs[0].legend()
        axs[0].set_title("Random Points")

        # plot the dirichlet condition
        axs[1].scatter(input_sb_c_0_[:, 0].detach().numpy(), output_sb_c_0_.detach().numpy(), label="Dirichlet Boundary Points")
        axs[1].scatter(input_sb_dc_l_[:, 0].detach().numpy(), output_sb_dc_l_.detach().numpy(), label="Dirichlet Boundary Points")
        axs[1].set_ylabel("T")
        axs[1].set_xlabel("t")
        axs[1].set_title("Dirichlet Condition mixed left and right")
        fig_path = os.path.join(main_path, "input_points_task2.png")
        fig.savefig(fig_path)

    
    n_epochs = 1
    optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                                lr=float(0.5),
                                max_iter=500,
                                max_eval=500,
                                history_size=150,
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
        fig_path = os.path.join(main_path, "train_loss_task2.png")
        fig.savefig(fig_path)

    if args.plot == "True":
        # plot the predicted solution
        fig_path = os.path.join(main_path, "plot_task2.png")
        pinn.plotting(name=fig_path)


    # load given data
    data_path = os.path.join(main_path, "DataSolution.txt")
    data = pd.read_csv(data_path, sep=",")
    inp = torch.tensor(data.values, dtype=torch.float32).to(device)

    # inference of the given data
    output = pinn.approximate_solution(inp[:,0:2]).to("cpu")
    inp = inp.to("cpu")

    if args.plot == "True":
        # Plot the measurement points
        fig, axs = plt.subplots(1, 3, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(inp[:, 1].detach(), inp[:, 0].detach(), c=inp[:, 2].detach().numpy(), cmap="jet", vmin=1, vmax=4)

        # plot the predicted fluid temperature
        im2 = axs[1].scatter(inp[:, 1].detach(), inp[:, 0].detach(), c=output[:, 0].detach().numpy(), cmap="jet", vmin=1, vmax=4)

        # plot the predicted solid temperature
        im3 = axs[2].scatter(inp[:, 1].detach(), inp[:, 0].detach(), c=output[:, 1].detach().numpy(), cmap="jet", vmin=1, vmax=4)

        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("t")
        axs[0].set_title("Given Points")
        axs[1].set_title("Predicted Fluid Temperature")
        axs[2].set_title("Predicted Solid Temperature")

        cbar = fig.colorbar(im1, ax=axs)
        cbar.set_ticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

        fig_path = os.path.join(main_path, "prediction_task2.png")
        fig.savefig(fig_path)


    # remove fluid temperature
    # add to dataframe
    data = data.drop("tf", axis=1)
    data["ts"] = output.detach().numpy()[:,1]

    # save dataframe
    save_path = os.path.join(parent_path, "Task2.txt")
    data.to_csv(save_path, sep=",", index=False)
