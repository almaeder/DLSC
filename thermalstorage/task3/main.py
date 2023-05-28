"""
This script is used to solve the task 3 of the thermal storage project.
"""
import os
import sys
import pandas as pd
import numpy as np
import numpy.typing as npt
import deepxde as dde
import typing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch

torch.manual_seed(128)
np.random.seed(128)

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, ".."))

def generate_training_data(
    data_raw: pd.DataFrame,
    num_branch: int
) -> typing.Tuple[typing.Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
                  npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Generate the training data for the task 3.

    Args:
        data_raw (pd.DataFrame): Dataframe containing time, fluid temperature and solid temperature
        num_branch (int): Number of time samples for branch net

    Returns:
        typing.Tuple[typing.Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
                  npt.NDArray[np.float32]]:
        First tuple of training input and then array of training output.
    """
    assert num_branch > 0
    assert num_branch + 1 <= data_raw.shape[0]
    num_train = data_raw.shape[0] - num_branch
    # double the input data due to both fluid and solid temperature
    x_branch_train = np.empty((num_train, 2*num_branch), dtype=np.float32)
    x_trunk_train = np.empty((num_train, 1), dtype=np.float32)
    y_train_fluid = np.empty((num_train, 1), dtype=np.float32)
    y_train_solid = np.empty((num_train, 1), dtype=np.float32)

    # set the branch and trunk net input
    # and the solution samples
    for j in range(num_train):
        # slice right part of the data

        # time for both fluid and solid
        x_trunk_train[j,:] = data_raw["t"].astype(np.float32).values[j+num_branch:j+num_branch+1]

        # fluid and solid temperature flattened
        x_branch_train[j,:num_branch] = data_raw["tf0"].astype(np.float32).values[j:j+num_branch]
        x_branch_train[j,num_branch:] = data_raw["ts0"].astype(np.float32).values[j:j+num_branch]

        # fluid and solid temperature flattened
        y_train_fluid[j,:] = data_raw["tf0"].astype(np.float32).values[j+num_branch:j+num_branch+1]
        y_train_solid[j,:] = data_raw["ts0"].astype(np.float32).values[j+num_branch:j+num_branch+1]

    return (x_branch_train, x_trunk_train), y_train_fluid, y_train_solid

def normalize(
      x: np.ndarray,
      x_mean: np.float64,
      x_std: np.float64
) -> np.ndarray:
    """
    Normalize the data.

    Args:
        x (np.ndarray): Data to be normalized
        x_mean (np.float64): Mean
        x_std (np.float64): Standard deviation

    Returns:
        np.ndarray: Normalized data
    """
    return (x - x_mean) / x_std

def denormalize(
      x: np.ndarray,
      x_mean: np.float64,
      x_std: np.float64
) -> np.ndarray:
    """
    Denormalize the data.

    Args:
        x (np.ndarray): Normalized data
        x_mean (np.float64): Mean
        x_std (np.float64): Standard deviation

    Returns:
        np.ndarray: Denormalized data
    """
    return x * x_std + x_mean



if __name__ == "__main__":
    # load the data
    data_path = os.path.join(main_path, "TrainingData.txt")
    data_r = pd.read_csv(data_path, sep=",")

    # normalize the data
    time_mean = data_r["t"].astype(np.float32).mean()
    time_std = data_r["t"].astype(np.float32).std()
    fluid_mean = data_r["tf0"].astype(np.float32).mean()
    fluid_std = data_r["tf0"].astype(np.float32).std()
    solid_mean = data_r["ts0"].astype(np.float32).mean()
    solid_std = data_r["ts0"].astype(np.float32).std()

    data_norm_r = data_r.copy()
    data_norm_r["t"] = normalize(data_r["t"].astype(np.float32), time_mean, time_std)
    data_norm_r["tf0"] = normalize(data_r["tf0"].astype(np.float32), fluid_mean, fluid_std)
    data_norm_r["ts0"] = normalize(data_r["ts0"].astype(np.float32), solid_mean, solid_std)

    # define number of samples
    num_b = 40
    # generate the training data
    x_t, y_t_f, y_t_s = generate_training_data(data_norm_r, num_b)


    # split the data into training and test data

    # Define the train-test split ratio
    test_ratio = 0.05
    # Compute the size of the test set
    test_size = int(y_t_f.shape[0] * test_ratio)

    # Create a random permutation of indices
    idx = np.random.permutation(y_t_f.shape[0])

    # Split the indices into train and test
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]

    X_train = (x_t[0][train_idx], x_t[1][train_idx])
    X_test = (x_t[0][test_idx], x_t[1][test_idx])

    y_train_f = y_t_f[train_idx]
    y_test_f = y_t_f[test_idx]
    y_train_s = y_t_s[train_idx]
    y_test_s = y_t_s[test_idx]

    data_f = dde.data.Triple(
        X_train=X_train, y_train=y_train_f, X_test=X_test, y_test=y_test_f
    )
    data_s = dde.data.Triple(
        X_train=X_train, y_train=y_train_s, X_test=X_test, y_test=y_test_s
    )

    # see https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_unaligned.html
    # for the model

    # Choose a network
    net_f = dde.nn.deeponet.DeepONet(
        [2*num_b, 20, 40, 40, 40],
        [1, 20, 40],
        "tanh",
        "Glorot uniform",
    )
    net_s = dde.nn.deeponet.DeepONet(
        [2*num_b, 20, 40, 40, 40],
        [1, 20, 40],
        "tanh",
        "Glorot uniform",
    )

    # Define a Model
    mode_f = dde.Model(data_f, net_f)
    mode_s = dde.Model(data_s, net_s)

    # Compile and train.
    mode_f.compile("adam", lr=0.005, metrics=["mean l2 relative error"])
    mode_s.compile("adam", lr=0.005, metrics=["mean l2 relative error"])
    losshistory_f, train_state_f = mode_f.train(iterations=3000)
    losshistory_s, train_state_s = mode_s.train(iterations=3000)

    # Plot the loss trajectory
    dde.utils.plot_loss_history(losshistory_f)
    dde.utils.plot_loss_history(losshistory_s)
    fig_path = os.path.join(main_path, "train_loss_task3.png")
    plt.savefig(fig_path)

    # plot prediction on training data
    y_pred_f = mode_f.predict(x_t)
    y_pred_s = mode_s.predict(x_t)
    # denormalize the output
    y_pred_f = denormalize(y_pred_f, fluid_mean, fluid_std)
    y_pred_s = denormalize(y_pred_s, solid_mean, solid_std)


    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)

    y_true_f = data_r["tf0"].astype(np.float32).values[num_b:]
    y_true_s = data_r["ts0"].astype(np.float32).values[num_b:]

    axs[0].plot(data_r["t"].astype(np.float32).values[num_b:], y_true_f, label="True Fluid Solution")
    axs[1].plot(data_r["t"].astype(np.float32).values[num_b:], y_true_s, label="True Solid Solution")
    axs[0].plot(data_r["t"].astype(np.float32).values[num_b:], y_pred_f, label="Fluid Prediction")
    axs[1].plot(data_r["t"].astype(np.float32).values[num_b:], y_pred_s, label="Solid Prediction")
    axs[0].set_ylabel("Temperature")
    axs[1].set_ylabel("Temperature")
    axs[0].grid(True, which="both", ls=":")
    axs[1].grid(True, which="both", ls=":")
    axs[0].legend()
    axs[1].legend()
    axs[0].set_title("Fluid Temperature")
    axs[1].set_title("Solid Temperature")
    fig_path = os.path.join(main_path, "train_train_pred_task3.png")
    fig.savefig(fig_path)

    # prediction on test data
    # load the data
    data_path = os.path.join(main_path, "TestingData.txt")
    data_q = pd.read_csv(data_path, sep=",")

    # normalize the data
    data_norm_q = data_q.copy()
    data_norm_q["t"] = normalize(data_q["t"], time_mean, time_std)
    time = data_norm_q["t"].astype(np.float32).values

    # output buffer
    num_test = data_q["t"].shape[0]
    temp_fluid = np.empty((num_test, 1), dtype=np.float32)
    temp_solid = np.empty((num_test, 1), dtype=np.float32)

    # define the initial temperature
    temp_fluid_in = np.empty((1, num_b), dtype=np.float32)
    temp_solid_in = np.empty((1, num_b), dtype=np.float32)
    temp_fluid_in[0,:] = data_norm_r["tf0"].astype(np.float32).values[-num_b:]
    temp_solid_in[0,:] = data_norm_r["ts0"].astype(np.float32).values[-num_b:]

    # iterativly predict the temperature
    for i in range(num_test):
        time_in = time[i].reshape(-1,1)
        temp_in = np.concatenate((temp_fluid_in, temp_solid_in), axis=1).reshape(1,-1)
        y_p_f = mode_f.predict((temp_in, time_in))
        y_p_s = mode_s.predict((temp_in, time_in))

        # prepare next input
        temp_fluid_in = np.roll(temp_fluid_in, -1, axis=1)
        temp_solid_in = np.roll(temp_solid_in, -1, axis=1)
        temp_fluid_in[0,num_b-1] = y_p_f
        temp_solid_in[0,num_b-1] = y_p_s

        # denormalize the output
        temp_fluid[i] = denormalize(y_p_f, fluid_mean, fluid_std)
        temp_solid[i] = denormalize(y_p_s, solid_mean, solid_std)

    # plot the prediction
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)


    # concatenate with the previous predictions
    time_plt = np.concatenate((data_r["t"].astype(np.float32).values[num_b:], data_q["t"].astype(np.float32).values), axis=0)
    temp_fluid_plt = np.concatenate((y_pred_f, temp_fluid), axis=0)
    temp_solid_plt = np.concatenate((y_pred_s, temp_solid), axis=0)
    axs[0].plot(time_plt, temp_fluid_plt, label="Fluid Prediction")
    axs[1].plot(time_plt, temp_solid_plt, label="Solid Prediction")
    axs[0].set_ylabel("Temperature")
    axs[1].set_ylabel("Temperature")
    axs[0].grid(True, which="both", ls=":")
    axs[1].grid(True, which="both", ls=":")
    axs[0].legend()
    axs[1].legend()
    axs[0].set_title("Fluid Temperature")
    axs[1].set_title("Solid Temperature")
    fig_path = os.path.join(main_path, "train_test_pred_task3.png")
    fig.savefig(fig_path)

    # save the prediction
    data_q["tf0"] = temp_fluid
    data_q["ts0"] = temp_solid
    save_path = os.path.join(parent_path, "Task3.txt")
    data_q.to_csv(save_path, sep=",", index=False)
