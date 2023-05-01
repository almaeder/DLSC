"""
PINNs for the given problem of the thermal storage equations.
Created by Roberto Molinaro and others, 21.03.2023
Adapted by Alexander Maeder, Leonard Deuschle
"""
import torch
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import typing
import matplotlib.pyplot as plt
import os
import sys

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, ".."))
sys.path.append(parent_path)

import common

class Pinns:
    """
    Class to create pinns for the thermal storage equation of task1.
    """
    def __init__(
        self: object,
        n_int_: int,
        n_sb_: int,
        n_tb_: int,
        batchsize_int: int,
        batchsize_bc: int,
        device: torch.device
    ):
        """Initialize the PINN

        Args:
            n_int_  (int): Number of interior points
            n_sb_   (int): Number of spatial boundary points
            n_tb_   (int): Number of temporal boundary points
            batchsize_int   (int): Batchsize for interior points
            batchsize_bc    (int): Batchsize for boundary points
            device  (torch.device): Device to use for training
        """
        # used device
        self.device = device

        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_
        self.batchsize_int = batchsize_int
        self.batchsize_bc = batchsize_bc

        # Extrema of the solution domain (t,x) in [0,1]x[0,1]
        self.domain_extrema = torch.tensor([[0, 1],   # Time dimension
                                            [0, 1]])  # Space dimension

        # Number of space dimensions
        self.space_dimensions = 1

        # Parameter to balance role of data and PDE
        self.lambda_d = 20
        self.lambda_n = 20
        self.lambda_t = 20

        # Dense NN to approximate the solution of the underlying equations
        # todo tune network
        self.approximate_solution = common.NeuralNet(
            input_dimension=self.domain_extrema.shape[0], # x and t
            output_dimension=2, # T_f and T_s
            n_hidden_layers=5,
            neurons=100,
            regularization_param=0.,
            regularization_exp=2.,
            retrain_seed=42
        ).to(device)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Physical paramters
        self.alphaf     = 0.05
        self.alphas     = 0.08
        self.hf         = 5
        self.hs         = 6
        self.temp_hot   = 4
        self.temp_0     = 1
        self.uf         = 1

        # Training sets as torch dataloader
        self.training_set_sb_n0, self.training_set_sb_nl, self.training_set_sb_d0, self.training_set_tb, self.training_set_int = self.assemble_datasets()

    ############################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(
        self: object,
        tens: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert a tensor whose values are between 0 and 1 
        to a tensor whose values are between the domain extrema

        Args:
            tens (torch.Tensor): Input tensor

        Returns:
            (torch.Tensor): Output tensor
        """
        assert tens.shape[1] == self.domain_extrema.shape[0]
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    # Initial conditions
    def dirichlet_condition_tf(
        self: object,
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the dirichlet boundary condition
        for the fluid temperature

        Args:
            time (torch.Tensor): Time tensor

        Returns:
            torch.Tensor: Output tensor
        """
        out = (torch.div(self.temp_hot - self.temp_0,
                         1 + torch.exp(-200*(time - 0.25))) + self.temp_0)
        return out


    ################################################################################################
    # Function returning the input-output tensor required
    # to assemble the training set S_tb corresponding to the temporal boundary
    def add_temporal_boundary_points(self):
        """
        Creates temporal boundary condition training
        points for both the fluid and solid temperature.
        Both share the same boundary condition

        Returns:
            Tuple(
                torch.Tensor, Input for fluid/solid
                torch.Tensor  Output for fluid/solid
            )
        """
        # initial time
        t0 = self.domain_extrema[0, 0]

        # transform into temporal domain
        input_tb = self.convert(self.soboleng.draw(self.n_sb))

        # fix temporal coordinate and random spatial one
        # fluid tmp
        input_tb[:, 0]   = torch.full(input_tb[:, 0].shape, t0)

        # fix dirichlet conditions for both fluid and solid
        output_tb = self.temp_0 * torch.ones((input_tb.shape[0], 2))

        return input_tb, output_tb

    # Function returning the input-output tensor required
    # to assemble the training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        """Creates training data for spatial boundary conditions.
        There is a dirichlet boundary condition 
        at only the left boundary for the fluid temperature.
        At the other boundaries
        both fluid and solid have homogeneous neumann boundary conditions.

        Returns:
            Tuple(
                torch.Tensor, Input homogeneous neumann at left boundary
                torch.Tensor, Output homogeneous neumann  at left boundary
                torch.Tensor, Input homogeneous neumann at right boundary
                torch.Tensor, Output homogeneous neumann at right boundary
                torch.Tensor, Input dirichlet at left boundary
                torch.Tensor  Output dirichlet  at left boundary
            )
        """
        # left and right boundary coordinates
        x_0 = self.domain_extrema[1, 0]
        x_l = self.domain_extrema[1, 1]

        # transform into temporal domain
        input_sb = self.convert(self.soboleng.draw(self.n_sb))

        # fix spatial coordinate and random temporal one
        input_sb_n0         = torch.clone(input_sb)
        input_sb_n0[:, 1]   = torch.full(input_sb_n0[:, 1].shape, x_0)

        input_sb_nl         = torch.clone(input_sb)
        input_sb_nl[:, 1]   = torch.full(input_sb_nl[:, 1].shape, x_l)

        # both left and right boundary should be zero
        output_sb_n0        = torch.zeros((input_sb.shape[0], 1))
        output_sb_nl        = torch.zeros((input_sb.shape[0], 1))

        # dirichlet inputs/outpus
        input_sb_d0         = torch.clone(input_sb)
        input_sb_d0[:, 1]   = torch.full(input_sb_d0[:, 1].shape, x_0)
        output_sb_d0        = self.dirichlet_condition_tf(input_sb[:, 0])

        return input_sb_n0, output_sb_n0, input_sb_nl, output_sb_nl, input_sb_d0, output_sb_d0

    # Function returning the input-output tensor required
    # to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        """
        Creates sobol random interior points.
        As the solution is not know, output is just zero

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Input and Output data
        """
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        """
        Creates five dataloader for training.

        Returns:
            Left neumann dataloader
            Right neumann dataloader
            Left dirichlet dataloader
            Temporal dataloader
            Interior dataloader
        """
        input_sb_n0, output_sb_n0, input_sb_nl, output_sb_nl, input_sb_d0, output_sb_d0 = self.add_spatial_boundary_points()   # S_sb
        input_tb, output_tb     = self.add_temporal_boundary_points()  # S_tb
        input_int, output_int   = self.add_interior_points()         # S_int

        # load data to device
        input_sb_n0, output_sb_n0 = input_sb_n0.to(self.device), output_sb_n0.to(self.device)
        input_sb_nl, output_sb_nl = input_sb_nl.to(self.device), output_sb_nl.to(self.device)
        input_sb_d0, output_sb_d0 = input_sb_d0.to(self.device), output_sb_d0.to(self.device)
        input_tb, output_tb       = input_tb.to(self.device), output_tb.to(self.device)
        input_int, output_int     = input_int.to(self.device), output_int.to(self.device)

        # create dataloaders
        training_set_sb_n0      = DataLoader(torch.utils.data.TensorDataset(input_sb_n0, output_sb_n0), batch_size=self.batchsize_bc, shuffle=False)
        training_set_sb_nl      = DataLoader(torch.utils.data.TensorDataset(input_sb_nl, output_sb_nl), batch_size=self.batchsize_bc, shuffle=False)
        training_set_sb_d0      = DataLoader(torch.utils.data.TensorDataset(input_sb_d0, output_sb_d0), batch_size=self.batchsize_bc, shuffle=False)

        training_set_tb         = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb),   batch_size=self.batchsize_bc, shuffle=False)
        training_set_int        = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.batchsize_int, shuffle=False)

        return training_set_sb_n0, training_set_sb_nl, training_set_sb_d0, training_set_tb, training_set_int

    ################################################################################################
    def compute_boundary_residual(
        self: object,
        input_0: torch.Tensor,
        input_l: torch.Tensor
    ) -> typing.Tuple[torch.Tensor,
                      torch.Tensor,
                      torch.Tensor]:
        """
        Calculates the residual for the three neumann boundary conditions

        Args:
            input_0 (torch.Tensor): left/0 input points
            input_l (torch.Tensor): right/l input points

        Returns:
            typing.Tuple[
                torch.Tensor, solid left residual
                torch.Tensor, fluid right residual
                torch.Tensor  solid right residual
            ]
        """
        # enable auto differentiation
        input_0.requires_grad = True
        input_l.requires_grad = True

        # two dimensional output of tf and ts
        temp_0 = self.approximate_solution(input_0)
        temp_l = self.approximate_solution(input_l)

        # split output into fluid and solid temperature
        ts_0 = temp_0[:,1]
        tf_l = temp_l[:,0]
        ts_l = temp_l[:,1]

        # need three gradients according to boundary condition
        # which are directly the residuals
        grad_ts_0 = torch.autograd.grad(ts_0.sum(), input_0, create_graph=True)[0]
        grad_tf_l = torch.autograd.grad(tf_l.sum(), input_l, create_graph=True)[0]
        grad_ts_l = torch.autograd.grad(ts_l.sum(), input_l, create_graph=True)[0]

        return grad_ts_0.reshape(-1, ), grad_tf_l.reshape(-1, ), grad_ts_l.reshape(-1, )

    # Function to compute the PDE residuals
    def compute_pde_residual(
        self: object,
        input_int: torch.Tensor
    ):
        """
        Calculates the two residuals of the system of equations.

        Args:
            input_int (torch.Tensor): Input points

        Returns:
            Tuple(torch.Tensor, torch.Tensor): residual of both equations of the system
        """
        # enable auto differentiation
        input_int.requires_grad = True

        # two dimensional ouput of tf and ts
        temp = self.approximate_solution(input_int)
        # split output into fluid and solid temperature
        tf = temp[:,0]
        ts = temp[:,1]

        # grad compute the gradient of a "SCALAR" function L with respect to some input nxm TENSOR Z=[[x1, y1],[x2,y2],[x3,y3],...,[xn,yn]], m=2
        # it returns grad_L = [[dL/dx1, dL/dy1],[dL/dx2, dL/dy2],[dL/dx3, dL/dy3],...,[dL/dxn, dL/dyn]]
        # Note: pytorch considers a tensor [u1, u2,u3, ... ,un] a vectorial function
        # whereas sum_u = u1 + u2 + u3 + u4 + ... + un as a "scalar" one

        # In our case ui = u(xi), therefore the line below returns:
        # grad_u = [[dsum_u/dx1, dsum_u/dy1],[dsum_u/dx2, dsum_u/dy2],[dsum_u/dx3, dL/dy3],...,[dsum_u/dxm, dsum_u/dyn]]
        # and dsum_u/dxi = d(u1 + u2 + u3 + u4 + ... + un)/dxi = d(u(x1) + u(x2) u3(x3) + u4(x4) + ... + u(xn))/dxi = dui/dxi
        grad_tf = torch.autograd.grad(tf.sum(), input_int, create_graph=True)[0]
        grad_ts = torch.autograd.grad(ts.sum(), input_int, create_graph=True)[0]

        # calculate the different gradients
        grad_tf_t = grad_tf[:, 0]
        grad_tf_x = grad_tf[:, 1]
        grad_ts_t = grad_ts[:, 0]
        grad_ts_x = grad_ts[:, 1]
        grad_tf_xx = torch.autograd.grad(grad_tf_x.sum(), input_int, create_graph=True)[0][:, 1]
        grad_ts_xx = torch.autograd.grad(grad_ts_x.sum(), input_int, create_graph=True)[0][:, 1]
        # compute difference between temperatures
        diff_t = tf - ts

        # the residuals for the two equations described in the README
        residual_tf = grad_tf_t + self.uf * grad_tf_x - self.alphaf * grad_tf_xx + self.hf * diff_t
        residual_ts = grad_ts_t                       - self.alphas * grad_ts_xx - self.hs * diff_t
        return residual_tf.reshape(-1, ), residual_ts.reshape(-1, )

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(
            self: object,
            inp_train_sb_n0: torch.Tensor,
            inp_train_sb_nl: torch.Tensor,
            inp_train_sb_d0: torch.Tensor,
            out_train_sb_d0: torch.Tensor,
            inp_train_tb: torch.Tensor,
            out_train_tb: torch.Tensor,
            inp_train_int: torch.Tensor,
            verbose=True
        ) -> torch.Tensor:
        """Calculates the loss of the system of equations.

        Args:
            inp_train_sb_n0 (torch.Tensor): Left neumann training data
            inp_train_sb_nl (torch.Tensor): Right neumann training data
            inp_train_sb_d0 (torch.Tensor): Left dirichlet training data
            out_train_sb_d0 (torch.Tensor): Left dirichlet training data
            inp_train_tb (torch.Tensor): Temporal boundary training data
            out_train_tb (torch.Tensor): Temporal boundary training data
            inp_train_int (torch.Tensor): Interior training data
            verbose (bool, optional): Verbosity. Defaults to True.

        Returns:
            (torch.Tensor): Loss
        """
        # predict the different temperatures
        out_pred_sb_d0  = self.approximate_solution(inp_train_sb_d0)[:,0] # only fluid temperature
        out_pred_tb     = self.approximate_solution(inp_train_tb)

        # a bit of sanity
        assert out_pred_sb_d0.shape == out_train_sb_d0.shape
        assert out_pred_tb.shape    == out_train_tb.shape


        # get the different residuals
        r_int_tf, r_int_ts = self.compute_pde_residual(inp_train_int)
        r_sb_ts_0, r_sb_tf_l, r_sb_ts_l = self.compute_boundary_residual(inp_train_sb_n0, inp_train_sb_nl)
        r_sb_d0 = out_train_sb_d0 - out_pred_sb_d0
        r_tb    = out_train_tb - out_pred_tb

        # spatial dirichlet/neumann, temporal, interior residuals
        loss_sb_d   = torch.mean(abs(r_sb_d0) ** 2)
        loss_sb_n   = torch.mean(abs(r_sb_ts_0) ** 2) + torch.mean(abs(r_sb_tf_l) ** 2) + torch.mean(abs(r_sb_ts_l) ** 2)
        loss_tb     = torch.mean(abs(r_tb[:,0]) ** 2)  + torch.mean(abs(r_tb[:,1]) ** 2)
        loss_int    = torch.mean(abs(r_int_tf) ** 2) + torch.mean(abs(r_int_ts) ** 2)

        # total loss
        loss = torch.log10(self.lambda_d * loss_sb_d + self.lambda_n * loss_sb_n + self.lambda_t * loss_tb + loss_int)

        if verbose:
            print("Total loss: ",     round(loss.item(), 4))
            # scale certain terms for fair comparison

            print("Boundary Loss: ",  round(torch.log10(self.lambda_d * loss_sb_d + self.lambda_n * loss_sb_n + self.lambda_t * loss_tb).item(), 4))
            print("Function Loss: ",  round(torch.log10(loss_int).item(), 4))
            print("Temporal Loss: ",  round(torch.log10(self.lambda_t *(loss_tb)).item(), 4))
            print("Dirichlet Loss: ", round(torch.log10(self.lambda_d *(loss_sb_d)).item(), 4))
            print("Neumann Loss: ",   round(torch.log10(self.lambda_n *(loss_sb_n)).item(), 4))
        return loss

    ############################################################################
    def fit(
        self: object,
        num_epochs: int,
        optimizer: Optimizer,
        verbose: str = True
    ) -> typing.List[float]:
        """
        Trains the model

        Args:
            num_epochs (int): Number of epochs to train
            optimizer (Optimizer): Which optimizer to use
            verbose (str, optional): If additional information should be printed. Defaults to True.

        Returns:
            typing.List[float]: List of losses every epoch
        """
        history = []

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            # get the different training data
            for _, ((inp_train_sb_n0, _), (inp_train_sb_nl, _), (inp_train_sb_d0, out_train_sb_d0), (inp_train_tb, out_train_tb), (inp_train_int, _)) in enumerate(zip(
                                                                                                                    self.training_set_sb_n0,
                                                                                                                    self.training_set_sb_nl,
                                                                                                                    self.training_set_sb_d0,
                                                                                                                    self.training_set_tb,
                                                                                                                    self.training_set_int
                                                                                                                    )):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(
                                            inp_train_sb_n0,
                                            inp_train_sb_nl,
                                            inp_train_sb_d0,
                                            out_train_sb_d0,
                                            inp_train_tb,
                                            out_train_tb,
                                            inp_train_int,
                                            verbose=verbose
                                            )
                    loss.backward()

                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

        print("Final Loss: ", history[-1])

        return history

    ################################################################################################
    def plotting(
        self: object,
        name: str = "plot_task1.png"
    ):
        """
        Plot the learned fluid and solid temperature

        Args:
            name (str, optional): Name of the plot. Defaults to "plot.png".
        """
        inputs = self.soboleng.draw(100000)
        inputs = self.convert(inputs)
        output = self.approximate_solution(inputs.to(self.device)).detach().cpu()

        # plot both fluid and solid temperature
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=output[:,0], cmap="jet", vmin=1, vmax=4)
        im2 = axs[1].scatter(inputs[:, 1].detach(), inputs[:, 0].detach(), c=output[:,1], cmap="jet", vmin=1, vmax=4)

        # set the labels
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        axs[0].grid(True, which="both", ls=":")
        axs[1].grid(True, which="both", ls=":")

        # create a colorbar
        cbar = fig.colorbar(im1, ax=axs)
        cbar.set_ticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

        # set the titles
        axs[0].set_title("Fluid Temperature")
        axs[1].set_title("Solid Temperature")
        
        plt.show()
        fig.savefig(name)
