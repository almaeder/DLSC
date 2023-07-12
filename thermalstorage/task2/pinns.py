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
import pandas as pd

main_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.join(main_path, ".."))
sys.path.append(parent_path)

import common

class Pinns:
    """
    Class to create pinns for the thermal storage equation of task2.
    """
    def __init__(
        self: object,
        n_int_: int,
        n_sb_: int,
        n_tb_: int,
        batchsize_int: int,
        batchsize_bc: int,
        batchsize_meas: int,
        device: torch.device
    ):
        """Initialize the PINN

        Args:
            n_int_  (int): Number of interior points
            n_sb_   (int): Number of spatial boundary points
            n_tb_   (int): Number of temporal boundary points
            batchsize_int   (int): Batchsize for interior points
            batchsize_bc    (int): Batchsize for boundary points
            batchsize_meas  (int): Batchsize for measurement points
            device  (torch.device): Device to use for training
        """
        # used device
        self.device = device

        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_
        self.batchsize_int = batchsize_int
        self.batchsize_bc = batchsize_bc
        self.batchsize_meas = batchsize_meas

        # Extrema of the solution domain (t,x) in [0,1]x[0,1]
        self.domain_extrema = torch.tensor([[0, 8],   # Time dimension
                                            [0, 1]])  # Space dimension
        
        self.phase_charge       = torch.tensor([[0,1],[4,5]])
        self.phase_discharge    = torch.tensor([[2,3],[6,7]])
        self.phase_idle         = torch.tensor([[1,2],[3,4],[5,6],[7,8]])

        # Number of space dimensions
        self.space_dimensions = 1

        # Parameter to balance role of data and PDE
        self.lambda_d = 20
        self.lambda_n = 20
        self.lambda_t = 200
        self.lambda_meas = 4000

        # Dense NN to approximate the solution of the underlying equations
        # todo tune network
        self.approximate_solution = common.NeuralNet(
            input_dimension=self.domain_extrema.shape[0], # x and t
            output_dimension=2, # T_f and T_s
            n_hidden_layers=5,
            neurons=85,
            regularization_param=0.1,
            regularization_exp=2.,
            retrain_seed=42
        ).to(device)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Physical paramters
        self.alphaf     = 0.005
        self.hf         = 5
        self.temp_hot   = 4
        self.temp_cold  = 1
        self.temp_0     = 1

        # Training sets as torch dataloader
        (self.training_set_sb_c_0, self.training_set_sb_c_l,
        self.training_set_sb_dc_0, self.training_set_sb_dc_l,
        self.training_set_sb_i_0, self.training_set_sb_i_l,
        self.training_set_tb, self.training_set_int, self.training_set_meas) = self.assemble_datasets()

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

    def convert_boundary(
        self: object,
        tens: torch.Tensor,
        domain_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert a tensor whose values are between 0 and 1 
        to a tensor whose values are between the spatial domain extrema
        and the input time domain

        Args:
            tens (torch.Tensor): Input tensor
            domain_time (torch.Tensor): Time range where the boundary points should be located

        Returns:
            (torch.Tensor): Output tensor
        """
        assert tens.shape[1] == self.domain_extrema.shape[0]
        assert domain_time.shape[0] == 2
        domain_end = torch.tensor([domain_time[1], self.domain_extrema[1,1]])
        domain_start = torch.tensor([domain_time[0], self.domain_extrema[1,0]])
        return tens * (domain_end - domain_start) + domain_start


    def masking_charge(
        self: object,
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        Return true for all the points inside the charging phase

        Args:
            time (torch.Tensor): Time tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # mask_charge = torch.logical_or(torch.logical_and(time >= 0, time < 1),
        #                                torch.logical_and(time >= 4, time < 5))
        mask_charge = (time.ge(0) * time.lt(1)) + (time.ge(4) * time.lt(5))
        return mask_charge

    def masking_discharge(
        self: object,
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        Return true for all the points inside the discharging phase

        Args:
            time (torch.Tensor): Time tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # mask_discharge = torch.logical_or(torch.logical_and(time >= 2, time < 3),
        #                                   torch.logical_and(time >= 6, time < 7))
        mask_discharge = (time.ge(2) * time.lt(3)) + (time.ge(6) * time.lt(7))
        #mask_discharge = ((time >= 2 + time < 3) == 1) + ((time >= 6 + time < 7) == 1) > 0
        return mask_discharge

    def masking_idle(
        self: object,
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        Return true for all the points inside the idle phase

        Args:
            time (torch.Tensor): Time tensor

        Returns:
            torch.Tensor: Output tensor
        """
        mask_idle = (time.ge(1) * time.lt(2)) + (time.ge(3) * time.lt(4)) + (time.ge(5) * time.lt(6)) + (time.ge(7) * time.le(8))
        # mask_idle = torch.logical_or(torch.logical_or(torch.logical_and(time >= 1, time < 2),
        #                                               torch.logical_and(time >= 3, time < 4)),
        #                              torch.logical_or(torch.logical_and(time >= 5, time < 6),
        #                                               torch.logical_and(time >= 7, time <= 8)))
        return mask_idle

    def fluid_velocity(
        self: object,
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the fluid velocity

        Args:
            time (torch.Tensor): Time tensor

        Returns:
            torch.Tensor: Output tensor
        """
        out = torch.empty_like(time)
        # masks for all the phases
        mask_charge     = self.masking_charge(time)
        mask_discharge  = self.masking_discharge(time)
        mask_idle       = self.masking_idle(time)

        # fill in the right values
        out.masked_fill_(mask_charge, 1)
        out.masked_fill_(mask_discharge, -1)
        out.masked_fill_(mask_idle, 0)

        return out

    ################################################################################################
    # Function returning the input-output tensor required
    # to assemble the training set S_tb corresponding to the temporal boundary
    def add_temporal_boundary_points(self):
        """
        Creates temporal boundary condition training
        points for both the fluid temperature.

        Returns:
            Tuple(
                torch.Tensor, Input for fluid
                torch.Tensor  Output for fluid
            )
        """
        # initial time
        t0 = self.domain_extrema[0, 0]

        # transform into temporal domain
        input_tb = self.convert(self.soboleng.draw(self.n_sb))

        # fix temporal coordinate and random spatial one
        # fluid tmp
        input_tb[:, 0]   = torch.full(input_tb[:, 0].shape, t0)

        # fix dirichlet conditions for both fluid
        output_tb = self.temp_0 * torch.ones((input_tb.shape[0]))

        return input_tb, output_tb

    # Function returning the input-output tensor required
    # to assemble the training set S_sb corresponding to the spatial boundary
    def add_sbp_charge(self):
        """Creates training data for spatial boundary conditions
        for the two charging phases
        Output are the concatenation of the two phases

        Returns:
            Tuple(
                torch.Tensor, Input spatial boundary condition at left boundary
                torch.Tensor, Output spatial boundary condition at left boundary
                torch.Tensor, Input spatial boundary condition at right boundary
                torch.Tensor  Output spatial boundary condition at right boundary
            )
        """
        # left and right boundary coordinates
        x_0 = self.domain_extrema[1, 0]
        x_l = self.domain_extrema[1, 1]

        # transform into temporal domain
        # the two charge phases are identical
        input_sb_first = self.convert_boundary(self.soboleng.draw(self.n_sb), self.phase_charge[0,:])
        input_sb_second = self.convert_boundary(self.soboleng.draw(self.n_sb), self.phase_charge[1,:])

        # concatenate the two phases
        input_sb = torch.cat((input_sb_first, input_sb_second), 0)

        # fix spatial coordinate and random temporal one
        input_sb_0         = torch.clone(input_sb)
        input_sb_0[:, 1]   = torch.full(input_sb_0[:, 1].shape, x_0)

        input_sb_l         = torch.clone(input_sb)
        input_sb_l[:, 1]   = torch.full(input_sb_l[:, 1].shape, x_l)

        # initialize both left and right boundary with zero
        output_sb_0        = self.temp_hot * torch.ones((input_sb.shape[0]))
        output_sb_l        = torch.zeros((input_sb.shape[0], 1))

        return input_sb_0, output_sb_0, input_sb_l, output_sb_l

    # Function returning the input-output tensor required
    # to assemble the training set S_sb corresponding to the spatial boundary
    def add_sbp_discharge(self):
        """Creates training data for spatial boundary conditions
        for the two discharging phases
        Output are the concatenation of the two phases

        Returns:
            Tuple(
                torch.Tensor, Input spatial boundary condition at left boundary
                torch.Tensor, Output spatial boundary condition at left boundary
                torch.Tensor, Input spatial boundary condition at right boundary
                torch.Tensor  Output spatial boundary condition at right boundary
            )
        """
        # left and right boundary coordinates
        x_0 = self.domain_extrema[1, 0]
        x_l = self.domain_extrema[1, 1]

        # transform into temporal domain
        # the two charge phases are identical
        input_sb_first = self.convert_boundary(self.soboleng.draw(self.n_sb), self.phase_discharge[0,:])
        input_sb_second = self.convert_boundary(self.soboleng.draw(self.n_sb), self.phase_discharge[1,:])

        # concatenate the two phases
        input_sb = torch.cat((input_sb_first, input_sb_second), 0)

        # fix spatial coordinate and random temporal one
        input_sb_0         = torch.clone(input_sb)
        input_sb_0[:, 1]   = torch.full(input_sb_0[:, 1].shape, x_0)

        input_sb_l         = torch.clone(input_sb)
        input_sb_l[:, 1]   = torch.full(input_sb_l[:, 1].shape, x_l)

        # initialize both left and right boundary with zero
        output_sb_0        = torch.zeros((input_sb.shape[0], 1))
        output_sb_l        = self.temp_cold * torch.ones((input_sb.shape[0]))

        return input_sb_0, output_sb_0, input_sb_l, output_sb_l

    # Function returning the input-output tensor required
    # to assemble the training set S_sb corresponding to the spatial boundary
    def add_sbp_idle(self):
        """Creates training data for spatial boundary conditions
        for the four idle phases
        Output are the concatenation of the phases

        Returns:
            Tuple(
                torch.Tensor, Input spatial boundary condition at left boundary
                torch.Tensor, Output spatial boundary condition at left boundary
                torch.Tensor, Input spatial boundary condition at right boundary
                torch.Tensor  Output spatial boundary condition at right boundary
            )
        """
        # left and right boundary coordinates
        x_0 = self.domain_extrema[1, 0]
        x_l = self.domain_extrema[1, 1]

        # transform into temporal domain
        # the two charge phases are identical
        input_sb_first  = self.convert_boundary(self.soboleng.draw(self.n_sb), self.phase_idle[0,:])
        input_sb_second = self.convert_boundary(self.soboleng.draw(self.n_sb), self.phase_idle[1,:])
        input_sb_third  = self.convert_boundary(self.soboleng.draw(self.n_sb), self.phase_idle[2,:])
        input_sb_fourth = self.convert_boundary(self.soboleng.draw(self.n_sb), self.phase_idle[3,:])

        # concatenate the two phases
        input_sb = torch.cat((input_sb_first, input_sb_second, input_sb_third, input_sb_fourth), 0)

        # fix spatial coordinate and random temporal one
        input_sb_0         = torch.clone(input_sb)
        input_sb_0[:, 1]   = torch.full(input_sb_0[:, 1].shape, x_0)

        input_sb_l         = torch.clone(input_sb)
        input_sb_l[:, 1]   = torch.full(input_sb_l[:, 1].shape, x_l)

        # initialize both left and right boundary with zero
        output_sb_0        = torch.zeros((input_sb.shape[0], 1))
        output_sb_l        = torch.zeros((input_sb.shape[0], 1))

        return input_sb_0, output_sb_0, input_sb_l, output_sb_l

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
        Creates nine dataloader for training.

        Returns:
            Left BC charge dataloader
            Right BC charge dataloader
            Left BC discharge dataloader
            Right BC discharge dataloader
            Left BC idle dataloader
            Right BC idle dataloader
            Temporal dataloader
            Interior dataloader
            Measurement dataloader
        """
        # create spatial/temporal BC and interior data
        # S_sb
        input_sb_c_0, output_sb_c_0, input_sb_c_l, output_sb_c_l = self.add_sbp_charge()
        input_sb_dc_0, output_sb_dc_0, input_sb_dc_l, output_sb_dc_l = self.add_sbp_discharge()
        input_sb_i_0, output_sb_i_0, input_sb_i_l, output_sb_i_l = self.add_sbp_idle()
        input_tb, output_tb     = self.add_temporal_boundary_points() # S_tb
        input_int, output_int   = self.add_interior_points() # S_int

        # load given solution
        solution_path = os.path.join(main_path, "DataSolution.txt")
        data = pd.read_csv(solution_path, sep=",")
        data = torch.tensor(data.values, dtype=torch.float32).to(self.device)
        input_meas = data[:,0:2]
        output_meas = data[:,2]

        # load data to device
        input_sb_c_0, output_sb_c_0 = input_sb_c_0.to(self.device), output_sb_c_0.to(self.device)
        input_sb_c_l, output_sb_c_l = input_sb_c_l.to(self.device), output_sb_c_l.to(self.device)
        input_sb_dc_0, output_sb_dc_0 = input_sb_dc_0.to(self.device), output_sb_dc_0.to(self.device)
        input_sb_dc_l, output_sb_dc_l = input_sb_dc_l.to(self.device), output_sb_dc_l.to(self.device)
        input_sb_i_0, output_sb_i_0 = input_sb_i_0.to(self.device), output_sb_i_0.to(self.device)
        input_sb_i_l, output_sb_i_l = input_sb_i_l.to(self.device), output_sb_i_l.to(self.device)

        input_tb, output_tb       = input_tb.to(self.device), output_tb.to(self.device)
        input_int, output_int     = input_int.to(self.device), output_int.to(self.device)

        # create dataloaders
        training_set_sb_c_0     = DataLoader(torch.utils.data.TensorDataset(input_sb_c_0, output_sb_c_0), batch_size=self.batchsize_bc*2, shuffle=False)
        training_set_sb_c_l     = DataLoader(torch.utils.data.TensorDataset(input_sb_c_l, output_sb_c_l), batch_size=self.batchsize_bc*2, shuffle=False)
        training_set_sb_dc_0     = DataLoader(torch.utils.data.TensorDataset(input_sb_dc_0, output_sb_dc_0), batch_size=self.batchsize_bc*2, shuffle=False)
        training_set_sb_dc_l     = DataLoader(torch.utils.data.TensorDataset(input_sb_dc_l, output_sb_dc_l), batch_size=self.batchsize_bc*2, shuffle=False)
        # not all the same amount points (idle 2x points) -> adapt batchsize
        training_set_sb_i_0     = DataLoader(torch.utils.data.TensorDataset(input_sb_i_0, output_sb_i_0), batch_size=self.batchsize_bc*4, shuffle=False)
        training_set_sb_i_l     = DataLoader(torch.utils.data.TensorDataset(input_sb_i_l, output_sb_i_l), batch_size=self.batchsize_bc*4, shuffle=False)

        training_set_tb         = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb),   batch_size=self.batchsize_bc, shuffle=False)
        training_set_int        = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.batchsize_int, shuffle=False)

        training_set_meas       = DataLoader(torch.utils.data.TensorDataset(input_meas, output_meas), batch_size=self.batchsize_meas, shuffle=False)

        return (training_set_sb_c_0, training_set_sb_c_l,
                training_set_sb_dc_0, training_set_sb_dc_l,
                training_set_sb_i_0, training_set_sb_i_l,
                training_set_tb, training_set_int, training_set_meas)

    ################################################################################################
    def compute_boundary_residual(
        self: object,
        input_c_l: torch.Tensor,
        input_dc_0: torch.Tensor,
        input_i_0: torch.Tensor,
        input_i_l: torch.Tensor
    ) -> typing.Tuple[torch.Tensor,
                      torch.Tensor,
                      torch.Tensor,
                      torch.Tensor]:
        """
        Calculates the residual for the neumann boundary conditions

        Args:
            input_c_l (torch.Tensor): right points for charge phase
            input_dc_0 (torch.Tensor): left points for discharge phase
            input_i_0 (torch.Tensor): left points for idle phase
            input_i_l (torch.Tensor): right points for idle phase

        Returns:
            typing.Tuple[
                torch.Tensor, right charge residual
                torch.Tensor, left discharge residual
                torch.Tensor, left idle residual
                torch.Tensor  right idle residual
            ]
        """
        # enable auto differentiation
        input_c_l.requires_grad = True
        input_dc_0.requires_grad = True
        input_i_0.requires_grad = True
        input_i_l.requires_grad = True

        # two dimensional output of tf and ts
        temp_c_l = self.approximate_solution(input_c_l)
        temp_dc_0 = self.approximate_solution(input_dc_0)
        temp_i_0 = self.approximate_solution(input_i_0)
        temp_i_l = self.approximate_solution(input_i_l)

        # split output into fluid and solid temperature
        tf_c_l = temp_c_l[:,0]
        tf_dc_0 = temp_dc_0[:,0]
        tf_i_0 = temp_i_0[:,0]
        tf_i_l = temp_i_l[:,0]

        # need three gradients according to boundary condition
        # which are directly the residuals
        grad_d_l = torch.autograd.grad(tf_c_l.sum(), input_c_l, create_graph=True)[0]
        grad_dc_0 = torch.autograd.grad(tf_dc_0.sum(), input_dc_0, create_graph=True)[0]
        grad_i_0 = torch.autograd.grad(tf_i_0.sum(), input_i_0, create_graph=True)[0]
        grad_i_l = torch.autograd.grad(tf_i_l.sum(), input_i_l, create_graph=True)[0]

        return grad_d_l.reshape(-1, ), grad_dc_0.reshape(-1, ), grad_i_0.reshape(-1, ), grad_i_l.reshape(-1, )

    # Function to compute the PDE residuals
    def compute_pde_residual(
        self: object,
        input_int: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the residuals for the equation.

        Args:
            input_int (torch.Tensor): Input points

        Returns:
            torch.Tensor: residual of both equations of the system
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

        # calculate the different gradients
        grad_tf_t = grad_tf[:, 0]
        grad_tf_x = grad_tf[:, 1]

        grad_tf_xx = torch.autograd.grad(grad_tf_x.sum(), input_int, create_graph=True)[0][:, 1]

        # compute difference between temperatures
        diff_t = tf - ts

        # get fluid velocity
        uf = self.fluid_velocity(input_int[:,0])

        # the residuals for the two equations described in the README
        residual_tf = grad_tf_t + uf * grad_tf_x - self.alphaf * grad_tf_xx + self.hf * diff_t

        return residual_tf.reshape(-1, )

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(
            self: object,
            inp_train_sb_c_0: torch.Tensor,
            inp_train_sb_c_l: torch.Tensor,
            inp_train_sb_dc_0: torch.Tensor,
            inp_train_sb_dc_l: torch.Tensor,
            inp_train_sb_i_0: torch.Tensor,
            inp_train_sb_i_l: torch.Tensor,
            out_train_sb_c_0: torch.Tensor,
            out_train_sb_dc_l: torch.Tensor,
            inp_train_tb: torch.Tensor,
            out_train_tb: torch.Tensor,
            inp_train_int: torch.Tensor,
            inp_train_meas: torch.Tensor,
            out_train_meas: torch.Tensor,
            verbose=True
        ) -> torch.Tensor:
        """Calculates the total loss of the system.

        Args:
            inp_train_sb_c_0 (torch.Tensor): Charge input boundary points at x=0
            inp_train_sb_c_l (torch.Tensor): Charge input boundary points at x=L
            inp_train_sb_dc_0 (torch.Tensor): Discharge input boundary points at x=0
            inp_train_sb_dc_l (torch.Tensor): Discharge input boundary points at x=L
            inp_train_sb_i_0 (torch.Tensor): Idle boundary points at x=0
            inp_train_sb_i_l (torch.Tensor): Idle boundary points at x=L
            out_train_sb_c_0 (torch.Tensor): Charge output boundary points at x=0
            out_train_sb_dc_l (torch.Tensor): Discharge output boundary points at x=L
            inp_train_tb (torch.Tensor): Temporal boundary points
            out_train_tb (torch.Tensor): Temporal output points
            inp_train_int (torch.Tensor): Interior points
            verbose (bool, optional): If loss should be printed. Defaults to True.

        Returns:
            torch.Tensor: _description_
        """
        # predict the temperatures
        # only fluid temperature needed for loss
        out_pred_sb_c_0  = self.approximate_solution(inp_train_sb_c_0)[:,0]
        out_pred_sb_dc_l  = self.approximate_solution(inp_train_sb_dc_l)[:,0]
        out_pred_tb     = self.approximate_solution(inp_train_tb)[:,0]
        out_pred_meas     = self.approximate_solution(inp_train_meas)[:,0]

        # a bit of sanity
        assert out_pred_sb_c_0.shape == out_train_sb_c_0.shape
        assert out_pred_sb_dc_l.shape == out_train_sb_dc_l.shape
        assert out_pred_tb.shape    == out_train_tb.shape
        assert out_train_meas.shape    == out_pred_meas.shape


        # get the different residuals
        r_int_tf = self.compute_pde_residual(inp_train_int)

        r_sb_c_l, r_sb_dc_0, r_sb_i_0, r_sb_i_l = self.compute_boundary_residual(inp_train_sb_c_l,
                                                                                 inp_train_sb_dc_0,
                                                                                 inp_train_sb_i_0,
                                                                                 inp_train_sb_i_l)
        r_sb_c_0    = out_train_sb_c_0 - out_pred_sb_c_0
        r_sb_dc_l   = out_train_sb_dc_l - out_pred_sb_dc_l
        r_tb        = out_train_tb - out_pred_tb
        r_meas      = out_train_meas - out_pred_meas

        # spatial dirichlet/neumann, temporal, interior residuals
        loss_sb_d   = torch.mean(abs(r_sb_c_0) ** 2) + torch.mean(abs(r_sb_dc_l) ** 2)
        loss_sb_n   = (torch.mean(abs(r_sb_c_l) ** 2) +
                       torch.mean(abs(r_sb_dc_0) ** 2) +
                       torch.mean(abs(r_sb_i_0) ** 2) +
                       torch.mean(abs(r_sb_i_l) ** 2))
        loss_tb     = torch.mean(abs(r_tb) ** 2)
        loss_int    = torch.mean(abs(r_int_tf) ** 2)
        loss_meas   = torch.mean(abs(r_meas)**2)


        # total loss
        loss = torch.log10(self.lambda_d * loss_sb_d +
                           self.lambda_n * loss_sb_n +
                           self.lambda_t * loss_tb +
                           loss_int +
                           self.lambda_meas * loss_meas)

        if verbose:
            print("Total loss: ",       round(loss.item(), 4))
            # scale certain terms for fair comparison

            print("Boundary Loss: ",    round(torch.log10(self.lambda_d * loss_sb_d + self.lambda_n * loss_sb_n + self.lambda_t * loss_tb).item(), 4))
            print("Function Loss: ",    round(torch.log10(loss_int).item(), 4))
            print("Temporal Loss: ",    round(torch.log10(self.lambda_t *(loss_tb)).item(), 4))
            print("Dirichlet Loss: ",   round(torch.log10(self.lambda_d *(loss_sb_d)).item(), 4))
            print("Neumann Loss: ",     round(torch.log10(self.lambda_n *(loss_sb_n)).item(), 4))
            print("Measurement Loss: ", round(torch.log10(self.lambda_meas *(loss_meas)).item(), 4))
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

            # Loop over batches
            for _, ((inp_train_sb_c_0, out_train_sb_c_0), (inp_train_sb_c_l, _),
                    (inp_train_sb_dc_0, _), (inp_train_sb_dc_l, out_train_sb_dc_l),
                    (inp_train_sb_i_0, _), (inp_train_sb_i_l, _),
                    (inp_train_tb, out_train_tb),
                    (inp_train_int, _),
                    (inp_train_meas, out_train_meas),
                    ) in enumerate(zip(
                                                    self.training_set_sb_c_0,
                                                    self.training_set_sb_c_l,
                                                    self.training_set_sb_dc_0,
                                                    self.training_set_sb_dc_l,
                                                    self.training_set_sb_i_0,
                                                    self.training_set_sb_i_l,
                                                    self.training_set_tb,
                                                    self.training_set_int,
                                                    self.training_set_meas
                                                    )):
                def closure():

                    verbose=True
                    optimizer.zero_grad()
                    loss = self.compute_loss(
                                inp_train_sb_c_0,
                                inp_train_sb_c_l,
                                inp_train_sb_dc_0,
                                inp_train_sb_dc_l,
                                inp_train_sb_i_0,
                                inp_train_sb_i_l,
                                out_train_sb_c_0,
                                out_train_sb_dc_l,
                                inp_train_tb,
                                out_train_tb,
                                inp_train_int,
                                inp_train_meas,
                                out_train_meas,
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
        name: str = "plot_task2.png"
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
