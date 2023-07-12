import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch import optim
import common
import typing
import matplotlib.pyplot as plt
import copy
class Pinns:
    """
    Class to create pinns for the thermal storage equation of task2.
    """
    def __init__(
        self: object,
        n_int: int,
        n_sb: int,
        batchsize_int: int,
        batchsize_sb: int,
        xl: float,
        xr: float,
        ub: float,
        device: torch.device
    ):
        self.device = device
        self.n_int = n_int
        self.n_sb = n_sb
        self.batchsize_int = batchsize_int
        self.batchsize_sb = batchsize_sb
        self.ub = ub
        self.domain_extrema = torch.tensor([[xl, xr]])
        self.domain_extrema = self.domain_extrema.to(self.device)
        self.space_dimensions = 1.0
        self.alpha_sb = 10.0
        self.alpha_norm  = 4000.0
        self.alpha_ortho = 1000.0
        self.alpha_drive = 200
        self.alpha_regu = 1
        self.c = 20.0
        self.num_eigenfunctions = 4
        self.num_layer = 2
        self.size_layer = 20
        self.approximate_solution = common.NeuralNet(
            input_dimension=self.domain_extrema.shape[0],
            output_dimension=1,
            n_hidden_layers=self.num_layer,
            neurons=self.size_layer,
            regularization_param=0.02,
            regularization_exp=2.,
            retrain_seed=42,
            eigenvalue_init=3.5,
            device=device
        ).to(device)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        self.training_set_int, self.training_set_sbl, self.training_set_sbr = self.assemble_datasets()
        self.eigenfunctions = []


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
        if tens.is_cuda:
            return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

        return tens * (self.domain_extrema[:, 1].cpu() - self.domain_extrema[:, 0].cpu()) + self.domain_extrema[:, 0].cpu()


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
    

    def add_boundary_points(self):
        input_sbl = self.domain_extrema[:,0]*torch.ones((self.n_sb, self.domain_extrema.shape[0]), device=self.device)
        input_sbr = self.domain_extrema[:,1]*torch.ones((self.n_sb, self.domain_extrema.shape[0]), device=self.device)
        output_sbl = torch.zeros((input_sbl.shape[0], 1), device=self.device)
        output_sbr = torch.zeros((input_sbr.shape[0], 1), device=self.device)
        return input_sbl, input_sbr, output_sbl, output_sbr

    def assemble_datasets(self):
        """
        Creates Dataloader

        Returns:
            Interior dataloader
        """
        # interior data, (spatial boundary conditions handled through ansatz, did not work yet)
        input_int, output_int   = self.add_interior_points()
        input_sbl, input_sbr, output_sbl, output_sbr = self.add_boundary_points()

        # load data to device
        input_int, output_int     = input_int.to(self.device), output_int.to(self.device)
        # input_sbl, input_sbr      = input_sbl.to(self.device), input_sbr.to(self.device)
        # output_sbl, output_sbr    = output_sbl.to(self.device), output_sbr.to(self.device)

        # create dataloader
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.batchsize_int, shuffle=False)
        training_set_sbl = DataLoader(torch.utils.data.TensorDataset(input_sbl, output_sbl), batch_size=self.batchsize_sb, shuffle=False)
        training_set_sbr = DataLoader(torch.utils.data.TensorDataset(input_sbr, output_sbr), batch_size=self.batchsize_sb, shuffle=False)

        return training_set_int, training_set_sbl, training_set_sbr

    def compute_ansatz(
        self: object,
        input_int: torch.Tensor,
        nn: torch.Tensor
    ) -> torch.Tensor:
        # return nn
        return (nn*
                (1-torch.exp(-10*(input_int-self.domain_extrema[:,0])))*
                (1-torch.exp(10*(input_int-self.domain_extrema[:,1]))) + self.ub)
    


    def compute_pde_residual(
        self: object,
        input_int: torch.Tensor,
        func: torch.Tensor,
        eigenvalue: torch.Tensor
    ) -> torch.Tensor:
        # grad compute the gradient of a "SCALAR" function L with respect to some input nxm TENSOR Z=[[x1, y1],[x2,y2],[x3,y3],...,[xn,yn]], m=2
        # it returns grad_L = [[dL/dx1, dL/dy1],[dL/dx2, dL/dy2],[dL/dx3, dL/dy3],...,[dL/dxn, dL/dyn]]
        # Note: pytorch considers a tensor [u1, u2,u3, ... ,un] a vectorial function
        # whereas sum_u = u1 + u2 + u3 + u4 + ... + un as a "scalar" one

        # In our case ui = u(xi), therefore the line below returns:
        # grad_u = [[dsum_u/dx1, dsum_u/dy1],[dsum_u/dx2, dsum_u/dy2],[dsum_u/dx3, dL/dy3],...,[dsum_u/dxm, dsum_u/dyn]]
        # and dsum_u/dxi = d(u1 + u2 + u3 + u4 + ... + un)/dxi = d(u(x1) + u(x2) u3(x3) + u4(x4) + ... + u(xn))/dxi = dui/dxi
        grad_func = torch.autograd.grad(func.sum(), input_int, create_graph=True)[0]

        grad_func_x = grad_func[:, 0]

        grad_func_xx = torch.autograd.grad(grad_func_x.sum(), input_int, create_graph=True)[0][:, 0]

        residual_pde = grad_func_xx + eigenvalue**2 * func

        return residual_pde


    def compute_drive_loss(
        self: object,
        eigenvalue: torch.Tensor
    ) -> torch.Tensor:
        # lower and upper limit for eigenvalue
        return torch.exp(10*(-self.c + eigenvalue)) + torch.exp(-10*eigenvalue)

    def compute_norm_loss(
        self: object,
        func: torch.Tensor
    ) -> torch.Tensor:
        num_samples = func.shape[0]
        norm = torch.sum(func**2)/num_samples
        print("Norm: ",  round(norm.item(), 4))
        return (norm - 1)**2

    def compute_ortho_loss(
        self: object,
        input_int: torch.Tensor,
        func: torch.Tensor
    ) -> torch.Tensor:
        num_samples = func.shape[0]
        loss = torch.zeros(1, device=self.device)
        func_sum_prev = torch.zeros_like(func, requires_grad=False)
        for eigenfunction in self.eigenfunctions:
            func_prev = self.compute_ansatz(input_int,eigenfunction(input_int))
            ortho = torch.sum(func*func_prev)/num_samples
            func_sum_prev += self.compute_ansatz(input_int,eigenfunction(input_int))
            loss += (ortho)**2
            print("Ortho: ",  round(ortho.item(), 4))

        return loss + (torch.sum(func*func_sum_prev))**2

    def compute_loss(
        self: object,
        input_int: torch.Tensor,
        input_sbl: torch.Tensor,
        input_sbr: torch.Tensor,
        output_sbl: torch.Tensor,
        output_sbr: torch.Tensor,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Computes the residual for the harmonic oscillator eigenvalue equation

        Args:
            self (object): _description_
            input_int (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """

        # enable auto differentiation
        input_int.requires_grad = True

        func = self.compute_ansatz(input_int,self.approximate_solution(input_int))

        eigenvalue = self.approximate_solution.eigenvalue

        residual_pde = self.compute_pde_residual(input_int,func,eigenvalue)
        residual_sbl = self.compute_ansatz(input_sbl,self.approximate_solution(input_sbl)) - output_sbl
        residual_sbr = self.compute_ansatz(input_sbr,self.approximate_solution(input_sbr)) - output_sbr

        loss_pde = torch.mean(abs(residual_pde)**2)
        loss_sbl = self.alpha_sb*torch.mean(abs(residual_sbl)**2)
        loss_sbr = self.alpha_sb*torch.mean(abs(residual_sbr)**2)

        loss_norm = self.alpha_norm*self.compute_norm_loss(func)

        loss_drive = self.alpha_drive*self.compute_drive_loss(eigenvalue)

        loss_ortho = self.alpha_ortho*self.compute_ortho_loss(input_int,func)

        loss_regu = self.alpha_regu*self.approximate_solution.regularization()

        loss = torch.log10(loss_pde +
                           loss_sbl +
                           loss_sbr +
                           loss_norm +
                           loss_drive +
                           loss_ortho +
                           loss_regu)

        if verbose:
            print("Total loss: ",       round(loss.item(), 4))
            # scale certain terms for fair comparison

            print("PDE Loss: ",    round(torch.log10(loss_pde).item(), 4))
            print("Boundary Loss: ",    round(torch.log10(loss_sbl+loss_sbr).item(), 4))
            print("Drive Loss: ",    round(torch.log10(loss_drive).item(), 4))
            print("Ortho Loss: ",    round(torch.log10(loss_ortho).item(), 4))
            print("Norm Loss: ",   round(torch.log10(loss_norm).item(), 4))
            print("Regu Loss: ",   round(torch.log10(loss_regu).item(), 4))
            print("Eigenvalue: ",   round(eigenvalue.item(), 4))

        return loss

    def compute_loss_no_boundary(
        self: object,
        input_int: torch.Tensor,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Computes the residual for the harmonic oscillator eigenvalue equation

        Args:
            self (object): _description_
            input_int (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """

        # enable auto differentiation
        input_int.requires_grad = True

        func = self.compute_ansatz(input_int,self.approximate_solution(input_int))

        eigenvalue = self.approximate_solution.eigenvalue

        residual_pde = self.compute_pde_residual(input_int,func,eigenvalue)

        loss_pde = torch.mean(abs(residual_pde)**2)

        loss_norm = self.alpha_norm*self.compute_norm_loss(func)

        loss_drive = self.alpha_drive*self.compute_drive_loss(eigenvalue)

        loss_ortho = self.alpha_ortho*self.compute_ortho_loss(input_int,func)

        loss_regu = self.alpha_regu*self.approximate_solution.regularization()

        loss = torch.log10(loss_pde +
                           loss_norm +
                           loss_drive +
                           loss_ortho +
                           loss_regu)

        if verbose:
            print("Total loss: ",       round(loss.item(), 4))
            # scale certain terms for fair comparison

            print("PDE Loss: ",    round(torch.log10(loss_pde).item(), 4))
            print("Drive Loss: ",    round(torch.log10(loss_drive).item(), 4))
            print("Ortho Loss: ",    round(torch.log10(loss_ortho).item(), 4))
            print("Norm Loss: ",   round(torch.log10(loss_norm).item(), 4))
            print("Regu Loss: ",   round(torch.log10(loss_regu).item(), 4))
            print("Eigenvalue: ",   round(eigenvalue.item(), 4))

        return loss


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
            for _, (
                    (inp_train_int, _),
                    (inp_train_sbl, out_train_sbl),
                    (inp_train_sbr, out_train_sbr)
                    ) in enumerate(zip(
                    self.training_set_int,
                    self.training_set_sbl,
                    self.training_set_sbr)):
                def closure():

                    verbose=True
                    optimizer.zero_grad()
                    loss = self.compute_loss(
                                inp_train_int,
                                inp_train_sbl,
                                inp_train_sbr,
                                out_train_sbl,
                                out_train_sbr,
                                verbose=verbose
                                            )
                    loss.backward()

                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

        print("Final Loss: ", history[-1])

        return history

    def fit_no_boundary(
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
            for _, (
                    (inp_train_int, _),
                    ) in enumerate(zip(
                    self.training_set_int)):
                def closure():

                    verbose=True
                    optimizer.zero_grad()
                    loss = self.compute_loss_no_boundary(
                                inp_train_int,
                                verbose=verbose
                                            )
                    loss.backward()

                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

        print("Final Loss: ", history[-1])

        return history


    def fit_multiple(
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
        max_iter = 20
        max_eval = 20
        lr = 0.5
        # Loop over eigenfunctions
        for i in range(self.num_eigenfunctions):
            if i == 1:
                self.alpha_ortho *= 2
                self.alpha_norm *= 10
            if i == 2:
                self.approximate_solution.init_xavier()
                self.alpha_ortho *= 20
                self.alpha_norm *= 4
            if i == 3:
                self.alpha_ortho *= 2
                self.alpha_norm *= 2

            history += self.fit_no_boundary(num_epochs, optimizer, verbose=verbose)
            solution_copy = common.NeuralNet(
                        input_dimension=self.domain_extrema.shape[0],
                        output_dimension=1,
                        n_hidden_layers=self.num_layer,
                        neurons=self.size_layer,
                        regularization_param=0.01,
                        regularization_exp=2.,
                        retrain_seed=42,
                        eigenvalue_init=self.approximate_solution.eigenvalue.item(),
                        device=self.device
                    ).to(self.device)
            solution_copy.load_state_dict(copy.deepcopy(self.approximate_solution.state_dict()))
            for param in solution_copy.parameters():
                param.requires_grad = False
            solution_copy.eigenvalue.requires_grad = False
            self.eigenfunctions.append(solution_copy)
            self.approximate_solution.eigenvalue = torch.tensor([2*self.approximate_solution.eigenvalue[:]], requires_grad=True, device=self.device)
            # self.approximate_solution.init_xavier()
            parameters = list(self.approximate_solution.parameters()) + [self.approximate_solution.eigenvalue]
            optimizer = optim.LBFGS(parameters,
                            lr=float(lr),
                            max_iter=max_iter,
                            max_eval=max_eval,
                            history_size=100,
                            line_search_fn="strong_wolfe",
                            tolerance_change=1.0 * np.finfo(float).eps)
        return history

    ################################################################################################
    def plotting(
        self: object,
        name: str = "plot.png"
    ):
        """
        Plot the learned fluid and solid temperature

        Args:
            name (str, optional): Name of the plot. Defaults to "plot.png".
        """
        inputs = self.soboleng.draw(100)
        inputs = self.convert(inputs)
        inputs = inputs.to(self.device)
        
        output = self.compute_ansatz(inputs,self.approximate_solution(inputs)).detach().cpu()

        # plot both fluid and solid temperature
        fig, axs = plt.subplots(1, 1, figsize=(16, 8), dpi=150)
        axs.scatter(inputs[:, 0].detach().cpu(), output[:,0])

        # set the labels
        axs.set_xlabel("x")
        axs.set_ylabel("u")
        axs.grid(True, which="both", ls=":")

        plt.show()
        fig.savefig(name + ".png")

    def plotting_multiple(
        self: object,
        name: str = "plot"
    ):
        """
        Plot the learned fluid and solid temperature

        Args:
            name (str, optional): Name of the plot. Defaults to "plot.png".
        """
        inputs = self.soboleng.draw(100)
        inputs = self.convert(inputs)
        inputs = inputs.to(self.device)
        inputss = inputs[:, 0].detach().cpu().numpy()
        for i,eigenfunction in enumerate(self.eigenfunctions):
            print(eigenfunction.eigenvalue.detach().item())

            output = self.compute_ansatz(inputs,eigenfunction(inputs)).detach().cpu()

            # plot both fluid and solid temperature
            fig, axs = plt.subplots(1, 1, figsize=(16, 8), dpi=150)
            
            axs.scatter(inputss, output[:,0])
            axs.scatter(inputss,-np.sqrt(2)*np.sin((i+1)*np.pi*inputss))

            # set the labels
            axs.set_xlabel("x")
            axs.set_ylabel("u")
            axs.grid(True, which="both", ls=":")

            fig.savefig(name + "_" + str(i) + ".png")
