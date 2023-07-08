import torch
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
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
        n_: int,
        batchsize_: int,
        xl_: float,
        xr_: float,
        ub_: float,
        device: torch.device
    ):
        self.device = device
        self.n_int = n_
        self.batchsize = batchsize_
        self.ub = ub_
        self.domain_extrema = torch.tensor([[xl_, xr_]])
        self.space_dimensions = 1
        self.alpha_norm = 1.0
        self.alpha_ortho = 100.0
        self.alpha_drive = 1.0
        self.c = 14.0
        self.num_eigenfunctions = 2

        self.approximate_solution = common.NeuralNet(
            input_dimension=self.domain_extrema.shape[0],
            output_dimension=1,
            n_hidden_layers=5,
            neurons=100,
            regularization_param=0.1,
            regularization_exp=2.,
            retrain_seed=42,
            eigenvalue_range=self.c,
            device=device
        ).to(device)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        self.training_set_int = self.assemble_datasets()
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
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    def set_drive(
        self: object,
        value: float
    ):
        self.c = value

    def get_drive(self) ->  float:
        return self.c


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

    def assemble_datasets(self):
        """
        Creates Dataloader

        Returns:
            Interior dataloader
        """
        # interior data, spatial boundary conditions handled through ansatz
        input_int, output_int   = self.add_interior_points() # S_int

        # load data to device
        input_int, output_int     = input_int.to(self.device), output_int.to(self.device)

        # create dataloader
        training_set_int        = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.batchsize, shuffle=False)
        
        return training_set_int

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

        # calculate the different gradients
        grad_func_x = grad_func[:, 0]

        grad_func_xx = torch.autograd.grad(grad_func_x.sum(), input_int, create_graph=True)[0][:, 0]

        # the residuals for the two equations described in the README
        residual_pde = grad_func_xx - eigenvalue * func

        return residual_pde.reshape(-1, )

    def compute_drive_loss(
        self: object,
        eigenvalue: torch.Tensor
    ) -> torch.Tensor:
        return torch.exp(-self.c + eigenvalue)

    def compute_norm_loss(
        self: object,
        func: torch.Tensor
    ) -> torch.Tensor:
        num_samples = func.shape[0]*func.shape[1]
        domain = 1.0
        for i in range(self.domain_extrema.shape[0]):
            domain *= (self.domain_extrema[i,1] - self.domain_extrema[i,0])
        return (torch.sum(func**2) - num_samples/domain)**2

    def compute_ortho_loss(
        self: object,
        input_int: torch.Tensor,
        func: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            func_prev = torch.zeros_like(func, requires_grad=False)    
            for eigenfunction in self.eigenfunctions:
                func_prev += eigenfunction(input_int)*(1-torch.exp(-(input_int-self.domain_extrema[:,0])**2))*(1-torch.exp(-(input_int-self.domain_extrema[:,1])**2)) + self.ub
        return torch.sum(func*func_prev)

    def compute_loss(
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
        self.domain_extrema = self.domain_extrema.to(self.device)
        func = self.approximate_solution(input_int)*(1-torch.exp(-(input_int-self.domain_extrema[:,0])**2))*(1-torch.exp(-(input_int-self.domain_extrema[:,1])**2)) + self.ub

        eigenvalue = self.approximate_solution.eigenvalue

        residual_pde = self.compute_pde_residual(input_int,func,eigenvalue)
        loss_pde = torch.mean(abs(residual_pde)**2)

        loss_norm = self.compute_norm_loss(func)

        loss_drive = self.compute_drive_loss(eigenvalue)

        loss_ortho = self.compute_ortho_loss(input_int,func)

        loss = torch.log10(loss_pde + self.alpha_norm*loss_norm + self.alpha_drive*loss_drive + self.alpha_ortho*loss_ortho)

        if verbose:
            print("Total loss: ",       round(loss.item(), 4))
            # scale certain terms for fair comparison

            print("PDE Loss: ",    round(torch.log10(loss_pde).item(), 4))
            print("Drive Loss: ",    round(torch.log10(self.alpha_drive*(loss_drive)).item(), 4))
            print("Ortho Loss: ",    round(torch.log10(self.alpha_ortho*(loss_ortho)).item(), 4))
            print("Norm Loss: ",   round(torch.log10(self.alpha_norm *(loss_norm)).item(), 4))
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
                    ) in enumerate(zip(self.training_set_int)):
                def closure():

                    verbose=True
                    optimizer.zero_grad()
                    loss = self.compute_loss(
                                inp_train_int,
                                verbose=verbose
                                            )
                    loss.backward(retain_graph=True)

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

        # Loop over eigenfunctions
        for i in range(self.num_eigenfunctions):
            history += self.fit(num_epochs, optimizer, verbose=verbose)
            solution_copy = common.NeuralNet(
                        input_dimension=self.domain_extrema.shape[0],
                        output_dimension=1,
                        n_hidden_layers=5,
                        neurons=100,
                        regularization_param=0.1,
                        regularization_exp=2.,
                        retrain_seed=42,
                        eigenvalue_range=self.c,
                        device=self.device
                    ).to(self.device)
            solution_copy.load_state_dict(copy.deepcopy(self.approximate_solution.state_dict()))
            for param in solution_copy.parameters():
                param.requires_grad = False
            solution_copy.eigenvalue = copy.deepcopy(self.approximate_solution.eigenvalue.detach())
            self.eigenfunctions.append(solution_copy)

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
        inputs = inputs.to(self.device)
        inputs = self.convert(inputs)
        output = (self.approximate_solution(inputs)*(1-torch.exp(-(inputs-self.domain_extrema[:,0])**2))*(1-torch.exp(-(inputs-self.domain_extrema[:,1])**2)) + self.ub).detach().cpu()

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
        inputs = inputs.to(self.device)
        inputs = self.convert(inputs)

        for i,eigenfunction in enumerate(self.eigenfunctions):
            print(eigenfunction.eigenvalue.detach().item())

            output = (eigenfunction(inputs)*(1-torch.exp(-(inputs-self.domain_extrema[:,0])**2))*(1-torch.exp(-(inputs-self.domain_extrema[:,1])**2)) + self.ub).detach().cpu()

            # plot both fluid and solid temperature
            fig, axs = plt.subplots(1, 1, figsize=(16, 8), dpi=150)
            axs.scatter(inputs[:, 0].detach().cpu(), output[:,0])

            # set the labels
            axs.set_xlabel("x")
            axs.set_ylabel("u")
            axs.grid(True, which="both", ls=":")

            fig.savefig(name + "_" + str(i) + ".png")
