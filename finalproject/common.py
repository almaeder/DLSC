"""
Given python script from the class.
Created by Roberto Molinaro and others, 21.03.2023
Adapted by Alexander Maeder, Leonard Deuschle
"""
import torch.nn as nn
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)


class NeuralNet(nn.Module):
    """Basic Neural Network with n hidden layers and m neurons per layer.
    With tanh activation function and L2 regularization.
    """
    def __init__(
        self: object,
        input_dimension: int,
        output_dimension: int,
        n_hidden_layers: int,
        neurons: int,
        regularization_param: int,
        regularization_exp: int,
        retrain_seed: int,
        eigenvalue_init: float,
        device
    ):
        """Initialize the Neural Network

        Args:
            input_dimension         (int): Input dimension of the network
            output_dimension        (int): Output dimension of the network
            n_hidden_layers         (int): Number of hidden layers
            neurons                 (int): Number of neurons per layer
            regularization_param    (int): Regularization parameter
            regularization_exp      (int): Type of regularization
            retrain_seed            (int): Random seed for weight initialization
            eigenvalue_init         (int):
        Returns:
            _type_: _description_
        """
        super().__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        # self.activation = nn.Tanh()
        self.activation = torch.sin
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp

        # self.eigenvalue = torch.tensor([eigenvalue_init], requires_grad=False, device=device)
        self.eigenvalue = nn.Parameter(torch.tensor([eigenvalue_init], requires_grad=True, device=device))

        self.input_layer = nn.Linear(self.input_dimension + 1, self.neurons)
        # self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons)
                                            for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.retrain_seed = retrain_seed
        # Random Seed for weight initialization
        self.init_xavier()
        self.device = device

    def forward(self,
        input_data
    ):
        """Calculates the forward pass

        Args:
            input_data (_type_): Input tensor

        Returns:
            _type_: Data after forward pass
        """
        # The forward function performs
        # the set of affine and non-linear transformations defining the network
        input_d = torch.cat((input_data, self.eigenvalue*torch.ones(input_data.shape[0], 1, device=self.device)), dim=1)
        hidden_data = self.activation(self.input_layer(input_d))
        for _, layer in enumerate(self.hidden_layers):
            hidden_data = self.activation(layer(hidden_data))
        return self.output_layer(hidden_data)
    
    def forward_old(self,
        input_data
    ):
        """Calculates the forward pass

        Args:
            input_data (_type_): Input tensor

        Returns:
            _type_: Data after forward pass
        """
        # The forward function performs
        hidden_data = self.activation(self.input_layer(input_data))
        for _, layer in enumerate(self.hidden_layers):
            hidden_data = self.activation(layer(hidden_data))
        return self.output_layer(hidden_data)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss

def compute_potential_infinite_well(
    input_int: torch.Tensor
) -> torch.Tensor:
    return torch.zeros_like(input_int)

def compute_potential_double_well(
    input_int: torch.Tensor
) -> torch.Tensor:
    pos_s = 0.4
    pos_e = 0.6
    return 200*torch.logical_and((pos_s <= input_int),(pos_e >= input_int))


def compute_potential_rtd(
    input_int: torch.Tensor
) -> torch.Tensor:
    pos_ls = 1.9
    pos_le = 2.25
    pos_rs = 2.75
    pos_re = 3.1
    return 30*torch.logical_or( torch.logical_and((pos_ls <= input_int),(pos_le >= input_int)),
            torch.logical_and((pos_rs <= input_int),(pos_re >= input_int))).double()

def compute_potential(
    input_int: torch.Tensor
) -> torch.Tensor:
    return compute_potential_rtd(input_int)
