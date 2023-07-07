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
        eigenvalue_range: float,
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
            eigenvalue_range        (int): Range to in which the eigenvalue is initialized
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
        self.activation = nn.Tanh()
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp

        self.eigenvalue = eigenvalue_range*torch.rand(1, requires_grad=True, device=device)

        self.input_layer = nn.Linear(self.input_dimension + 1, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons)
                                            for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.retrain_seed = retrain_seed
        # Random Seed for weight initialization
        self.init_xavier()
        self.device = device

    def get_eigenvalue(self):
        """_summary_

        Returns:
            (torch.Tensor): eigenvalue tensor
        """
        return self.eigenvalue


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


def fit(model, training_set, num_epochs, optimizer, p, verbose=True):
    history = list()

    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose: print("################################ ", epoch, " ################################")

        running_loss = list([0])

        # Loop over batches
        for j, (x_train_, u_train_) in enumerate(training_set):
            def closure():
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                u_pred_ = model(x_train_)
                # Item 1. below
                loss = torch.mean((u_pred_.reshape(-1, ) - u_train_.reshape(-1, )) ** p) + model.regularization()
                # Item 2. below
                loss.backward()
                # Compute average training loss over batches for the current epoch
                running_loss[0] += loss.item()
                return loss

            # Item 3. below
            optimizer.step(closure=closure)

        if verbose: print('Loss: ', (running_loss[0] / len(training_set)))
        history.append(running_loss[0])

    return history
