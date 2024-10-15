import sys
import os
import numpy as np
from tqdm import tqdm
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

from tools.utils import logger, tqdm_enumerate

import inputools.Trajectory as it


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



""" Big model """


class ConvEncoderTG(nn.Module):

    def __init__(self, input_channels, output_size):

        super(ConvEncoderTG, self).__init__()

        # Adjust convolutional layers to handle 100x100 input
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)  # Output: 50x50
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # Output: 25x25
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Output: 13x13

        # Calculate the size for the Linear layer dynamically or
        # use a fixed size based on the above calculations
        self.fc = nn.Linear(13 * 8 * 8, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 13*8*8)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc(x))
        return x


class Attention(nn.Module):

    def __init__(self, hidden_size):

        super(Attention, self).__init__()

        self.attn = nn.Linear(50, 1)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        hidden = hidden.repeat(seq_len, 1, 1)  # Repeat decoder hidden state seq_len times
        combined = torch.cat((hidden, encoder_outputs), 2)
        attention_weights = F.softmax(self.attn(combined), dim=0)
        context = torch.bmm(attention_weights.transpose(0, 1), encoder_outputs.transpose(0, 1))
        return context.squeeze(1), attention_weights


class TrajectoryGenerator(nn.Module):

    def __init__(self, context_size: int, hidden_size: int,
                 num_layers: int, seq_len: int):

        """
        The input is:
        - connectivity matrix 100 x 100
        - current and target positions (context vector)

        Parameters
        ----------
        context_size : int
            The size of the context
        hidden_size : int
            The hidden size
        output_size : int
            The output size
        num_layers : int
            The number of layers
        seq_len : int
            The sequence length 
        """

        super(TrajectoryGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        # Convolutional encoder for the grid
        self.conv_encoder = ConvEncoderTG(input_channels=1,
                                        output_size=hidden_size)

        # Linear layer for integrating context vector
        self.context_fc = nn.Linear(context_size, hidden_size)

        # Decoder RNN
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, 100)

        # Attention
        self.attention = Attention(hidden_size)

    def forward(self, grid, context):

        logger.debug(f"{grid.shape}, {context.shape}")

        batch_size = grid.size(0)
        encoder_output = self.conv_encoder(grid).unsqueeze(0)  # Process grid
        logger.debug(f"{encoder_output.shape}")

        context = F.relu(self.context_fc(context))  # Process context vector
        logger.debug(f"{context.shape}")

        # Combine encoder output and context to form initial input and hidden state for RNN
        # logger.debug(f"{encoder_output.shape}, {context.shape}")
        rnn_input = torch.cat((encoder_output, context), dim=1).reshape(-1, 1, 100)  # Initial input

        logger.debug(f"{rnn_input.shape}")
        hidden = torch.zeros(self.num_layers, batch_size, 
                             self.hidden_size).to(grid.device)  # Initial hidden state
        logger.debug(f"{hidden.shape}")

        outputs = []
        for _ in range(self.seq_len):
            rnn_output, hidden = self.rnn(rnn_input, hidden)
            logger.debug(f"{rnn_output.shape}, {hidden.shape}")

            logger.debug(f"{encoder_output.shape}")
            context, attn_weights = self.attention(hidden,
                            encoder_output.unsqueeze(0).repeat(self.seq_len, 1, 1))
            rnn_output = rnn_output.squeeze(0)  # Squeeze the time dimension
            output = self.out(torch.cat((rnn_output, context), 1))
            outputs.append(output)
            rnn_input = output.unsqueeze(0)  # Use current output as next input

        return torch.stack(outputs, dim=1)  # Stack outputs along the sequence dimension


class CRLmodel(nn.Module):

    def __init__(self, hidden_nodes: list = [256, 128], 
                 recurrent_nodes: int =100, seq_len: int =10):

        """
        Adapted model for input dimensions of 100x100
        and an output of 100.

        Parameters
        ----------
        hidden_nodes : list
            The list of hidden nodes. Default is [256, 128]
        recurrent_nodes : int
            The number of recurrent nodes. Default is 100
        seq_len : int
            The sequence length. Default is 10
        """

        super(CRLmodel, self).__init__()

        self.seq_len = seq_len

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5,
                               stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5,
                               stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)

        # Adjusting the size of the flattened features
        # for 100x100 input after 3 pooling layers
        self.num_flattened = 64 * (100 // (2**3)) * (100 // (2**3))

        # Linear layers
        self.fc1 = nn.Linear(self.num_flattened, hidden_nodes[0])
        self.fc2 = nn.Linear(hidden_nodes[0], hidden_nodes[1])

        # RNN layers
        self.rnn1 = nn.RNN(input_size=hidden_nodes[1],
                           hidden_size=recurrent_nodes,
                           num_layers=2, batch_first=True)

        # Output layer
        self.fc_final = nn.Linear(recurrent_nodes, 100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """

        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten
        x = x.view(-1, self.num_flattened)

        # Linear layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Reshape for RNN layers - assuming batch size,
        # we now need to add a sequence dimension that
        # matches our expectations for Seq2Seq
        # For simplicity, let's assume we reshape x to
        # have a sequence length equal to a specified value
        # This might involve reshaping x to have dimensions
        # [batch_size, seq_len, feature_size]
        # However, x currently lacks a real sequence
        # dimension relevant for RNN processing
        # So, we will simulate a sequence by repeating
        # x for a sequence of a fixed length
        # For a dynamic sequence length, you would adjust
        # this based on your actual input sequence lengths

        # Repeat x seq_len times along the sequence dimension
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)  

        # RNN layers
        x, _ = self.rnn1(x)

        # Apply the output layer to each time step
        # No need to select the last timestep,
        # process the entire sequence
        return F.relu(self.fc_final(x))


class CustomDataset(Dataset):

    """ Custom PyTorch dataset for the generated data. """

    def __init__(self, name: str, path: str=None):

        """
        Parameters
        ----------
        name : str
            The name of the dataset
        path : str
            The path to the dataset. Default is None
        """

        logger.debug(os.getcwd())
        if path is None:

            # go up one level to the parent directory 
            # os.chdir('..')

            with open(f"cache/{name}.json", 'r') as f:
                data = json.load(f)

            # go back to the current directory
            # os.chdir('src')

        else:
            logger.debug(os.getcwd())
            with open(os.path.join(path,
                        f"{name}.json"), 'r') as f:
                data = json.load(f)

        self.name = name
        self.data = data
        self.seq_len = self.data['seq_len']
        del self.data['seq_len']

    def __repr__(self):

        return f"CustomDataset(name={self.name}, keys_len={self.__len__()})"

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:

        """
        Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The index of the item
        """

        item = self.data[str(idx)]
        input_matrix = torch.tensor(item['connectivity'], dtype=torch.float32)
        current_position = torch.tensor(item['start'], dtype=torch.float32)
        target_position = torch.tensor(item['end'], dtype=torch.float32)
        activations = torch.tensor(item['activations'], dtype=torch.float32)

        # Input: Concatenate initial and target connectivity matrices along with current and target positions
        input_positions = torch.cat((current_position.unsqueeze(0), target_position.unsqueeze(0)), dim=0)

        return input_matrix, input_positions, activations


def train_model(dataset: object, model: object,
                **kwargs) -> tuple:

    """
    Train the model using the provided dataset and model.

    Parameters
    ----------
    dataset : object
        The dataset
    model : object
        The model
    **kwargs : dict
        batch_size : int
            The batch size. Default is 4
        epochs : int
            The number of epochs. Default is 10
        lr : float
            The learning rate. Default is 0.001
        epoch_log : int 
            The epoch log. Default is 1

    Returns
    -------
    tuple
        The trained model and the losses
    """

    model.to(DEVICE)

    # Unpack keyword arguments
    batch_size = kwargs.get('batch_size', 1)
    epochs = kwargs.get('epochs', 10)
    lr = kwargs.get('lr', 0.001)
    epoch_log = kwargs.get('epoch_log', 1)  # Add logging frequency

    # DataLoader
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True)

    # Loss and optimizer
    criterion = nn.MSELoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []  # Corrected typo from 'lossess' to 'losses'
    epoch_loss = -1

    # Training loop
    for epoch in range(epochs):

        model.train()  # Ensure model is in training mode
        running_loss = 0.0

        for batch_idx, (input_matrices, 
            input_positions, activations) in \
            tqdm_enumerate(data_loader,
                           desc=f"Epoch {epoch} - Loss: {epoch_loss:.6f}"):

            input_matrices = input_matrices.to(DEVICE)
            input_positions = input_positions.to(DEVICE)
            activations = activations.to(DEVICE)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            # outputs = model(grid=input_matrices,
            # context=input_positions)#.unsqueeze(0))
            # logger.debug(f"{input_matrices.shape}, {input_positions.shape}")
            outputs = model(x=torch.cat((input_matrices,
                                         input_positions), dim=1))

            # Compute loss
            loss = criterion(outputs, activations)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(data_loader)
        losses.append(epoch_loss)

        # Log progress
        # if (epoch + 1) % epoch_log == 0:
        #     pbar.set_description(f"Epoch: {epoch + 1}, Loss: {epoch_loss:.6f}")

    # logging.info(f'Final loss: {losses[-1]:.6f}')

    return model, losses


""" Autoencoder """


class Autoencoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: list ):
        """
        Parameters
        ----------
        input_dim : int
            The input dimension
        hidden_dims : list
            The list of hidden dimensions
        """
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.ModuleList()
        for i, dim in enumerate(hidden_dims):
            self.encoder.append(nn.Linear(input_dim if i == 0 else hidden_dims[i-1], dim))

        # Decoder
        reversed_dims = list(reversed(hidden_dims))
        self.decoder = nn.ModuleList()
        for i, dim in enumerate(reversed_dims):
            self.decoder.append(nn.Linear(reversed_dims[i], reversed_dims[i+1] if i + 1 < len(reversed_dims) else input_dim))

    def forward(self, x: torch.Tensor,
                return_innermost_layer: bool=False) -> torch.Tensor:

        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor
        return_innermost_layer : bool
            Whether to return the innermost layer. Default is False

        Returns
        -------
        torch.Tensor
            The output tensor
        """

        # Encoder
        for i, layer in enumerate(self.encoder):
            x = F.sigmoid(layer(x))
            if return_innermost_layer and i == len(self.encoder) - 1:
                innermost_layer = x

        # Decoder
        for layer in self.decoder[:-1]:
            x = F.sigmoid(layer(x))
        x = F.relu(self.decoder[-1](x))

        if return_innermost_layer:
            return x, innermost_layer
        else:
            return x


class Encoder(nn.Module):

    """ Encoder-only model """

    def __init__(self, input_dim: int, hidden_dims: list):

        """
        Parameters
        ----------
        input_dim : int
            The input dimension
        hidden_dims : list
            The list of hidden dimensions
        """

        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList()
        for i, dim in enumerate(hidden_dims):
            self.encoder.append(nn.Linear(input_dim if i == 0 else hidden_dims[i-1], dim))

        self.N = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """

        for layer in self.encoder:
            x = F.relu(layer(x))
        return x

    def step(self, position: np.ndarray, **kwargs) -> np.ndarray:

        """
        Forward pass.

        Parameters
        ----------
        position : np.ndarray
            The input array
        **kwargs : dict
            max_rate : float
                The maximum spike rate.
                Default is 1

        Returns
        -------
        np.ndarray
            The output array
        """

        # settings
        max_rate = kwargs.get('max_rate', 1)

        # initialize activations
        return self(x=torch.tensor(trajectory, dtype=torch.float32)
                    ).detach().numpy() * max_rate

    def parse_trajectory(self, trajectory: np.ndarray,
                         **kwargs) -> np.ndarray:

        """
        Parse a trajectory into the input layer.

        Parameters
        ----------
        trajectory : numpy.ndarray
            Trajectory to parse into the input layer.
        **kwargs : dict
            max_rate : float
                Maximum spike rate of the neurons in the layer.
                Default: 1

        Returns
        -------
        activations : numpy.ndarray
            Activations of the neurons in the layer.
        """

        return self.step(position=trajectory, **kwargs)


class ConvAutoencoder(nn.Module):

    def __init__(self, hidden_dims: list, **kwargs):

        """
        Convolutional Autoencoder.
        It assumes the input is a 60x60 image.

        Parameters
        ----------
        hidden_dims : list
            The list of hidden dimensions, where the first element 
            is the dimension of the first linear layer, and the last
            element is the dimension of the innermost layer
        """

        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=10, stride=2, padding=1),
            nn.ReLU(),

            # Flatten the output for the linear layers
            nn.Flatten(),
            nn.Linear(32*11*11, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], 32*15*15),
            nn.ReLU(),

            # Unflatten to get back to the shape before the linear layers
            nn.Unflatten(1, (32, 15, 15)),
            nn.ConvTranspose2d(in_channels=32, out_channels=16,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """

        x = self.encoder(x)
        x = self.decoder(x)

        return x, None


class ConvEncoder(nn.Module):

    """ Convolutional Encoder-only model """

    def __init__(self, hidden_dims: list, **kwargs):

        """
        Parameters
        ----------
        hidden_dims : list
            The list of hidden dimensions
        """

        super(ConvEncoder, self).__init__()

        # Define the convolutional encoder architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Flatten the output for the linear layers
            nn.Flatten(),
            nn.Linear(32*15*15, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )

        self.N = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """

        x = self.encoder(x)
        return x

    def step(self, position: np.ndarray, **kwargs) -> np.ndarray:

        """
        Forward pass.

        Parameters
        ----------
        position : np.ndarray
            The input array
        **kwargs : dict
            max_rate : float
                The maximum spike rate.
                Default is 1

        Returns
        -------
        np.ndarray
            The output array
        """

        # settings
        max_rate = kwargs.get('max_rate', 1)

        # initialize activations
        return self(x=torch.tensor(trajectory, dtype=torch.float32)
                    ).detach().numpy() * max_rate

    def parse_trajectory(self, trajectory: np.ndarray,
                         **kwargs) -> np.ndarray:

        """
        Parse a trajectory into the input layer.

        Parameters
        ----------
        trajectory : numpy.ndarray
            Trajectory to parse into the input layer.
        **kwargs : dict
            max_rate : float
                Maximum spike rate of the neurons in the layer.
                Default: 1

        Returns
        -------
        activations : numpy.ndarray
            Activations of the neurons in the layer.
        """

        return self.step(position=trajectory, **kwargs)


def load_autoencoder(path: str,
                     kind: str='Autoencoder') -> object:

    """
    Load an autoencoder model from a saved model.

    Parameters
    ----------
    path : str
        The path to the saved model
    kind : str
        The kind of model to load.
        Options: 'Autoencoder', 'ConvAutoencoder'
        Default is 'Autoencoder'

    Returns
    -------
    Autoencoder
        The autoencoder model
    """

    # load the state dictionary from the saved model
    state_dict = torch.load(path)

    # inspect the state dictionary to extract encoder shapes
    encoder_shapes = []

    if kind == 'Autoencoder':
        i = 0
        for name, param in state_dict.items():
            if name.startswith('encoder'):
                if 'weight' in name:
                    encoder_shapes.append(param.size(0))

                    # Extract input dimension from the first layer
                    if i == 0:
                        input_dim = param.size(1) 
                i += 1

        # create the autoencoder model
        autoencoder = Autoencoder(input_dim=input_dim,
                                  hidden_dims=encoder_shapes)
    elif kind == 'ConvAutoencoder':
        i = 0
        for name, param in state_dict.items():
            if name.startswith('encoder'):

                if 'weight' in name and i > 2:
                    encoder_shapes.append(param.size(0))
            i += 1

        # create the autoencoder model
        autoencoder = ConvAutoencoder(hidden_dims=encoder_shapes)

    # load the autoencoder-specific parameters
    autoencoder_state_dict = {name: param for name,
        param in state_dict.items() \
        if name.startswith('encoder') or name.startswith('decoder')}

    # load the state dictionary into the model
    autoencoder.load_state_dict(autoencoder_state_dict)

    # set autoencoder to evaluation mode
    autoencoder.eval()

    return autoencoder


def load_encoder(path: str, kind: str='Encoder') -> Encoder:

    """
    Make an encoder model from a saved model.

    Parameters
    ----------
    path : str
        The path to the saved model
    kind : str
        The kind of model to load.
        Options: 'Encoder', 'ConvEncoder'
        Default is 'Encoder'

    Returns
    -------
    Encoder
        The encoder model
    """

    # load the state dictionary from the saved model
    state_dict = torch.load(path)

    # inspect the state dictionary to extract encoder shapes
    encoder_shapes = []

    if kind == 'Encoder':
        i = 0
        for name, param in state_dict.items():
            if name.startswith('encoder'):
                if 'weight' in name:
                    encoder_shapes.append(param.size(0))

                    # Extract input dimension from the first layer
                    if i == 0:
                        input_dim = param.size(1) 
            i += 1

        # create the encoder model
        encoder = Encoder(input_dim=input_dim,
                          hidden_dims=encoder_shapes)

    # create the encoder model
    elif kind == 'ConvEncoder':
        i = 0
        for name, param in state_dict.items():
            if name.startswith('encoder'):

                if 'weight' in name and i > 2:
                    encoder_shapes.append(param.size(0))
            i += 1

        print(encoder_shapes)
        # create the encoder model
        encoder = ConvEncoder(hidden_dims=encoder_shapes)

    # load the encoder-specific parameters
    encoder_state_dict = {name: param for name,
        param in state_dict.items() \
        if name.startswith('encoder')}

    # load the state dictionary into the model
    encoder.load_state_dict(encoder_state_dict)

    # set encoder to evaluation mode
    encoder.eval()

    return encoder



""" Training functions """


def custom_loss(output: torch.Tensor, target: torch.Tensor,
                innermost_layer: torch.Tensor, alpha: float=1e-5) -> torch.Tensor:

    """
    Custom loss function: MSE + alpha * L1 norm on the innermost layer's output.

    Parameters
    ----------
    output : torch.Tensor
        The output tensor
    target : torch.Tensor
        The target tensor
    innermost_layer : torch.Tensor
        The innermost layer
    alpha : float
        The alpha parameter. Default is 1e-5

    Returns
    -------
    torch.Tensor
        The loss
    """

    mse_loss = F.mse_loss(output, target)

    regularization_loss = torch.norm(innermost_layer, p=1)
    total_loss = mse_loss + alpha * regularization_loss

    return total_loss


def train_autoencoder(input_array: np.ndarray, epochs: int, model: object,
                      loss_fn: object=custom_loss, **kwargs) -> tuple:

    """
    Trains an autoencoder model.

    Parameters
    ----------
    input_array : np.ndarray
        The input array
    epochs : int
        The number of epochs
    model : nn.Module
        The model
    loss_fn : nn.Module
        The loss function.
        Default is `custom_loss`
    **kwargs : dict
        lr : float
            The learning rate. Default is 1e-3
        alpha : float
            The alpha parameter. Default is 1e-5
        epoch_log : float
            The epoch log. Default is 0.1

    Returns
    -------
    tuple
        The trained model and the losses
    """

    model.to(DEVICE)

    # Set the parameters
    lr = kwargs.get('lr', 1e-3)
    alpha = kwargs.get('alpha', 1e-5)
    epoch_log = int(kwargs.get('epoch_log', 0.1) * epochs)

    # Convert input_array to a torch tensor
    input_tensor = torch.tensor(input_array, dtype=torch.float32).to(DEVICE)

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Set the model to training mode
    model.train()

    losses = []

    # Setup tqdm progress bar
    pbar = tqdm(range(epochs), desc="Training Progress")
    for epoch in pbar:

        optimizer.zero_grad()
        output, innermost_layer = model(input_tensor, return_innermost_layer=True)

        if loss_fn == F.mse_loss:
            loss = loss_fn(output, input_tensor)
        else:
            loss = custom_loss(output, input_tensor, innermost_layer, alpha=alpha)

        # Backpropagation
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Update tqdm progress bar description with current epoch and loss
        if epoch % epoch_log == 0:
            pbar.set_description(f"Epoch: {epoch + 1}, Loss: {loss.item():.6f}")

    logger.info(f'Final loss: {loss.item():.6f}')

    return model, losses


def generate(model: object, input_array: np.ndarray) -> np.ndarray:

    """
    Generates reconstructions for the given input array using the autoencoder model.

    Parameters
    ----------
    model : nn.Module
        The autoencoder model
    input_array : np.ndarray
        The input array

    Returns
    -------
    np.ndarray
        The reconstructed array
    """

    # Ensure the model is in evaluation mode
    model.eval()

    # Convert input_array to a torch tensor and move it to the same device as the model
    input_tensor = torch.tensor(input_array, dtype=torch.float32).to(DEVICE)

    # Perform inference (no need to compute gradients)
    with torch.no_grad():
        reconstructed, _ = model(input_tensor)

    # Convert the reconstructed tensor to a numpy array and return
    return reconstructed.cpu().numpy()



""" Simulation functions """


def save_model(model: object, name: str, path: str=None) -> None:

    """
    Save the model.

    Parameters
    ----------
    model : object
        The model
    name : str
        The name of the model
    path : str
        The path to save the model. Default is None
    """

    if path is None:
        path = f"cache_torch/{name}.pt"
    else:
        path = os.path.join(path, f"{name}.pt")

    torch.save(model.state_dict(), path)

    logger(f"Model saved at `{path}`")


def simulate_1():

    """
    simulation of the Autoencoder model
    """

    logger("Simulation 1")
    logger(f"{DEVICE}")

    # settings from record 
    duration = 10
    dt = 1e-3
    speed = 1e-2
    prob_turn = 0.004
    k_average = 300

    # Creation of the data 
    whole_track_1 = trajectory=it.make_whole_walk(dx=0.05)

    # option 1) direct border input
    whole_track = calc_input_trajectory(whole_track_1).reshape(-1, 6)

    # option 2) grid + border input
    layer = it.InputNetwork(layers=[it.BorderLayer(N=4*2, sigma=0.01), 
                                    it.GridLayer(N=10, Nz=0, sigma=8,
                                                scale=np.array([1.1, 1.]))
                                    ])
    #whole_track = layer.parse_trajectory(trajectory=whole_track_1)

    logger(f"Whole track, shape={whole_track.shape}")

    # Model initialization 
    N = layer.N
    N = 6
    model = Autoencoder(input_dim=N, hidden_dims=[10, 20, 40])

    # Run 
    model, losses = train_autoencoder(input_array=whole_track.copy(),
                                      epochs=50_000,
                                      model=model,
                                      lr=1e-3,
                                      alpha=1e-3)

    # Save the model
    save_model(model=model, name="autoencoder_1")


def simulate_conv(name: str, hidden_dims: list, **kwargs):

    """
    simulation of the ConvAutoencoder model

    Parameters
    ----------
    name : str
        The name of the model
    hidden_dims : list
        The list of hidden dimensions
    **kwargs : dict
        num_input : int 
            The number of input images. Default is 10
        epochs : int
            The number of epochs. Default is 1_000
    """

    logger("Simulation `ConvAutoencoder`")
    logger(f"{DEVICE}")

    num_input = kwargs.get('num_input', 10)
    epochs = kwargs.get('epochs', 1_000)

    # -- Data --
    input_data = np.random.binomial(1, 0.1, size=(num_input, 1, 60, 60))  

    logger(f"{input_data.shape}")

    # -- Model --
    model = ConvAutoencoder(hidden_dims=hidden_dims)

    # -- Training --
    model, losses = train_autoencoder(input_array=input_data,
                                      epochs=epochs, model=model,
                                      lr=1e-3, alpha=0, loss_fn=F.mse_loss)

    # Save the model
    save_model(model=model, name=name)


def simulate_crl(name: str, dataset_name: str='dataset_1',
                 hidden_dims: list=[256, 128],
                 recurrent_nodes: int=100, **kwargs):

    """
    simulation of the CRL model

    Parameters
    ----------
    name : str
        The name of the model
    dataset_name : str
        The name of the dataset. Default is 'dataset_1'.
    hidden_nodes : list
        The list of hidden nodes. Default is [256, 128]
    recurrent_nodes : int
        The number of recurrent nodes. Default is 100
    **kwargs : dict
        num_input : int 
            The number of input images. Default is 10
        epochs : int
            The number of epochs. Default is 1_000
        epoch_log : float
            The epoch log. Default is 0.1
        lr : float
            The learning rate. Default is 1e-3
        save : bool
            Whether to save the model. Default is True
        data_path : str
            The path to the dataset. Default is None
    """

    logger("Simulation `CRL`")
    logger(f"{DEVICE}")

    num_input = kwargs.get('num_input', 10)
    epochs = kwargs.get('epochs', 1_000)
    epoch_log = kwargs.get('epoch_log', 0.1)
    lr = kwargs.get('lr', 1e-3)

    # -- Data --
    cd = CustomDataset(name=dataset_name,
                       path=kwargs.get('data_path', None))

    logger(f"{cd}")

    # -- Model --
    model = CRLmodel(hidden_nodes=hidden_dims,
                     recurrent_nodes=recurrent_nodes,
                     seq_len=cd.seq_len)
    logger(f"{model}")

    # -- Training --
    model, losses = train_model(
        dataset=cd, model=model, epochs=epochs,
        epoch_log=epoch_log, lr=lr)

    # Save the model
    if kwargs.get('save', True):
        save_model(model=model, name=name)


""" Other functions """


def calc_input(position: np.ndarray) -> np.ndarray:

    """
    Calculate the input.

    Parameters
    ----------
    position : np.ndarray
        The position.

    Returns
    -------
    input : np.ndarray
        The input.
    """

    # distance from the walls
    dist_walls = np.array([abs(position[0] - BOUNDS[0]),
                           abs(position[0] - BOUNDS[1]),
                           abs(position[1] - BOUNDS[0]),
                           abs(position[1] - BOUNDS[1])])

    return np.concatenate((position, dist_walls)).reshape(-1, 1)


def calc_input_trajectory(trajectory: np.ndarray) -> np.ndarray:

    """
    Calculate the input trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory.

    Returns
    -------
    input : np.ndarray
        The input trajectory.
    """

    return np.array([calc_input(pos) for pos in trajectory])



""" MLP 4 PC """


class DNN(nn.Module):

    def __init__(self, hidden_dims: list=[10, 10], activation: str=None):

        super(DNN, self).__init__()


        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        else:
            self.activation = lambda x: x

        # Encoder
        self.N = len(hidden_dims)
        self.layers = nn.ModuleList()
        for i, dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(2 if i == 0 else hidden_dims[i-1], dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """

        # Encoder
        for i, layer in enumerate(self.layers):
            # if (i+1) == self.N:
            #     x = layer(x)
            #     break
            # x = F.relu(layer(x))
            # x = F.sigmoid(layer(x))
            # x = layer(x)
            x = self.activation(layer(x))

        return x


def train_dnn(input_array: np.ndarray, target_array: np.ndarray,
              epochs: int, model: object,
              **kwargs) -> tuple:

    """
    Trains an autoencoder model.

    Parameters
    ----------
    input_array : np.ndarray
        The input array
    target_array : np.ndarray
        The target array
    epochs : int
        The number of epochs
    model : nn.Module
        The model
    **kwargs : dict
        lr : float
            The learning rate. Default is 1e-3
        epoch_log : float
            The epoch log. Default is 0.1

    Returns
    -------
    tuple
        The trained model and the losses
    """

    model.to(DEVICE)

    # Set the parameters
    lr = kwargs.get('lr', 1e-3)
    epoch_log = int(kwargs.get('epoch_log', 
                               0.1) * epochs)

    # Convert input_array to a torch tensor
    input_tensor = torch.tensor(input_array,
                                dtype=torch.float32).to(DEVICE)
    target_tensor = torch.tensor(target_array,
                                 dtype=torch.float32).to(DEVICE)

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=lr)

    # Set the model to training mode
    model.train()

    losses = []

    # Setup tqdm progress bar
    pbar = tqdm(range(epochs), desc="Training Progress")
    for epoch in pbar:

        optimizer.zero_grad()
        output = model(input_tensor)

        # logger.debug(input_tensor)
        # logger.debug(target_tensor)

        loss = F.mse_loss(output, target_tensor)

        # Backpropagation
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Update tqdm progress bar description with current epoch and loss
        if epoch % epoch_log == 0:
            pbar.set_description(f"Epoch: {epoch + 1}, Loss: {loss.item():.6f}")

    logger.info(f'Final loss: {loss.item():.6f}')

    return model, losses, model(input_tensor).cpu().detach().numpy()


def simulate_dnn(nj: int, sigma: float=0.01, dx: float=0.009,
                 hidden_dims: list=[32, 128], 
                 activation: str='none',
                 epochs: int=100,
                 name: str='dnn_pc'):

    """
    train a DNN to replicate PC fields
    """

    logger("Simulation DNN")
    logger(f"DEVICE=`{DEVICE}`")

    # settings from record 
    nj = nj
    Nj = nj**2
    layer = it.PlaceLayer(N=Nj, sigma=sigma)

    input_array = it.make_whole_walk(dx=dx)
    target_array = layer.parse_trajectory(trajectory=input_array)

    logger(f"Input shape={input_array.shape}")
    logger(f"Target shape={target_array.shape}")

    # Model initialization 
    model = DNN(hidden_dims=hidden_dims + [Nj])

    # Run
    model, losses, preds = train_dnn(
        input_array=input_array.copy(),
        target_array=target_array.copy(),
        model=model, epochs=epochs)

    # Save the model
    save_model(model=model, name=name)



def load_dnnpc(path: str, activation: str) -> object:

    """
    Load an DNN PC model from a saved model.
    It assumes there is a bias term.

    Parameters
    ----------
    path : str
        The path to the saved model
    activation : str
        The activation function.
        Options: 'relu', 'sigmoid'

    Returns
    -------
    DNN : object
    """

    # load the state dictionary from the saved model
    state_dict = torch.load(path)

    # extract layer sizes 
    hidden_dims = []
    for name, param in state_dict.items():
        if name.startswith('layers'):
            if 'bias' in name:
                hidden_dims.append(param.numpy().size)

    # create the DNN model
    dnn = DNN(hidden_dims=hidden_dims, 
              activation=activation)

    # load the state dictionary into the model
    dnn.load_state_dict(state_dict)

    # set autoencoder to evaluation mode
    dnn.eval()

    logger(f"DNN loaded from `{path}`")

    return dnn




if __name__ == '__main__':

    # simulate_1()
    # simulate_conv(name="conv_autoencoder_1",
    #               hidden_dims=[100, 60], 
    #               num_input=30, epochs=50_000)

    # simulate_crl(name="crl_1", dataset_name='dataset_1',
    #              hidden_dims=[256, 128], recurrent_nodes=100,
    #              epochs=5, epoch_log=1, lr=1e-3, 
    #              save=False)

    simulate_dnn(nj=5, hidden_dims=[64],
                 activation='sigmoid',
                 epochs=10_000)
