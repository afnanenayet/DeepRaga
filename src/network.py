"""
A neural network defined using PyTorch for DeepRaga.

This network is a GRU/LSTM model that takes transformed MIDI data that can be
used for training or music generation.
"""

from torch import nn


class RagaNet(nn.Module):
    """
    The actual module that implements RagaNet. This network is based on a
    simple LSTM structure that takes transformed MIDI data as input, and
    trains on it in order to generate music.
    """

    def __init__(self, input_size: int, dropout: float = 0.5, num_layers: int = 2):
        """
        Initialize the RagaNet network

        Args:
            - input_size: The anticipated size of the input feature set
            - dropout: The dropout probability
            - num_layers: The number of hidden layers for the LSTM network
        """
        super(RagaNet, self).__init__()

        # network params
        self.lstm = nn.LSTM(
            input_size=input_size, dropout=dropout, num_layers=num_layers
        )

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return (output, hidden)
