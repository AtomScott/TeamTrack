import torch
import numpy as np
import torch.nn.functional as F
from scipy.interpolate import interp1d
import math

import torch
import torch.nn.functional as F
from einops import rearrange


class SportSpecificTransform(torch.nn.Module):
    def __init__(self, sport: str):
        super().__init__()
        self.sport = sport.lower()
        self.field_dimensions = self.get_sport_dimensions()

    def get_sport_dimensions(self):
        # Define typical field/court dimensions for different sports (width, height)
        dimensions = {"soccer": (105, 68), "basketball": (28, 15), "handball": (40, 20)}

        if self.sport not in dimensions:
            raise ValueError(f"Sport '{self.sport}' not recognized or dimensions not defined.")
        return dimensions[self.sport]

    def forward(self, players_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the given (x, y) coordinates of players to the range [-1, 1] based on the sport's field/court dimensions.
        Clipping values to the range [-1, 1].

        players_coordinates: tensor of shape (n_players, 2) representing [(x1, y1), (x2, y2), ...]
        returns: tensor of normalized coordinates of shape (n_players, 2) representing [(nx1, ny1), (nx2, ny2), ...]
        """
        width, height = self.field_dimensions

        # Normalize such that center is (0,0) and edges are (-1,1)
        # nx = 2 * (players_coordinates[:, 0] / width) - 1
        # ny = 2 * (players_coordinates[:, 1] / height) - 1

        players_coordinates[:, :, 0] = torch.clamp(players_coordinates[:, :, 0], 0, width)
        players_coordinates[:, :, 1] = torch.clamp(players_coordinates[:, :, 1], 0, height)

        return players_coordinates

    def inverse_transform(self, normalized_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Undoes the normalization to convert normalized coordinates back to original scale.

        normalized_coordinates: tensor of shape (n_players, 2) representing [(nx1, ny1), (nx2, ny2), ...]
        returns: tensor of coordinates of shape (n_players, 2) representing [(x1, y1), (x2, y2), ...]
        """
        # width, height = self.field_dimensions

        # x = ((normalized_coordinates[:, 0] + 1) / 2) * width
        # y = ((normalized_coordinates[:, 1] + 1) / 2) * height

        # return torch.stack((x, y), dim=1)
        return normalized_coordinates


class Resample(torch.nn.Module):
    def __init__(self, original_fps, new_fps):
        """
        Resample a batch of trajectories using linear interpolation.

        Parameters:
            original_fps (int): Original frame per second (sampling rate).
            new_fps (int): New frame per second (sampling rate).
        """
        super().__init__()
        self.original_fps = original_fps
        self.new_fps = new_fps

    def forward(self, x):
        """
        Resample a batch of trajectories using linear interpolation.

        Parameters:
            x (Tensor): trajectory with shape (batch_size, seq_len, 2).
                        Each entry represents a 2D point (x, y) in a sequence.

        Returns:
            Tensor: Resampled trajectory with shape (batch_size, new_seq_len, 2).
        """

        needs_to_be_reshaped = False
        if x.dim() == 4:
            needs_to_be_reshaped = True
            B = x.shape[0]
            x = rearrange(x, "B L N D -> (B N) L D")

        batch_size, seq_len, _ = x.shape
        # Compute the new sequence length based on the new fps using accurate rounding
        new_seq_len = math.ceil((seq_len * self.new_fps) / self.original_fps)

        # Reshape to (batch_size, channels, seq_len) as required by interpolate
        x = x.permute(0, 2, 1)

        # Use linear interpolation
        x_resampled = F.interpolate(x, size=new_seq_len, mode="linear", align_corners=True)

        # Reshape back to (batch_size, new_seq_len, 2)
        x_resampled = x_resampled.permute(0, 2, 1)

        if needs_to_be_reshaped:
            x_resampled = rearrange(x_resampled, "(B N) L D -> B L N D", B=B)
        return x_resampled


class Noise(torch.nn.Module):
    def __init__(self, mean=0, std=0.1):
        """
        Add Gaussian noise to a batch of trajectories.

        Parameters:
            mean (float): Mean of the Gaussian distribution.
            std (float): Standard deviation of the Gaussian distribution.
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        """
        Add Gaussian noise to a batch of trajectories.

        Parameters:
            x (Tensor): trajectory with shape (batch_size, seq_len, 2).
                        Each entry represents a 2D point (x, y) in a sequence.

        Returns:
            Tensor: Noisy trajectory with shape (batch_size, seq_len, 2).
        """
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise
