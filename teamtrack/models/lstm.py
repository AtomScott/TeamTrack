import torch
from torch import nn
from torch.nn import functional as F


class LSTM(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, n_layers: int = 1, dropout: float = 0.5, n_mixtures: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_mixtures = n_mixtures

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc0 = nn.Linear(hidden_dim * 2, 6 * n_mixtures)

    def init_hidden(self, batch_size, h_0=None, c_0=None):
        device = next(self.parameters()).device
        if h_0 is None:
            h_0 = torch.zeros(2 * self.n_layers, batch_size, self.hidden_dim).to(device)
        if c_0 is None:
            c_0 = torch.zeros(2 * self.n_layers, batch_size, self.hidden_dim).to(device)
        return h_0, c_0

    def forward(self, x, h_0=None, c_0=None):
        batch_size = x.size(0)

        h_0, c_0 = self.init_hidden(batch_size, h_0, c_0)

        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        last_lstm_out = lstm_out[:, -1, :]
        gaussian_params = self.fc0(last_lstm_out)

        pi, mu, sigma, corr = torch.split(
            gaussian_params,
            [
                self.n_mixtures,
                2 * self.n_mixtures,
                2 * self.n_mixtures,
                self.n_mixtures,
            ],
            dim=1,
        )

        pi = F.softmax(pi, dim=1)
        sigma = torch.exp(sigma)
        corr = torch.tanh(corr)

        return pi, mu, sigma, corr, (h_n, c_n)

    def predict(self, x, h_0=None, c_0=None, stochastic=False):
        """
        Predict the next position given the input sequence.

        Parameters:
        - x (torch.Tensor): Input sequence.
        - h_0, c_0 (torch.Tensor): Initial hidden and cell states. Default to None.
        - stochastic (bool): If True, sample from the mixture for prediction.
                            If False, use the mean of the mixture with the highest weight. Default to False.

        Returns:
        - torch.Tensor: Predicted next position.
        """
        pi, mu, sigma, corr, (h_n, c_n) = self(x, h_0=h_0, c_0=c_0)

        if stochastic:
            # Sample from the bivariate mixture model
            next_position = self.sample_from_mixture(pi, mu, sigma, corr) + x[:, -1, :]
        else:
            # Take the mean of the Gaussian with the highest mixture coefficient
            pi_max_idx = pi.argmax(dim=1)

            # Considering mu_x and mu_y for the chosen mixture component
            mu_x_chosen = mu[torch.arange(mu.size(0)), pi_max_idx]
            mu_y_chosen = mu[torch.arange(mu.size(0)), pi_max_idx + self.n_mixtures]

            next_position = torch.stack([mu_x_chosen, mu_y_chosen], dim=1) + x[:, -1, :]

        return next_position

    def sample_from_mixture(self, pi, mu, sigma, corr):
        """Sample from bivariate Gaussian mixture model."""

        # Asserting input shapes
        batch_size, n_mixtures = pi.shape
        assert mu.shape == (batch_size, 2 * n_mixtures), "Incorrect shape for mu"
        assert sigma.shape == (batch_size, 2 * n_mixtures), "Incorrect shape for sigma"
        assert corr.shape == (batch_size, n_mixtures), "Incorrect shape for corr"

        # Choose a component from the mixture
        mixture_idx = torch.multinomial(pi, 1).squeeze(1)
        batch_size = pi.size(0)

        # Create expanded indices for gathering
        batch_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, 2).to(mixture_idx.device)
        mixture_indices = (mixture_idx * 2).unsqueeze(1)
        mixture_indices = torch.cat([mixture_indices, mixture_indices + 1], dim=1)

        # Gather the parameters of the chosen Gaussian component
        chosen_mu = mu[batch_indices, mixture_indices]
        chosen_sigma = sigma[batch_indices, mixture_indices]
        chosen_corr = corr[torch.arange(batch_size), mixture_idx]

        # Compute elements for Cholesky decomposition of the covariance matrix
        L_00 = chosen_sigma[:, 0]
        L_01 = torch.zeros_like(L_00)
        L_10 = chosen_corr * chosen_sigma[:, 1]
        L_11 = chosen_sigma[:, 1] * torch.sqrt(1 - chosen_corr**2 + 1e-6)

        # Get two samples from a standard normal
        z = torch.randn(batch_size, 2).to(chosen_mu.device)

        # Apply the Cholesky decomposition and adjust with the mean
        sample_0 = chosen_mu[:, 0] + L_00 * z[:, 0] + L_01 * z[:, 1]
        sample_1 = chosen_mu[:, 1] + L_10 * z[:, 0] + L_11 * z[:, 1]
        return torch.stack([sample_0, sample_1], dim=1)

    def roll_out(self, x, n_steps, y_gt=None, stochastic=False, return_all_params=False):
        """
        Roll out prediction over multiple steps.

        Parameters:
        - x (torch.Tensor): Input sequence.
        - n_steps (int): Number of prediction steps.
        - y_gt (torch.Tensor, optional): Ground truth for guided roll-out. Default to None.
        - stochastic (bool): If True, sample from the mixture for prediction.
                            If False, use the mean of the mixture with the highest weight. Default to False.
        - return_all_params (bool): If True, return next_positions, pis, mus, and sigmas.
                                If False, return only next_positions. Default to False.

        Returns:
        - torch.Tensor: Depending on return_all_params, returns predicted trajectory or tuple of predicted trajectory and Gaussian parameters.
        """
        next_positions = []
        pis, mus, sigmas, corrs = [], [], [], []
        h_n, c_n = None, None
        for i in range(n_steps):
            pi, mu, sigma, corr, (h_n, c_n) = self(x, h_0=h_n, c_0=c_n)

            if stochastic:
                next_position = self.sample_from_mixture(pi, mu, sigma, corr) + x[:, -1, :]
            else:
                # Take the mean of the Gaussian with the highest mixture coefficient
                pi_max_idx = pi.argmax(dim=1)

                # Considering mu_x and mu_y for the chosen mixture component
                mu_x_chosen = mu[torch.arange(mu.size(0)), pi_max_idx]
                mu_y_chosen = mu[torch.arange(mu.size(0)), pi_max_idx + self.n_mixtures]

                next_position = torch.stack([mu_x_chosen, mu_y_chosen], dim=1) + x[:, -1, :]

            if y_gt is not None:
                x = y_gt[:, i : i + 1, :]
            else:
                x = next_position.unsqueeze(1)

            next_positions.append(next_position)

            if return_all_params:
                pis.append(pi)
                mus.append(mu)
                sigmas.append(sigma)
                corrs.append(corr)

        if return_all_params:
            return torch.stack(next_positions, dim=1), torch.stack(pis, dim=1), torch.stack(mus, dim=1), torch.stack(sigmas, dim=1), torch.stack(corrs, dim=1)
        else:
            return torch.stack(next_positions, dim=1)


if __name__ == "__main__":
    n_mixtures = 3
    lstm = LSTM(input_dim=2, hidden_dim=32, n_layers=2, dropout=0.5, n_mixtures=n_mixtures)

    batch_size = 13
    seq_len = 10
    dim = 2
    x = torch.randn(batch_size, seq_len, dim)

    pi, mu, sigma, corr, (h_n, c_n) = lstm(x)

    # Check shapes and values
    outputs_shapes = {
        "pi_shape": pi.shape,
        "mu_shape": mu.shape,
        "sigma_shape": sigma.shape,
        "corr_shape": corr.shape,
        "h_n_shape": h_n.shape,
        "c_n_shape": c_n.shape,
        "pi_sum": pi.sum(dim=1),  # Should be close to 1 due to softmax
        "corr_min_max": (corr.min(), corr.max()),  # Should be between -1 and 1 due to tanh
    }

    print(outputs_shapes)
    print(lstm.predict(x, stochastic=True).shape)
    print(lstm.roll_out(x, 5, stochastic=True).shape)

    from teamtrack.metrics import bivariate_nll_loss, rmse_loss, trajectory_nll_loss

    mu_x, mu_y = torch.split(mu, [n_mixtures, n_mixtures], 1)
    sigma_x, sigma_y = torch.split(sigma, [n_mixtures, n_mixtures], 1)

    nll = bivariate_nll_loss(x[:, 0], pi, mu_x, mu_y, sigma_x, sigma_y, corr)
    print(nll)

    next_positions, pi_traj, mu_traj, sigma_traj, corr_traj = lstm.roll_out(x, n_steps=5, return_all_params=True)
    print(f"{pi_traj.shape=}")
    print(f"{mu_traj.shape=}")
    print(f"{sigma_traj.shape=}")
    print(f"{corr_traj.shape=}")

    mu_x_traj, mu_y_traj = torch.split(mu_traj, [n_mixtures, n_mixtures], 2)
    sigma_x_traj, sigma_y_traj = torch.split(sigma_traj, [n_mixtures, n_mixtures], 2)

    traj_nll = trajectory_nll_loss(x, pi_traj, mu_x_traj, mu_y_traj, sigma_x_traj, sigma_y_traj, corr_traj)
    print(f"{traj_nll=}")
