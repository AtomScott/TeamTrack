import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

import torch
from torch import nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GENConv, aggr, GENConv, DeepGCNLayer
from torch.nn import LayerNorm, Linear, ReLU
from torch_geometric.data import Data, Batch


def get_norm_layer(norm_method, dim):
    if norm_method == "layer":
        return pyg_nn.LayerNorm(dim, affine=True)
    if norm_method == "graph":
        return pyg_nn.GraphNorm(dim)
    if norm_method == "batch":
        return pyg_nn.BatchNorm(dim, affine=True)
    if norm_method == "instance":
        return pyg_nn.InstanceNorm(dim, affine=True)
    if norm_method == "none" or norm_method is None:
        return nn.Identity()


def get_aggr(arg_method):
    if arg_method == "softmax":
        return aggr.SoftmaxAggregation(learn=True)
    if arg_method == "powermean":
        return aggr.PowerMeanAggregation(learn=True)
    return aggr.MultiAggregation(arg_method)


class GCNEncoder(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        dropout=0.0,
        norm="layer",
        local_aggr="softmax",
        local_norm="layer",
        num_layers=2,
        use_complete_graph=False,
        jk=None,
    ):
        super().__init__()

        self.node_encoder = Linear(input_channels, hidden_channels)
        self.use_complete_graph = use_complete_graph

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(
                hidden_channels,
                hidden_channels,
                aggr=local_aggr,
                t=1.0,
                learn_t=True,
                num_layers=2,
                norm=local_norm,
                jk=jk,
            )
            norm = get_norm_layer(norm, hidden_channels)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block="plain", dropout=dropout)
            self.layers.append(layer)

        self.fc1 = nn.Linear(hidden_channels, output_channels)

        # TODO: This is a hack to get the complete graph. It should be refactored in a function that caches the edge index and the graph index
        self.edge_index_cache = {}
        self.graph_index_cache = {}

    def forward(self, batch):
        batch_size = batch.shape[0]
        num_nodes = batch.shape[1]

        x = rearrange(batch, "b n f -> (b n) f")
        edge_index = self.get_edge_index(num_nodes).to(self.get_device())

        for layer in self.layers:
            x = layer(x, edge_index)

        out = self.fc1(x)

        out = rearrange(out, "(b n) f -> b n f", b=batch_size)
        return out

    # def forward(self, batch):
    #     """
    #     x is a batch of graphs with shape (batch_size, num_nodes, num_features)
    #     """

    #     # Reshape x to (num_nodes, num_features) since batches are considered as unconected graphs in torch_geometric
    #     batch_size = batch.shape[0]
    #     num_nodes = batch.shape[1]

    #     x = rearrange(batch, "b n f -> (b n) f")
    #     edge_index = self.get_edge_index(num_nodes).to(self.get_device())

    #     # x = self.node_encoder(x)

    #     x = self.layers[0](x, edge_index)

    #     for layer in self.layers[1:]:
    #         x = layer(x, edge_index)

    #     x = self.layers[0].act(self.layers[0].norm(x))
    #     out = self.fc1(x)

    #     # Reshape x to (batch_size, num_nodes, num_features)
    #     out = rearrange(out, "(b n) f -> b n f", b=batch_size)
    #     return out

    def get_device(self):
        return next(self.parameters()).device

    def get_edge_index(self, num_nodes):
        if self.edge_index_cache.get(num_nodes) is None:
            edge_index = []
            edge_index0 = []
            edge_index1 = []
            device = self.get_device()
            for i in range(num_nodes):
                for j in range(num_nodes):
                    edge_index0.append(i)
                    edge_index1.append(j)
            edge_index.append(edge_index0)
            edge_index.append(edge_index1)

            edge_index = torch.tensor(edge_index, dtype=torch.long)

            self.edge_index_cache[num_nodes] = edge_index

        return self.edge_index_cache[num_nodes]

    def get_graph_index(self, num_nodes, batch_size):
        if self.graph_index_cache.get((num_nodes, batch_size)) is None:
            edge_index = self.get_edge_index(num_nodes)
            batch_list = []
            for _ in range(batch_size):
                batch_list.append(Data(edge_index=edge_index, num_nodes=num_nodes))

            batch = Batch.from_data_list(batch_list)
            graph_index = batch.batch

            self.graph_index_cache[(num_nodes, batch_size)] = graph_index

        return self.graph_index_cache[(num_nodes, batch_size)]


class GNN(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        n_mixtures=3,
        use_complete_graph=True,
        n_layers=3,
        dropout=0.0,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.n_mixtures = n_mixtures
        self.n_layers = n_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_channels // 2,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

        self.gcn = GCNEncoder(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            output_channels=hidden_channels,
            num_layers=n_layers,
            use_complete_graph=use_complete_graph,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_channels, 6 * n_mixtures)

    def forward(self, x):
        # B: batch size, L: sequence length, N: number of agents, D: dimension
        batch_size = x.shape[0]
        num_agents = x.shape[2]

        lstm_input = rearrange(x, "B L N D -> (B N) L D")
        lstm_output, *_ = self.lstm(lstm_input)
        last_lstm_output = lstm_output[:, -1, :]  # (batch_size, hidden_channels)
        gcn_input = rearrange(
            last_lstm_output,
            "(B N) F -> B N F",
            B=batch_size,
            N=num_agents,
        )

        # gcn_input = rearrange(x, "B L N D -> B N (L D)")

        gcn_output = F.relu(self.gcn(gcn_input))
        gaussian_params = self.fc(gcn_output)  # (batch_size, n_players, 6 * n_mixtures)
        pi, mu, sigma, corr = torch.split(
            gaussian_params,
            [
                self.n_mixtures,
                2 * self.n_mixtures,
                2 * self.n_mixtures,
                self.n_mixtures,
            ],
            dim=2,
        )

        pi = F.softmax(pi, dim=2)
        sigma = torch.exp(sigma)
        corr = torch.tanh(corr)

        return pi, mu, sigma, corr

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
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        n_players = x.shape[2]

        if y_gt is not None:
            y_gt = rearrange(y_gt, "B L N D -> (B N) L D")

        for i in range(n_steps):
            pi, mu, sigma, corr = self(x)

            pi = rearrange(pi, "B N M -> (B N) M")
            mu = rearrange(mu, "B N M -> (B N) M")
            sigma = rearrange(sigma, "B N M -> (B N) M")
            corr = rearrange(corr, "B N M -> (B N) M")
            x = rearrange(x, "B L N D -> (B N) L D")

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
                x = torch.cat([x[:, 1:, :], y_gt[:, i, :].unsqueeze(1)], dim=1)
            else:
                x = torch.cat([x[:, 1:, :], next_position.unsqueeze(1)], dim=1)
            x = rearrange(x, "(B N) L D -> B L N D", B=batch_size, N=n_players)
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

    batch_size = 64
    dim = 2
    seq_len = 96
    n_players = 22
    x = torch.randn(batch_size, seq_len, n_players, dim)

    input_channels = dim * seq_len
    gnn = GNN(
        input_channels=input_channels,
        hidden_channels=32,
        n_layers=2,
        dropout=0.5,
        n_mixtures=n_mixtures,
    )

    pi, mu, sigma, corr = gnn(x)

    # Check shapes and values
    outputs_shapes = {
        "pi_shape": pi.shape,
        "mu_shape": mu.shape,
        "sigma_shape": sigma.shape,
        "corr_shape": corr.shape,
        "pi_sum": pi.sum(dim=2),  # Should be close to 1 due to softmax
        "corr_min_max": (corr.min(), corr.max()),  # Should be between -1 and 1 due to tanh
    }

    print(outputs_shapes)
    # print(gnn.predict(x, stochastic=True).shape)
    # print(gnn.roll_out(x, 5, stochastic=True).shape)

    from teamtrack.metrics import bivariate_nll_loss, rmse_loss, trajectory_nll_loss

    pi = rearrange(pi, "B N M -> (B N) M")
    mu = rearrange(mu, "B N M -> (B N) M")
    sigma = rearrange(sigma, "B N M -> (B N) M")
    corr = rearrange(corr, "B N M -> (B N) M")
    x = rearrange(x, "B L N M -> (B N) L M")

    mu_x, mu_y = torch.split(mu, [n_mixtures, n_mixtures], 1)
    sigma_x, sigma_y = torch.split(sigma, [n_mixtures, n_mixtures], 1)

    nll = bivariate_nll_loss(x[:, 0], pi, mu_x, mu_y, sigma_x, sigma_y, corr)
    print(nll)

    x = torch.randn(batch_size, seq_len, n_players, dim)
    next_positions, pi_traj, mu_traj, sigma_traj, corr_traj = gnn.roll_out(x, n_steps=5, return_all_params=True)
    print(f"{pi_traj.shape=}")
    print(f"{mu_traj.shape=}")
    print(f"{sigma_traj.shape=}")
    print(f"{corr_traj.shape=}")

    mu_x_traj, mu_y_traj = torch.split(mu_traj, [n_mixtures, n_mixtures], 2)
    sigma_x_traj, sigma_y_traj = torch.split(sigma_traj, [n_mixtures, n_mixtures], 2)

    x = rearrange(x, "B L N M -> (B N) L M")
    traj_nll = trajectory_nll_loss(x, pi_traj, mu_x_traj, mu_y_traj, sigma_x_traj, sigma_y_traj, corr_traj)
    print(f"{traj_nll=}")
