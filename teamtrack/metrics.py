import torch
import numpy as np
from torchmetrics.functional import mean_squared_error
import warnings


def trajectory_nll_loss(y_true, pi, mu_x, mu_y, sigma_x, sigma_y, rhos):
    """
    Compute the MDN loss for bivariate Gaussian mixtures over a trajectory rollout.

    Parameters:
    - y_true: target values, shape (batch_size, seq_length, 2)
    - pi, mu_x, mu_y, sigma_x, sigma_y, rho: Parameters of the bivariate Gaussian mixtures for each timestep.
      Each of these parameters has the shape (batch_size, seq_length, num_mixtures).
    """

    batch_size, seq_length, _ = pi.shape
    nll_seq = torch.zeros((batch_size, seq_length))

    for t in range(seq_length):
        nll_seq[:, t] = bivariate_nll_loss(y_true[:, t], pi[:, t], mu_x[:, t], mu_y[:, t], sigma_x[:, t], sigma_y[:, t], rhos[:, t])

    return torch.mean(nll_seq)


# def bivariate_nll_loss(y, pi, mu_x, mu_y, sigma_x, sigma_y, rhos):
#     # Compute the bivariate Gaussian PDF
#     x, y_actual = y[:, 0], y[:, 1]
#     x = x.unsqueeze(1)
#     y_actual = y_actual.unsqueeze(1)

#     z = ((x - mu_x) / sigma_x) ** 2 + ((y_actual - mu_y) / sigma_y) ** 2 - (2 * rhos * (x - mu_x) * (y_actual - mu_y)) / (sigma_x * sigma_y)
#     normalization = 2 * torch.pi * sigma_x * sigma_y * torch.sqrt(1 - rhos**2)

#     # Small constant to prevent log(0) or division by zero
#     epsilon = 1e-20

#     # clamp normalization to prevent division by zero
#     normalization = torch.clamp(normalization, epsilon, 1e10)

#     gaussian = torch.exp(-z / (2 * (1 - rhos**2))) / normalization

#     # Combine the Gaussians using the mixing coefficients (pi)
#     mixture = pi * gaussian
#     mixture = torch.sum(mixture, dim=1)  # Sum across the mixtures

#     # Compute the negative log likelihood
#     nll = -torch.log(mixture + epsilon)

#     # print("y", y.min(), y.max(), y.mean())
#     # print("pi", pi.min(), pi.max(), pi.mean())
#     # print("mu_x", mu_x.min(), mu_x.max(), mu_x.mean())
#     # print("mu_y", mu_y.min(), mu_y.max(), mu_y.mean())
#     # print("sigma_x", sigma_x.min(), sigma_x.max(), sigma_x.mean())
#     # print("sigma_y", sigma_y.min(), sigma_y.max(), sigma_y.mean())
#     # print("rhos", rhos.min(), rhos.max(), rhos.mean())

#     # print("z", z.min(), z.max(), z.mean())
#     # print("normalization", normalization.min(), normalization.max(), normalization.mean())
#     # print("gaussian", gaussian.min(), gaussian.max(), gaussian.mean())
#     # print("mixture", mixture.min(), mixture.max(), mixture.mean())

#     # raise error if nll is nan
#     if torch.isnan(nll).any():
#         print("y", y.min(), y.max(), y.mean())
#         print("pi", pi.min(), pi.max(), pi.mean())
#         print("mu_x", mu_x.min(), mu_x.max(), mu_x.mean())
#         print("mu_y", mu_y.min(), mu_y.max(), mu_y.mean())
#         print("sigma_x", sigma_x.min(), sigma_x.max(), sigma_x.mean())
#         print("sigma_y", sigma_y.min(), sigma_y.max(), sigma_y.mean())
#         print("rhos", rhos.min(), rhos.max(), rhos.mean())

#         print("z", z.min(), z.max(), z.mean())
#         print("normalization", normalization.min(), normalization.max(), normalization.mean())
#         print("gaussian", gaussian.min(), gaussian.max(), gaussian.mean())
#         print("mixture", mixture.min(), mixture.max(), mixture.mean())


#         raise ValueError("nll is nan")
#     return torch.mean(nll)  # Return the average NLL
def bivariate_nll_loss(y, pi, mu_x, mu_y, sigma_x, sigma_y, rhos):
    epsilon = 1e-12  # Small constant to prevent log(0) or division by zero
    x, y_actual = y[:, 0], y[:, 1]
    x = x.unsqueeze(1)
    y_actual = y_actual.unsqueeze(1)

    # Clamp rhos to ensure they are in a valid range (-1, 1)
    rhos = torch.clamp(rhos, -1 + epsilon, 1 - epsilon)
    sigma_x = torch.clamp(sigma_x, epsilon, 1e12)  # Adjust these limits according to your specific use-case
    sigma_y = torch.clamp(sigma_y, epsilon, 1e12)  # Adjust these limits according to your specific use-case

    z_numerator = (
        ((x - mu_x) / (sigma_x + epsilon)) ** 2 + ((y_actual - mu_y) / (sigma_y + epsilon)) ** 2 - 2 * rhos * (x - mu_x) * (y_actual - mu_y) / ((sigma_x * sigma_y) + epsilon)
    )
    z_denominator = 2 * (1 - rhos**2)

    z = z_numerator / (z_denominator + epsilon)

    normalization = 2 * torch.pi * sigma_x * sigma_y * torch.sqrt(1 - rhos**2 + epsilon)
    normalization = torch.clamp(normalization, epsilon, 1e12)

    gaussian = torch.exp(-z) / normalization

    # Combine the Gaussians
    mixture = pi * gaussian
    mixture = torch.sum(mixture, dim=1)  # Sum across the mixtures

    # Compute the negative log likelihood
    nll = -torch.log(mixture + epsilon)

    if torch.isnan(nll).any():
        warnings.warn("NaN values detected in the loss. Diagnostics info printed below.")

        print("y", y.min().item(), y.max().item(), y.mean().item())
        print("pi", pi.min().item(), pi.max().item(), pi.mean().item())
        print("mu_x", mu_x.min().item(), mu_x.max().item(), mu_x.mean().item())
        print("mu_y", mu_y.min().item(), mu_y.max().item(), mu_y.mean().item())
        print("sigma_x", sigma_x.min().item(), sigma_x.max().item(), sigma_x.mean().item())
        print("sigma_y", sigma_y.min().item(), sigma_y.max().item(), sigma_y.mean().item())
        print("rhos", rhos.min().item(), rhos.max().item(), rhos.mean().item())

        print("z", z.min().item(), z.max().item(), z.mean().item())
        print("normalization", normalization.min().item(), normalization.max().item(), normalization.mean().item())
        print("gaussian", gaussian.min().item(), gaussian.max().item(), gaussian.mean().item())
        print("mixture", mixture.min().item(), mixture.max().item(), mixture.mean().item())

    # Replace NaNs with zeros
    nll = torch.where(torch.isnan(nll), torch.zeros_like(nll), nll)

    return nll

    # print("z", z.min(), z.max(), z.mean())
    # print("normalization", normalization.min(), normalization.max(), normalization.mean())
    # print("gaussian", gaussian.min(), gaussian.max(), gaussian.mean())
    # print("mixture", mixture.min(), mixture.max(), mixture.mean())
    # pi = torch.clamp(pi, epsilon, 1 - epsilon)
    # sigma_x = torch.clamp(sigma_x, epsilon, 3)  # Assuming you don't expect larger std deviations
    # sigma_y = torch.clamp(sigma_y, epsilon, 3)
    # rhos = torch.clamp(rhos, -1 + epsilon, 1 - epsilon)


# def nll_loss(y, pi_logits, mu, sigma_logits):
#     """
#     Computes the negative log likelihood (NLL) for a mixture of Gaussians.

#     Parameters:
#     - y (torch.Tensor): Target values.
#         Shape: [batch_size, sequence_length, output_dim]
#         where:
#         - batch_size: Number of samples in the batch.
#         - sequence_length: Length of the trajectory or sequence.
#         - output_dim: Dimensionality of the output (e.g., 2 for 2D trajectories).

#     - pi_logits (torch.Tensor): Raw logits for the mixture weights of the Gaussian components.
#         Shape: [batch_size, sequence_length, num_gaussians]
#         where:
#         - num_gaussians: Number of Gaussian components in the mixture.

#     - mu (torch.Tensor): Means of the Gaussian components.
#         Shape: [batch_size, sequence_length, num_gaussians, output_dim]

#     - sigma_logits (torch.Tensor): Raw logits for the standard deviations of the Gaussian components.
#         Shape: [batch_size, sequence_length, num_gaussians, output_dim]

#     Returns:
#     - torch.Tensor: Negative log likelihood value.
#     """

#     # Transformations
#     pi = torch.softmax(pi_logits, dim=-1)
#     sigma = torch.exp(sigma_logits)

#     # Create the Gaussian distribution
#     m = torch.distributions.Normal(loc=mu, scale=sigma)

#     # Compute log probability for each Gaussian component
#     log_prob = m.log_prob(y.unsqueeze(1))

#     # Weight the log probabilities by the mixture probabilities
#     weighted_log_prob = torch.log(pi) + log_prob.sum(dim=-1)

#     # LogSumExp trick to compute the log of the sum of exponentials
#     max_log_prob = weighted_log_prob.max(dim=-1, keepdim=True).values
#     logsumexp = max_log_prob + torch.log((weighted_log_prob - max_log_prob).exp().sum(dim=-1, keepdim=True))

#     # Return the negative log likelihood
#     return -logsumexp.mean()


def rmse_loss(y_gt, y_pred):
    mse = mean_squared_error(y_pred, y_gt)
    return torch.sqrt(mse)
