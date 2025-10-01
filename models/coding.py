import math
import numpy as np
import torch
import torch.nn.functional as F
import constriction as cs

def _gaussian_cdf(x, mu, sigma):
    inv_sqrt2 = 0.7071067811865476
    z = (x - mu) / sigma.clamp_min(1e-6)
    return 0.5 * (1.0 + torch.erf(z * inv_sqrt2))

def discretized_gaussian_pmf(mu, sigma, Q=255):
    """
    PMF over integer support [-Q, Q] (size 2Q+1).
    mu, sigma: broadcastable to output.
    """
    device, dtype = mu.device, mu.dtype
    support = torch.arange(-Q, Q + 1, device=device, dtype=dtype)  # [2Q+1]
    left  = support - 0.5
    right = support + 0.5
    pmf = (_gaussian_cdf(right, mu, sigma) - _gaussian_cdf(left, mu, sigma)).clamp_min(1e-12)
    pmf = pmf / pmf.sum(dim=-1, keepdim=True)
    return pmf  # [..., 2Q+1]

def pmf_to_int_cdf(pmf, precision=16):
    """
    Convert PMF -> integer CDF for constriction:
    returns np.uint32 array shaped [..., 2Q+2], strictly increasing, last == 2^precision.
    """
    total = 1 << precision
    cdf = pmf.cumsum(dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # prepend 0
    cdf = (cdf * total).round().clamp_(0, total)
    # enforce monotonicity; last must be total
    ar = torch.arange(cdf.shape[-1], device=cdf.device, dtype=cdf.dtype)
    cdf = torch.maximum(cdf, ar)
    cdf[..., -1] = total
    return cdf.to(torch.uint32).cpu().numpy()

# ---------- encode / decode with constriction (queue) ----------

def encode_y_with_gaussian_constriction(y_hat, mu, var, Q=255, precision=16):
    """
    y_hat: int tensor (N,M,L) in [-Q, Q]
    mu, var: float tensors (N,M,L)
    returns: bytes bitstream
    """
    assert y_hat.shape == mu.shape == var.shape
    N, M, L = y_hat.shape

    enc = cs.stream.queue.RangeEncoder()

    # flatten in raster order; encode forward (queue encoder)
    y_flat  = y_hat.reshape(-1).tolist()
    mu_flat = mu.reshape(-1).tolist()
    var_flat= var.reshape(-1).tolist()

    for s_int, m, v in zip(y_flat, mu_flat, var_flat):
        s_int = int(max(-Q, min(Q, s_int)))
        sigma = math.sqrt(max(1e-9, float(v)))

        pmf = discretized_gaussian_pmf(torch.tensor(m), torch.tensor(sigma), Q=Q)
        cdf = pmf_to_int_cdf(pmf, precision=precision)  # np.uint32 [2Q+2]

        s_shift = s_int + Q  # map [-Q, Q] -> [0, 2Q]
        enc.encode_symbol(s_shift, cdf)

    return enc.get_compressed()


def decode_y_with_gaussian_constriction(bitstream, mu, var, Q=255, precision=16, device="cpu"):
    """
    Decode into torch.int32 tensor shaped (N,M,L) in [-Q, Q]
    """
    N, M, L = mu.shape
    dec = cs.stream.queue.RangeDecoder(bitstream)

    out = torch.empty((N*M*L,), dtype=torch.int32, device=device)
    mu_flat = mu.reshape(-1).tolist()
    var_flat= var.reshape(-1).tolist()

    for i, (m, v) in enumerate(zip(mu_flat, var_flat)):
        sigma = math.sqrt(max(1e-9, float(v)))
        pmf = discretized_gaussian_pmf(torch.tensor(m), torch.tensor(sigma), Q=Q)
        cdf = pmf_to_int_cdf(pmf, precision=precision)
        s_shift = dec.decode_symbol(cdf)     # [0..2Q]
        out[i]  = int(s_shift) - Q           # back to [-Q, Q]

    return out.view(N, M, L)