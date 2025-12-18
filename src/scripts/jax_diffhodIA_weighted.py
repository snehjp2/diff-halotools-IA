from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from diffhodIA_utils import (
    load_cleaned_catalogs_from_h5,
    mask_bad_halocat,
    plot_diagnostic,
)
from jax import config, jit, lax, random
from jax.nn import sigmoid

Array = jnp.ndarray


@jit
def _unitize(v: Array, eps: float = 1e-12) -> Array:
    """
    Normalize a tensor to have unit length.
    v: [N, 3] tensor
    Returns: [N, 3] tensor of unit vectors
    """
    return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)


@jit
def _erfi_real(x: Array) -> Array:
    """
    Stable real-valued erfi for JAX arrays.
    Matches your piecewise series/asymptotic form, with bounded iteration counts.
    """

    def small_branch(xs: Array) -> Array:
        # erfi(x) = 2/sqrt(pi) * sum_{n>=0} x^{2n+1} / (n! (2n+1))
        def body_fun(n, carry):
            term, s = carry
            term = term * (xs * xs) / n  # multiply by x^2 / n
            s = s + term / (2 * n + 1)
            return term, s

        term0 = xs
        s0 = xs
        termN, sN = lax.fori_loop(1, 20, body_fun, (term0, s0))
        return (2.0 / math.sqrt(math.pi)) * sN

    def large_branch(xl: Array) -> Array:
        inv = 1.0 / xl
        inv2 = inv * inv
        series = 1.0 + 0.5 * inv2 + 0.75 * inv2 * inv2 + 1.875 * inv2 * inv2 * inv2
        return jnp.exp(xl * xl) * series / (math.sqrt(math.pi) * xl)

    y_small = small_branch(x)
    y_large = large_branch(x)
    y = jnp.where(jnp.abs(x) <= 3.0, y_small, y_large)
    return y.astype(x.dtype)


@partial(jit, static_argnums=(3,))
def _sample_t_watson(key: Array, kappa: Array, u: Array, n_newton: int = 6) -> Array:
    eps = 1e-12

    def one(k_i, u_i):
        def neg_branch(_):
            km = jnp.sqrt(-k_i)
            den = jax.scipy.special.erf(km)
            arg = jnp.clip((2.0 * u_i - 1.0) * den, -1.0 + 1e-7, 1.0 - 1e-7)
            return jax.scipy.special.erfinv(arg) / (km + eps)

        def zer_branch(_):
            return 2.0 * u_i - 1.0

        def pos_branch(_):
            sp = jnp.sqrt(k_i)
            tp0 = 2.0 * u_i - 1.0
            den = _erfi_real(sp) + 1e-30

            def body(_, tp_curr):
                F = 0.5 * (_erfi_real(sp * tp_curr) / den + 1.0)
                pdf = (sp / math.sqrt(math.pi)) * jnp.exp(k_i * tp_curr * tp_curr) / den
                tp_next = jnp.clip(
                    tp_curr - (F - u_i) / (pdf + 1e-30), -1.0 + 1e-6, 1.0 - 1e-6
                )
                return tp_next

            return lax.fori_loop(0, n_newton, body, tp0)

        return lax.cond(
            k_i < -1e-12,
            neg_branch,
            lambda _: lax.cond(k_i > 1e-12, pos_branch, zer_branch, operand=None),
            operand=None,
        )

    t = jax.vmap(one)(kappa, u)
    return jnp.clip(t, -1.0 + 1e-6, 1.0 - 1e-6)


@partial(jit, static_argnums=(3,))  # n_newton is the 4th argument (index 3)
def sample_watson_orientations(
    key: Array, ref_dirs: Array, mu: Array | float, n_newton: int = 6
) -> Array:
    """
    Dimroth–Watson axial samples about ref_dirs (unit vectors).
    """

    u_axis = _unitize(ref_dirs)
    N = u_axis.shape[0]
    mu_arr = jnp.asarray(mu, dtype=u_axis.dtype)

    if mu_arr.ndim == 0:
        mu = jnp.full((N,), mu_arr, dtype=u_axis.dtype)
    else:
        mu = mu_arr.reshape(-1)
        if mu.shape[0] == 1:
            mu = jnp.repeat(mu, N, axis=0)

    mu = jnp.clip(mu, -1.0 + 1e-6, 1.0 - 1e-6)

    kappa = jnp.tan(0.5 * math.pi * mu)
    kappa = jnp.clip(kappa, -1e6, 1e6)

    key_u, key_phi = random.split(key)

    u_uni = jnp.clip(
        random.uniform(key_u, (N,), minval=0.0, maxval=1.0), 1e-7, 1 - 1e-7
    )
    t = _sample_t_watson(key_u, kappa, u_uni, n_newton=n_newton)

    xhat = jnp.broadcast_to(
        jnp.array([1.0, 0.0, 0.0], dtype=u_axis.dtype), u_axis.shape
    )
    yhat = jnp.broadcast_to(
        jnp.array([0.0, 1.0, 0.0], dtype=u_axis.dtype), u_axis.shape
    )
    alt = jnp.where(jnp.abs(u_axis[:, 0:1]) > 0.9, yhat, xhat)
    b1 = _unitize(jnp.cross(u_axis, alt))
    b2 = jnp.cross(u_axis, b1)

    phi = 2.0 * math.pi * random.uniform(key_phi, (N, 1))
    sinth = jnp.sqrt(jnp.clip(1.0 - t * t, 0.0)).reshape(N, 1)
    costh = t.reshape(N, 1)

    n = _unitize(costh * u_axis + sinth * (jnp.cos(phi) * b1 + jnp.sin(phi) * b2))
    return n


@jit
def Ncen(M: Array, logMmin: float, sigma_logM: float) -> Array:
    log_M = jnp.log10(M)
    term = (log_M - logMmin) / (jnp.abs(sigma_logM) + 1e-6)
    return 0.5 * (1.0 + jax.scipy.special.erf(term))


@jit
def Nsat(
    M: Array,
    logMmin: float,
    sigma_logM: float,
    logM0: float,
    logM1: float,
    alpha: float,
) -> Array:
    sigma_logM = jnp.abs(sigma_logM) + 1e-6
    alpha = jnp.abs(alpha) + 1e-6
    M0 = 10.0**logM0
    M1 = 10.0**logM1
    x = jnp.clip((M - M0) / M1, 0.0, None)
    return Ncen(M, logMmin, sigma_logM) * jnp.power(x, alpha)


@jit
def _logit(p: Array) -> Array:
    p = jnp.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
    return jnp.log(p) - jnp.log1p(-p)  # now safe


@jit
def sample_centrals_diffhod(key, mean_N_cen, *, relaxed: bool = True, tau: float = 0.1):
    p = jnp.clip(mean_N_cen, 0.0, 1.0)
    dtype = p.dtype
    tau = jnp.asarray(tau, dtype=dtype)
    pred = jnp.asarray(relaxed, dtype=bool)  # supports Python bool or scalar JAX bool

    def discrete_branch(_):
        u = random.uniform(key, p.shape, dtype=dtype)
        Hc = (u < p).astype(dtype)
        return Hc, Hc

    def relaxed_branch(_):
        u = jnp.clip(random.uniform(key, p.shape, dtype=dtype), 1e-6, 1 - 1e-6)
        eps_logistic = jnp.log(u) - jnp.log1p(-u)
        z = jax.nn.sigmoid((_logit(p).astype(dtype) + eps_logistic) / tau)
        z_hard = (z >= 0.5).astype(dtype)
        z_st = z_hard + (z - lax.stop_gradient(z))
        return z_hard, z_st

    return lax.cond(pred, relaxed_branch, discrete_branch, operand=None)


@partial(jit, static_argnames=["N_max", "relaxed"])
def sample_satellites_diffhod(
    key: jnp.ndarray,
    mean_N_sat: jnp.ndarray,
    *,
    N_max: int = 48,
    relaxed: bool = True,
    tau: float = 0.1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Binomial(N_max, p=lambda/N_max) via N_max independent (relaxed) Bernoullis.
    Returns (hard_counts:int32, straight_through:float32)
    """
    lam = jnp.clip(mean_N_sat, 0.0)
    invN = jnp.asarray(1.0 / N_max, dtype=lam.dtype)
    p = jnp.clip(lam * invN, 0.0, 1.0)

    H = p.shape[0]
    pred = jnp.asarray(relaxed, dtype=bool)

    def discrete_branch(_):
        u = random.uniform(key, (H, N_max), dtype=lam.dtype)
        trials = jnp.sum(u < p[:, None], axis=1)
        Kh = trials.astype(jnp.int32)
        return Kh, Kh.astype(jnp.float32)

    def relaxed_branch(_):
        u = random.uniform(key, (H, N_max), dtype=lam.dtype)
        u = jnp.clip(u, 1e-6, 1.0 - 1e-6)
        eps = jnp.log(u) - jnp.log1p(-u)
        logits = _logit(p).astype(lam.dtype)[:, None]
        z = sigmoid((logits + eps) / jnp.asarray(tau, lam.dtype))
        z_hard = (z >= 0.5).astype(z.dtype)
        z_st = z_hard + (z - lax.stop_gradient(z))
        Kh_soft = jnp.sum(z, axis=1)
        Kh_hard = jnp.sum(z_hard, axis=1).astype(jnp.int32)
        Kh_st = Kh_hard.astype(jnp.float32) + (Kh_soft - lax.stop_gradient(Kh_soft))
        return Kh_hard, Kh_st

    return lax.cond(pred, relaxed_branch, discrete_branch, operand=None)


@partial(jax.jit, static_argnames=["n_newton", "per_host_cap"])
def sample_nfw_about_hosts(
    key: Array,
    host_centers: Array,  # [H, 3]
    host_rvir: Array,  # [H]
    counts_per_host: Array,  # [H], int-like
    *,
    conc: float = 5.0,
    n_newton: int = 6,
    per_host_cap: int = 64,  # STATIC: max draws per host
):
    """
    Vectorized NFW radii via inverse CDF with fixed Newton iterations.

    Returns fixed-shape outputs plus a mask:
      pos_full:   [H*per_host_cap, 3]
      host_idx:   [H*per_host_cap]
      use_mask:   [H*per_host_cap]  (True for the first 'counts_per_host[h]' of each host h)
    """
    H = host_centers.shape[0]
    dtype = host_centers.dtype

    conc = jnp.asarray(conc, dtype=dtype)
    conc = jnp.maximum(conc, jnp.asarray(1e-8, dtype=dtype))

    counts = jnp.clip(counts_per_host.astype(jnp.int32), 0, per_host_cap)

    host_ids = jnp.arange(H, dtype=jnp.int32)
    host_idx = jnp.repeat(host_ids, per_host_cap)
    idx_in_host = jnp.tile(jnp.arange(per_host_cap, dtype=jnp.int32), (H,))
    use_mask = idx_in_host < counts[host_idx]

    rvir = host_rvir[host_idx]
    rs = rvir / conc

    key_u, key_dir = random.split(key)

    u = jnp.clip(
        random.uniform(key_u, (H * per_host_cap,), dtype=dtype), 1e-7, 1 - 1e-7
    )
    mc = jnp.log1p(conc) - conc / (1 + conc)
    y = u * mc
    x0 = conc * u

    def body_fun(_, x_curr):
        fx = jnp.log1p(x_curr) - x_curr / (1 + x_curr) - y
        dfx = x_curr / jnp.square(1 + x_curr)
        return jnp.clip(x_curr - fx / (dfx + 1e-30), 0.0)

    x_fin = lax.fori_loop(0, n_newton, body_fun, x0)
    r = x_fin * rs

    z = random.normal(key_dir, (H * per_host_cap, 3), dtype=dtype)
    z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)

    pos_full = host_centers[host_idx] + r[:, None] * z

    return pos_full, host_idx, use_mask


# -------------------- per-host softmax over ranks --------------------
@jit
def per_host_softmax_over_ranks(
    sub_host_ids: Array, sub_mvir: Array, t_rank: float = 0.5
) -> Array:
    """
    Compute softmax over (within-host) ranks (massive-first).
    Exactly mirrors the idea in your torch code but fully vectorized:

    1) Sort by (host_id asc, mass desc) to get ranks (0,1,2,...) per host.
    2) logits = -rank / t_rank
    3) per-host softmax via segment_max/sum.
    4) Scatter back to original order.
    """
    t_rank = jnp.asarray(t_rank, dtype=sub_mvir.dtype)
    t_rank = jnp.maximum(t_rank, jnp.asarray(1e-12, dtype=sub_mvir.dtype))

    n = sub_mvir.shape[0]
    if n == 0:
        return jnp.zeros((0,), dtype=sub_mvir.dtype)

    order = jnp.lexsort((-sub_mvir, sub_host_ids))
    inv_order = jnp.empty_like(order)
    inv_order = inv_order.at[order].set(jnp.arange(n, dtype=order.dtype))

    h_sorted = sub_host_ids[order]

    is_start = jnp.concatenate([jnp.array([True]), (h_sorted[1:] != h_sorted[:-1])])
    is_end = jnp.concatenate([(h_sorted[1:] != h_sorted[:-1]), jnp.array([True])])
    idx = jnp.arange(n, dtype=jnp.int32)
    start_idx_mark = jnp.where(is_start, idx, -1)

    def scan_max(carry, x):
        new = jnp.maximum(carry, x)
        return new, new

    _, last_start_idx = lax.scan(scan_max, jnp.int32(-1), start_idx_mark)
    ranks_sorted = idx - last_start_idx

    logits_sorted = -ranks_sorted.astype(sub_mvir.dtype) / t_rank

    def run_max(carry, inp):
        start, val = inp
        cur = jnp.where(start, val, jnp.maximum(carry, val))
        return cur, cur

    _, runmax = lax.scan(
        run_max, jnp.array(-jnp.inf, logits_sorted.dtype), (is_start, logits_sorted)
    )

    def backfill_max(carry, inp):
        end, v = inp
        out = jnp.where(end, v, carry)
        return out, out

    _, max_per_pos_rev = lax.scan(
        backfill_max,
        jnp.array(-jnp.inf, logits_sorted.dtype),
        (is_end[::-1], runmax[::-1]),
    )
    max_per_pos = max_per_pos_rev[::-1]

    exp_shift = jnp.exp(logits_sorted - max_per_pos)

    def run_sum(carry, inp):
        start, val = inp
        cur = jnp.where(start, val, carry + val)
        return cur, cur

    _, runsum = lax.scan(
        run_sum, jnp.array(0.0, exp_shift.dtype), (is_start, exp_shift)
    )

    def backfill_sum(carry, inp):
        end, v = inp
        out = jnp.where(end, v, carry)
        return out, out

    _, sum_per_pos_rev = lax.scan(
        backfill_sum, jnp.array(0.0, exp_shift.dtype), (is_end[::-1], runsum[::-1])
    )
    sum_per_pos = sum_per_pos_rev[::-1]

    q_sorted = exp_shift / (sum_per_pos + 1e-30)

    q = jnp.empty_like(q_sorted)
    q = q.at[order].set(q_sorted)
    return q


@jit
def _satellite_mu_from_radius(r_over_rvir: Array, a: float, gamma: float) -> Array:
    r_safe = jnp.nan_to_num(r_over_rvir, nan=0.0, posinf=1e3, neginf=0.0)
    r_clipped = jnp.clip(r_safe, 1e-5, 1e3)
    mu = a * jnp.power(r_clipped, gamma)
    return jnp.clip(mu, -0.999, 0.999)

# @jit
def compute_projected_separation_and_los(
    pos_i: Array, pos_j: Array, los_axis: int = 2
) -> tuple[Array, Array]:
    """
    Compute projected separation rp and line-of-sight separation pi.
    
    Args:
        pos_i, pos_j: [3] position vectors
        los_axis: which axis is line-of-sight (default: z=2)
    
    Returns:
        rp: projected separation
        pi: line-of-sight separation
    """
    delta = pos_j - pos_i
    
    # Line-of-sight component
    pi = jnp.abs(delta[los_axis])
    
    # Projected separation (perpendicular to LOS)
    if los_axis == 0:
        rp = jnp.sqrt(delta[1]**2 + delta[2]**2)
    elif los_axis == 1:
        rp = jnp.sqrt(delta[0]**2 + delta[2]**2)
    else:  # los_axis == 2
        rp = jnp.sqrt(delta[0]**2 + delta[1]**2)
    
    return rp, pi


# @jit
def bin_selection_projected(
    rp: Array, pi: Array, rp_min: float, rp_max: float, pi_max: float
) -> Array:
    """
    Returns 1.0 if pair falls in bin, 0.0 otherwise.
    Equivalent to B_i(x_α, x_β) in paper Eq. 14.
    """
    in_rp_bin = (rp >= rp_min) & (rp < rp_max)
    in_pi_range = pi < pi_max
    return (in_rp_bin & in_pi_range).astype(rp.dtype)


# @partial(jit, static_argnames=['n_bins', 'pi_max', 'los_axis'])
def compute_wp_differentiable(
    positions: Array,      # [N, 3]
    orientations: Array,   # [N, 3]  
    weights: Array,        # [N]
    rp_bins: Array,        # [n_bins+1]
    pi_max: float = 40.0,
    los_axis: int = 2,
    n_bins: int = None
) -> Array:
    """
    Differentiable projected correlation function wp(rp).
    
    Implements the weighted pair counting from SHAMNet §4.3.
    
    Returns:
        wp: [n_bins] array of projected correlation function
    """
    N = positions.shape[0]
    if n_bins is None:
        n_bins = len(rp_bins) - 1
    
    # Compute DD (data-data weighted pairs) for each bin
    def compute_bin_dd(bin_idx):
        rp_min = rp_bins[bin_idx]
        rp_max = rp_bins[bin_idx + 1]
        
        def pair_contribution(i):
            def inner_pair(j):
                # Skip self-pairs
                skip = (i == j).astype(weights.dtype)
                
                rp, pi = compute_projected_separation_and_los(
                    positions[i], positions[j], los_axis
                )
                
                in_bin = bin_selection_projected(
                    rp, pi, rp_min, rp_max, pi_max
                )
                
                # Weight by product of probabilities (Eq. 14)
                weighted = (1.0 - skip) * weights[i] * weights[j] * in_bin
                return weighted
            
            # Sum over all j partners for this i
            return jnp.sum(jax.vmap(inner_pair)(jnp.arange(N)))
        
        # Sum over all i
        dd = jnp.sum(jax.vmap(pair_contribution)(jnp.arange(N)))
        return dd
    
    # Compute for all bins
    DD = jax.vmap(compute_bin_dd)(jnp.arange(n_bins))
    
    # Compute RR (random-random) following paper Eq. 15-17
    # For each bin, RR = (Σw)² * (1 - 1/N_eff) * (bin_volume / total_volume)
    total_weight = jnp.sum(weights)
    weight_sq_sum = jnp.sum(weights**2)
    N_eff = total_weight**2 / (weight_sq_sum + 1e-30)
    
    # Bin volumes (annulus area × 2*pi_max for projected)
    bin_areas = jnp.pi * (rp_bins[1:]**2 - rp_bins[:-1]**2)
    total_area = jnp.pi * rp_bins[-1]**2
    
    RR = (total_weight**2) * (1.0 - 1.0/N_eff) * (bin_areas / total_area)
    
    # Projected correlation function (Eq. 13)
    xi_rp = DD / (RR + 1e-30) - 1.0
    
    # Integrate to get wp = 2 * Σ ξ(rp, pi) * dpi
    # For simplicity, assuming ξ is constant in pi:
    wp = 2.0 * pi_max * xi_rp
    
    return wp


# @partial(jit, static_argnames=['n_bins'])
def compute_position_orientation_correlation(
    positions: Array,      # [N, 3]
    orientations: Array,   # [N, 3]
    weights: Array,        # [N]
    rp_bins: Array,        # [n_bins+1]
    los_axis: int = 2,
    n_bins: int = None
) -> Array:
    """
    Differentiable position-orientation correlation.
    
    Measures correlation between position and orientation alignment.
    For each separation bin, compute <cos(θ)> where θ is angle 
    between orientation and separation vector.
    
    Returns:
        w_p_theta: [n_bins] mean cos(theta) in each bin
    """
    N = positions.shape[0]
    if n_bins is None:
        n_bins = len(rp_bins) - 1
    
    def compute_bin_correlation(bin_idx):
        rp_min = rp_bins[bin_idx]
        rp_max = rp_bins[bin_idx + 1]
        
        def pair_contribution(i):
            def inner_pair(j):
                skip = (i == j).astype(weights.dtype)
                
                # Separation vector
                delta = positions[j] - positions[i]
                rp, _ = compute_projected_separation_and_los(
                    positions[i], positions[j], los_axis
                )
                
                in_bin = ((rp >= rp_min) & (rp < rp_max)).astype(weights.dtype)
                
                # Orientation alignment with separation
                delta_norm = delta / (jnp.linalg.norm(delta) + 1e-12)
                cos_theta = jnp.dot(orientations[i], delta_norm)
                
                # Weighted contribution
                weight_ij = (1.0 - skip) * weights[i] * weights[j] * in_bin
                
                return weight_ij * cos_theta, weight_ij
            
            results = jax.vmap(inner_pair)(jnp.arange(N))
            return results[0].sum(), results[1].sum()
        
        results = jax.vmap(pair_contribution)(jnp.arange(N))
        numerator = results[0].sum()
        denominator = results[1].sum()
        
        return numerator / (denominator + 1e-30)
    
    return jax.vmap(compute_bin_correlation)(jnp.arange(n_bins))


# @partial(jit, static_argnames=['n_bins'])
def compute_orientation_orientation_correlation(
    positions: Array,      # [N, 3]
    orientations: Array,   # [N, 3]
    weights: Array,        # [N]
    rp_bins: Array,        # [n_bins+1]
    los_axis: int = 2,
    n_bins: int = None
) -> Array:
    """
    Differentiable orientation-orientation correlation.
    
    Measures correlation between orientations of galaxy pairs.
    For each separation bin, compute <|n_i · n_j|> where n are
    orientation unit vectors.
    
    Returns:
        w_theta_theta: [n_bins] mean |cos(angle)| between orientations
    """
    N = positions.shape[0]
    if n_bins is None:
        n_bins = len(rp_bins) - 1
    
    def compute_bin_correlation(bin_idx):
        rp_min = rp_bins[bin_idx]
        rp_max = rp_bins[bin_idx + 1]
        
        def pair_contribution(i):
            def inner_pair(j):
                skip = (i == j).astype(weights.dtype)
                
                rp, _ = compute_projected_separation_and_los(
                    positions[i], positions[j], los_axis
                )
                
                in_bin = ((rp >= rp_min) & (rp < rp_max)).astype(weights.dtype)
                
                # Orientation alignment (absolute value for axial symmetry)
                cos_theta = jnp.abs(jnp.dot(orientations[i], orientations[j]))
                
                weight_ij = (1.0 - skip) * weights[i] * weights[j] * in_bin
                
                return weight_ij * cos_theta, weight_ij
            
            results = jax.vmap(inner_pair)(jnp.arange(N))
            return results[0].sum(), results[1].sum()
        
        results = jax.vmap(pair_contribution)(jnp.arange(N))
        numerator = results[0].sum()
        denominator = results[1].sum()
        
        return numerator / (denominator + 1e-30)
    
    return jax.vmap(compute_bin_correlation)(jnp.arange(n_bins))


@dataclass
class DiffHalotoolsIA:
    """
    JAX version of your DiffHalotoolsIA.
    subcat: a dict-like with columns (same keys you required).
    params: [7] -> [mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha]
    """

    subcat: any
    params: Array  # shape (7,)
    do_discrete: bool = True
    do_nfw_fallback: bool = True
    alignment_model: str = "radial"  # or "subhalo"
    alignment_strength: str = "constant"  # or "radial"
    a_strength: float = 0.8
    gamma_strength: float = 1.0
    relaxed: bool = True
    tau: float = 0.1
    Nmax_sat: int = 256
    t_rank: float = 0.5
    max_output_galaxies: int = 1000000
    seed: Optional[int] = None
    rp_bins: Array = None  # projected separation bins
    pi_max: float = 40.0   # line-of-sight integration limit

    def __post_init__(self):
        import numpy as np

        req = [
            "halo_id",
            "halo_upid",
            "halo_hostid",
            "halo_mvir",
            "halo_x",
            "halo_y",
            "halo_z",
            "halo_rvir",
            "halo_axisA_x",
            "halo_axisA_y",
            "halo_axisA_z",
            "halo_b_to_a",
        ]

        missing = [k for k in req if k not in self.subcat.colnames]
        if missing:
            raise KeyError(f"subcat missing required keys: {missing}")

        if self.alignment_strength.lower() == "radial":
            config.update("jax_enable_x64", True)

        halo_id = np.asarray(self.subcat["halo_id"], dtype=np.int64)
        halo_upid = np.asarray(self.subcat["halo_upid"], dtype=np.int64)
        halo_hostid = np.asarray(self.subcat["halo_hostid"], dtype=np.int64)

        halo_mvir = np.asarray(self.subcat["halo_mvir"], dtype=np.float32)
        halo_x = np.asarray(self.subcat["halo_x"], dtype=np.float32)
        halo_y = np.asarray(self.subcat["halo_y"], dtype=np.float32)
        halo_z = np.asarray(self.subcat["halo_z"], dtype=np.float32)
        halo_rvir = np.asarray(self.subcat["halo_rvir"], dtype=np.float32)

        ax_x = np.asarray(self.subcat["halo_axisA_x"], dtype=np.float32)
        ax_y = np.asarray(self.subcat["halo_axisA_y"], dtype=np.float32)
        ax_z = np.asarray(self.subcat["halo_axisA_z"], dtype=np.float32)

        host_mask = halo_upid == -1
        sub_mask = ~host_mask

        host_keys = halo_hostid[host_mask]
        parent_keys = halo_hostid[sub_mask]
        host_index_of = {int(k): i for i, k in enumerate(host_keys.tolist())}
        idx_list = []
        keep_mask = np.zeros(parent_keys.shape[0], dtype=bool)
        for i, k in enumerate(parent_keys):
            j = host_index_of.get(int(k), None)
            if j is not None:
                idx_list.append(j)
                keep_mask[i] = True
        sub_host_idx = np.asarray(idx_list, dtype=np.int64)

        host_pos = jnp.asarray(
            np.stack([halo_x[host_mask], halo_y[host_mask], halo_z[host_mask]], 1)
        )
        host_rvir = jnp.asarray(halo_rvir[host_mask])
        host_mvir = jnp.asarray(halo_mvir[host_mask])

        ba_all = np.asarray(self.subcat["halo_b_to_a"], dtype=np.float32)

        host_ba = ba_all[host_mask]
        sub_ba = ba_all[sub_mask][keep_mask]

        self.host_b_to_a = jnp.asarray(np.clip(host_ba, 0.1, 1.0))
        self.sub_b_to_a = jnp.asarray(np.clip(sub_ba, 0.1, 1.0))

        sub_pos = jnp.asarray(
            np.stack(
                [
                    halo_x[sub_mask][keep_mask],
                    halo_y[sub_mask][keep_mask],
                    halo_z[sub_mask][keep_mask],
                ],
                1,
            )
        )
        sub_mvir = jnp.asarray(halo_mvir[sub_mask][keep_mask])
        sub_host_ids = jnp.asarray(sub_host_idx, dtype=jnp.int32)

        host_axis = jnp.asarray(
            np.stack([ax_x[host_mask], ax_y[host_mask], ax_z[host_mask]], 1)
        )
        sub_axis = jnp.asarray(
            np.stack(
                [
                    ax_x[sub_mask][keep_mask],
                    ax_y[sub_mask][keep_mask],
                    ax_z[sub_mask][keep_mask],
                ],
                1,
            )
        )

        self.host_pos = jnp.asarray(host_pos)
        self.host_rvir = host_rvir
        self.host_mvir = host_mvir

        self.sub_pos = jnp.asarray(sub_pos)
        self.sub_mvir = sub_mvir
        self.sub_host_ids = sub_host_ids

        self.host_axis = _unitize(host_axis)
        self.sub_axis = _unitize(sub_axis)

        self.host_halo_id = jnp.asarray(halo_id[host_mask], dtype=jnp.int32)
        self.sub_halo_id = jnp.asarray(halo_id[sub_mask][keep_mask], dtype=jnp.int32)

        self.alignment_model = self.alignment_model.lower()
        self.alignment_strength = self.alignment_strength.lower()
        self.a_strength = float(self.a_strength)
        self.gamma_strength = float(self.gamma_strength)
        if self.rp_bins is None:
            self.rp_bins = jnp.logspace(-1, 1.5, 15)  # 0.1 to ~30 Mpc/h

        if self.host_pos.size == 0:
            raise ValueError("No host halos found after filtering (halo_upid == -1).")

        self.key = random.PRNGKey(0 if self.seed is None else int(self.seed))

    @property
    def mean_central_per_host(self) -> Array:
        mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha = self.params
        return Ncen(self.host_mvir, logMmin.astype(float), sigma_logM.astype(float))

    @property
    def mean_satellite_per_host(self) -> Array:
        mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha = self.params
        return Nsat(
            self.host_mvir,
            logMmin.astype(float),
            sigma_logM.astype(float),
            logM0.astype(float),
            logM1.astype(float),
            alpha.astype(float),
        )

    def compute_weights_and_attributes(self) -> tuple[Array, Array, Array, Array]:
        """
        Compute galaxy weights and attributes without hard sampling.
        
        Returns:
            positions: [N_sources, 3] - all possible galaxy positions
            orientations: [N_sources, 3] - all possible galaxy orientations  
            weights: [N_sources] - probability each galaxy exists
            is_central: [N_sources] - mask for central vs satellite
        """
        (mu_cen, mu_sat, logMmin, sigma_logM, 
         logM0, logM1, alpha) = [
            jnp.asarray(x, dtype=self.host_mvir.dtype) 
            for x in jnp.asarray(self.params)
        ]
        
        # Central weights (probabilities)
        mean_N_cen = Ncen(self.host_mvir, logMmin, sigma_logM)
        
        # Generate central orientations for ALL hosts
        k_oric, self.key = random.split(self.key)
        cent_ori = sample_watson_orientations(
            k_oric, self.host_axis, mu_cen
        )
        
        # Satellite weights via softmax
        mean_N_sat = Nsat(
            self.host_mvir, logMmin, sigma_logM, 
            logM0, logM1, alpha
        )
        q = per_host_softmax_over_ranks(
            self.sub_host_ids, self.sub_mvir, t_rank=self.t_rank
        )
        sat_w = q * mean_N_sat[self.sub_host_ids]
        
        # Satellite orientations for ALL subhalos
        if self.alignment_model == "subhalo":
            ref_s = self.sub_axis
        else:
            base = self.sub_pos - self.host_pos[self.sub_host_ids]
            ref_s = _unitize(base)
        
        if self.alignment_strength == "constant":
            mu_sub = jnp.full((self.sub_pos.shape[0],), 
                            mu_sat.astype(ref_s.dtype), dtype=ref_s.dtype)
        else:
            base_vec = self.sub_pos - self.host_pos[self.sub_host_ids]
            r = jnp.linalg.norm(base_vec, axis=1)
            rvir_sel = self.host_rvir[self.sub_host_ids]
            r_over = r / (rvir_sel + 1e-12)
            mu_sub = _satellite_mu_from_radius(
                r_over, self.a_strength, self.gamma_strength
            )
        
        k_oris, self.key = random.split(self.key)
        sat_ori = sample_watson_orientations(k_oris, ref_s, mu_sub)
        
        # Concatenate all sources
        all_pos = jnp.concatenate([self.host_pos, self.sub_pos], axis=0)
        all_ori = jnp.concatenate([cent_ori, sat_ori], axis=0)
        all_weights = jnp.concatenate([mean_N_cen, sat_w], axis=0)
        
        n_hosts = self.host_pos.shape[0]
        is_central = jnp.concatenate([
            jnp.ones(n_hosts, dtype=bool),
            jnp.zeros(self.sub_pos.shape[0], dtype=bool)
        ])
        
        return all_pos, all_ori, all_weights, is_central
    
    def compute_positions_and_weights(self, params: Array | None = None):
        """
        Deterministic 'soft' HOD: positions + weights only.

        Returns:
            pos_all: [N_sources, 3] host and sub positions
            w_all:   [N_sources]    expected galaxy count / probability
            is_central: [N_sources] bool mask
        """
        if params is None:
            params = self.params

        (mu_cen, mu_sat, logMmin, sigma_logM,
         logM0, logM1, alpha) = [
            jnp.asarray(x, dtype=self.host_mvir.dtype)
            for x in jnp.asarray(params)
        ]

        # Centrals: one potential galaxy per host
        mean_N_cen = Ncen(self.host_mvir, logMmin, sigma_logM)    # [H]

        # Satellites: Nsat per host, distributed over subhalos via softmax
        mean_N_sat = Nsat(
            self.host_mvir, logMmin, sigma_logM,
            logM0, logM1, alpha
        )  # [H]

        q = per_host_softmax_over_ranks(
            self.sub_host_ids, self.sub_mvir, t_rank=self.t_rank
        )  # [N_sub]
        sat_w = q * mean_N_sat[self.sub_host_ids]                 # [N_sub]

        # Positions: host centers + subhalo centers
        pos_hosts = self.host_pos                                  # [H,3]
        pos_subs  = self.sub_pos                                   # [N_sub,3]

        pos_all = jnp.concatenate([pos_hosts, pos_subs], axis=0)   # [N,3]
        w_all   = jnp.concatenate([mean_N_cen, sat_w], axis=0)     # [N]

        n_hosts = pos_hosts.shape[0]
        is_central = jnp.concatenate([
            jnp.ones(n_hosts, dtype=bool),
            jnp.zeros(pos_subs.shape[0], dtype=bool),
        ])

        return pos_all, w_all, is_central

    def return_catalog(self) -> Array:
        """
        Build and return (Ng, 6) array [x,y,z,nx,ny,nz]
        """
        (mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha) = [
            jnp.asarray(x, dtype=self.host_mvir.dtype) for x in jnp.asarray(self.params)
        ]

        mean_N_cen = Ncen(self.host_mvir, logMmin, sigma_logM)  # [H]
        mean_N_sat = Nsat(self.host_mvir, logMmin, sigma_logM, logM0, logM1, alpha)

        k1, self.key = random.split(self.key)
        Hc, Hc_st = sample_centrals_diffhod(
            k1, mean_N_cen, relaxed=self.relaxed, tau=self.tau
        )

        if self.do_discrete:
            sel_c = Hc.astype(bool)
            cat_cent = self.host_pos[sel_c]
            ref_c = self.host_axis[sel_c]
            k_oric, self.key = random.split(self.key)
            ori_cent = sample_watson_orientations(k_oric, ref_c, mu_cen)
        else:
            sel = mean_N_cen > 0
            cat_cent = self.host_pos[sel]
            ref_c = self.host_axis[sel]
            k_oric, self.key = random.split(self.key)
            ori_cent = sample_watson_orientations(k_oric, ref_c, mu_cen)

        q = per_host_softmax_over_ranks(
            self.sub_host_ids, self.sub_mvir, t_rank=self.t_rank
        )
        sat_w = q * mean_N_sat[self.sub_host_ids]

        k2, self.key = random.split(self.key)
        Kh, Kh_st = sample_satellites_diffhod(
            k2, mean_N_sat, N_max=self.Nmax_sat, relaxed=self.relaxed, tau=self.tau
        )

        n_host = self.host_pos.shape[0]
        n_sub = self.sub_pos.shape[0]

        if self.do_discrete:
            order = jnp.lexsort((-q, self.sub_host_ids))
            h_sorted = self.sub_host_ids[order]
            is_start = jnp.concatenate(
                [jnp.array([True]), (h_sorted[1:] != h_sorted[:-1])]
            )
            idx = jnp.arange(n_sub, dtype=jnp.int32)
            start_idx_mark = jnp.where(is_start, idx, -1)

            def scan_max(carry, x):
                new = jnp.maximum(carry, x)
                return new, new

            _, last_start_idx = lax.scan(scan_max, jnp.int32(-1), start_idx_mark)
            ranks_in_host = idx - last_start_idx
            keep_sorted = ranks_in_host < Kh[h_sorted]
            chosen_sorted_idx = order[keep_sorted]
            sat_counts = (
                jnp.zeros((n_sub,), dtype=jnp.int32).at[chosen_sorted_idx].add(1)
            )

            placed_per_host = (
                jnp.zeros((n_host,), dtype=jnp.int32)
                .at[self.sub_host_ids]
                .add(sat_counts)
            )
            deficit = jnp.clip(Kh - placed_per_host, 0)

            cat_sat_sub = jnp.repeat(self.sub_pos, sat_counts, axis=0)
            chosen_sub_idx = jnp.nonzero(
                sat_counts, size=sat_counts.size, fill_value=-1
            )[0]
            chosen_sub_idx = chosen_sub_idx[chosen_sub_idx >= 0]

            if self.alignment_model == "subhalo":
                ref_s_sub = self.sub_axis[chosen_sub_idx]
            else:
                base = self.sub_pos - self.host_pos[self.sub_host_ids]
                ref_s_sub = _unitize(base)[chosen_sub_idx]

            if self.alignment_strength == "constant":
                mu_sub = jnp.full(
                    (ref_s_sub.shape[0],), mu_sat.astype(float), dtype=ref_s_sub.dtype
                )
            else:
                base_vec = (self.sub_pos - self.host_pos[self.sub_host_ids])[
                    chosen_sub_idx
                ]
                r = jnp.linalg.norm(base_vec, axis=1)
                rvir_sel = self.host_rvir[self.sub_host_ids[chosen_sub_idx]]
                r_over = r / (rvir_sel + 1e-12)
                mu_sub = _satellite_mu_from_radius(
                    r_over, float(self.a_strength), float(self.gamma_strength)
                )

            k3, self.key = random.split(self.key)
            mu_sub = jnp.nan_to_num(mu_sub, nan=0.0)
            mu_sub = jnp.clip(mu_sub, -0.999, 0.999)
            ori_sat_sub = sample_watson_orientations(k3, ref_s_sub, mu_sub)

            if self.do_nfw_fallback:
                k4, self.key = random.split(self.key)
                pos_full, host_idx_full, use_mask = sample_nfw_about_hosts(
                    k4,
                    self.host_pos,
                    self.host_rvir,
                    deficit,
                    conc=5.0,
                    n_newton=6,
                    per_host_cap=64,
                )

                nfw_pts = pos_full[use_mask]
                nfw_host_idx = host_idx_full[use_mask]

                if nfw_pts.shape[0] > 0:
                    disp = nfw_pts - self.host_pos[nfw_host_idx]
                    r_hat = _unitize(disp)

                    if self.alignment_strength == "constant":
                        mu_nfw = jnp.full(
                            (r_hat.shape[0],), mu_sat.astype(float), dtype=r_hat.dtype
                        )
                    else:
                        r = jnp.linalg.norm(disp, axis=1)
                        rvir_sel = self.host_rvir[nfw_host_idx]
                        r_over = r / (rvir_sel + 1e-12)
                        mu_nfw = _satellite_mu_from_radius(
                            r_over, float(self.a_strength), float(self.gamma_strength)
                        )

                    k5, self.key = random.split(self.key)
                    ori_sat_nfw = sample_watson_orientations(k5, r_hat, mu_nfw)
                else:
                    ori_sat_nfw = nfw_pts  # empty

                cat_sat = jnp.concatenate([cat_sat_sub, nfw_pts], axis=0)
                ori_sat = jnp.concatenate([ori_sat_sub, ori_sat_nfw], axis=0)
            else:
                cat_sat = cat_sat_sub
                ori_sat = ori_sat_sub

        else:
            k_soft = jnp.rint(sat_w).astype(jnp.int32)
            cat_sat = jnp.repeat(self.sub_pos, k_soft, axis=0)
            if self.alignment_model == "subhalo":
                ref_s_sub = jnp.repeat(self.sub_axis, k_soft, axis=0)
            else:
                base = self.sub_pos - self.host_pos[self.sub_host_ids]
                ref_s_sub = jnp.repeat(_unitize(base), k_soft, axis=0)

            if self.alignment_strength == "constant":
                mu_soft = jnp.full(
                    (ref_s_sub.shape[0],), mu_sat.astype(float), dtype=ref_s_sub.dtype
                )
            else:
                base = self.sub_pos - self.host_pos[self.sub_host_ids]
                r_all = jnp.linalg.norm(base, axis=1)
                rvir_all = self.host_rvir[self.sub_host_ids]
                r_over_all = r_all / (rvir_all + 1e-12)
                r_over_rep = jnp.repeat(r_over_all, k_soft, axis=0)
                mu_soft = _satellite_mu_from_radius(
                    r_over_rep, float(self.a_strength), float(self.gamma_strength)
                )

            k6, self.key = random.split(self.key)
            ori_sat = sample_watson_orientations(k6, ref_s_sub, mu_soft)

        if cat_cent.size == 0 and cat_sat.size == 0:
            return jnp.zeros((0, 6), dtype=self.host_pos.dtype)
        pos = jnp.concatenate([cat_cent, cat_sat], axis=0)
        ori = jnp.concatenate([ori_cent, ori_sat], axis=0)
        return jnp.concatenate([pos, ori], axis=1)

    def return_catalog_fixed_shape(self) -> tuple[Array, Array]:
        """
        Build and return FIXED-SHAPE catalog suitable for JIT/HMC.
        Returns:
            catalog: [max_output_galaxies, 6] array [x,y,z,nx,ny,nz]
            mask: [max_output_galaxies] bool array indicating valid galaxies
        """
        (mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha) = [
            jnp.asarray(x, dtype=self.host_mvir.dtype) for x in jnp.asarray(self.params)
        ]

        mean_N_cen = Ncen(self.host_mvir, logMmin, sigma_logM)
        mean_N_sat = Nsat(self.host_mvir, logMmin, sigma_logM, logM0, logM1, alpha)

        k1, self.key = random.split(self.key)
        Hc, Hc_st = sample_centrals_diffhod(
            k1, mean_N_cen, relaxed=self.relaxed, tau=self.tau
        )

        n_hosts = self.host_pos.shape[0]

        k_oric, self.key = random.split(self.key)
        cent_ori = sample_watson_orientations(k_oric, self.host_axis, mu_cen)

        q = per_host_softmax_over_ranks(
            self.sub_host_ids, self.sub_mvir, t_rank=self.t_rank
        )
        sat_w = q * mean_N_sat[self.sub_host_ids]

        k2, self.key = random.split(self.key)
        Kh, Kh_st = sample_satellites_diffhod(
            k2, mean_N_sat, N_max=self.Nmax_sat, relaxed=self.relaxed, tau=self.tau
        )

        n_sub = self.sub_pos.shape[0]

        if self.alignment_model == "subhalo":
            ref_s = self.sub_axis
        else:
            base = self.sub_pos - self.host_pos[self.sub_host_ids]
            ref_s = _unitize(base)

        if self.alignment_strength == "constant":
            mu_sub = jnp.full((n_sub,), mu_sat.astype(float), dtype=ref_s.dtype)
        else:
            base_vec = self.sub_pos - self.host_pos[self.sub_host_ids]
            r = jnp.linalg.norm(base_vec, axis=1)
            rvir_sel = self.host_rvir[self.sub_host_ids]
            r_over = r / (rvir_sel + 1e-12)
            mu_sub = _satellite_mu_from_radius(
                r_over, float(self.a_strength), float(self.gamma_strength)
            )

        k3, self.key = random.split(self.key)
        sat_ori = sample_watson_orientations(k3, ref_s, mu_sub)

        all_pos = jnp.concatenate([self.host_pos, self.sub_pos], axis=0)
        all_ori = jnp.concatenate([cent_ori, sat_ori], axis=0)

        cent_weights = Hc_st
        sat_weights = sat_w
        all_weights = jnp.concatenate([cent_weights, sat_weights], axis=0)

        valid_mask_full = all_weights > 1e-6

        max_gal = self.max_output_galaxies
        total_sources = n_hosts + n_sub

        if total_sources >= max_gal:
            catalog = jnp.concatenate([all_pos[:max_gal], all_ori[:max_gal]], axis=1)
            valid_mask = valid_mask_full[:max_gal]
        else:
            pad_size = max_gal - total_sources
            all_pos_padded = jnp.pad(all_pos, ((0, pad_size), (0, 0)), mode="constant")
            all_ori_padded = jnp.pad(all_ori, ((0, pad_size), (0, 0)), mode="constant")
            catalog = jnp.concatenate([all_pos_padded, all_ori_padded], axis=1)
            valid_mask = jnp.pad(
                valid_mask_full,
                ((0, pad_size),),
                mode="constant",
                constant_values=False,
            )

        return catalog, valid_mask

    def return_catalog_with_ids(self):
        """
        Returns:
        catalog: [Ng, 6] float array [x,y,z,nx,ny,nz]
        meta: dict with
            - host_idx_per_gal: [Ng] int host index (into host arrays)
            - host_halo_id_per_gal: [Ng] int64 host halo_id for each galaxy
            - sub_halo_id_per_gal: [Ng] int64 subhalo_id for satellites placed on subs, -1 for centrals/NFW
            - has_gal_host_mask: [n_hosts] bool mask: which hosts ended up with ≥1 galaxy
            - hosts_with_galaxy_ids: [n_sel] int64 array of host halo_id with any galaxy
        """
        (mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha) = [
            jnp.asarray(x, dtype=self.host_mvir.dtype) for x in jnp.asarray(self.params)
        ]
        mean_N_cen = Ncen(self.host_mvir, logMmin, sigma_logM)  # [H]
        mean_N_sat = Nsat(self.host_mvir, logMmin, sigma_logM, logM0, logM1, alpha)

        k1, self.key = random.split(self.key)
        Hc, _ = sample_centrals_diffhod(
            k1, mean_N_cen, relaxed=self.relaxed, tau=self.tau
        )
        sel_c = Hc.astype(bool)
        cat_cent = self.host_pos[sel_c]
        k_oric, self.key = random.split(self.key)
        ori_cent = sample_watson_orientations(k_oric, self.host_axis[sel_c], mu_cen)

        q = per_host_softmax_over_ranks(
            self.sub_host_ids, self.sub_mvir, t_rank=self.t_rank
        )
        k2, self.key = random.split(self.key)
        Kh, _ = sample_satellites_diffhod(
            k2, mean_N_sat, N_max=self.Nmax_sat, relaxed=self.relaxed, tau=self.tau
        )

        n_host = self.host_pos.shape[0]
        n_sub = self.sub_pos.shape[0]
        order = jnp.lexsort((-q, self.sub_host_ids))
        h_sorted = self.sub_host_ids[order]
        is_start = jnp.concatenate([jnp.array([True]), (h_sorted[1:] != h_sorted[:-1])])
        idx = jnp.arange(n_sub, dtype=jnp.int32)
        start_idx_mark = jnp.where(is_start, idx, -1)

        def scan_max(carry, x):
            new = jnp.maximum(carry, x)
            return new, new

        _, last_start_idx = lax.scan(scan_max, jnp.int32(-1), start_idx_mark)
        ranks_in_host = idx - last_start_idx
        keep_sorted = ranks_in_host < Kh[h_sorted]
        chosen_sorted_idx = order[keep_sorted]

        sat_counts_on_sub = (
            jnp.zeros((n_sub,), dtype=jnp.int32).at[chosen_sorted_idx].add(1)
        )
        placed_on_sub_per_host = (
            jnp.zeros((n_host,), dtype=jnp.int32)
            .at[self.sub_host_ids]
            .add(sat_counts_on_sub)
        )

        deficit = jnp.clip(Kh - placed_on_sub_per_host, 0)
        k4, self.key = random.split(self.key)
        pos_full, host_idx_full, use_mask = sample_nfw_about_hosts(
            k4,
            self.host_pos,
            self.host_rvir,
            deficit,
            conc=5.0,
            n_newton=6,
            per_host_cap=64,
        )
        nfw_pts = pos_full[use_mask]
        nfw_host_idx = host_idx_full[use_mask]

        cat_sat_sub = jnp.repeat(self.sub_pos, sat_counts_on_sub, axis=0)
        chosen_sub_idx = jnp.nonzero(
            sat_counts_on_sub, size=sat_counts_on_sub.size, fill_value=-1
        )[0]
        chosen_sub_idx = chosen_sub_idx[chosen_sub_idx >= 0]

        if self.alignment_model == "subhalo":
            ref_s_sub = self.sub_axis[chosen_sub_idx]
        else:
            base = self.sub_pos - self.host_pos[self.sub_host_ids]
            ref_s_sub = _unitize(base)[chosen_sub_idx]

        if self.alignment_strength == "constant":
            mu_sub = jnp.full(
                (ref_s_sub.shape[0],), mu_sat.astype(ref_s_sub.dtype), dtype=ref_s_sub.dtype
            )
        else:
            base_vec = (self.sub_pos - self.host_pos[self.sub_host_ids])[chosen_sub_idx]
            r = jnp.linalg.norm(base_vec, axis=1)
            rvir_sel = self.host_rvir[self.sub_host_ids[chosen_sub_idx]]
            r_over = r / (rvir_sel + 1e-12)
            mu_sub = _satellite_mu_from_radius(
                r_over, float(self.a_strength), float(self.gamma_strength)
            )

        k3, self.key = random.split(self.key)
        ori_sat_sub = sample_watson_orientations(
            k3, ref_s_sub, jnp.clip(jnp.nan_to_num(mu_sub, nan=0.0), -0.999, 0.999)
        )

        if nfw_pts.shape[0] > 0:
            disp = nfw_pts - self.host_pos[nfw_host_idx]
            r_hat = _unitize(disp)
            if self.alignment_strength == "constant":
                mu_nfw = jnp.full((r_hat.shape[0],), mu_sat.astype(r_hat.dtype), dtype=r_hat.dtype)
            else:
                r = jnp.linalg.norm(disp, axis=1)
                rvir_sel = self.host_rvir[nfw_host_idx]
                mu_nfw = _satellite_mu_from_radius(
                    r / (rvir_sel + 1e-12),
                    float(self.a_strength),
                    float(self.gamma_strength),
                )
            k5, self.key = random.split(self.key)
            ori_sat_nfw = sample_watson_orientations(k5, r_hat, mu_nfw)
        else:
            ori_sat_nfw = nfw_pts

        cat_sat = jnp.concatenate([cat_sat_sub, nfw_pts], axis=0)
        ori_sat = jnp.concatenate([ori_sat_sub, ori_sat_nfw], axis=0)
        catalog = jnp.concatenate(
            [
                jnp.concatenate([cat_cent, cat_sat], axis=0),
                jnp.concatenate([ori_cent, ori_sat], axis=0),
            ],
            axis=1,
        )

        host_idx_cent = jnp.nonzero(sel_c, size=sel_c.size, fill_value=-1)[0]
        host_idx_cent = host_idx_cent[host_idx_cent >= 0]
        host_idx_sat_sub = self.sub_host_ids[chosen_sub_idx]

        host_idx_per_gal = jnp.concatenate(
            [host_idx_cent, host_idx_sat_sub, nfw_host_idx], axis=0
        ).astype(jnp.int32)

        host_halo_id_per_gal = self.host_halo_id[host_idx_per_gal]  # always host id
        sub_ids_for_sub = (
            self.sub_halo_id[chosen_sub_idx]
            if chosen_sub_idx.shape[0] > 0
            else jnp.zeros((0,), dtype=jnp.int64)
        )
        sub_ids_pad_cent = -jnp.ones((host_idx_cent.shape[0],), dtype=jnp.int64)
        sub_ids_pad_nfw = -jnp.ones((nfw_host_idx.shape[0],), dtype=jnp.int64)
        sub_halo_id_per_gal = jnp.concatenate(
            [sub_ids_pad_cent, sub_ids_for_sub, sub_ids_pad_nfw], axis=0
        )

        has_gal_host_mask = sel_c | (Kh > 0)
        hosts_with_galaxy_ids = self.host_halo_id[has_gal_host_mask]

        meta = dict(
            host_idx_per_gal=host_idx_per_gal,
            host_halo_id_per_gal=host_halo_id_per_gal,
            sub_halo_id_per_gal=sub_halo_id_per_gal,
            has_gal_host_mask=has_gal_host_mask,
            hosts_with_galaxy_ids=hosts_with_galaxy_ids,
        )
        return catalog, meta
    
    # def return_catalog_with_weights(self):
    #     """
    #     Returns soft catalog with weights for differentiable correlation functions.
    #     Must use relaxed=True to get meaningful weights.
        
    #     Returns:
    #         catalog: [N_sources, 6] array [x,y,z,nx,ny,nz] for all hosts+subs
    #         weights: [N_sources] HOD occupation weights
    #     """
    #     if self.do_discrete:
    #         raise ValueError(
    #             "return_catalog_with_weights requires relaxed=True and do_discrete=False. "
    #             "With do_discrete=True, you filter galaxies and lose the weight information."
    #         )
        
    #     (mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha) = [
    #         jnp.asarray(x, dtype=self.host_mvir.dtype) for x in jnp.asarray(self.params)
    #     ]

    #     mean_N_cen = Ncen(self.host_mvir, logMmin, sigma_logM)
    #     mean_N_sat = Nsat(self.host_mvir, logMmin, sigma_logM, logM0, logM1, alpha)

    #     # Generate orientations for ALL hosts
    #     k_oric, self.key = random.split(self.key)
    #     cent_ori = sample_watson_orientations(k_oric, self.host_axis, mu_cen)

    #     # Satellite weights via softmax
    #     q = per_host_softmax_over_ranks(
    #         self.sub_host_ids, self.sub_mvir, t_rank=self.t_rank
    #     )
    #     sat_w = q * mean_N_sat[self.sub_host_ids]

    #     # Generate orientations for ALL subhalos
    #     if self.alignment_model == "subhalo":
    #         ref_s = self.sub_axis
    #     else:
    #         base = self.sub_pos - self.host_pos[self.sub_host_ids]
    #         ref_s = _unitize(base)

    #     if self.alignment_strength == "constant":
    #         mu_sub = jnp.full((self.sub_pos.shape[0],), mu_sat.astype(ref_s.dtype), dtype=ref_s.dtype)
    #     else:
    #         base_vec = self.sub_pos - self.host_pos[self.sub_host_ids]
    #         r = jnp.linalg.norm(base_vec, axis=1)
    #         rvir_sel = self.host_rvir[self.sub_host_ids]
    #         r_over = r / (rvir_sel + 1e-12)
    #         mu_sub = _satellite_mu_from_radius(
    #             r_over, float(self.a_strength), float(self.gamma_strength)
    #         )

    #     k_oris, self.key = random.split(self.key)
    #     sat_ori = sample_watson_orientations(k_oris, ref_s, mu_sub)

    #     # Concatenate ALL sources (hosts + subs)
    #     all_pos = jnp.concatenate([self.host_pos, self.sub_pos], axis=0)
    #     all_ori = jnp.concatenate([cent_ori, sat_ori], axis=0)
    #     all_weights = jnp.concatenate([mean_N_cen, sat_w], axis=0)

    #     catalog = jnp.concatenate([all_pos, all_ori], axis=1)
        
    #     return catalog, all_weights
    
    # def return_catalog_with_weights(self):
    #     """
    #     Returns soft catalog with weights that matches the discrete sampling.
        
    #     For centrals: Use the straight-through estimator from sample_centrals_diffhod
    #     For satellites: Use the softmax weights but only for the TOP-ranked subhalos
        
    #     Returns:
    #         catalog: [N_sources, 6] array [x,y,z,nx,ny,nz]
    #         weights: [N_sources] HOD occupation weights
    #     """
    #     if self.do_discrete:
    #         raise ValueError(
    #             "return_catalog_with_weights requires do_discrete=False. "
    #             "With do_discrete=True, use the discrete catalog."
    #         )
        
    #     (mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha) = [
    #         jnp.asarray(x, dtype=self.host_mvir.dtype) for x in jnp.asarray(self.params)
    #     ]

    #     mean_N_cen = Ncen(self.host_mvir, logMmin, sigma_logM)
    #     mean_N_sat = Nsat(self.host_mvir, logMmin, sigma_logM, logM0, logM1, alpha)

    #     # ===== CENTRALS: Use straight-through estimator =====
    #     k1, self.key = random.split(self.key)
    #     Hc_hard, Hc_st = sample_centrals_diffhod(
    #         k1, mean_N_cen, relaxed=self.relaxed, tau=self.tau
    #     )
        
    #     # Filter to only hosts that have Hc_st > threshold
    #     # This matches the structure of discrete sampling
    #     cent_threshold = 0.5 if self.relaxed else 0.99
    #     sel_c = Hc_st > cent_threshold
        
    #     cat_cent = self.host_pos[sel_c]
    #     cent_weights = Hc_st[sel_c]  # Use straight-through values
        
    #     # Generate orientations only for selected centrals
    #     k_oric, self.key = random.split(self.key)
    #     ori_cent = sample_watson_orientations(k_oric, self.host_axis[sel_c], mu_cen)

    #     # ===== SATELLITES: Match discrete sampling structure =====
    #     # Use the same ranking logic as return_catalog()
    #     q = per_host_softmax_over_ranks(
    #         self.sub_host_ids, self.sub_mvir, t_rank=self.t_rank
    #     )
        
    #     # Sample how many satellites each host gets (for structure)
    #     k2, self.key = random.split(self.key)
    #     Kh_hard, Kh_st = sample_satellites_diffhod(
    #         k2, mean_N_sat, N_max=self.Nmax_sat, relaxed=self.relaxed, tau=self.tau
    #     )
        
    #     # Sort subhalos by (host_id, -mass) to get ranks
    #     n_sub = self.sub_pos.shape[0]
    #     order = jnp.lexsort((-q, self.sub_host_ids))
    #     h_sorted = self.sub_host_ids[order]
        
    #     # Find rank of each subhalo within its host
    #     is_start = jnp.concatenate([jnp.array([True]), (h_sorted[1:] != h_sorted[:-1])])
    #     idx = jnp.arange(n_sub, dtype=jnp.int32)
    #     start_idx_mark = jnp.where(is_start, idx, -1)
        
    #     def scan_max(carry, x):
    #         new = jnp.maximum(carry, x)
    #         return new, new
        
    #     _, last_start_idx = lax.scan(scan_max, jnp.int32(-1), start_idx_mark)
    #     ranks_in_host = idx - last_start_idx
        
    #     # Keep only subhalos with rank < Kh (matching discrete sampling)
    #     # Use Kh_st for soft version
    #     keep_sorted = ranks_in_host < Kh_st[h_sorted]  # Soft comparison
        
    #     # Map back to original order
    #     keep_original = jnp.zeros(n_sub, dtype=bool)
    #     keep_original = keep_original.at[order].set(keep_sorted)
        
    #     # Filter subhalos and compute their weights
    #     cat_sat = self.sub_pos[keep_original]
    #     sat_weights = q[keep_original] * mean_N_sat[self.sub_host_ids[keep_original]]
        
    #     # Generate orientations for selected satellites
    #     if self.alignment_model == "subhalo":
    #         ref_s = self.sub_axis[keep_original]
    #     else:
    #         base = self.sub_pos - self.host_pos[self.sub_host_ids]
    #         ref_s = _unitize(base)[keep_original]
        
    #     if self.alignment_strength == "constant":
    #         mu_sub = jnp.full((ref_s.shape[0],), mu_sat.astype(ref_s.dtype), dtype=ref_s.dtype)
    #     else:
    #         base_vec = self.sub_pos - self.host_pos[self.sub_host_ids]
    #         r = jnp.linalg.norm(base_vec, axis=1)
    #         rvir_sel = self.host_rvir[self.sub_host_ids]
    #         r_over = r / (rvir_sel + 1e-12)
    #         mu_all = _satellite_mu_from_radius(
    #             r_over, float(self.a_strength), float(self.gamma_strength)
    #         )
    #         mu_sub = mu_all[keep_original]
        
    #     k_oris, self.key = random.split(self.key)
    #     ori_sat = sample_watson_orientations(k_oris, ref_s, mu_sub)

    #     # Concatenate centrals and satellites
    #     all_pos = jnp.concatenate([cat_cent, cat_sat], axis=0)
    #     all_ori = jnp.concatenate([ori_cent, ori_sat], axis=0)
    #     all_weights = jnp.concatenate([cent_weights, sat_weights], axis=0)

    #     catalog = jnp.concatenate([all_pos, all_ori], axis=1)
        
    #     return catalog, all_weights
    
    def return_catalog_with_weights(self):
        """
        Returns catalog with IDENTICAL structure to return_catalog() but with soft weights.
        
        Key insight: We must use relaxed=False internally to get the same discrete
        selection, then compute soft weights for those selected galaxies.
        """
        (mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha) = [
            jnp.asarray(x, dtype=self.host_mvir.dtype) for x in jnp.asarray(self.params)
        ]

        mean_N_cen = Ncen(self.host_mvir, logMmin, sigma_logM)
        mean_N_sat = Nsat(self.host_mvir, logMmin, sigma_logM, logM0, logM1, alpha)

        # ===== CENTRALS: Use DISCRETE sampling for selection =====
        k1, self.key = random.split(self.key)
        # Sample with relaxed=False to get exact same selection as return_catalog()
        Hc_hard, _ = sample_centrals_diffhod(
            k1, mean_N_cen, relaxed=False, tau=self.tau
        )
        
        sel_c = Hc_hard.astype(bool)
        cat_cent = self.host_pos[sel_c]
        
        # Soft weights: use the mean occupation probability for selected centrals
        cent_weights = mean_N_cen[sel_c]
        
        k_oric, self.key = random.split(self.key)
        ori_cent = sample_watson_orientations(k_oric, self.host_axis[sel_c], mu_cen)

        # ===== SATELLITES: Use DISCRETE sampling for selection =====
        q = per_host_softmax_over_ranks(
            self.sub_host_ids, self.sub_mvir, t_rank=self.t_rank
        )
        
        k2, self.key = random.split(self.key)
        # Sample with relaxed=False for exact same selection
        Kh_hard, _ = sample_satellites_diffhod(
            k2, mean_N_sat, N_max=self.Nmax_sat, relaxed=False, tau=self.tau
        )
        
        n_host = self.host_pos.shape[0]
        n_sub = self.sub_pos.shape[0]
        
        # Exact same ranking logic as return_catalog()
        order = jnp.lexsort((-q, self.sub_host_ids))
        h_sorted = self.sub_host_ids[order]
        
        is_start = jnp.concatenate([jnp.array([True]), (h_sorted[1:] != h_sorted[:-1])])
        idx = jnp.arange(n_sub, dtype=jnp.int32)
        start_idx_mark = jnp.where(is_start, idx, -1)
        
        def scan_max(carry, x):
            new = jnp.maximum(carry, x)
            return new, new
        
        _, last_start_idx = lax.scan(scan_max, jnp.int32(-1), start_idx_mark)
        ranks_in_host = idx - last_start_idx
        
        # Use HARD Kh for selection (identical to discrete)
        keep_sorted = ranks_in_host < Kh_hard[h_sorted]
        chosen_sorted_idx = order[keep_sorted]
        
        sat_counts = jnp.zeros((n_sub,), dtype=jnp.int32).at[chosen_sorted_idx].add(1)
        
        # Positions from subhalos
        cat_sat_sub = jnp.repeat(self.sub_pos, sat_counts, axis=0)
        chosen_sub_idx = jnp.nonzero(sat_counts, size=sat_counts.size, fill_value=-1)[0]
        chosen_sub_idx = chosen_sub_idx[chosen_sub_idx >= 0]
        
        # Soft weights for satellites: q * Nsat(host)
        sat_weights_sub = q[chosen_sub_idx] * mean_N_sat[self.sub_host_ids[chosen_sub_idx]]
        
        # Orientations for subhalo satellites
        if self.alignment_model == "subhalo":
            ref_s_sub = self.sub_axis[chosen_sub_idx]
        else:
            base = self.sub_pos - self.host_pos[self.sub_host_ids]
            ref_s_sub = _unitize(base)[chosen_sub_idx]
        
        if self.alignment_strength == "constant":
            mu_sub = jnp.full(
                (ref_s_sub.shape[0],), mu_sat.astype(ref_s_sub.dtype), dtype=ref_s_sub.dtype
            )
        else:
            base_vec = (self.sub_pos - self.host_pos[self.sub_host_ids])[chosen_sub_idx]
            r = jnp.linalg.norm(base_vec, axis=1)
            rvir_sel = self.host_rvir[self.sub_host_ids[chosen_sub_idx]]
            r_over = r / (rvir_sel + 1e-12)
            mu_sub = _satellite_mu_from_radius(
                r_over, float(self.a_strength), float(self.gamma_strength)
            )
        
        k3, self.key = random.split(self.key)
        mu_sub = jnp.nan_to_num(mu_sub, nan=0.0)
        mu_sub = jnp.clip(mu_sub, -0.999, 0.999)
        ori_sat_sub = sample_watson_orientations(k3, ref_s_sub, mu_sub)
        
        # ===== NFW FALLBACK (if enabled) =====
        if self.do_nfw_fallback:
            placed_per_host = (
                jnp.zeros((n_host,), dtype=jnp.int32)
                .at[self.sub_host_ids]
                .add(sat_counts)
            )
            deficit = jnp.clip(Kh_hard - placed_per_host, 0)
            
            k4, self.key = random.split(self.key)
            pos_full, host_idx_full, use_mask = sample_nfw_about_hosts(
                k4, self.host_pos, self.host_rvir, deficit,
                conc=5.0, n_newton=6, per_host_cap=64,
            )
            nfw_pts = pos_full[use_mask]
            nfw_host_idx = host_idx_full[use_mask]
            
            if nfw_pts.shape[0] > 0:
                disp = nfw_pts - self.host_pos[nfw_host_idx]
                r_hat = _unitize(disp)
                
                if self.alignment_strength == "constant":
                    mu_nfw = jnp.full(
                        (r_hat.shape[0],), mu_sat.astype(r_hat.dtype), dtype=r_hat.dtype
                    )
                else:
                    r = jnp.linalg.norm(disp, axis=1)
                    rvir_sel = self.host_rvir[nfw_host_idx]
                    r_over = r / (rvir_sel + 1e-12)
                    mu_nfw = _satellite_mu_from_radius(
                        r_over, float(self.a_strength), float(self.gamma_strength)
                    )
                
                k5, self.key = random.split(self.key)
                ori_sat_nfw = sample_watson_orientations(k5, r_hat, mu_nfw)
                
                # NFW satellites: weight = expected satellites for that host / deficit
                # Simplified: just use weight = 1 (they're drawn to fill the deficit)
                nfw_weights = jnp.ones(nfw_pts.shape[0], dtype=cat_sat_sub.dtype)
                
                cat_sat = jnp.concatenate([cat_sat_sub, nfw_pts], axis=0)
                ori_sat = jnp.concatenate([ori_sat_sub, ori_sat_nfw], axis=0)
                sat_weights = jnp.concatenate([sat_weights_sub, nfw_weights], axis=0)
            else:
                cat_sat = cat_sat_sub
                ori_sat = ori_sat_sub
                sat_weights = sat_weights_sub
        else:
            cat_sat = cat_sat_sub
            ori_sat = ori_sat_sub
            sat_weights = sat_weights_sub

        # Concatenate all
        all_pos = jnp.concatenate([cat_cent, cat_sat], axis=0)
        all_ori = jnp.concatenate([ori_cent, ori_sat], axis=0)
        all_weights = jnp.concatenate([cent_weights, sat_weights], axis=0)

        catalog = jnp.concatenate([all_pos, all_ori], axis=1)
        
        return catalog, all_weights


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from halotools.sim_manager import CachedHaloCatalog
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="resource_tracker: There appear to be .* leaked semaphore objects"
    )


    catalog, columns = load_cleaned_catalogs_from_h5(
        "/Users/snehpandya/Projects/iaemu_v2/src/data/v2/cleaned_catalogs.h5",
        as_list=True,
        debug=True,
    )
    inputs = jnp.load(
        "/Users/snehpandya/Projects/iaemu_v2/src/data/v2/cleaned_inputs.npy",
        allow_pickle=True,
    )
    halocat = CachedHaloCatalog(
        simname="bolplanck",
        halo_finder="rockstar",
        redshift=0,
        version_name="halotools_v0p4",
    )
    mask_bad_halocat(halocat)
    subcat = halocat.halo_table[
        [
            "halo_id",
            "halo_upid",
            "halo_mvir",
            "halo_x",
            "halo_y",
            "halo_z",
            "halo_axisA_x",
            "halo_axisA_y",
            "halo_axisA_z",
            "halo_rvir",
            "halo_hostid",
            "halo_b_to_a",
        ]
    ]
    idx = 2
    # params = jnp.asarray(inputs[idx], dtype=jnp.float32)
    params = np.asarray([0.55, 0.03, 13.61, 0.26, 11.8, 12.6, 1.0])
    original_catalog = catalog[idx]

    builder = DiffHalotoolsIA(
        subcat=subcat,
        params=params,
        do_discrete=True,
        do_nfw_fallback=True,
        seed=1234,
        alignment_model="radial",
        alignment_strength="constant",
        relaxed=True,
        tau=0.1,
        Nmax_sat=512,
        t_rank=0.5,
    )

    gal_cat = builder.return_catalog()
    print("Generated catalog shape:", gal_cat.shape)

    fig, axs = plot_diagnostic(
        builder, gal_cat=gal_cat, 
        orig_catalog=original_catalog
    )
    plt.show()
