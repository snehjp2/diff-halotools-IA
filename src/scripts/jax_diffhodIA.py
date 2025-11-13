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
                (ref_s_sub.shape[0],), float(mu_sat), dtype=ref_s_sub.dtype
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
                mu_nfw = jnp.full((r_hat.shape[0],), float(mu_sat), dtype=r_hat.dtype)
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from halotools.sim_manager import CachedHaloCatalog

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
    for idx in range(len(inputs)):
        print(f"Running test for catalog index {idx}...")
        # idx = 2
        params = jnp.asarray(inputs[idx], dtype=jnp.float32)
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
        gal_cat = np.asarray(gal_cat)
        print("Generated catalog shape:", gal_cat.shape)

        fig, axs = plot_diagnostic(
            builder, gal_cat=gal_cat, orig_catalog=original_catalog
        )
        plt.show()
