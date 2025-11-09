from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

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

# config.update("jax_enable_x64", True)      # better numeric headroom
# config.update("jax_debug_nans", True)  # crash when a NaN is created
# config.update("jax_debug_infs", True)  # ditto for infs

def dbg(name, x):
    jax.debug.print(
        "{name}: shape={s}, any_nan={nan}, any_inf={inf}, min={mn}, max={mx}",
        name=name,
        s=x.shape,
        nan=jnp.isnan(x).any(),
        inf=~jnp.isfinite(x).all(),
        mn=jnp.nanmin(x),
        mx=jnp.nanmax(x),
    )


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
    # x64 = x.astype(jnp.float64)
    # ax = jnp.abs(x64)

    def small_branch(xs: Array) -> Array:
        # Series: erfi(x) = 2/sqrt(pi) * sum_{n>=0} x^{2n+1} / (n! (2n+1))
        # We implement via a stable recurrence on 'term' and accumulate into s
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


@jit
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


@jit
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


@partial(jit, static_argnames=("N_max",))
def sample_satellites_diffhod(
    key: jnp.ndarray,
    mean_N_sat: jnp.ndarray,
    *,
    N_max: int = 48,  # must be static for JAX (used in shapes)
    relaxed: bool = True,  # also static here; we still branch via lax.cond
    tau: float = 0.1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Binomial(N_max, p=lambda/N_max) via N_max independent (relaxed) Bernoullis.
    Returns (hard_counts:int32, straight_through:float32)
    """
    # keep dtype consistent with inputs (usually float32)
    lam = jnp.clip(mean_N_sat, 0.0)
    invN = jnp.asarray(1.0 / N_max, dtype=lam.dtype)  # avoid float(N_max)
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
        logits = _logit(p).astype(lam.dtype)[:, None]  # ensure dtype matches
        z = sigmoid((logits + eps) / jnp.asarray(tau, lam.dtype))  # [H, N_max]
        z_hard = (z >= 0.5).astype(z.dtype)
        z_st = z_hard + (z - lax.stop_gradient(z))
        Kh_soft = jnp.sum(z, axis=1)
        Kh_hard = jnp.sum(z_hard, axis=1).astype(jnp.int32)
        Kh_st = Kh_hard.astype(jnp.float32) + (Kh_soft - lax.stop_gradient(Kh_soft))
        return Kh_hard, Kh_st

    return lax.cond(pred, relaxed_branch, discrete_branch, operand=None)


@partial(jax.jit, static_argnames=("n_newton", "per_host_cap"))
def sample_nfw_about_hosts(
    key: Array,
    host_centers: Array,         # [H, 3]
    host_rvir: Array,            # [H]
    counts_per_host: Array,      # [H], int-like
    *,
    conc: float = 5.0,
    n_newton: int = 6,
    per_host_cap: int = 64,      # STATIC: max draws per host
):
    """
    Vectorized NFW radii via inverse CDF with fixed Newton iters.

    Returns fixed-shape outputs plus a mask:
      pos_full:   [H*per_host_cap, 3]
      host_idx:   [H*per_host_cap]
      use_mask:   [H*per_host_cap]  (True for the first 'counts_per_host[h]' of each host h)
    """
    H = host_centers.shape[0]
    dtype = host_centers.dtype

    conc = jnp.asarray(conc, dtype=dtype)
    conc = jnp.maximum(conc, jnp.asarray(1e-8, dtype=dtype))

    # Cap requested counts to per_host_cap so the mask is always valid.
    counts = jnp.clip(counts_per_host.astype(jnp.int32), 0, per_host_cap)

    # Build static index grids
    host_ids   = jnp.arange(H, dtype=jnp.int32)                      # [H]
    host_idx   = jnp.repeat(host_ids, per_host_cap)                  # [H*K]
    # idx_in_host= jnp.tile(jnp.arange(per_host_cap, jnp.int32), (H,)) # [H*K]
    # Option A: minimal fix
    idx_in_host = jnp.tile(jnp.arange(per_host_cap, dtype=jnp.int32), (H,))  # [H*K]


    # Mask: keep only first counts[h] samples within each host
    use_mask = idx_in_host < counts[host_idx]                        # [H*K], bool

    # Per-sample parameters
    rvir = host_rvir[host_idx]                                       # [H*K]
    rs   = rvir / conc                                               # [H*K]

    # RNG
    key_u, key_dir = random.split(key)

    # Draw uniforms/normals for the full static size (masked later)
    u  = jnp.clip(random.uniform(key_u, (H * per_host_cap,), dtype=dtype), 1e-7, 1 - 1e-7)
    mc = jnp.log1p(conc) - conc / (1 + conc)
    y  = u * mc
    x0 = conc * u

    # Newton iterations (vectorized, fixed count)
    def body_fun(_, x_curr):
        fx  = jnp.log1p(x_curr) - x_curr / (1 + x_curr) - y
        dfx = x_curr / jnp.square(1 + x_curr)
        return jnp.clip(x_curr - fx / (dfx + 1e-30), 0.0)

    x_fin = lax.fori_loop(0, n_newton, body_fun, x0)
    r     = x_fin * rs                                              # [H*K]

    # Random directions
    z = random.normal(key_dir, (H * per_host_cap, 3), dtype=dtype)
    z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)    # _unitize

    pos_full = host_centers[host_idx] + r[:, None] * z              # [H*K, 3]

    # Return fixed-shape arrays + mask (caller can slice with the mask or carry it forward)
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

    # order by host asc, then by mass desc  (lexsort uses last key as primary)
    order = jnp.lexsort((-sub_mvir, sub_host_ids))
    inv_order = jnp.empty_like(order)
    inv_order = inv_order.at[order].set(jnp.arange(n, dtype=order.dtype))

    h_sorted = sub_host_ids[order]

    # block starts (new host)
    is_start = jnp.concatenate([jnp.array([True]), (h_sorted[1:] != h_sorted[:-1])])
    is_end = jnp.concatenate([(h_sorted[1:] != h_sorted[:-1]), jnp.array([True])])

    # ranks within each host block, without segment_min:
    # compute last-start index via a running max scan over start markers
    idx = jnp.arange(n, dtype=jnp.int32)
    start_idx_mark = jnp.where(is_start, idx, -1)

    def scan_max(carry, x):
        new = jnp.maximum(carry, x)
        return new, new

    _, last_start_idx = lax.scan(scan_max, jnp.int32(-1), start_idx_mark)
    ranks_sorted = idx - last_start_idx

    # logits from ranks
    logits_sorted = -ranks_sorted.astype(sub_mvir.dtype) / t_rank

    # Per-block MAX via forward run+backward propagate
    def run_max(carry, inp):
        start, val = inp
        cur = jnp.where(start, val, jnp.maximum(carry, val))
        return cur, cur

    _, runmax = lax.scan(
        run_max, jnp.array(-jnp.inf, logits_sorted.dtype), (is_start, logits_sorted)
    )

    def backfill_max(carry, inp):
        end, v = inp  # v is runmax
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
        end, v = inp  # v is runsum
        out = jnp.where(end, v, carry)
        return out, out

    _, sum_per_pos_rev = lax.scan(
        backfill_sum, jnp.array(0.0, exp_shift.dtype), (is_end[::-1], runsum[::-1])
    )
    sum_per_pos = sum_per_pos_rev[::-1]

    q_sorted = exp_shift / (sum_per_pos + 1e-30)

    # back to original order
    q = jnp.empty_like(q_sorted)
    q = q.at[order].set(q_sorted)
    return q  # sums to 1 per host


def _satellite_mu_from_radius(r_over_rvir: Array, a: float, gamma: float) -> Array:
    r_safe = jnp.nan_to_num(r_over_rvir, nan=0.0, posinf=1e3, neginf=0.0)
    r_clipped = jnp.clip(r_safe, 1e-5, 1e3)  # avoid negatives / zero
    mu = a * jnp.power(r_clipped, gamma)
    return jnp.clip(mu, -0.999, 0.999)  # stay far from ±1


# -------------------- main builder --------------------
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
    alignment_strength: str = "constant"
    a_strength: float = 0.8
    gamma_strength: float = 1.0
    relaxed: bool = True
    tau: float = 0.1
    Nmax_sat: int = 256
    t_rank: float = 0.5
    seed: Optional[int] = None

    # cached tensors (device-resident)
    def __post_init__(self):
        # Build host/sub tensors (same logic as your helper; using numpy is fine for I/O)
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
        ]
        missing = [k for k in req if k not in self.subcat.colnames]
        if missing:
            raise KeyError(f"subcat missing required keys: {missing}")
        
        if self.alignment_strength.lower() == 'radial':
            config.update("jax_enable_x64", True)      # better numeric headroom

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

        # Map subs to host indices by halo_hostid
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
        self.alignment_model = self.alignment_model.lower()
        self.alignment_strength = self.alignment_strength.lower()
        self.a_strength = float(self.a_strength)
        self.gamma_strength = float(self.gamma_strength)

        if self.host_pos.size == 0:
            raise ValueError("No host halos found after filtering (halo_upid == -1).")

        # RNG key
        self.key = random.PRNGKey(0 if self.seed is None else int(self.seed))

    # ----------- basic HOD means -----------
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

    # ----------- public API -----------
    def return_catalog(self) -> Array:
        """
        Build and return (Ng, 6) array [x,y,z,nx,ny,nz]
        """
        (mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha) = [
            float(x) for x in jnp.asarray(self.params)
        ]

        mean_N_cen = Ncen(self.host_mvir, logMmin, sigma_logM)  # [H]
        mean_N_sat = Nsat(
            self.host_mvir, logMmin, sigma_logM, logM0, logM1, alpha
        )  # [H]

        # Centrals
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
            # "soft" placement: keep hosts with nonzero mean (approx like your torch shortcut)
            sel = mean_N_cen > 0
            cat_cent = self.host_pos[sel]
            ref_c = self.host_axis[sel]
            k_oric, self.key = random.split(self.key)
            ori_cent = sample_watson_orientations(k_oric, ref_c, mu_cen)

        # Satellites: per-host softmax over ranks (prefers massive subs)
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
            # Vectorized per-host "top-k by q" without Python loops:
            # 1) sort by (host asc, prob desc)
            order = jnp.lexsort((-q, self.sub_host_ids))
            h_sorted = self.sub_host_ids[order]
            is_start = jnp.concatenate(
                [jnp.array([True]), (h_sorted[1:] != h_sorted[:-1])]
            )
            # ranks within each host block without segment ops
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

            # NFW fallback if needed
            placed_per_host = (
                jnp.zeros((n_host,), dtype=jnp.int32)
                .at[self.sub_host_ids]
                .add(sat_counts)
            )
            deficit = jnp.clip(Kh - placed_per_host, 0)

            # Subhalo-placed satellites
            cat_sat_sub = jnp.repeat(self.sub_pos, sat_counts, axis=0)
            chosen_sub_idx = jnp.nonzero(
                sat_counts, size=sat_counts.size, fill_value=-1
            )[0]
            chosen_sub_idx = chosen_sub_idx[chosen_sub_idx >= 0]

            if self.alignment_model == "subhalo":
                ref_s_sub = self.sub_axis[chosen_sub_idx]
            else:
                # radial: from host center to sub position
                base = self.sub_pos - self.host_pos[self.sub_host_ids]
                ref_s_sub = _unitize(base)[chosen_sub_idx]

            if self.alignment_strength == "constant":
                mu_sub = jnp.full(
                    (ref_s_sub.shape[0],), float(mu_sat), dtype=ref_s_sub.dtype
                )
            else:
                # radius for each chosen subhalo relative to its host
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
                # nfw_pts, nfw_host_idx = sample_nfw_about_hosts(
                #     k4, self.host_pos, self.host_rvir, deficit, conc=5.0
                # )
                pos_full, host_idx_full, use_mask = sample_nfw_about_hosts(
                    k4, self.host_pos, self.host_rvir, deficit,
                conc=5.0, n_newton=6, per_host_cap=64
            )

                # If you need ragged outputs *outside* jit:
                nfw_pts     = pos_full[use_mask]
                nfw_host_idx= host_idx_full[use_mask]

                if nfw_pts.shape[0] > 0:
                    disp = nfw_pts - self.host_pos[nfw_host_idx]
                    r_hat = _unitize(disp)

                    # alignment strength for fallback points
                    if self.alignment_strength == "constant":
                        mu_nfw = jnp.full(
                            (r_hat.shape[0],), float(mu_sat), dtype=r_hat.dtype
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

            # Sanity (optional): total satellites equals sum(Kh)
            # assert cat_sat.shape[0] == int(Kh.sum()), "Satellite total mismatch"
        else:
            # "soft" satellites: expand by rounded expected allocations (diagnostics)
            k_soft = jnp.rint(sat_w).astype(jnp.int32)
            cat_sat = jnp.repeat(self.sub_pos, k_soft, axis=0)
            if self.alignment_model == "subhalo":
                ref_s_sub = jnp.repeat(self.sub_axis, k_soft, axis=0)
            else:
                base = self.sub_pos - self.host_pos[self.sub_host_ids]
                ref_s_sub = jnp.repeat(_unitize(base), k_soft, axis=0)

            # compute per-satellite mu for the repeated selection
            if self.alignment_strength == "constant":
                mu_soft = jnp.full(
                    (ref_s_sub.shape[0],), float(mu_sat), dtype=ref_s_sub.dtype
                )
            else:
                # radii for the repeated subhalo list
                base = self.sub_pos - self.host_pos[self.sub_host_ids]
                r_all = jnp.linalg.norm(base, axis=1)
                rvir_all = self.host_rvir[self.sub_host_ids]
                r_over_all = r_all / (rvir_all + 1e-12)
                # repeat r_over according to k_soft to match ref_s_sub length
                r_over_rep = jnp.repeat(r_over_all, k_soft, axis=0)
                mu_soft = _satellite_mu_from_radius(
                    r_over_rep, float(self.a_strength), float(self.gamma_strength)
                )

            k6, self.key = random.split(self.key)
            ori_sat = sample_watson_orientations(k6, ref_s_sub, mu_soft)

        # Concatenate centrals + satellites
        if cat_cent.size == 0 and cat_sat.size == 0:
            return jnp.zeros((0, 6), dtype=self.host_pos.dtype)
        pos = jnp.concatenate([cat_cent, cat_sat], axis=0)
        ori = jnp.concatenate([ori_cent, ori_sat], axis=0)
        return jnp.concatenate([pos, ori], axis=1)


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
            Nmax_sat=256,
            t_rank=0.5,
        )

        gal_cat = builder.return_catalog()
        gal_cat = np.asarray(gal_cat)
        print("Generated catalog shape:", gal_cat.shape)

        fig, axs = plot_diagnostic(builder, gal_cat=gal_cat, orig_catalog=original_catalog)
        plt.show()

# def sample_centrals_diffhod(
#     key: Array, mean_N_cen: Array, *, relaxed: bool = True, tau: float = 0.1
# ) -> Tuple[Array, Array]:
#     """
#     Bernoulli or Relaxed Bernoulli (Gumbel-Logistic) + straight-through.
#     Returns (hard_0_1, straight_through)
#     """
#     p = jnp.clip(mean_N_cen, 0.0, 1.0)
#     ## make if statements static for jitting
#     if not relaxed:
#         u = random.uniform(key, p.shape)
#         Hc = (u < p).astype(p.dtype)
#         return Hc, Hc

#     # relaxed
#     key_u = key
#     u = jnp.clip(random.uniform(key_u, p.shape), 1e-6, 1 - 1e-6)
#     eps_logistic = jnp.log(u) - jnp.log1p(-u)
#     z = sigmoid((_logit(p) + eps_logistic) / tau)
#     z_hard = (z >= 0.5).astype(z.dtype)
#     z_st = z_hard + (z - lax.stop_gradient(z))
#     return z_hard, z_st


# def sample_satellites_diffhod(
#     key: Array,
#     mean_N_sat: Array,
#     *,
#     N_max: int = 48,
#     relaxed: bool = True,
#     tau: float = 0.1,
# ) -> Tuple[Array, Array]:
#     """
#     Binomial(N_max, p=lambda/N_max) via N_max independent (relaxed) Bernoullis.
#     Returns (hard_counts:int, straight_through:float)
#     """
#     lam = jnp.clip(mean_N_sat, 0.0)
#     p = jnp.clip(lam / float(N_max), 0.0, 1.0)

#     if not relaxed:
#         key_u = key
#         u = random.uniform(key_u, (p.shape[0], N_max))
#         trials = jnp.sum(u < p[:, None], axis=1)
#         Kh = trials.astype(jnp.int32)
#         return Kh, Kh.astype(jnp.float32)

#     # relaxed independent trials
#     key_u = key
#     u = jnp.clip(random.uniform(key_u, (p.shape[0], N_max)), 1e-6, 1 - 1e-6)
#     eps = jnp.log(u) - jnp.log1p(-u)
#     logits = _logit(p)[:, None]
#     z = sigmoid((logits + eps) / tau)  # [H, N_max]
#     z_hard = (z >= 0.5).astype(z.dtype)
#     z_st = z_hard + (z - lax.stop_gradient(z))
#     Kh_soft = jnp.sum(z, axis=1)
#     Kh_hard = jnp.sum(z_hard, axis=1).astype(jnp.int32)
#     Kh_st = Kh_hard.astype(jnp.float32) + (Kh_soft - lax.stop_gradient(Kh_soft))
#     return Kh_hard, Kh_st


# -------------------- NFW sampling about hosts --------------------
# @jit
# @partial(jit, static_argnames=('n_newton',))
# def sample_nfw_about_hosts(
#     key: Array,
#     host_centers: Array,
#     host_rvir: Array,
#     counts_per_host: Array,
#     *,
#     conc: float = 5.0,
#     n_newton: int = 6,
# ) -> Tuple[Array, Array]:
#     """
#     Vectorized NFW radii via inverse CDF with fixed Newton iters.
#     Returns:
#       pos: [total, 3], host_idx: [total]
#     """
#     conc = jnp.asarray(conc, dtype=host_centers.dtype)
#     conc = jnp.maximum(conc, jnp.asarray(1e-8, dtype=host_centers.dtype))

#     counts = jnp.clip(counts_per_host, 0).astype(jnp.int32)
#     total = jnp.sum(counts).astype(jnp.int32)
    
#     # if total == 0:
#     #     return jnp.zeros((0, 3), host_centers.dtype), jnp.zeros((0,), jnp.int32)

#     host_ids = jnp.arange(host_centers.shape[0], dtype=jnp.int32)
#     host_idx = jnp.repeat(host_ids, counts)  # [total]
#     rvir = host_rvir[host_idx]
#     rs = rvir / conc

#     key_u, key_dir = random.split(key)
#     u = jnp.clip(random.uniform(key_u, (total,)), 1e-7, 1 - 1e-7)
#     mc = jnp.log1p(conc) - conc / (1 + conc)
#     y = u * mc

#     x = conc * u

#     def body_fun(_, x_curr):
#         fx = jnp.log1p(x_curr) - x_curr / (1 + x_curr) - y
#         dfx = x_curr / jnp.square(1 + x_curr)
#         return jnp.clip(x_curr - fx / (dfx + 1e-30), 0.0)

#     x_fin = lax.fori_loop(0, n_newton, body_fun, x)
#     r = x_fin * rs

#     z = random.normal(key_dir, (total, 3))
#     z = _unitize(z)
#     pos = host_centers[host_idx] + r[:, None] * z
#     return pos, host_idx