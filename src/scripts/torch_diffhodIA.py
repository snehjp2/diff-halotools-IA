import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.special import erfinv
from diffhodIA_utils import (
    load_cleaned_catalogs_from_h5,
    mask_bad_halocat,
    plot_diagnostic,
)


# -------------------- small helpers --------------------
def _unitize(v):
    """
    Normalize a tensor to have unit length.
    v: [N, 3] tensor
    Returns: [N, 3] tensor of unit vectors
    """
    return v / (v.norm(dim=1, keepdim=True) + 1e-12)


def _erfi_real(x):
    """
    Stable real-valued erfi for torch tensors.
    This is used for computing the inverse error function, which is needed for sampling from the Watson distribution.
    x: tensor
    Returns: tensor of same shape as x
    """
    # Stable real-valued erfi for torch tensors
    x64 = x.to(torch.float64)
    ax = x64.abs()
    y = torch.empty_like(x64)

    small = ax <= 3.0
    if small.any():
        xs = x64[small]
        term = xs.clone()
        s = xs.clone()
        x2 = xs * xs
        for n in range(1, 20):
            term = term * x2 / n
            s = s + term / (2 * n + 1)
        y[small] = (2.0 / math.sqrt(math.pi)) * s

    if (~small).any():
        xl = x64[~small]
        inv = 1.0 / xl
        inv2 = inv * inv
        series = 1.0 + 0.5 * inv2 + 0.75 * inv2 * inv2 + 1.875 * inv2 * inv2 * inv2
        y[~small] = torch.exp(xl * xl) * series / (math.sqrt(math.pi) * xl)

    return y.to(x.dtype)


def _sample_t_watson(kappa, u, n_newton=6):
    """
    Sample t = cos(theta) in (-1,1) from the Dimroth-Watson distribution using inverse transform sampling.
    for kappa < 0, use the erf inverse method. The inverse is known analytically.

    for kappa = 0, this is uniform in (-1,1), so t = 2u - 1.

    for kappa > 0, use Newton's method to invert the CDF with t_0 = 2u - 1 as the initial guess.
    Number of iterations is controlled by n_newton.

    kappa: [N] tensor of concentration parameters
    u: [N] tensor of uniform random numbers in (0,1)
    n_newton: number of Newton iterations to use for kappa > 0

    Returns: [N] tensor of samples t in (-1,1)
    """
    eps = 1e-12
    t = torch.empty_like(u)
    pos = kappa > 1e-12
    neg = kappa < -1e-12
    zer = ~(pos | neg)

    if neg.any():
        km = torch.sqrt(-kappa[neg])
        den = torch.erf(km)
        arg = ((2.0 * u[neg] - 1.0) * den).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        t[neg] = erfinv(arg) / (km + eps)

    if zer.any():
        t[zer] = 2.0 * u[zer] - 1.0

    if pos.any():
        kp = kappa[pos]
        sp = torch.sqrt(kp)
        tp = 2.0 * u[pos] - 1.0
        den = _erfi_real(sp) + 1e-30
        for _ in range(n_newton):
            F = 0.5 * (_erfi_real(sp * tp) / den + 1.0)
            pdf = (sp / math.sqrt(math.pi)) * torch.exp(kp * tp * tp) / den
            tp = (tp - (F - u[pos]) / (pdf + 1e-30)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        t[pos] = tp

    return t.clamp(-1.0 + 1e-6, 1.0 - 1e-6)


def sample_watson_orientations(ref_dirs, mu, n_newton=6, generator=None):
    """
    Sample orientations from the Dimroth-Watson distribution about reference directions.

    step 0) normalize ref_dirs to unit vectors
    step 1) compute kappa from mu (kappa = tan(0.5 * pi * mu))
    step 2) sample t = cos(theta) from the Watson distribution using inverse transform sampling
    step 3) compute the orthonormal basis (u_axis, b1, b2)
    step 4) sample phi uniformly in [0, 2pi) for the azimuthal angle
    step 5) compute the sampled orientations n = cos(theta) * u_axis + sin(theta) * (cos(phi) * b1 + sin(phi) * b2)

    ref_dirs: [N, 3] tensor of reference directions (need not be unit vectors)
    mu: scalar or [N] tensor of concentration parameters in (-1,1)
    n_newton: number of Newton iterations to use for kappa > 0
    generator: optional torch.Generator for RNG

    Returns: [N, 3] tensor of sampled unit orientation vectors
    """
    u_axis = _unitize(ref_dirs)
    N = u_axis.shape[0]
    if N == 0:
        return u_axis.clone()

    if not torch.is_tensor(mu):
        mu = torch.tensor(mu, dtype=u_axis.dtype, device=u_axis.device)
    mu = mu.to(dtype=u_axis.dtype, device=u_axis.device).reshape(-1)
    if mu.numel() == 1:
        mu = mu.expand(N)
    mu = mu.clamp(-1 + 1e-6, 1 - 1e-6)

    kappa = torch.tan(0.5 * math.pi * mu)
    u_uni = torch.rand(N, device=u_axis.device, generator=generator).clamp_(
        1e-7, 1 - 1e-7
    )

    t = _sample_t_watson(kappa, u_uni, n_newton=n_newton)

    xhat = torch.tensor(
        [1.0, 0.0, 0.0], device=u_axis.device, dtype=u_axis.dtype
    ).expand_as(u_axis)
    yhat = torch.tensor(
        [0.0, 1.0, 0.0], device=u_axis.device, dtype=u_axis.dtype
    ).expand_as(u_axis)
    alt = torch.where((u_axis[:, 0].abs() > 0.9).unsqueeze(1), yhat, xhat)
    b1 = _unitize(torch.cross(u_axis, alt, dim=1))
    b2 = torch.cross(u_axis, b1, dim=1)

    phi = 2.0 * math.pi * torch.rand(N, 1, device=u_axis.device, generator=generator)
    sinth = torch.sqrt((1 - t * t).clamp_min(0)).unsqueeze(1)
    costh = t.unsqueeze(1)

    n = _unitize(costh * u_axis + sinth * (torch.cos(phi) * b1 + torch.sin(phi) * b2))
    return n


def Ncen(M, logMmin, sigma_logM):
    """
    Mean central occupation function.
    Known.

    M: [H] tensor of host masses
    logMmin, sigma_logM: HOD parameters (scalars)
    Returns: [H] tensor of mean central occupation per host
    """
    log_M = torch.log10(M)
    term = (log_M - logMmin) / sigma_logM
    return 0.5 * (1.0 + torch.erf(term))


def Nsat(M, logMmin, sigma_logM, logM0, logM1, alpha):
    """
    Mean satellite occupation function.
    Known.

    M: [H] tensor of host masses
    logMmin, sigma_logM, logM0, logM1, alpha: HOD parameters (scalars)
    Returns: [H] tensor of mean satellite occupation per host
    """
    M0 = 10.0**logM0
    M1 = 10.0**logM1
    x = torch.clamp((M - M0) / M1, min=0.0)
    return Ncen(M, logMmin, sigma_logM) * x.pow(alpha)


def sample_centrals_diffhod(mean_N_cen, relaxed=True, tau=0.1, generator=None):
    """
    Sample central occupation per host using a Bernoulli distribution.
    This is following Benjamin Horowitz's paper.

    mean_N_cen: [H] tensor of mean central occupation per host
    relaxed: if True, use relaxed (differentiable) sampling; else hard sampling
    tau: temperature parameter for relaxed sampling
    generator: optional torch.Generator for RNG
    Returns: Hc_hard, Hc_st
        Hc_hard: [H] tensor of hard 0/1 samples
        Hc_st: [H] tensor of straight-through estimator samples (hard + soft - soft.detach())
    """
    p = mean_N_cen.clamp(0, 1)
    if not relaxed:
        g = torch.Generator(device=p.device) if generator is None else generator
        Hc = torch.bernoulli(p, generator=g).float()
        return Hc, Hc
    logit_p = torch.log(p + 1e-8) - torch.log1p(-p + 1e-8)
    g = torch.Generator(device=p.device) if generator is None else generator
    # u = torch.rand_like(p, generator=g).clamp_(1e-6, 1 - 1e-6)
    u = torch.rand(p.shape, device=p.device, dtype=p.dtype, generator=g).clamp_(
        1e-6, 1 - 1e-6
    )
    eps_logistic = torch.log(u) - torch.log1p(-u)
    z = torch.sigmoid((logit_p + eps_logistic) / tau)
    Hc_hard = (z >= 0.5).float()
    Hc_st = Hc_hard + (z - z.detach())
    return Hc_hard, Hc_st


def sample_satellites_diffhod(
    mean_N_sat, N_max=48, relaxed=True, tau=0.1, generator=None
):
    """
    Satellite counts via Binomial(N_max, p=lambda/N_max) (Poisson approximation for small p).
    This is following Benjamin Horowitz's paper.

    mean_N_sat: [H] tensor of mean satellite counts per host
    N_max: maximum number of satellites to consider per host (truncation)
    relaxed: if True, use relaxed (differentiable) sampling; else hard sampling
    tau: temperature parameter for relaxed sampling
    generator: optional torch.Generator for RNG
    Returns: Kh_hard, Kh_st
        Kh_hard: [H] tensor of hard integer samples
        Kh_st: [H] tensor of straight-through estimator samples (hard + soft - soft.detach())
    """
    if N_max <= 0:
        raise ValueError(f"N_max must be a positive integer; got {N_max}")

    lam = mean_N_sat.clamp_min(0.0)
    p = (lam / float(N_max)).clamp(0.0, 1.0)
    g = torch.Generator(device=lam.device) if generator is None else generator

    if not relaxed:
        u = torch.rand(p.shape[0], N_max, device=lam.device, generator=g)
        trials = (u < p.unsqueeze(1)).sum(dim=1)
        Kh = trials.to(torch.long)
        return Kh, Kh

    logit_p = torch.log(p + 1e-8) - torch.log1p(-p + 1e-8)
    u = torch.rand(p.shape[0], N_max, device=lam.device, generator=g).clamp_(
        1e-6, 1 - 1e-6
    )
    eps_logistic = torch.log(u) - torch.log1p(-u)
    z = torch.sigmoid((logit_p.unsqueeze(1) + eps_logistic) / tau)  # [H, N_max]
    z_hard = (z >= 0.5).float()
    z_st = z_hard + (z - z.detach())
    Kh_soft = z.sum(dim=1)
    Kh_hard = z_hard.sum(dim=1).to(torch.long)
    Kh_st = Kh_hard.float() + (Kh_soft - Kh_soft.detach())
    return Kh_hard, Kh_st


def sample_nfw_about_hosts(
    host_centers, host_rvir, counts_per_host, conc=5.0, n_newton=6, generator=None
):
    """
    Sample points from NFW profile about host centers.
    The NFW fallback is used when there are no resolved subhalos to place satellites.

    host_centers: [n_host, 3] tensor of host positions
    host_rvir: [n_host] tensor of host virial radii
    counts_per_host: [n_host] tensor of number of points to sample per host
    conc: concentration parameter (scalar)
    n_newton: number of Newton iterations to use for inverse CDF
    generator: optional torch.Generator for RNG
    Returns: pos, host_idx
        pos: [total_count, 3] tensor of sampled positions
        host_idx: [total_count] tensor of host indices corresponding to each sampled position
    """
    if conc <= 0:
        raise ValueError(f"NFW concentration 'conc' must be > 0; got {conc}")
    if n_newton < 1:
        raise ValueError(f"n_newton must be >= 1; got {n_newton}")

    device = host_centers.device
    counts = counts_per_host.clamp(min=0).to(torch.long)
    total = int(counts.sum().item())
    if total == 0:
        return host_centers.new_zeros((0, 3)), torch.empty(
            0, dtype=torch.long, device=device
        )

    host_idx = torch.repeat_interleave(
        torch.arange(host_centers.shape[0], device=device), counts
    )
    rvir = host_rvir[host_idx]
    conc_t = torch.as_tensor(conc, dtype=rvir.dtype, device=device)
    rs = rvir / conc_t

    g = torch.Generator(device=device) if generator is None else generator
    u = torch.rand(total, device=device, generator=g).clamp_(1e-7, 1 - 1e-7)
    mc = torch.log1p(conc_t) - conc_t / (1 + conc_t)
    y = u * mc

    x = conc_t * u
    for _ in range(n_newton):
        fx = torch.log1p(x) - x / (1 + x) - y
        dfx = x / (1 + x).pow(2)
        x = (x - fx / dfx).clamp_min(0.0)

    r = x * rs

    z = torch.randn(total, 3, device=device, generator=g)
    z = z / (z.norm(dim=1, keepdim=True) + 1e-12)
    pos = host_centers[host_idx] + r.unsqueeze(1) * z
    return pos, host_idx


def per_host_softmax_over_ranks(sub_host_ids, sub_mvir, t_rank=0.5):
    """
    Compute softmax over ranks for each host.
    This avoids computing a non-differentiable sorting operation that isolate the top-k (where k is number of satellites) subhalos.

    sub_host_ids: [n_sub] tensor of host indices for each subhalo
    sub_mvir: [n_sub] tensor of subhalo masses
    t_rank: temperature parameter for ranking softmax
    Returns: q: [n_sub] tensor of softmax weights summing to 1 per host
    """
    if t_rank <= 0:
        raise ValueError(f"t_rank must be > 0; got {t_rank}")
    if sub_host_ids.numel() == 0:
        return torch.empty(0, dtype=sub_mvir.dtype, device=sub_mvir.device)

    # numpy is fine here; inputs are constants from the catalog (no grads needed)
    h = sub_host_ids.cpu().numpy()
    m = sub_mvir.cpu().numpy()
    n = len(m)

    order = np.lexsort((-m, h))  # group by host, then sort by -mass
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)
    h_sorted = h[order]

    new_block = np.ones(n, dtype=bool)
    new_block[1:] = h_sorted[1:] != h_sorted[:-1]
    starts = np.flatnonzero(new_block)
    start_pos = starts[np.cumsum(new_block) - 1]
    ranks_sorted = np.arange(n) - start_pos
    ranks = ranks_sorted[inv_order].astype(np.float32)

    ranks_t = torch.from_numpy(ranks).to(device=sub_mvir.device, dtype=sub_mvir.dtype)

    logits = -ranks_t / float(t_rank)  # prefer massive subs
    logits_sorted = logits[order]

    block_ids = torch.from_numpy(np.cumsum(new_block) - 1).to(logits_sorted.device)
    n_blocks = int(block_ids.max().item()) + 1
    max_per_block = torch.full((n_blocks,), -torch.inf, device=logits_sorted.device)
    max_per_block = max_per_block.scatter_reduce_(
        0, block_ids, logits_sorted, reduce="amax", include_self=True
    )
    max_aligned = max_per_block[block_ids]
    exp_shift = torch.exp(logits_sorted - max_aligned)
    sum_per_block = torch.zeros_like(max_per_block).scatter_add_(
        0, block_ids, exp_shift
    )
    denom = sum_per_block[block_ids]
    q_sorted = exp_shift / denom
    q = torch.empty_like(q_sorted)
    q[order] = q_sorted
    return q  # [n_sub], sums to 1 per host


def build_host_sub_tensors_hostid_hash(subcat):
    """
    Build tensors for host and subhalo properties from a subhalo catalog.
    This version uses halo_hostid and a hash map to link subhalos to hosts.
    This is a utility function used in the DiffHODCatalog class.

    subcat: dictionary-like object with keys:
        "halo_id", "halo_upid", "halo_hostid", "halo_mvir", "halo_x", "halo_y", "halo_z", "halo_rvir",
        "halo_axisA_x", "halo_axisA_y", "halo_axisA_z"

    Returns:
        host_pos: [n_host, 3] tensor of host positions
        host_rvir: [n_host] tensor of host virial radii
        host_mvir: [n_host] tensor of host masses
        sub_pos: [n_sub_kept, 3] tensor of subhalo positions (only those with valid hosts)
        sub_mvir: [n_sub_kept] tensor of subhalo masses
        sub_host_ids: [n_sub_kept] tensor of host indices for each subhalo
        host_axis: [n_host, 3] tensor of host major axes (unit vectors)
        sub_axis: [n_sub_kept, 3] tensor of subhalo major axes (unit vectors)
    """
    required = [
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
    missing = [k for k in required if k not in subcat.colnames]
    if missing:
        raise KeyError(f"subcat is missing required keys: {missing}")

    halo_id = np.asarray(subcat["halo_id"], dtype=np.int64)
    halo_upid = np.asarray(subcat["halo_upid"], dtype=np.int64)
    halo_hostid = np.asarray(subcat["halo_hostid"], dtype=np.int64)

    halo_mvir = np.asarray(subcat["halo_mvir"], dtype=np.float32)
    halo_x = np.asarray(subcat["halo_x"], dtype=np.float32)
    halo_y = np.asarray(subcat["halo_y"], dtype=np.float32)
    halo_z = np.asarray(subcat["halo_z"], dtype=np.float32)
    halo_rvir = np.asarray(subcat["halo_rvir"], dtype=np.float32)

    ax_x = np.asarray(subcat["halo_axisA_x"], dtype=np.float32)
    ax_y = np.asarray(subcat["halo_axisA_y"], dtype=np.float32)
    ax_z = np.asarray(subcat["halo_axisA_z"], dtype=np.float32)

    host_mask = halo_upid == -1
    sub_mask = ~host_mask

    host_keys = halo_hostid[host_mask]  # [n_host]
    parent_keys = halo_hostid[sub_mask]  # [n_sub_all]

    host_index_of = {int(k): i for i, k in enumerate(host_keys.tolist())}
    idx_list = []
    keep_mask = np.zeros(parent_keys.shape[0], dtype=bool)
    for i, k in enumerate(parent_keys):
        j = host_index_of.get(int(k), None)
        if j is not None:
            idx_list.append(j)
            keep_mask[i] = True
    sub_host_idx = np.asarray(idx_list, dtype=np.int64)  # [n_sub_kept]

    host_pos = torch.tensor(
        np.stack([halo_x[host_mask], halo_y[host_mask], halo_z[host_mask]], axis=1)
    ).float()
    host_rvir = torch.tensor(halo_rvir[host_mask]).float()
    host_mvir = torch.tensor(halo_mvir[host_mask]).float()

    sub_pos = torch.tensor(
        np.stack(
            [
                halo_x[sub_mask][keep_mask],
                halo_y[sub_mask][keep_mask],
                halo_z[sub_mask][keep_mask],
            ],
            axis=1,
        )
    ).float()
    sub_mvir = torch.tensor(halo_mvir[sub_mask][keep_mask]).float()
    sub_host_ids = torch.tensor(sub_host_idx, dtype=torch.long)

    host_axis = torch.tensor(
        np.stack([ax_x[host_mask], ax_y[host_mask], ax_z[host_mask]], axis=1)
    ).float()
    sub_axis = torch.tensor(
        np.stack(
            [
                ax_x[sub_mask][keep_mask],
                ax_y[sub_mask][keep_mask],
                ax_z[sub_mask][keep_mask],
            ],
            axis=1,
        )
    ).float()

    host_axis = host_axis / (host_axis.norm(dim=1, keepdim=True) + 1e-12)
    sub_axis = sub_axis / (sub_axis.norm(dim=1, keepdim=True) + 1e-12)

    if host_pos.numel() == 0:
        raise ValueError("No host halos found after filtering (halo_upid == -1).")
    if sub_pos.numel() == 0:
        # not fatal for NFW fallback workflows, but warn loudly
        import warnings

        warnings.warn(
            "No subhalos retained after hostid mapping; satellites will rely on NFW fallback."
        )

    # consistency
    assert sub_pos.shape[0] == sub_mvir.shape[0] == sub_host_ids.shape[0]
    return (
        host_pos,
        host_rvir,
        host_mvir,
        sub_pos,
        sub_mvir,
        sub_host_ids,
        host_axis,
        sub_axis,
    )


class DiffHalotoolsIA:
    """
    Build differentiable galaxy catalogs from a halo catalog and HOD parameters.

    subcat: dictionary-like object with halo catalog data (see build_host_sub_tensors_hostid_hash for required keys)
    params: 1D torch tensor of length 7 with HOD parameters:
        [mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha]
    do_discrete: if True, use hard sampling for discrete galaxy counts; else use relaxed (differentiable) sampling
    do_nfw_fallback: if True, use NFW profile to place satellites when no resolved subhalos are available
    seed: optional integer seed for RNG
    satellite_alignment: 'subhalo' to align satellites with subhalo major axes; 'radial' to align radially towards host center
    relaxed: if True, use relaxed (differentiable) sampling for centrals and satellites; else hard sampling
    tau: temperature parameter for relaxed sampling
    Nmax_sat: maximum number of satellites to consider per host (truncation)
    t_rank: temperature parameter for ranking softmax when selecting subhalos for satellites

    Returns: object with method return_catalog() that returns a (num_galaxies, 6) tensor:
        [x,y,z,nx,ny,nz] for centrals+satellites

    """

    def __init__(
        self,
        subcat,
        params: torch.Tensor,
        *,
        do_discrete: bool = True,
        do_nfw_fallback: bool = True,
        seed: Optional[int] = None,
        satellite_alignment: str = "radial",
        relaxed: bool = True,
        tau: float = 0.1,
        Nmax_sat: int = 256,
        t_rank: float = 0.5,
    ):
        if params.ndim != 1 or params.numel() != 7:
            raise ValueError(
                f"params must be 1D tensor of length 7: [mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha]; got {params.shape}"
            )

        self.subcat = subcat
        self.params = params
        self.do_discrete = do_discrete
        self.do_nfw_fallback = do_nfw_fallback
        self.relaxed = relaxed
        self.tau = tau
        self.Nmax_sat = Nmax_sat
        self.t_rank = t_rank

        if satellite_alignment not in ("subhalo", "radial"):
            raise ValueError(
                f"satellite_alignment must be 'subhalo' or 'radial'; got {satellite_alignment!r}"
            )

        self.satellite_alignment = satellite_alignment

        # cache (filled on first build)
        self._built = False
        self._cache = {}

        # build static tensors from subcat
        (
            self.host_pos,
            self.host_rvir,
            self.host_mvir,
            self.sub_pos,
            self.sub_mvir,
            self.sub_host_ids,
            self.host_axis,
            self.sub_axis,
        ) = build_host_sub_tensors_hostid_hash(self.subcat)

        self.generator = None
        if seed is not None:
            self.generator = torch.Generator(device=self.host_pos.device)
            self.generator.manual_seed(int(seed))

    # ----------- basic HOD means -----------
    @property
    def mean_central_per_host(self):
        mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha = self._unpack_params()
        sigma_logM = torch.abs(sigma_logM) + 1e-6
        return Ncen(self.host_mvir, logMmin, sigma_logM)

    @property
    def mean_satellite_per_host(self):
        mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha = self._unpack_params()
        sigma_logM = torch.abs(sigma_logM) + 1e-6
        alpha = torch.abs(alpha) + 1e-6
        return Nsat(self.host_mvir, logMmin, sigma_logM, logM0, logM1, alpha)

    @property
    def num_centrals(self):
        if self._built and self.do_discrete:
            return int(self._cache["cat_cent"].shape[0])
        # expected count
        return float(self.mean_central_per_host.sum())

    @property
    def num_satellites(self):
        if self._built and self.do_discrete:
            return int(self._cache["cat_sat"].shape[0])
        # expected count
        return float(self.mean_satellite_per_host.sum())

    @property
    def cent_w(self):
        return self._cache.get("cent_w", None)

    @property
    def sat_w(self):
        return self._cache.get("sat_w", None)

    @property
    def Hc_st(self):
        return self._cache.get("Hc", None)

    @property
    def Kh_st(self):
        return self._cache.get("Kh", None)

    # ----------- public API -----------
    def return_catalog(self) -> torch.Tensor:
        """
        Build (or rebuild) and return a (num_galaxies, 6) tensor:
        [x,y,z,nx,ny,nz] for centrals+satellites.
        """
        self._build_if_needed()
        cat_cent = self._cache["cat_cent"]
        cat_sat = self._cache["cat_sat"]
        ori_cent = self._cache["ori_cent"]
        ori_sat = self._cache["ori_sat"]

        if cat_cent.numel() == 0 and cat_sat.numel() == 0:
            return torch.empty(
                0, 6, device=self.host_pos.device, dtype=self.host_pos.dtype
            )

        pos = torch.cat([cat_cent, cat_sat], dim=0)
        ori = torch.cat([ori_cent, ori_sat], dim=0)
        return torch.cat([pos, ori], dim=1)

    # ----------- internal -----------
    def _unpack_params(self):
        mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha = self.params.unbind()
        return mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha

    def _build_if_needed(self):
        self.params = self.params.to(
            device=self.host_mvir.device, dtype=self.host_mvir.dtype
        )

        # always allow rebuild (params might have grads / changed)
        (mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha) = (
            self._unpack_params()
        )
        sigma_logM = torch.abs(sigma_logM) + 1e-6
        alpha = torch.abs(alpha) + 1e-6

        mean_N_cen = Ncen(self.host_mvir, logMmin, sigma_logM)  # [H]
        mean_N_sat = Nsat(
            self.host_mvir, logMmin, sigma_logM, logM0, logM1, alpha
        )  # [H]

        self._cache["cent_w"] = mean_N_cen

        # centrals
        if self.do_discrete:
            Hc, Hc_st = sample_centrals_diffhod(
                mean_N_cen, relaxed=self.relaxed, tau=self.tau, generator=self.generator
            )
            self._cache["Hc"] = Hc_st
            sel_c = Hc.bool()
            cat_cent = self.host_pos[sel_c]
            ref_c = self.host_axis[sel_c]
            ori_cent = sample_watson_orientations(
                ref_c, mu_cen, generator=self.generator
            )
        else:
            # "soft" catalog: place expected centrals at host centers; orientations drawn for diagnostics
            cat_cent = self.host_pos[
                mean_N_cen > 0
            ]  # not strictly correct, but avoids empty tensors
            ref_c = self.host_axis[mean_N_cen > 0]
            ori_cent = sample_watson_orientations(
                ref_c, mu_cen, generator=self.generator
            )

        # satellites
        q = per_host_softmax_over_ranks(
            self.sub_host_ids, self.sub_mvir, t_rank=self.t_rank
        )  # [n_sub]
        sat_w = q * mean_N_sat[self.sub_host_ids]
        self._cache["sat_w"] = sat_w
        n_host = self.host_pos.shape[0]
        n_sub = self.sub_pos.shape[0]
        with_sub = torch.bincount(self.sub_host_ids, minlength=n_host) > 0

        if self.do_discrete:
            Kh, Kh_st = sample_satellites_diffhod(
                mean_N_sat,
                N_max=self.Nmax_sat,
                relaxed=self.relaxed,
                tau=self.tau,
                generator=self.generator,
            )
            self._cache["Kh"] = Kh_st

            # per-host slices in the order-sorted view
            order = torch.argsort(self.sub_host_ids)
            sub_host_sorted = self.sub_host_ids[order]
            is_start = torch.ones_like(sub_host_sorted, dtype=torch.bool)
            is_start[1:] = sub_host_sorted[1:] != sub_host_sorted[:-1]
            start_idx = torch.nonzero(is_start, as_tuple=True)[0]
            end_idx = torch.cat(
                [start_idx[1:], torch.tensor([n_sub], device=start_idx.device)]
            )
            host_vals = sub_host_sorted[start_idx]

            def slice_for_host(h):
                pos = torch.searchsorted(host_vals, h)
                if pos >= start_idx.numel() or host_vals[pos] != h:
                    return slice(0, 0)
                s = int(start_idx[pos])
                e = int(end_idx[pos])
                return slice(s, e)

            sat_counts = torch.zeros(
                n_sub, dtype=torch.long, device=self.sub_pos.device
            )
            hosts_draw = torch.nonzero((Kh > 0) & with_sub, as_tuple=True)[0]
            for h in hosts_draw.tolist():
                sl = slice_for_host(h)
                if sl.start == sl.stop:
                    continue
                k = int(Kh[h].item())
                idx_sorted = order[sl]
                probs = q[idx_sorted]  # per-host probs sum to 1
                top_local = torch.topk(
                    probs, k=min(k, probs.numel()), largest=True
                ).indices
                sat_counts[idx_sorted[top_local]] += 1

            chosen_sub_mask = sat_counts > 0
            chosen_sub_idx = torch.nonzero(chosen_sub_mask, as_tuple=True)[0]
            cat_sat_sub = self.sub_pos.repeat_interleave(sat_counts.clamp(min=0), dim=0)

            # orientation reference for satellites (new flag)
            if self.satellite_alignment == "subhalo":
                ref_s_sub = self.sub_axis[chosen_sub_idx]
            else:  # "radial"
                ref_s_sub = _unitize(
                    self.sub_pos[chosen_sub_idx]
                    - self.host_pos[self.sub_host_ids[chosen_sub_idx]]
                )

            ori_sat_sub = sample_watson_orientations(
                ref_s_sub, mu_sat, generator=self.generator
            )

            if self.do_nfw_fallback:
                placed_per_host = torch.zeros_like(Kh).index_add_(
                    0, self.sub_host_ids, sat_counts
                )
                deficit = (Kh - placed_per_host).clamp(min=0)
                nfw_pts, nfw_host_idx = sample_nfw_about_hosts(
                    self.host_pos,
                    self.host_rvir,
                    deficit,
                    conc=5.0,
                    generator=self.generator,
                )
                if nfw_pts.shape[0] > 0:
                    r_hat = _unitize(nfw_pts - self.host_pos[nfw_host_idx])
                    ori_sat_nfw = sample_watson_orientations(
                        r_hat, mu_sat, generator=self.generator
                    )
                else:
                    ori_sat_nfw = nfw_pts  # empty [0,3]
                cat_sat = torch.cat([cat_sat_sub, nfw_pts], dim=0)
                ori_sat = torch.cat([ori_sat_sub, ori_sat_nfw], dim=0)
            else:
                cat_sat = cat_sat_sub
                ori_sat = ori_sat_sub

            assert cat_sat.shape[0] == int(Kh.sum().item()), (
                f"satellite total mismatch: expected {int(Kh.sum().item())}, got {cat_sat.shape[0]}"
            )
        else:
            # "soft" satellites: use subhalo locations weighted by mean allocation (useful for losses/diagnostics)
            sat_w = q * mean_N_sat[self.sub_host_ids]  # [n_sub]
            self._cache["sat_w"] = sat_w
            # deterministically expand by rounding (keeps grad via weights used elsewhere)
            k = sat_w.round().to(torch.long)
            cat_sat = self.sub_pos.repeat_interleave(k.clamp(min=0), dim=0)
            if self.satellite_alignment == "subhalo":
                ref_s_sub = self.sub_axis.repeat_interleave(k.clamp(min=0), dim=0)
            elif self.satellite_alignment == "radial":
                base = self.sub_pos - self.host_pos[self.sub_host_ids]
                ref_s_sub = _unitize(base).repeat_interleave(k.clamp(min=0), dim=0)
            ori_sat = sample_watson_orientations(
                ref_s_sub, mu_sat, generator=self.generator
            )

        self._cache.update(
            dict(
                cat_cent=cat_cent,
                cat_sat=cat_sat,
                ori_cent=ori_cent,
                ori_sat=ori_sat,
            )
        )

        self._built = True


if __name__ == "__main__":
    from halotools.sim_manager import CachedHaloCatalog

    catalog, columns = load_cleaned_catalogs_from_h5(
        "/Users/snehpandya/Projects/iaemu_v2/src/data/v2/cleaned_catalogs.h5",
        as_list=True,
        debug=True,
    )
    inputs = np.load(
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

    idx = 4
    params = torch.tensor(np.asarray(inputs[idx], dtype=np.float32), requires_grad=True)
    original_catalog = catalog[idx]

    builder = DiffHalotoolsIA(
        subcat=subcat,
        params=params,
        do_discrete=True,
        do_nfw_fallback=True,
        seed=1234,
        satellite_alignment="radial",
        relaxed=False,
        tau=0.1,
        Nmax_sat=256,
        t_rank=0.5,
    )

    gal_cat = builder.return_catalog()
    print("Generated catalog shape:", gal_cat.shape)

    fig, axs = plot_diagnostic(builder=builder, original_catalog=original_catalog, orig_catalog=gal_cat)
    plt.show()


# def sample_axial_orientations(ref_dirs, mu, generator=None):
#     ref = _unitize(ref_dirs)
#     N = ref.shape[0]
#     if N == 0:
#         return ref.clone()

#     if not torch.is_tensor(mu):
#         mu = torch.tensor(mu, dtype=ref.dtype, device=ref.device)
#     mu = mu.to(dtype=ref.dtype, device=ref.device)
#     if mu.ndim == 0:
#         mu = mu.expand(N)
#     else:
#         mu = mu.reshape(-1)
#         if mu.numel() == 1:
#             mu = mu.expand(N)
#         elif mu.numel() != N:
#             raise ValueError(f"mu must be scalar or length-N (N={N}); got {mu.numel()}")

#     amp = mu.abs().clamp(0.0, 1.0 - 1e-6)
#     a = amp / (1.0 - amp + 1e-6)

#     xhat = torch.tensor([1.0, 0.0, 0.0], device=ref.device, dtype=ref.dtype).expand_as(
#         ref
#     )
#     yhat = torch.tensor([0.0, 1.0, 0.0], device=ref.device, dtype=ref.dtype).expand_as(
#         ref
#     )
#     use_y = (ref[:, 0].abs() > 0.9).unsqueeze(1)
#     alt = torch.where(use_y, yhat, xhat)
#     b1 = _unitize(torch.cross(ref, alt, dim=1))
#     b2 = torch.cross(ref, b1, dim=1)

#     phi = 2.0 * torch.pi * torch.rand(N, 1, device=ref.device, generator=generator)
#     perp = torch.cos(phi) * b1 + torch.sin(phi) * b2
#     ref_eff = torch.where((mu < 0).unsqueeze(1), perp, ref)

#     eps = torch.randn(N, 3, device=ref.device, generator=generator)
#     orient = _unitize(a.unsqueeze(1) * ref_eff + eps)
#     return orient
