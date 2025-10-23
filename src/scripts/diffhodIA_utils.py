import h5py
import numpy as np
from halotools.mock_observables import ed_3d, ee_3d
from tqdm import tqdm
from typing import Optional, Tuple, Sequence
import matplotlib.pyplot as plt

# --------------------- Loading helpers --------------------


def mask_bad_halocat(halocat):
    bad_mask = (
        (halocat.halo_table["halo_axisA_x"] == 0)
        & (halocat.halo_table["halo_axisA_y"] == 0)
        & (halocat.halo_table["halo_axisA_z"] == 0)
    )
    halocat._halo_table = halocat.halo_table[~bad_mask]


def load_cleaned_catalogs_from_h5(path, as_list=True, debug=False):
    """
    Load cleaned catalogs stored with save_cleaned_catalogs_to_h5.

    Parameters
    ----------
    path : str
        Path to .h5 file.
    as_list : bool, default=True
        - True: return list of (n_i, ncols) numpy arrays
        - False: return dict {index: np.ndarray}
    debug : bool, default=False
        If True, only load the first 10 catalogs for quick testing.

    Returns
    -------
    data : list or dict
    columns : list of str or None
    """
    with h5py.File(path, "r") as f:
        n_catalogs = f.attrs.get("n_catalogs")
        ncols = f.attrs.get("ncols")
        columns = None
        if "columns" in f.attrs:
            columns = [c.decode("utf-8") for c in f.attrs["columns"]]

        g = f["catalogs"]

        if as_list:
            catalogs = []
            limit = 5 if debug else n_catalogs
            for i in tqdm(range(limit), desc="loading catalogs"):
                ds = g[f"{i:08d}"]
                catalogs.append(ds[...])  # (nrows, ncols), dtype=float32
        else:
            catalogs = {}
            keys = list(g.keys())
            if debug:
                keys = keys[:10]
            for k in tqdm(keys, desc="loading catalogs"):
                catalogs[int(k)] = g[k][...]

    return catalogs, columns


# -------------------- plotting helpers --------------------


def _as_xyz(a, name):
    """
    Ensure input is a (N, 3) numpy array of floats.
    a: array-like
    name: string, name of the variable (for error messages)

    Returns: (N, 3) numpy array of floats
    """
    a = np.asarray(a, dtype=float)
    # If flat, try to reinterpret as triplets
    if a.ndim == 1 and a.size % 3 == 0:
        a = a.reshape(-1, 3)
    # If transposed (3, N), fix it
    if a.ndim == 2 and a.shape[0] == 3 and a.shape[1] != 3:
        a = a.T
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3); got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(a)


def _unit_vectors(v, name):
    """
    Normalize a set of vectors to unit length.
    v: array-like of shape (N, 3)
    name: string, name of the variable (for error messages)

    Returns: (N, 3) numpy array of unit vectors
    """
    v = _as_xyz(v, name)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    good = n[:, 0] > 0
    out = np.zeros_like(v)
    out[good] = v[good] / n[good]
    if (~good).any():
        m = (~good).sum()
        rnd = np.random.normal(size=(m, 3))
        rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
        out[~good] = rnd
    return out


def compute_ed_ee(orig_pos, orig_ori, gen_pos, gen_ori, L, rbins=None, num_threads=1):
    """
    Compute the 3D ED and EE correlation functions for two catalogs.
    orig_pos, gen_pos: (N, 3) numpy arrays of positions
    orig_ori, gen_ori: (N, 3) numpy arrays of unit orientation vectors
    L: box size (scalar)
    rbins: optional array of bin edges for r; if None, use default log-spaced bins from 0.1 to 16 Mpc/h
    num_threads: number of threads to use (Halotools)

    Returns: dictionary with keys:
    - "r": (N,) array of radial bin centers
    - "rbins": (N+1,) array of radial bin edges
    - "ed_true": (N,) array of true ED values
    - "ed_gen": (N,) array of generated ED values
    - "ee_true": (N,) array of true EE values
    - "ee_gen": (N,) array of generated EE values
    """
    # sanitize + periodic wrap
    orig_pos = _as_xyz(orig_pos, "orig_pos")
    orig_pos = np.mod(orig_pos, L)
    gen_pos = _as_xyz(gen_pos, "gen_pos")
    gen_pos = np.mod(gen_pos, L)

    orig_ori = _unit_vectors(orig_ori, "orig_ori")
    gen_ori = _unit_vectors(gen_ori, "gen_ori")

    # length checks (Halotools will also check later, but this is clearer)
    if len(orig_pos) != len(orig_ori):
        raise ValueError(
            f"orig_pos (N={len(orig_pos)}) and orig_ori (N={len(orig_ori)}) must match"
        )
    if len(gen_pos) != len(gen_ori):
        raise ValueError(
            f"gen_pos (N={len(gen_pos)}) and gen_ori (N={len(gen_ori)}) must match"
        )

    if rbins is None:
        rbins = np.logspace(np.log10(0.1), np.log10(16), 20)

    # Auto-correlations (pass sample2 = sample1)
    ed_true = ed_3d(
        sample1=orig_pos,
        sample2=orig_pos,
        orientations1=orig_ori,
        rbins=rbins,
        period=L,
        num_threads=num_threads,
    )
    ed_gen = ed_3d(
        sample1=gen_pos,
        sample2=gen_pos,
        orientations1=gen_ori,
        rbins=rbins,
        period=L,
        num_threads=num_threads,
    )

    ee_true = ee_3d(
        sample1=orig_pos,
        sample2=orig_pos,
        orientations1=orig_ori,
        orientations2=orig_ori,
        rbins=rbins,
        period=L,
        num_threads=num_threads,
    )
    ee_gen = ee_3d(
        sample1=gen_pos,
        sample2=gen_pos,
        orientations1=gen_ori,
        orientations2=gen_ori,
        rbins=rbins,
        period=L,
        num_threads=num_threads,
    )

    r_mid = 0.5 * (rbins[1:] + rbins[:-1])
    return {
        "r": r_mid,
        "rbins": rbins,
        "ed_true": ed_true,
        "ed_gen": ed_gen,
        "ee_true": ee_true,
        "ee_gen": ee_gen,
    }

from typing import Optional, Tuple, Literal, Union, Dict

def compute_correlations(
    builder=None,
    gal_cat: Optional["np.ndarray"] = None,
    *,
    L: Optional[float] = 250.0,
    rbins_2pcf: Optional["np.ndarray"] = None,
    rbins_ia: Optional["np.ndarray"] = None,
    num_threads: int = 4,
    which: Literal["all", "xi", "omega", "eta"] = "all",
) -> Union[
    Dict[str, "np.ndarray"],
    Tuple["np.ndarray", "np.ndarray"]
]:
    """
    Compute 2pcf (xi), ED 3D (omega), and EE 3D (eta) for a single catalog.

    Parameters
    ----------
    builder : object, optional
        If provided and `gal_cat` is None, used as `builder.return_catalog()`.
    gal_cat : (N, >=6) array-like, optional
        Columns [:3] = positions (x,y,z), [3:6] = orientations.
    L : float or None, default 250.0
        Period for periodic boundary conditions. If None, no periodicity.
    rbins_2pcf : array-like, optional
        Radial bins for xi(r). Default: logspace(0.1, 8.0) with 10 bins.
    rbins_ia : array-like, optional
        Radial bins for alignment stats. Defaults to `rbins_2pcf`.
    num_threads : int, default 4
        Threads for halotools routines.
    which : {"all", "xi", "omega", "eta"}, default "all"
        Select a single statistic or return all.

    Returns
    -------
    If which == "all":
        dict with keys:
          - "r_mid_2pcf": bin centers for xi
          - "r_mid_ia":   bin centers for omega/eta
          - "xi":   xi(r)
          - "omega": ed_3d(r)
          - "eta":  ee_3d(r)
    If which in {"xi", "omega", "eta"}:
        (r_mid, values) tuple for the requested stat.
    """
    import numpy as np
    from halotools.mock_observables import tpcf, ed_3d, ee_3d

    # --- catalog handling ---
    if gal_cat is None:
        if builder is None:
            raise ValueError("Provide either `gal_cat` or `builder`.")
        gal_cat = builder.return_catalog()

    arr = np.asarray(gal_cat, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError("`gal_cat` must have shape (N, >=6) with [:3]=pos and [3:6]=orientations.")
    pos = arr[:, :3]
    ori = arr[:, 3:6]

    # --- periodic vs non-periodic ---
    if L is not None:
        period_kw = {"period": float(L)}
        pos_stats = np.mod(pos, float(L))
    else:
        period_kw = {}
        pos_stats = pos

    # --- binning ---
    if rbins_2pcf is None:
        rbins_2pcf = np.logspace(np.log10(0.1), np.log10(8.0), 20)
    rbins_2pcf = np.asarray(rbins_2pcf, dtype=float)

    if rbins_ia is None:
        rbins_ia = rbins_2pcf
    rbins_ia = np.asarray(rbins_ia, dtype=float)

    r_mid_2pcf = 0.5 * (rbins_2pcf[1:] + rbins_2pcf[:-1])
    r_mid_ia   = 0.5 * (rbins_ia[1:]   + rbins_ia[:-1])

    # --- compute requested stats ---
    want_xi    = which in ("all", "xi")
    want_omega = which in ("all", "omega")
    want_eta   = which in ("all", "eta")

    xi = omega = eta = None

    if want_xi:
        xi = tpcf(sample1=pos_stats, rbins=rbins_2pcf, num_threads=num_threads, **period_kw)

    if want_omega:
        omega = ed_3d(
            sample1=pos_stats, sample2=pos_stats,
            orientations1=ori,
            rbins=rbins_ia, num_threads=num_threads, **period_kw
        )

    if want_eta:
        eta = ee_3d(
            sample1=pos_stats, sample2=pos_stats,
            orientations1=ori, orientations2=ori,
            rbins=rbins_ia, num_threads=num_threads, **period_kw
        )

    # --- return format ---
    if which == "xi":
        return r_mid_2pcf, xi
    if which == "omega":
        return r_mid_ia, omega
    if which == "eta":
        return r_mid_ia, eta

    return {
        "r_bins": r_mid_2pcf,
        "xi": xi,
        "omega": omega,
        "eta": eta,
    }

# -------------------- (optional) plotting diagnostic --------------------

def plot_diagnostic(
    builder,
    gal_cat: Optional["np.ndarray"] = None,
    orig_catalog: Optional["np.ndarray"] = None,
    *,
    L: Optional[float] = 250.0,
    nbins: int = 200,
    rbins_2pcf: Optional["np.ndarray"] = None,
    rbins_ia: Optional["np.ndarray"] = None,
    num_threads: int = 4,
    figsize: Optional[Tuple[int, int]] = None,
    axs: Optional[Sequence["plt.Axes"]] = None,
    fig: Optional["plt.Figure"] = None,
    clear: bool = False,
):
    """
    Same panels/behavior as your torch version. Converts to numpy for halotools/matplotlib.
    If `axs` is provided, draw into those axes (do not create new ones).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from halotools.mock_observables import ed_3d, ee_3d, tpcf

    if gal_cat is None:
        gal_cat = builder.return_catalog()
    gal_cat = np.asarray(gal_cat)
    if gal_cat is None or gal_cat.size == 0:
        raise RuntimeError("Generated catalog is empty.")
    gen = np.asarray(gal_cat)
    gen_pos, gen_ori = gen[:, :3], gen[:, 3:6]

    has_orig = orig_catalog is not None
    if has_orig:
        orig = np.asarray(orig_catalog, dtype=float)
        if orig.ndim != 2 or orig.shape[1] < 6:
            raise ValueError("orig_catalog must have shape (N, >=6)")
        orig_pos, orig_ori = orig[:, :3], orig[:, 3:6]

    if L is not None:
        xg, yg = np.mod(gen_pos[:, 0], L), np.mod(gen_pos[:, 1], L)
        if has_orig:
            xo, yo = np.mod(orig_pos[:, 0], L), np.mod(orig_pos[:, 1], L)
        extent = [0.0, L, 0.0, L]
        period_kw = {"period": L}
        pos_gen_for_stats = np.mod(gen_pos, L)
        pos_orig_for_stats = np.mod(orig_pos, L) if has_orig else None
    else:
        xg, yg = gen_pos[:, 0], gen_pos[:, 1]
        if has_orig:
            xo, yo = orig_pos[:, 0], orig_pos[:, 1]
        xmin = np.min(xg) if not has_orig else min(xg.min(), xo.min())
        xmax = np.max(xg) if not has_orig else max(xg.max(), xo.max())
        ymin = np.min(yg) if not has_orig else min(yg.min(), yo.min())
        ymax = np.max(yg) if not has_orig else max(yg.max(), yo.max())
        extent = [xmin, xmax, ymin, ymax]
        period_kw = {}
        pos_gen_for_stats = gen_pos
        pos_orig_for_stats = orig_pos if has_orig else None

    if has_orig:
        Ho, xe, ye = np.histogram2d(
            xo, yo, bins=nbins, range=[[extent[0], extent[1]], [extent[2], extent[3]]]
        )
        Hg, _, _ = np.histogram2d(xg, yg, bins=[xe, ye])
        Ho = Ho / (Ho.sum() + 1e-12)
        Hg = Hg / (Hg.sum() + 1e-12)
    else:
        Hg, xe, ye = np.histogram2d(
            xg, yg, bins=nbins, range=[[extent[0], extent[1]], [extent[2], extent[3]]]
        )
        Hg = Hg / (Hg.sum() + 1e-12)

    if rbins_2pcf is None:
        rbins_2pcf = np.logspace(np.log10(0.1), np.log10(8.0), 10)
    if rbins_ia is None:
        rbins_ia = rbins_2pcf
    r_mid_2pcf = 0.5 * (rbins_2pcf[1:] + rbins_2pcf[:-1])
    r_mid_ia = 0.5 * (rbins_ia[1:] + rbins_ia[:-1])

    xi_gen = tpcf(
        sample1=pos_gen_for_stats, rbins=rbins_2pcf, num_threads=num_threads, **period_kw
    )
    if has_orig:
        xi_orig = tpcf(
            sample1=pos_orig_for_stats, rbins=rbins_2pcf, num_threads=num_threads, **period_kw
        )

    ed_gen = ed_3d(
        sample1=pos_gen_for_stats,
        sample2=pos_gen_for_stats,
        orientations1=gen_ori,
        rbins=rbins_ia,
        num_threads=num_threads,
        **period_kw,
    )
    ee_gen = ee_3d(
        sample1=pos_gen_for_stats,
        sample2=pos_gen_for_stats,
        orientations1=gen_ori,
        orientations2=gen_ori,
        rbins=rbins_ia,
        num_threads=num_threads,
        **period_kw,
    )
    if has_orig:
        ed_orig = ed_3d(
            sample1=pos_orig_for_stats,
            sample2=pos_orig_for_stats,
            orientations1=orig_ori,
            rbins=rbins_ia,
            num_threads=num_threads,
            **period_kw,
        )
        ee_orig = ee_3d(
            sample1=pos_orig_for_stats,
            sample2=pos_orig_for_stats,
            orientations1=orig_ori,
            orientations2=orig_ori,
            rbins=rbins_ia,
            num_threads=num_threads,
            **period_kw,
        )

    # --- Figure/Axes handling ---
    ncols = 5 if has_orig else 4
    if axs is None:
        if figsize is None:
            figsize = (18, 4) if has_orig else (16, 4)
        fig, axs = plt.subplots(1, ncols, figsize=figsize, constrained_layout=True)
    else:
        # normalize axs to a flat list/array
        try:
            n_available = len(axs)
        except TypeError:
            raise TypeError("`axs` must be a sequence of matplotlib Axes with length >= required panels.")
        if n_available < ncols:
            raise ValueError(f"`axs` has {n_available} axes but {ncols} are required (has_orig={has_orig}).")
        if fig is None:
            # best-effort: take figure from the first axis
            fig = axs[0].figure
        if clear:
            for a in axs[:ncols]:
                a.cla()

    # --- Plotting ---
    col = 0
    if has_orig:
        axs[col].imshow(Ho.T, origin="lower", extent=extent, aspect="equal")
        axs[col].set_title("Original density")
        axs[col].set_xlabel("x [Mpc/h]")
        axs[col].set_ylabel("y [Mpc/h]")
        col += 1

    axs[col].imshow(Hg.T, origin="lower", extent=extent, aspect="equal")
    axs[col].set_title("Generated density")
    axs[col].set_xlabel("x [Mpc/h]")
    axs[col].set_ylabel("y [Mpc/h]")
    col += 1

    ax = axs[col]
    if has_orig:
        ax.loglog(r_mid_2pcf, xi_orig, label="Original")
        ax.loglog(r_mid_2pcf, xi_gen, "--", label="Generated")
        ax.legend()
    else:
        ax.loglog(r_mid_2pcf, xi_gen)
    ax.set_title(r"$\xi(r)$")
    ax.set_xlabel(r"$r$ [Mpc/h]")
    col += 1

    ax = axs[col]
    if has_orig:
        ax.semilogx(r_mid_ia, ed_orig, label="Original")
        ax.semilogx(r_mid_ia, ed_gen, "--", label="Generated")
        ax.legend()
    else:
        ax.semilogx(r_mid_ia, ed_gen)
    ax.set_title(r"$\omega(r)$")
    ax.set_xlabel(r"$r$ [Mpc/h]")
    col += 1

    ax = axs[col]
    if has_orig:
        ax.semilogx(r_mid_ia, ee_orig, label="Original")
        ax.semilogx(r_mid_ia, ee_gen, "--", label="Generated")
        ax.legend()
    else:
        ax.semilogx(r_mid_ia, ee_gen)
    ax.set_title(r"$\eta(r)$")
    ax.set_xlabel(r"$r$ [Mpc/h]$")

    # increase all font sizes
    for a in axs[:ncols]:
        a.tick_params(axis="both", which="major", labelsize=12)
        a.title.set_fontsize(14)
        a.xaxis.label.set_fontsize(14)
        a.yaxis.label.set_fontsize(14)

    return fig, axs
