import h5py
import numpy as np
from halotools.mock_observables import ed_3d, ee_3d
from tqdm import tqdm

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
