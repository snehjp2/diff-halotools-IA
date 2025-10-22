# jax_train_diffhodIA_reparam.py
import csv
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from diffhodIA_utils import load_cleaned_catalogs_from_h5, mask_bad_halocat
from halotools.sim_manager import CachedHaloCatalog

# Your JAX builder
from jax_diffhodIA import DiffHalotoolsIA


# ---------- logger ----------
class CatalogLogger:
    def __init__(self, root="runs", run_name=None, save_every=1, sample_n=5000):
        ts = time.strftime("%Y%m%d-%H%M%S") if run_name is None else run_name
        self.dir = os.path.join(root, ts)
        os.makedirs(self.dir, exist_ok=True)
        self.csv_path = os.path.join(self.dir, "scalars.csv")
        self.save_every = int(save_every)
        self.sample_n = sample_n
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["step", "loss", "N_true", "N_hat", "params_json", "seed", "notes"]
                )

    def _np(self, x):
        return np.asarray(x)

    def log_step(
        self, step, loss, N_true, N_hat, params, gal_cat, seed=12345, notes=""
    ):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    int(step),
                    float(np.asarray(loss)),
                    int(N_true),
                    float(np.asarray(N_hat)),
                    json.dumps(self._np(params).tolist()),
                    int(seed),
                    str(notes),
                ]
            )
        if (step % self.save_every) == 0:
            cat_np = self._np(gal_cat).copy()
            out_path = os.path.join(self.dir, f"catalog_step_{step:05d}.npz")
            sample = None
            if (self.sample_n is not None) and (cat_np.shape[0] > self.sample_n):
                rng = np.random.default_rng(seed + step)
                idx = rng.choice(cat_np.shape[0], size=self.sample_n, replace=False)
                sample = cat_np[idx]
            np.savez_compressed(
                out_path,
                gal_cat=cat_np,
                sample=(
                    sample if sample is not None else np.empty((0, cat_np.shape[1]))
                ),
                params=self._np(params).copy(),
                N_true=np.array([N_true], dtype=np.int64),
                N_hat=np.array([float(np.asarray(N_hat))], dtype=np.float32),
                loss=np.array([float(np.asarray(loss))], dtype=np.float32),
                seed=np.array([seed], dtype=np.int64),
                notes=np.array([notes]),
            )
            with open(os.path.join(self.dir, "latest.txt"), "w") as f:
                f.write(f"{step}\n")


# ---------- stable HOD means ----------
def _Ncen(M, logMmin, sigma_logM):
    sigma = jnp.abs(sigma_logM) + 1e-6
    term = (jnp.log10(M) - logMmin) / sigma
    out = 0.5 * (1.0 + jax.scipy.special.erf(term))
    return jnp.clip(out, 0.0, 1.0)


def _Nsat(M, logMmin, sigma_logM, logM0, logM1, alpha):
    sigma = jnp.abs(sigma_logM) + 1e-6
    alpha = jnp.abs(alpha) + 1e-6
    M0 = jnp.power(10.0, logM0)  # kept in stable range via reparam
    M1 = jnp.power(10.0, logM1)
    x = jnp.clip((M - M0) / M1, 0.0, None)
    ns = _Ncen(M, logMmin, sigma) * jnp.power(x, alpha)
    return jnp.nan_to_num(ns, posinf=0.0, neginf=0.0)


# ---------- reparameterization ----------
def unpack_params(theta):
    """
    theta: unconstrained R^7  -> physical params in the same order as your model:
      [mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha]
    """
    t0, t1, t2, t3, t4, t5, t6 = [theta[i] for i in range(7)]
    mu_cen = jnp.tanh(t0)  # (-1,1)
    mu_sat = jnp.tanh(t1)  # (-1,1)
    logMmin = 8.0 + 8.0 * jax.nn.sigmoid(t2)  # [8,16]
    sigma_logM = jax.nn.softplus(t3) + 1e-3  # > 0
    logM0_base = 8.0 + 7.0 * jax.nn.sigmoid(t4)  # [8,15]
    gap = jax.nn.softplus(t5) + 0.1  # >= 0.1
    logM1 = logM0_base + gap  # > logM0
    logM0 = logM0_base
    alpha = jax.nn.softplus(t6) + 1e-3  # > 0
    return jnp.array(
        [mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha], dtype=jnp.float32
    )


# ---------- differentiable surrogate ----------
def expected_counts_from_theta(theta, host_mvir):
    p = unpack_params(theta)
    mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha = p
    Nc = _Ncen(host_mvir, logMmin, sigma_logM)
    Ns = _Nsat(host_mvir, logMmin, sigma_logM, logM0, logM1, alpha)
    return jnp.sum(Nc) + jnp.sum(Ns), p


# ---------- data load ----------
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
host_mask = np.asarray(subcat["halo_upid"]) == -1
host_mvir = jnp.asarray(np.asarray(subcat["halo_mvir"], dtype=np.float32)[host_mask])

# ---------- training setup ----------
target = 500_000.0
num_steps = 200
base_seed = 12345
tau0, tau_min, decay = 0.5, 0.03, 0.98

# start from Torch values by inverting the reparam approximately
theta0 = jnp.array(
    [-1.75, -0.66, 0.8, 0.8, 0.8, 2.0, 0.1], dtype=jnp.float32
)  # decent seed


# optax: small LR + clip
lr = 1e-2
tx = optax.chain(
    optax.clip_by_global_norm(2.0),
    optax.adamw(lr, weight_decay=0.0),
)
opt_state = tx.init(theta0)

logger = CatalogLogger(
    root="runs", run_name="hod_maxN_jax_reparam", save_every=1, sample_n=10_000
)


# loss: relative MSE keeps scale ~1
def loss_and_aux(theta):
    N_hat, params_phys = expected_counts_from_theta(theta, host_mvir)
    rel = (N_hat - target) / (target + 1e-9)
    loss = rel * rel
    return loss, (N_hat, params_phys)


@jax.jit
def train_step(theta, opt_state):
    (loss_val, (N_hat, params_phys)), grads = jax.value_and_grad(
        loss_and_aux, has_aux=True
    )(theta)
    # scrub grads just in case
    grads = jax.tree_util.tree_map(
        lambda g: jnp.nan_to_num(g, posinf=0.0, neginf=0.0), grads
    )
    updates, opt_state2 = tx.update(grads, opt_state, theta)
    theta2 = optax.apply_updates(theta, updates)
    # very mild param scrub to avoid drift to NaN
    theta2 = jax.tree_util.tree_map(
        lambda x: jnp.nan_to_num(x, neginf=0.0, posinf=0.0), theta2
    )
    return theta2, opt_state2, loss_val, N_hat, params_phys


for step in range(num_steps):
    theta0, opt_state, loss_val, N_hat_avg, params_phys = train_step(theta0, opt_state)

    # build ONE concrete catalog for logging/visualization (using physical params)
    tau = max(tau_min, float(tau0 * (decay**step)))
    builder = DiffHalotoolsIA(
        subcat=subcat,
        params=params_phys,  # physical 7-vector
        do_discrete=True,
        do_nfw_fallback=True,
        seed=base_seed,
        satellite_alignment="radial",
        relaxed=True,
        tau=float(tau),
        Nmax_sat=48,
        t_rank=0.5,
    )
    gal_cat = np.asarray(builder.return_catalog())
    N_true = int(gal_cat.shape[0])

    # sanity guard
    if not (np.isfinite(loss_val) and np.isfinite(N_hat_avg)):
        raise FloatingPointError(
            f"NaN/Inf at step {step}: loss={loss_val}, N_hat={N_hat_avg}, params={np.asarray(params_phys)}"
        )

    logger.log_step(
        step, loss_val, N_true, N_hat_avg, params_phys, gal_cat, seed=base_seed
    )

    if (step % 2 == 0) or (step == num_steps - 1):
        print(
            f"[step {step:03d}] loss={float(loss_val):.6g}  "
            f"N_true={N_true:.0f}  N_hat={float(N_hat_avg):.1f}  "
            f"lr={lr:.3g}  tau={tau:.3f}"
        )
