#!/usr/bin/env python3
"""
Joint Hamiltonian Monte Carlo inference of HOD + IA parameters using ξ(r) and ω(r).
"""

# Suppress XLA warnings
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import time
from pathlib import Path
from scipy.spatial import cKDTree

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random

from jax_diffhodIA_weighted import (
    DiffHalotoolsIA,
    sample_watson_orientations,
    Ncen,
    Nsat,
    per_host_softmax_over_ranks,
)
from diffhodIA_utils import mask_bad_halocat
from halotools.sim_manager import CachedHaloCatalog

print(f"JAX devices: {jax.devices()}")

# ============================================================
# Configuration
# ============================================================
# Fiducial/true parameters for validation
TRUE_PARAMS = {
    'mu_cen': 0.79,
    'mu_sat': 0.30,
    'logMmin': 12.02,
    'sigma_logM': 0.26,
    'logM0': 11.38,
    'logM1': 13.31,
    'alpha': 1.06,
}

MAX_MU = 0.95
BOX_SIZE = 250.0

# Variance reduction: average over multiple orientation samples
N_ORIENTATION_SAMPLES = 5

# Seed for inference catalog structure
SEED_OPT = 999

# MCMC settings
N_WARMUP = 500
N_SAMPLES = 1000
N_CHAINS = 4

# ============================================================
# Load target observations and covariances
# ============================================================
print("\nLoading target ω(r) and covariance...")
OMEGA_TARGET = np.load('/n/home04/spandya/IASim/src/nick_mcmc/Illustris/omega_sample1.npy')
OMEGA_COV = np.load('/n/home04/spandya/IASim/src/nick_mcmc/Illustris/omega_sample1_cov.npy')

print("\nLoading target ξ(r) and covariance...")
XI_TARGET = np.load('/n/home04/spandya/IASim/src/nick_mcmc/Illustris/xi_sample1.npy')
XI_COV = np.load('/n/home04/spandya/IASim/src/nick_mcmc/Illustris/xi_sample1_cov.npy')

N_BINS_OMEGA = len(OMEGA_TARGET)
N_BINS_XI = len(XI_TARGET)

print(f"  ω(r): {N_BINS_OMEGA} bins")
print(f"  ξ(r): {N_BINS_XI} bins")

# Add regularization for numerical stability
OMEGA_COV_REG = OMEGA_COV + 1e-10 * np.eye(N_BINS_OMEGA)
XI_COV_REG = XI_COV + 1e-10 * np.eye(N_BINS_XI)

OMEGA_COV_JAX = jnp.array(OMEGA_COV_REG)
XI_COV_JAX = jnp.array(XI_COV_REG)
OMEGA_TARGET_JAX = jnp.array(OMEGA_TARGET)
XI_TARGET_JAX = jnp.array(XI_TARGET)

# Define r_bins (must match target measurements)
r_bins_omega = np.logspace(np.log10(0.1), np.log10(16.0), N_BINS_OMEGA + 1)
r_bins_xi = np.logspace(np.log10(0.1), np.log10(16.0), N_BINS_XI + 1)
R_BINS_OMEGA = jnp.array(r_bins_omega)
R_BINS_XI = jnp.array(r_bins_xi)
N_BINS_OMEGA_INT = N_BINS_OMEGA
N_BINS_XI_INT = N_BINS_XI

# ============================================================
# Load halo catalog
# ============================================================
print("\nLoading halo catalog...")
halocat = CachedHaloCatalog(
    simname="bolplanck",
    halo_finder="rockstar",
    redshift=0,
    version_name="halotools_v0p4",
)
mask_bad_halocat(halocat)
subcat = halocat.halo_table[[
    "halo_id", "halo_upid", "halo_mvir", "halo_x", "halo_y", "halo_z",
    "halo_axisA_x", "halo_axisA_y", "halo_axisA_z", "halo_rvir",
    "halo_hostid", "halo_b_to_a"
]]

# ============================================================
# Pre-compute halo catalog arrays (fixed across inference)
# ============================================================
print("\nPre-computing halo catalog arrays...")

# Extract host and subhalo information
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

# Build host-subhalo mapping
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

# Store as JAX arrays
HOST_POS = jnp.array(np.stack([halo_x[host_mask], halo_y[host_mask], halo_z[host_mask]], axis=1))
HOST_MVIR = jnp.array(halo_mvir[host_mask])
HOST_RVIR = jnp.array(halo_rvir[host_mask])
HOST_AXIS = jnp.array(np.stack([ax_x[host_mask], ax_y[host_mask], ax_z[host_mask]], axis=1))
HOST_AXIS = HOST_AXIS / (jnp.linalg.norm(HOST_AXIS, axis=-1, keepdims=True) + 1e-12)

SUB_POS = jnp.array(np.stack([
    halo_x[sub_mask][keep_mask],
    halo_y[sub_mask][keep_mask],
    halo_z[sub_mask][keep_mask]
], axis=1))
SUB_MVIR = jnp.array(halo_mvir[sub_mask][keep_mask])
SUB_HOST_IDS = jnp.array(sub_host_idx, dtype=jnp.int32)

N_HOSTS = HOST_POS.shape[0]
N_SUBS = SUB_POS.shape[0]

print(f"  N_hosts = {N_HOSTS}")
print(f"  N_subhalos = {N_SUBS}")

# ============================================================
# Helper functions
# ============================================================
def build_neighbor_pairs_numpy(pos: np.ndarray, r_max: float, box_size: float = 250.0):
    """Build neighbor pair list using scipy KDTree."""
    tree = cKDTree(pos, boxsize=box_size)
    pairs = tree.query_pairs(r_max, output_type='ndarray')
    
    if len(pairs) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    
    i_idx = np.concatenate([pairs[:, 0], pairs[:, 1]])
    j_idx = np.concatenate([pairs[:, 1], pairs[:, 0]])
    
    return i_idx.astype(np.int32), j_idx.astype(np.int32)


# ============================================================
# Correlation functions WITHOUT jit decorator (will be traced once)
# ============================================================
def compute_omega_from_catalog(
    pos: jnp.ndarray,
    ori: jnp.ndarray,
    i_idx: jnp.ndarray,
    j_idx: jnp.ndarray,
    r_bins: jnp.ndarray,
    n_bins: int,
    box_size: float = 250.0
) -> jnp.ndarray:
    """Compute ω(r) from galaxy positions and orientations."""
    ori_norm = ori / (jnp.linalg.norm(ori, axis=-1, keepdims=True) + 1e-12)
    
    diff = pos[j_idx] - pos[i_idx]
    diff = jnp.where(diff > box_size / 2, diff - box_size, diff)
    diff = jnp.where(diff < -box_size / 2, diff + box_size, diff)
    
    r = jnp.linalg.norm(diff, axis=-1)
    r_hat = diff / (r[:, None] + 1e-12)
    
    cos_theta = jnp.sum(ori_norm[i_idx] * r_hat, axis=-1)
    alignment = cos_theta ** 2
    
    bin_idx = jnp.searchsorted(r_bins, r) - 1
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)
    
    valid_mask = (r >= r_bins[0]) & (r < r_bins[-1])
    alignment = jnp.where(valid_mask, alignment, 0.0)
    valid_count = valid_mask.astype(jnp.float32)
    
    alignment_sum = jnp.zeros(n_bins).at[bin_idx].add(alignment)
    pair_counts = jnp.zeros(n_bins).at[bin_idx].add(valid_count)
    
    omega = (alignment_sum / (pair_counts + 1e-30)) - (1.0 / 3.0)
    return omega


def compute_xi_weighted(
    pos: jnp.ndarray,
    weights: jnp.ndarray,
    i_idx: jnp.ndarray,
    j_idx: jnp.ndarray,
    r_bins: jnp.ndarray,
    n_bins: int,
    box_size: float = 250.0
) -> jnp.ndarray:
    """Compute ξ(r) using galaxy weights."""
    diff = pos[j_idx] - pos[i_idx]
    diff = jnp.where(diff > box_size / 2, diff - box_size, diff)
    diff = jnp.where(diff < -box_size / 2, diff + box_size, diff)
    r = jnp.linalg.norm(diff, axis=-1)
    
    w_pairs = weights[i_idx] * weights[j_idx]
    
    bin_idx = jnp.searchsorted(r_bins, r, side='right') - 1
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)
    
    valid_mask = (r >= r_bins[0]) & (r < r_bins[-1])
    w_pairs = jnp.where(valid_mask, w_pairs, 0.0)
    
    DD = jnp.zeros(n_bins).at[bin_idx].add(w_pairs)
    
    total_weight = jnp.sum(weights)
    weight_sq_sum = jnp.sum(weights**2)
    N_eff_pairs = total_weight**2 - weight_sq_sum
    
    V_shells = (4.0 / 3.0) * jnp.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    V_total = box_size**3
    RR = N_eff_pairs * (V_shells / V_total)
    
    xi = DD / (RR + 1e-30) - 1.0
    return xi


# ============================================================
# Pre-compute neighbor pairs for all possible galaxy positions
# ============================================================
print("\nPre-computing neighbor pairs...")

# All possible galaxy positions: host centers + subhalo centers
ALL_POS = np.array(jnp.concatenate([HOST_POS, SUB_POS], axis=0))
r_max = max(r_bins_omega[-1], r_bins_xi[-1])

i_idx_all, j_idx_all = build_neighbor_pairs_numpy(ALL_POS, r_max, BOX_SIZE)
I_IDX = jnp.array(i_idx_all)
J_IDX = jnp.array(j_idx_all)
ALL_POS_JAX = jnp.array(ALL_POS)

print(f"  Total potential sources = {len(ALL_POS)}")
print(f"  N_pairs = {len(i_idx_all)}")

# Pre-compute reference directions for satellites (radial alignment)
SUB_REF_DIRS = SUB_POS - HOST_POS[SUB_HOST_IDS]
SUB_REF_DIRS = SUB_REF_DIRS / (jnp.linalg.norm(SUB_REF_DIRS, axis=-1, keepdims=True) + 1e-12)


# ============================================================
# Forward model: compute weights and orientations from parameters
# ============================================================
def compute_weights(logMmin, sigma_logM, logM0, logM1, alpha, t_rank=0.5):
    """
    Compute galaxy weights from HOD parameters.
    
    Returns weights for all potential sources (hosts + subhalos).
    """
    # Central weights
    mean_N_cen = Ncen(HOST_MVIR, logMmin, sigma_logM)
    
    # Satellite weights via softmax over subhalos
    mean_N_sat = Nsat(HOST_MVIR, logMmin, sigma_logM, logM0, logM1, alpha)
    q = per_host_softmax_over_ranks(SUB_HOST_IDS, SUB_MVIR, t_rank=t_rank)
    sat_weights = q * mean_N_sat[SUB_HOST_IDS]
    
    # Concatenate: [centrals, satellites]
    all_weights = jnp.concatenate([mean_N_cen, sat_weights], axis=0)
    
    return all_weights


def compute_orientations(mu_cen, mu_sat, key):
    """
    Compute galaxy orientations from IA parameters.
    
    Returns orientations for all potential sources (hosts + subhalos).
    """
    # Central orientations: aligned with host axis
    key_cen, key_sat = random.split(key)
    cen_ori = sample_watson_orientations(key_cen, HOST_AXIS, mu_cen)
    
    # Satellite orientations: radial alignment
    sat_ori = sample_watson_orientations(key_sat, SUB_REF_DIRS, mu_sat)
    
    # Concatenate: [centrals, satellites]
    all_ori = jnp.concatenate([cen_ori, sat_ori], axis=0)
    
    return all_ori


def forward_model(mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha, base_seed):
    """
    Full forward model: parameters -> (ξ, ω).
    
    Averages ω over multiple orientation samples for variance reduction.
    ξ depends only on weights (HOD), so no averaging needed.
    """
    # Clip parameters to valid ranges
    mu_cen = jnp.clip(mu_cen, -MAX_MU, MAX_MU)
    mu_sat = jnp.clip(mu_sat, -MAX_MU, MAX_MU)
    sigma_logM = jnp.abs(sigma_logM) + 1e-6
    alpha = jnp.abs(alpha) + 1e-6
    
    # Compute weights (deterministic given HOD params)
    weights = compute_weights(logMmin, sigma_logM, logM0, logM1, alpha)
    
    # Compute ξ(r) from weights
    xi_pred = compute_xi_weighted(
        ALL_POS_JAX, weights, I_IDX, J_IDX, R_BINS_XI, N_BINS_XI_INT, BOX_SIZE
    )
    
    # Compute ω(r) averaged over multiple orientation samples
    keys = jnp.array([random.PRNGKey(base_seed + i) for i in range(N_ORIENTATION_SAMPLES)])
    
    def single_omega(key):
        ori = compute_orientations(mu_cen, mu_sat, key)
        omega = compute_omega_from_catalog(
            ALL_POS_JAX, ori, I_IDX, J_IDX, R_BINS_OMEGA, N_BINS_OMEGA_INT, BOX_SIZE
        )
        return jnp.nan_to_num(omega, nan=0.0)
    
    omegas = jax.vmap(single_omega)(keys)
    omega_pred = jnp.mean(omegas, axis=0)
    
    return xi_pred, omega_pred


# ============================================================
# NumPyro model
# ============================================================
def joint_model():
    """
    NumPyro model for joint HOD + IA inference.
    
    Priors adapted from the literature with means near fiducial values.
    """
    # IA parameters: uniform priors
    mu_cen = numpyro.sample('mu_cen', dist.Uniform(-MAX_MU, MAX_MU))
    mu_sat = numpyro.sample('mu_sat', dist.Uniform(-MAX_MU, MAX_MU))
    
    # HOD parameters: Gaussian priors centered on fiducial values
    logMmin = numpyro.sample('logMmin', dist.Normal(12.0, 0.5))
    sigma_logM = numpyro.sample('sigma_logM', dist.TruncatedNormal(0.25, 0.2, low=0.01))
    logM0 = numpyro.sample('logM0', dist.Normal(11.25, 0.5))
    logM1 = numpyro.sample('logM1', dist.Normal(13.20, 0.5))
    alpha = numpyro.sample('alpha', dist.TruncatedNormal(1.0, 0.2, low=0.1))
    
    # Forward model
    BASE_SEED = 12345
    xi_pred, omega_pred = forward_model(
        mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha, BASE_SEED
    )
    
    # Likelihood for ξ(r)
    numpyro.sample(
        'xi_obs',
        dist.MultivariateNormal(xi_pred, covariance_matrix=XI_COV_JAX),
        obs=XI_TARGET_JAX
    )
    
    # Likelihood for ω(r)
    numpyro.sample(
        'omega_obs',
        dist.MultivariateNormal(omega_pred, covariance_matrix=OMEGA_COV_JAX),
        obs=OMEGA_TARGET_JAX
    )


def run_joint_mcmc(rng_key, num_warmup=500, num_samples=1000, num_chains=4):
    """Run joint HMC inference."""
    
    nuts_kernel = NUTS(
        joint_model,
        target_accept_prob=0.80,
        max_tree_depth=10,
    )
    
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    
    mcmc.run(rng_key)
    
    return mcmc


# ============================================================
# Main execution
# ============================================================
def main():
    print("\n" + "="*80)
    print("Joint Hamiltonian Monte Carlo Inference: HOD + IA Parameters")
    print("="*80)
    print("\nTrue parameters:")
    for name, val in TRUE_PARAMS.items():
        print(f"  {name}: {val}")
    print(f"\nN_ORIENTATION_SAMPLES = {N_ORIENTATION_SAMPLES}")
    print(f"MCMC settings: {N_WARMUP} warmup, {N_SAMPLES} samples, {N_CHAINS} chains")
    
    # Diagnostic: check forward model at true parameters
    print("\nDiagnostic: Forward model at true parameters...")
    xi_true, omega_true = forward_model(
        TRUE_PARAMS['mu_cen'], TRUE_PARAMS['mu_sat'],
        TRUE_PARAMS['logMmin'], TRUE_PARAMS['sigma_logM'],
        TRUE_PARAMS['logM0'], TRUE_PARAMS['logM1'], TRUE_PARAMS['alpha'],
        12345
    )
    
    chi2_xi = float((xi_true - XI_TARGET_JAX) @ jnp.linalg.inv(XI_COV_JAX) @ (xi_true - XI_TARGET_JAX))
    chi2_omega = float((omega_true - OMEGA_TARGET_JAX) @ jnp.linalg.inv(OMEGA_COV_JAX) @ (omega_true - OMEGA_TARGET_JAX))
    print(f"  χ²(ξ) at true params: {chi2_xi:.2f}")
    print(f"  χ²(ω) at true params: {chi2_omega:.2f}")
    
    # Run MCMC
    print("\nRunning joint HMC...")
    start_time = time.time()
    
    rng_key = random.PRNGKey(42)
    mcmc = run_joint_mcmc(rng_key, N_WARMUP, N_SAMPLES, N_CHAINS)
    
    elapsed = time.time() - start_time
    print(f"\nMCMC completed in {elapsed/60:.1f} minutes")
    
    # Print summary
    mcmc.print_summary()
    
    # Get samples
    samples = mcmc.get_samples()
    
    # Compute and print posterior statistics
    print("\n" + "="*80)
    print("Posterior Summary")
    print("="*80)
    
    param_names = ['mu_cen', 'mu_sat', 'logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha']
    
    results = {}
    for name in param_names:
        samps = np.array(samples[name])
        mean = np.mean(samps)
        std = np.std(samps)
        true_val = TRUE_PARAMS[name]
        bias = mean - true_val
        
        results[name] = {
            'samples': samps,
            'mean': mean,
            'std': std,
            'true': true_val,
            'bias': bias,
        }
        
        print(f"\n{name}:")
        print(f"  True:       {true_val:.4f}")
        print(f"  Mean ± Std: {mean:.4f} ± {std:.4f}")
        print(f"  Bias:       {bias:.4f} ({bias/std:.2f}σ)")
    
    # Compute correlation matrix
    print("\n" + "="*80)
    print("Posterior Correlation Matrix")
    print("="*80)
    
    all_samples = np.column_stack([results[name]['samples'] for name in param_names])
    corr_matrix = np.corrcoef(all_samples.T)
    
    # Print header
    print(f"{'':>12}", end='')
    for name in param_names:
        print(f"{name:>10}", end='')
    print()
    
    # Print matrix
    for i, name_i in enumerate(param_names):
        print(f"{name_i:>12}", end='')
        for j in range(len(param_names)):
            print(f"{corr_matrix[i,j]:>10.3f}", end='')
        print()
    
    # Save results
    output_dir = Path('joint_hmc_results')
    output_dir.mkdir(exist_ok=True)
    
    save_results = {
        'samples': {name: results[name]['samples'] for name in param_names},
        'means': {name: results[name]['mean'] for name in param_names},
        'stds': {name: results[name]['std'] for name in param_names},
        'true_params': TRUE_PARAMS,
        'correlation_matrix': corr_matrix,
        'param_names': param_names,
        'xi_target': np.array(XI_TARGET),
        'omega_target': np.array(OMEGA_TARGET),
        'xi_cov': np.array(XI_COV),
        'omega_cov': np.array(OMEGA_COV),
        'n_warmup': N_WARMUP,
        'n_samples': N_SAMPLES,
        'n_chains': N_CHAINS,
        'n_orientation_samples': N_ORIENTATION_SAMPLES,
        'elapsed_minutes': elapsed / 60,
    }
    
    results_file = output_dir / 'joint_hmc_results.npy'
    np.save(results_file, save_results, allow_pickle=True)
    print(f"\nResults saved to {results_file}")
    
    print("\n" + "="*80)
    print("Joint HMC inference complete!")
    print("="*80)


if __name__ == "__main__":
    main()