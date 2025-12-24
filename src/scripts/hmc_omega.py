#!/usr/bin/env python3
"""
Hamiltonian Monte Carlo inference of intrinsic alignment parameters using omega(r).
IMPROVED VERSION with better variance reduction.
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
)
from diffhodIA_utils import mask_bad_halocat
from halotools.sim_manager import CachedHaloCatalog

print(f"JAX devices: {jax.devices()}")

# ============================================================
# Configuration
# ============================================================
TARGET_PARAMS = np.asarray([0.79, 0.30, 12.54, 0.26, 12.68, 13.48, 1.0])
HOD_PARAMS_FIXED = TARGET_PARAMS[2:]

TRUE_MU_CEN = 0.79
TRUE_MU_SAT = 0.30

MAX_MU = 0.95
BOX_SIZE = 250.0

# IMPROVED: Increase samples per step for variance reduction
N_SAMPLES_PER_STEP = 5  # Was 1, now 10

# Seed for inference catalog
SEED_OPT = 999

# MCMC settings
N_WARMUP = 500
N_SAMPLES = 1000
N_CHAINS = 4

# ============================================================
# Load target omega and covariance
# ============================================================
print("\nLoading target ω(r) and covariance...")

OMEGA_TARGET = np.load('/Users/snehpandya/Projects/IAEmu/Illustris/measurements/omega_sample1.npy')
OMEGA_COV = np.load('/Users/snehpandya/Projects/IAEmu/Illustris/measurements/omega_sample1_cov.npy')

N_BINS = len(OMEGA_TARGET)
print(f"  N_bins = {N_BINS}")
print(f"  Target ω(r) range: [{np.min(OMEGA_TARGET):.4f}, {np.max(OMEGA_TARGET):.4f}]")
print(f"  Covariance shape: {OMEGA_COV.shape}")
print(f"  Diagonal std range: [{np.sqrt(np.min(np.diag(OMEGA_COV))):.4e}, {np.sqrt(np.max(np.diag(OMEGA_COV))):.4e}]")

# Add small regularization for numerical stability
OMEGA_COV_REG = OMEGA_COV + 1e-10 * np.eye(N_BINS)
OMEGA_COV_JAX = jnp.array(OMEGA_COV_REG)
OMEGA_TARGET_JAX = jnp.array(OMEGA_TARGET)

# Define r_bins to match the target
r_bins = np.logspace(np.log10(0.1), np.log10(16.0), N_BINS + 1)
R_BINS_TUPLE = tuple(r_bins.tolist())
R_MIDS = np.sqrt(r_bins[:-1] * r_bins[1:])

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


@partial(jax.jit, static_argnums=(5, 6))
def compute_omega_unweighted(
    pos: jnp.ndarray,
    ori: jnp.ndarray,
    weights: jnp.ndarray,
    i_idx: jnp.ndarray,
    j_idx: jnp.ndarray,
    r_bins: tuple,
    box_size: float = 250.0
) -> jnp.ndarray:
    """Unbiased omega with unit weights."""
    r_bins_arr = jnp.array(r_bins)
    n_bins = len(r_bins) - 1
    
    ori_norm = ori / (jnp.linalg.norm(ori, axis=-1, keepdims=True) + 1e-12)
    
    diff = pos[j_idx] - pos[i_idx]
    diff = jnp.where(diff > box_size / 2, diff - box_size, diff)
    diff = jnp.where(diff < -box_size / 2, diff + box_size, diff)
    
    r = jnp.linalg.norm(diff, axis=-1)
    r_hat = diff / (r[:, None] + 1e-12)
    
    cos_theta = jnp.sum(ori_norm[i_idx] * r_hat, axis=-1)
    alignment = cos_theta ** 2
    
    bin_idx = jnp.searchsorted(r_bins_arr, r) - 1
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)
    
    DD = jnp.zeros(n_bins).at[bin_idx].add(1.0)
    DO = jnp.zeros(n_bins).at[bin_idx].add(alignment)
    
    omega = (DO / (DD + 1e-12)) - (1.0 / 3.0)
    return omega


def setup_catalog(params, subcat, seed):
    """Set up a galaxy catalog with given HOD parameters."""
    builder = DiffHalotoolsIA(
        subcat=subcat, params=params,
        do_discrete=True, do_nfw_fallback=True, seed=seed,
        alignment_model="radial", alignment_strength="constant",
        relaxed=False, tau=0.1, Nmax_sat=512, t_rank=0.5,
    )
    
    catalog, meta = builder.return_catalog_with_ids()
    pos_fixed = jnp.array(catalog[:, 0:3])
    n_gal = len(pos_fixed)
    
    sub_halo_id = np.array(meta['sub_halo_id_per_gal'])
    host_idx = np.array(meta['host_idx_per_gal'])
    
    host_pos_np = np.array(builder.host_pos)
    is_central = np.zeros(n_gal, dtype=bool)
    
    for i in range(n_gal):
        if sub_halo_id[i] == -1:
            dist_val = np.linalg.norm(np.array(pos_fixed[i]) - host_pos_np[host_idx[i]])
            if dist_val < 1e-4:
                is_central[i] = True
    
    host_axis_np = np.array(builder.host_axis)
    ref_dirs = np.zeros((n_gal, 3))
    
    for i in range(n_gal):
        if is_central[i]:
            ref_dirs[i] = host_axis_np[host_idx[i]]
        else:
            radial = np.array(pos_fixed[i]) - host_pos_np[host_idx[i]]
            norm = np.linalg.norm(radial)
            if norm > 1e-6:
                ref_dirs[i] = radial / norm
            else:
                ref_dirs[i] = [1, 0, 0]
    
    return {
        'pos': jnp.array(pos_fixed),
        'ref_dirs': jnp.array(ref_dirs),
        'is_central': jnp.array(is_central),
        'n_gal': n_gal,
        'n_cent': int(np.sum(is_central)),
        'n_sat': int(np.sum(~is_central)),
    }


# ============================================================
# Setup inference catalog
# ============================================================
print(f"\nSetting up INFERENCE catalog (seed={SEED_OPT})...")
OPT_CATALOG = setup_catalog(
    np.concatenate([[0.5, 0.3], HOD_PARAMS_FIXED]),
    subcat, SEED_OPT
)
print(f"  N_galaxies = {OPT_CATALOG['n_gal']}")
print(f"  N_centrals = {OPT_CATALOG['n_cent']}")
print(f"  N_satellites = {OPT_CATALOG['n_sat']}")

i_idx_opt, j_idx_opt = build_neighbor_pairs_numpy(
    np.array(OPT_CATALOG['pos']), r_bins[-1], box_size=BOX_SIZE
)
OPT_CATALOG['i_idx'] = jnp.array(i_idx_opt)
OPT_CATALOG['j_idx'] = jnp.array(j_idx_opt)
print(f"  N_pairs = {len(i_idx_opt)}")


# ============================================================
# Forward model functions - IMPROVED
# ============================================================
def compute_omega_at_params_multi_key(mu_cen, mu_sat, base_seed):
    """
    IMPROVED: Compute omega(r) averaged over multiple orientation samples
    using DIFFERENT random keys for variance reduction.
    """
    mu_cen = jnp.clip(mu_cen, -MAX_MU, MAX_MU)
    mu_sat = jnp.clip(mu_sat, -MAX_MU, MAX_MU)
    
    # Use multiple different seeds
    keys = jnp.array([random.PRNGKey(base_seed + i) for i in range(N_SAMPLES_PER_STEP)])
    
    def single_omega(key):
        mu_per_gal = jnp.where(OPT_CATALOG['is_central'], mu_cen, mu_sat)
        ori = sample_watson_orientations(key, OPT_CATALOG['ref_dirs'], mu_per_gal)
        ori = jnp.where(jnp.isnan(ori), OPT_CATALOG['ref_dirs'], ori)
        
        omega = compute_omega_unweighted(
            OPT_CATALOG['pos'], ori, jnp.ones(OPT_CATALOG['n_gal']),
            OPT_CATALOG['i_idx'], OPT_CATALOG['j_idx'], R_BINS_TUPLE, BOX_SIZE
        )
        return jnp.nan_to_num(omega, nan=0.0)
    
    omegas = jax.vmap(single_omega)(keys)
    return jnp.mean(omegas, axis=0)


# ============================================================
# NumPyro model and MCMC - IMPROVED
# ============================================================
def run_mcmc(
    omega_obs, 
    omega_cov, 
    rng_key,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    use_diagonal=False,
):
    """Run MCMC inference with improved forward model."""
    
    # IMPROVED: Use multiple fixed seeds spread across parameter space
    # This reduces bias from using a single random realization
    BASE_SEED = 12345
    
    if use_diagonal:
        omega_std = jnp.sqrt(jnp.diag(omega_cov))
        
        def model():
            mu_cen = numpyro.sample('mu_cen', dist.Uniform(-MAX_MU, MAX_MU))
            mu_sat = numpyro.sample('mu_sat', dist.Uniform(-MAX_MU, MAX_MU))
            omega_pred = compute_omega_at_params_multi_key(mu_cen, mu_sat, BASE_SEED)
            numpyro.sample('omega', dist.Normal(omega_pred, omega_std), obs=omega_obs)
    else:
        def model():
            mu_cen = numpyro.sample('mu_cen', dist.Uniform(-MAX_MU, MAX_MU))
            mu_sat = numpyro.sample('mu_sat', dist.Uniform(-MAX_MU, MAX_MU))
            omega_pred = compute_omega_at_params_multi_key(mu_cen, mu_sat, BASE_SEED)
            numpyro.sample('omega', dist.MultivariateNormal(omega_pred, covariance_matrix=omega_cov), obs=omega_obs)
    
    nuts_kernel = NUTS(
        model,
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
    USE_DIAGONAL = False
    
    print("\n" + "="*80)
    print("Hamiltonian Monte Carlo Inference using ω(r) - IMPROVED")
    print(f"True parameters: μ_cen = {TRUE_MU_CEN:.4f}, μ_sat = {TRUE_MU_SAT:.4f}")
    print(f"N_SAMPLES_PER_STEP = {N_SAMPLES_PER_STEP} (variance reduction)")
    print(f"Covariance: {'Diagonal' if USE_DIAGONAL else 'Full Multivariate Normal'}")
    print(f"MCMC settings: {N_WARMUP} warmup, {N_SAMPLES} samples, {N_CHAINS} chains")
    print("="*80)
    
    # DIAGNOSTIC: Check forward model at known points
    print("\nDiagnostic: Forward model predictions...")
    test_points = [
        (0.80, 0.30, "Test point"),
        (HALOTOOLS_MU_CEN := 0.7925, HALOTOOLS_MU_SAT := 0.2937, "halotools posterior"),
        (TRUE_MU_CEN, TRUE_MU_SAT, "True values"),
    ]
    
    for mu_c, mu_s, label in test_points:
        omega_pred = compute_omega_at_params_multi_key(mu_c, mu_s, 12345)
        chi2 = float((omega_pred - OMEGA_TARGET_JAX) @ jnp.linalg.inv(OMEGA_COV_JAX) @ (omega_pred - OMEGA_TARGET_JAX))
        print(f"  {label}: μ_cen={mu_c:.4f}, μ_sat={mu_s:.4f} → χ² = {chi2:.2f}")
    
    # Run MCMC
    print("\nRunning MCMC...")
    start_time = time.time()
    
    rng_key = random.PRNGKey(0)
    
    mcmc = run_mcmc(
        OMEGA_TARGET_JAX,
        OMEGA_COV_JAX,
        rng_key,
        num_warmup=N_WARMUP,
        num_samples=N_SAMPLES,
        num_chains=N_CHAINS,
        use_diagonal=USE_DIAGONAL,
    )
    
    elapsed = time.time() - start_time
    print(f"\nMCMC completed in {elapsed/60:.1f} minutes")
    
    # Get samples
    samples = mcmc.get_samples()
    mu_cen_samples = np.array(samples['mu_cen'])
    mu_sat_samples = np.array(samples['mu_sat'])
    
    # Print summary
    mcmc.print_summary()
    
    # Compute statistics
    print("\n" + "="*80)
    print("Posterior Summary")
    print("="*80)
    
    mu_cen_mean = np.mean(mu_cen_samples)
    mu_cen_std = np.std(mu_cen_samples)
    mu_sat_mean = np.mean(mu_sat_samples)
    mu_sat_std = np.std(mu_sat_samples)
    
    print(f"\nμ_cen:")
    print(f"  True:       {TRUE_MU_CEN:.4f}")
    print(f"  Mean ± Std: {mu_cen_mean:.4f} ± {mu_cen_std:.4f}")
    print(f"  Bias:       {mu_cen_mean - TRUE_MU_CEN:.4f}")
    
    print(f"\nμ_sat:")
    print(f"  True:       {TRUE_MU_SAT:.4f}")
    print(f"  Mean ± Std: {mu_sat_mean:.4f} ± {mu_sat_std:.4f}")
    print(f"  Bias:       {mu_sat_mean - TRUE_MU_SAT:.4f}")
    
    # Compare to halotools
    print(f"\nComparison to halotools-IA posterior:")
    print(f"  halotools: μ_cen = 0.7925, μ_sat = 0.2937")
    print(f"  diffHOD:   μ_cen = {mu_cen_mean:.4f}, μ_sat = {mu_sat_mean:.4f}")
    print(f"  Δμ_cen = {mu_cen_mean - 0.7925:.4f} ({(mu_cen_mean - 0.7925)/mu_cen_std:.2f}σ)")
    print(f"  Δμ_sat = {mu_sat_mean - 0.2937:.4f} ({(mu_sat_mean - 0.2937)/mu_sat_std:.2f}σ)")
    
    # Compute correlation
    posterior_cov = np.cov(mu_cen_samples, mu_sat_samples)
    correlation = posterior_cov[0, 1] / np.sqrt(posterior_cov[0, 0] * posterior_cov[1, 1])
    print(f"\nPosterior Correlation(μ_cen, μ_sat): {correlation:.4f}")
    
    # Save results
    output_dir = Path('omega_hmc_results')
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'mu_cen_samples': mu_cen_samples,
        'mu_sat_samples': mu_sat_samples,
        'mu_cen_mean': mu_cen_mean,
        'mu_cen_std': mu_cen_std,
        'mu_sat_mean': mu_sat_mean,
        'mu_sat_std': mu_sat_std,
        'posterior_correlation': correlation,
        'posterior_cov': posterior_cov,
        'true_mu_cen': TRUE_MU_CEN,
        'true_mu_sat': TRUE_MU_SAT,
        'halotools_mu_cen': 0.7925,
        'halotools_mu_sat': 0.2937,
        'omega_target': np.array(OMEGA_TARGET),
        'omega_cov': OMEGA_COV,
        'r_mids': R_MIDS,
        'r_bins': r_bins,
        'n_warmup': N_WARMUP,
        'n_samples': N_SAMPLES,
        'n_chains': N_CHAINS,
        'n_samples_per_step': N_SAMPLES_PER_STEP,
        'use_diagonal': USE_DIAGONAL,
        'elapsed_minutes': elapsed / 60,
    }
    
    results_file = output_dir / 'hmc_results.npy'
    np.save(results_file, results, allow_pickle=True)
    print(f"\nResults saved to {results_file}")
    
    print("\n" + "="*80)
    print("HMC inference complete!")
    print("="*80)


if __name__ == "__main__":
    main()