#!/usr/bin/env python3
"""
Hamiltonian Monte Carlo inference of intrinsic alignment parameters using omega(r).
Uses NumPyro's NUTS sampler.
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
from numpyro import handlers
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
TARGET_PARAMS = np.asarray([0.78, 0.33, 12.54, 0.26, 12.68, 13.48, 1.0])
HOD_PARAMS_FIXED = TARGET_PARAMS[2:]

MAX_MU = 0.95

# Variance reduction settings
N_SAMPLES_PER_STEP = 5  # Average over this many orientation samples per likelihood eval
N_TARGET_SAMPLES = 20  # Samples to average for target omega

# Seeds
SEED_TARGET = 42
SEED_OPT = 999

# MCMC settings
N_WARMUP = 500
N_SAMPLES = 1000
N_CHAINS = 4

# ============================================================
# Load halo catalog (do once globally)
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
# Setup target and optimization catalogs (do once globally)
# ============================================================
print("\nSetting up catalogs...")

r_bins = np.logspace(np.log10(0.1), np.log10(16.0), 20)
R_BINS_TUPLE = tuple(r_bins.tolist())
R_MIDS = np.sqrt(r_bins[:-1] * r_bins[1:])
N_BINS = len(r_bins) - 1

# Target catalog
print(f"\nSetting up TARGET catalog (seed={SEED_TARGET})...")
TARGET_CATALOG = setup_catalog(TARGET_PARAMS, subcat, SEED_TARGET)
print(f"  N_galaxies = {TARGET_CATALOG['n_gal']}")
print(f"  N_centrals = {TARGET_CATALOG['n_cent']}")
print(f"  N_satellites = {TARGET_CATALOG['n_sat']}")

i_idx_target, j_idx_target = build_neighbor_pairs_numpy(
    np.array(TARGET_CATALOG['pos']), r_bins[-1], box_size=250.0
)
TARGET_CATALOG['i_idx'] = jnp.array(i_idx_target)
TARGET_CATALOG['j_idx'] = jnp.array(j_idx_target)
print(f"  N_pairs = {len(i_idx_target)}")

# Compute target omega (averaged)
print(f"\nComputing target ω(r) (averaged over {N_TARGET_SAMPLES} samples)...")
target_mu_cen, target_mu_sat = TARGET_PARAMS[0], TARGET_PARAMS[1]

omega_target_samples = []
for i in range(N_TARGET_SAMPLES):
    key = random.PRNGKey(SEED_TARGET + 1000 + i)
    mu_per_gal = jnp.where(TARGET_CATALOG['is_central'], target_mu_cen, target_mu_sat)
    ori = sample_watson_orientations(key, TARGET_CATALOG['ref_dirs'], mu_per_gal)
    
    omega_i = compute_omega_unweighted(
        TARGET_CATALOG['pos'], ori, jnp.ones(TARGET_CATALOG['n_gal']),
        TARGET_CATALOG['i_idx'], TARGET_CATALOG['j_idx'], R_BINS_TUPLE, 250.0
    )
    omega_target_samples.append(omega_i)

omega_target_samples = jnp.stack(omega_target_samples)
OMEGA_TARGET = jnp.mean(omega_target_samples, axis=0)
OMEGA_TARGET_STD = jnp.std(omega_target_samples, axis=0)

print(f"  Target ω(r) range: [{float(jnp.min(OMEGA_TARGET)):.4f}, {float(jnp.max(OMEGA_TARGET)):.4f}]")
print(f"  Target μ_cen = {target_mu_cen}, μ_sat = {target_mu_sat}")

# Optimization catalog (DIFFERENT seed)
print(f"\nSetting up INFERENCE catalog (seed={SEED_OPT})...")
OPT_CATALOG = setup_catalog(
    np.concatenate([TARGET_PARAMS[:2], HOD_PARAMS_FIXED]), 
    subcat, SEED_OPT
)
print(f"  N_galaxies = {OPT_CATALOG['n_gal']}")
print(f"  N_centrals = {OPT_CATALOG['n_cent']}")
print(f"  N_satellites = {OPT_CATALOG['n_sat']}")

i_idx_opt, j_idx_opt = build_neighbor_pairs_numpy(
    np.array(OPT_CATALOG['pos']), r_bins[-1], box_size=250.0
)
OPT_CATALOG['i_idx'] = jnp.array(i_idx_opt)
OPT_CATALOG['j_idx'] = jnp.array(j_idx_opt)
print(f"  N_pairs = {len(i_idx_opt)}")

# Compute covariance matrix for likelihood
print("\nEstimating ω(r) covariance matrix...")
n_cov_samples = 100
omega_cov_samples = []

for i in range(n_cov_samples):
    key = random.PRNGKey(SEED_OPT + 3000 + i)
    mu_per_gal = jnp.where(OPT_CATALOG['is_central'], target_mu_cen, target_mu_sat)
    ori = sample_watson_orientations(key, OPT_CATALOG['ref_dirs'], mu_per_gal)
    
    omega_i = compute_omega_unweighted(
        OPT_CATALOG['pos'], ori, jnp.ones(OPT_CATALOG['n_gal']),
        OPT_CATALOG['i_idx'], OPT_CATALOG['j_idx'], R_BINS_TUPLE, 250.0
    )
    omega_cov_samples.append(np.array(omega_i))

omega_cov_samples = np.array(omega_cov_samples)
OMEGA_COV = np.cov(omega_cov_samples.T)

# Add small regularization for numerical stability
OMEGA_COV_REG = OMEGA_COV + 1e-10 * np.eye(N_BINS)
OMEGA_PRECISION = jnp.array(np.linalg.inv(OMEGA_COV_REG))
OMEGA_COV_JAX = jnp.array(OMEGA_COV_REG)

print(f"  Covariance matrix shape: {OMEGA_COV.shape}")
print(f"  Diagonal std range: [{np.sqrt(np.min(np.diag(OMEGA_COV))):.4e}, {np.sqrt(np.max(np.diag(OMEGA_COV))):.4e}]")


# ============================================================
# Forward model functions (deterministic given key)
# ============================================================
def compute_omega_at_params(mu_cen, mu_sat, rng_key):
    """
    Compute omega(r) averaged over multiple orientation samples.
    This is deterministic given the rng_key.
    """
    mu_cen = jnp.clip(mu_cen, -MAX_MU, MAX_MU)
    mu_sat = jnp.clip(mu_sat, -MAX_MU, MAX_MU)
    
    keys = random.split(rng_key, N_SAMPLES_PER_STEP)
    
    def single_omega(key):
        mu_per_gal = jnp.where(OPT_CATALOG['is_central'], mu_cen, mu_sat)
        ori = sample_watson_orientations(key, OPT_CATALOG['ref_dirs'], mu_per_gal)
        ori = jnp.where(jnp.isnan(ori), OPT_CATALOG['ref_dirs'], ori)
        
        omega = compute_omega_unweighted(
            OPT_CATALOG['pos'], ori, jnp.ones(OPT_CATALOG['n_gal']),
            OPT_CATALOG['i_idx'], OPT_CATALOG['j_idx'], R_BINS_TUPLE, 250.0
        )
        return jnp.nan_to_num(omega, nan=0.0)
    
    omegas = jax.vmap(single_omega)(keys)
    return jnp.mean(omegas, axis=0)


# ============================================================
# NumPyro model
# ============================================================
def model(omega_obs, omega_cov, rng_key):
    """
    NumPyro model for IA parameter inference.
    
    Args:
        omega_obs: Observed omega(r) values [n_bins]
        omega_cov: Covariance matrix of omega(r) [n_bins, n_bins]
        rng_key: JAX random key for forward model stochasticity
    """
    # Priors: uniform on [-0.95, 0.95]
    mu_cen = numpyro.sample('mu_cen', dist.Uniform(-MAX_MU, MAX_MU))
    mu_sat = numpyro.sample('mu_sat', dist.Uniform(-MAX_MU, MAX_MU))
    
    # Forward model: compute predicted omega(r)
    omega_pred = compute_omega_at_params(mu_cen, mu_sat, rng_key)
    
    # Likelihood: multivariate normal
    numpyro.sample('omega', dist.MultivariateNormal(omega_pred, covariance_matrix=omega_cov), obs=omega_obs)


def model_diagonal(omega_obs, omega_std, rng_key):
    """
    Simplified model with diagonal covariance (independent bins).
    Faster but ignores bin correlations.
    """
    # Priors: uniform on [-0.95, 0.95]
    mu_cen = numpyro.sample('mu_cen', dist.Uniform(-MAX_MU, MAX_MU))
    mu_sat = numpyro.sample('mu_sat', dist.Uniform(-MAX_MU, MAX_MU))
    
    # Forward model
    omega_pred = compute_omega_at_params(mu_cen, mu_sat, rng_key)
    
    # Likelihood: independent normal per bin
    numpyro.sample('omega', dist.Normal(omega_pred, omega_std), obs=omega_obs)


# ============================================================
# Custom MCMC runner that handles stochastic forward model
# ============================================================
def run_mcmc_with_stochastic_model(
    omega_obs, 
    omega_cov, 
    rng_key,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    use_diagonal=False
):
    """
    Run MCMC with a stochastic forward model.
    Uses a fixed random key per MCMC iteration for the forward model.
    """
    
    if use_diagonal:
        omega_std = jnp.sqrt(jnp.diag(omega_cov))
        
        def conditioned_model(rng_key):
            return model_diagonal(omega_obs, omega_std, rng_key)
    else:
        def conditioned_model(rng_key):
            return model(omega_obs, omega_cov, rng_key)
    
    # For HMC, we need the forward model to be deterministic during gradient computation.
    # We fix the random key for the forward model and change it between MCMC steps.
    # One approach: use a fixed key (introduces bias but allows HMC to work)
    
    fixed_key = random.PRNGKey(12345)
    
    if use_diagonal:
        omega_std = jnp.sqrt(jnp.diag(omega_cov))
        
        def deterministic_model():
            mu_cen = numpyro.sample('mu_cen', dist.Uniform(-MAX_MU, MAX_MU))
            mu_sat = numpyro.sample('mu_sat', dist.Uniform(-MAX_MU, MAX_MU))
            omega_pred = compute_omega_at_params(mu_cen, mu_sat, fixed_key)
            numpyro.sample('omega', dist.Normal(omega_pred, omega_std), obs=omega_obs)
    else:
        def deterministic_model():
            mu_cen = numpyro.sample('mu_cen', dist.Uniform(-MAX_MU, MAX_MU))
            mu_sat = numpyro.sample('mu_sat', dist.Uniform(-MAX_MU, MAX_MU))
            omega_pred = compute_omega_at_params(mu_cen, mu_sat, fixed_key)
            numpyro.sample('omega', dist.MultivariateNormal(omega_pred, covariance_matrix=omega_cov), obs=omega_obs)
    
    nuts_kernel = NUTS(
        deterministic_model,
        target_accept_prob=0.8,
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
    print("Hamiltonian Monte Carlo Inference using ω(r)")
    print(f"Target: μ_cen = {TARGET_PARAMS[0]:.4f}, μ_sat = {TARGET_PARAMS[1]:.4f}")
    print(f"Using DIFFERENT catalogs: target seed={SEED_TARGET}, inference seed={SEED_OPT}")
    print(f"MCMC settings: {N_WARMUP} warmup, {N_SAMPLES} samples, {N_CHAINS} chains")
    print("="*80)
    
    # Use diagonal covariance for speed (can switch to full covariance)
    USE_DIAGONAL = False
    
    if USE_DIAGONAL:
        print("\nUsing diagonal covariance (independent bins) for likelihood")
    else:
        print("\nUsing full covariance matrix for likelihood")
    
    # Run MCMC
    print("\nRunning MCMC...")
    start_time = time.time()
    
    rng_key = random.PRNGKey(0)
    
    mcmc = run_mcmc_with_stochastic_model(
        OMEGA_TARGET,
        OMEGA_COV_JAX,
        rng_key,
        num_warmup=N_WARMUP,
        num_samples=N_SAMPLES,
        num_chains=N_CHAINS,
        use_diagonal=USE_DIAGONAL
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
    mu_cen_median = np.median(mu_cen_samples)
    mu_cen_q16 = np.percentile(mu_cen_samples, 16)
    mu_cen_q84 = np.percentile(mu_cen_samples, 84)
    
    mu_sat_mean = np.mean(mu_sat_samples)
    mu_sat_std = np.std(mu_sat_samples)
    mu_sat_median = np.median(mu_sat_samples)
    mu_sat_q16 = np.percentile(mu_sat_samples, 16)
    mu_sat_q84 = np.percentile(mu_sat_samples, 84)
    
    print(f"\nμ_cen:")
    print(f"  Target:     {TARGET_PARAMS[0]:.4f}")
    print(f"  Mean ± Std: {mu_cen_mean:.4f} ± {mu_cen_std:.4f}")
    print(f"  Median:     {mu_cen_median:.4f}")
    print(f"  68% CI:     [{mu_cen_q16:.4f}, {mu_cen_q84:.4f}]")
    print(f"  Error:      {abs(mu_cen_mean - TARGET_PARAMS[0]):.4f}")
    
    print(f"\nμ_sat:")
    print(f"  Target:     {TARGET_PARAMS[1]:.4f}")
    print(f"  Mean ± Std: {mu_sat_mean:.4f} ± {mu_sat_std:.4f}")
    print(f"  Median:     {mu_sat_median:.4f}")
    print(f"  68% CI:     [{mu_sat_q16:.4f}, {mu_sat_q84:.4f}]")
    print(f"  Error:      {abs(mu_sat_mean - TARGET_PARAMS[1]):.4f}")
    
    # Compute correlation
    cov_matrix = np.cov(mu_cen_samples, mu_sat_samples)
    correlation = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])
    print(f"\nCorrelation(μ_cen, μ_sat): {correlation:.4f}")
    
    # Save results
    output_dir = Path('omega_hmc_results')
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'mu_cen_samples': mu_cen_samples,
        'mu_sat_samples': mu_sat_samples,
        'mu_cen_mean': mu_cen_mean,
        'mu_cen_std': mu_cen_std,
        'mu_cen_median': mu_cen_median,
        'mu_cen_q16': mu_cen_q16,
        'mu_cen_q84': mu_cen_q84,
        'mu_sat_mean': mu_sat_mean,
        'mu_sat_std': mu_sat_std,
        'mu_sat_median': mu_sat_median,
        'mu_sat_q16': mu_sat_q16,
        'mu_sat_q84': mu_sat_q84,
        'correlation': correlation,
        'cov_matrix': cov_matrix,
        'target_mu_cen': TARGET_PARAMS[0],
        'target_mu_sat': TARGET_PARAMS[1],
        'omega_target': np.array(OMEGA_TARGET),
        'omega_target_std': np.array(OMEGA_TARGET_STD),
        'omega_cov': OMEGA_COV,
        'r_mids': R_MIDS,
        'n_warmup': N_WARMUP,
        'n_samples': N_SAMPLES,
        'n_chains': N_CHAINS,
        'elapsed_minutes': elapsed / 60,
    }
    
    results_file = output_dir / 'hmc_results.npy'
    np.save(results_file, results, allow_pickle=True)
    print(f"\nResults saved to {results_file}")
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        # Try to import corner, but don't fail if not available
        try:
            import corner
            HAS_CORNER = True
        except ImportError:
            HAS_CORNER = False
            print("Note: Install corner for corner plots (pip install corner)")
        
        if HAS_CORNER:
            # Corner plot
            fig = corner.corner(
                np.column_stack([mu_cen_samples, mu_sat_samples]),
                labels=[r'$\mu_{\rm cen}$', r'$\mu_{\rm sat}$'],
                truths=[TARGET_PARAMS[0], TARGET_PARAMS[1]],
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_fmt='.4f',
            )
            
            corner_file = output_dir / 'corner_plot.png'
            fig.savefig(corner_file, dpi=150, bbox_inches='tight')
            print(f"Corner plot saved to {corner_file}")
            plt.close()
        
        # Trace plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        n_total_samples = len(mu_cen_samples)
        samples_per_chain = n_total_samples // N_CHAINS
        
        # mu_cen trace
        ax = axes[0, 0]
        for chain in range(N_CHAINS):
            start_idx = chain * samples_per_chain
            end_idx = start_idx + samples_per_chain
            if end_idx > n_total_samples:
                end_idx = n_total_samples
            chain_samples = mu_cen_samples[start_idx:end_idx]
            ax.plot(chain_samples, alpha=0.7, label=f'Chain {chain}')
        ax.axhline(TARGET_PARAMS[0], color='r', linestyle='--', label='Target')
        ax.set_xlabel('Sample')
        ax.set_ylabel(r'$\mu_{\rm cen}$')
        ax.set_title(r'$\mu_{\rm cen}$ Trace')
        ax.legend(fontsize=8)
        
        # mu_sat trace
        ax = axes[0, 1]
        for chain in range(N_CHAINS):
            start_idx = chain * samples_per_chain
            end_idx = start_idx + samples_per_chain
            if end_idx > n_total_samples:
                end_idx = n_total_samples
            chain_samples = mu_sat_samples[start_idx:end_idx]
            ax.plot(chain_samples, alpha=0.7, label=f'Chain {chain}')
        ax.axhline(TARGET_PARAMS[1], color='r', linestyle='--', label='Target')
        ax.set_xlabel('Sample')
        ax.set_ylabel(r'$\mu_{\rm sat}$')
        ax.set_title(r'$\mu_{\rm sat}$ Trace')
        ax.legend(fontsize=8)
        
        # mu_cen histogram
        ax = axes[1, 0]
        ax.hist(mu_cen_samples, bins=50, density=True, alpha=0.7)
        ax.axvline(TARGET_PARAMS[0], color='r', linestyle='--', lw=2, label='Target')
        ax.axvline(mu_cen_mean, color='k', linestyle='-', lw=2, label='Mean')
        ax.set_xlabel(r'$\mu_{\rm cen}$')
        ax.set_ylabel('Density')
        ax.legend()
        
        # mu_sat histogram
        ax = axes[1, 1]
        ax.hist(mu_sat_samples, bins=50, density=True, alpha=0.7)
        ax.axvline(TARGET_PARAMS[1], color='r', linestyle='--', lw=2, label='Target')
        ax.axvline(mu_sat_mean, color='k', linestyle='-', lw=2, label='Mean')
        ax.set_xlabel(r'$\mu_{\rm sat}$')
        ax.set_ylabel('Density')
        ax.legend()
        
        plt.tight_layout()
        trace_file = output_dir / 'trace_plots.png'
        fig.savefig(trace_file, dpi=150, bbox_inches='tight')
        print(f"Trace plots saved to {trace_file}")
        plt.close()
        
    except ImportError:
        print("\nNote: Install matplotlib for plotting (pip install matplotlib)")
    
    print("\n" + "="*80)
    print("HMC inference complete!")
    print("="*80)


if __name__ == "__main__":
    main()