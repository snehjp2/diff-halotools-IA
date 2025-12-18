#!/usr/bin/env python3
"""
Parallel optimization of intrinsic alignment parameters using omega(r).
Runs multiple optimization trajectories in parallel using joblib.
"""

# Suppress XLA warnings
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from joblib import Parallel, delayed
import time
from pathlib import Path
from scipy.spatial import cKDTree

from jax_diffhodIA_weighted import (
    DiffHalotoolsIA,
    sample_watson_orientations,
    _unitize,
)
from diffhodIA_utils import mask_bad_halocat
from halotools.sim_manager import CachedHaloCatalog
from halotools.mock_observables import ed_3d
from jax import random

print(f"JAX devices: {jax.devices()}")

# ============================================================
# Configuration
# ============================================================
TARGET_PARAMS = np.asarray([0.78, 0.33, 12.54, 0.26, 12.68, 13.48, 1.0])
HOD_PARAMS_FIXED = TARGET_PARAMS[2:]

N_PARALLEL_RUNS = 50
N_ITERS = 1000
LEARNING_RATE = 0.01  # Initial learning rate for schedule
MAX_MU = 0.95

# Learning rate schedule settings
LR_DECAY_RATE = 0.65
LR_TRANSITION_STEPS = 100

# Variance reduction settings
N_SAMPLES_PER_STEP = 5  # Average over this many orientation samples per step
MIN_BIN_IDX = 0  # Don't skip any bins
N_TARGET_SAMPLES = 20  # Samples to average for target omega

# Seeds
MASTER_INIT_SEED = 42
SEED_TARGET = 42
SEED_OPT = 999

# ============================================================
# Generate fixed random initializations
# ============================================================
print(f"\nGenerating {N_PARALLEL_RUNS} random initializations...")
np.random.seed(MASTER_INIT_SEED)
FIXED_INITIALIZATIONS = []
for i in range(N_PARALLEL_RUNS):
    mu_c_init = np.random.uniform(-1.0, 1.0)
    mu_s_init = np.random.uniform(-1.0, 1.0)
    FIXED_INITIALIZATIONS.append((mu_c_init, mu_s_init))
    if i < 10 or i >= N_PARALLEL_RUNS - 3:
        print(f"  Init {i+1}: mu_c={mu_c_init:+.3f}, mu_s={mu_s_init:+.3f}")
    elif i == 10:
        print(f"  ...")

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
            dist = np.linalg.norm(np.array(pos_fixed[i]) - host_pos_np[host_idx[i]])
            if dist < 1e-4:
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
print(f"\nSetting up OPTIMIZATION catalog (seed={SEED_OPT})...")
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

# Compute inverse variance weights
print("\nEstimating ω(r) variance for weighting...")
n_var_samples = 20
omega_var_samples = []

for i in range(n_var_samples):
    key = random.PRNGKey(SEED_OPT + 2000 + i)
    mu_per_gal = jnp.where(OPT_CATALOG['is_central'], 0.5, 0.3)
    ori = sample_watson_orientations(key, OPT_CATALOG['ref_dirs'], mu_per_gal)
    
    omega_i = compute_omega_unweighted(
        OPT_CATALOG['pos'], ori, jnp.ones(OPT_CATALOG['n_gal']),
        OPT_CATALOG['i_idx'], OPT_CATALOG['j_idx'], R_BINS_TUPLE, 250.0
    )
    omega_var_samples.append(omega_i)

omega_var_samples = jnp.stack(omega_var_samples)
OMEGA_OPT_STD = jnp.std(omega_var_samples, axis=0)

INV_VAR_WEIGHTS = 1.0 / (OMEGA_OPT_STD**2 + 1e-6)
INV_VAR_WEIGHTS = INV_VAR_WEIGHTS.at[:MIN_BIN_IDX].set(0.0)
INV_VAR_WEIGHTS = INV_VAR_WEIGHTS / (jnp.sum(INV_VAR_WEIGHTS) + 1e-12)

print(f"  ω(r) std range: [{float(jnp.min(OMEGA_OPT_STD)):.4f}, {float(jnp.max(OMEGA_OPT_STD)):.4f}]")


# ============================================================
# Loss function
# ============================================================
def compute_single_omega(mu_cen, mu_sat, key):
    """Compute omega for a single orientation sample."""
    mu_cen = jnp.clip(mu_cen, -MAX_MU, MAX_MU)
    mu_sat = jnp.clip(mu_sat, -MAX_MU, MAX_MU)
    
    mu_per_gal = jnp.where(OPT_CATALOG['is_central'], mu_cen, mu_sat)
    ori = sample_watson_orientations(key, OPT_CATALOG['ref_dirs'], mu_per_gal)
    ori = jnp.where(jnp.isnan(ori), OPT_CATALOG['ref_dirs'], ori)
    
    omega = compute_omega_unweighted(
        OPT_CATALOG['pos'], ori, jnp.ones(OPT_CATALOG['n_gal']),
        OPT_CATALOG['i_idx'], OPT_CATALOG['j_idx'], R_BINS_TUPLE, 250.0
    )
    return jnp.nan_to_num(omega, nan=0.0)


def compute_omega_averaged(mu_cen, mu_sat, key):
    """Average omega over multiple samples."""
    keys = random.split(key, N_SAMPLES_PER_STEP)
    
    def single(k):
        return compute_single_omega(mu_cen, mu_sat, k)
    
    omegas = jax.vmap(single)(keys)
    return jnp.mean(omegas, axis=0)


def loss_fn(params, key):
    """Weighted MSE loss."""
    mu_cen, mu_sat = params[0], params[1]
    omega_pred = compute_omega_averaged(mu_cen, mu_sat, key)
    
    residuals = (omega_pred - OMEGA_TARGET)**2
    loss = jnp.sum(INV_VAR_WEIGHTS * residuals)
    return jnp.nan_to_num(loss, nan=1e6)


# ============================================================
# Single optimization run
# ============================================================
def run_single_optimization(run_id, mu_c_init, mu_s_init, n_iters=N_ITERS, lr=LEARNING_RATE, verbose=False):
    """Run one optimization trajectory with learning rate scheduling."""
    
    params = jnp.array([mu_c_init, mu_s_init], dtype=jnp.float32)
    
    # Learning rate schedule
    schedule = optax.exponential_decay(
        init_value=lr,
        transition_steps=LR_TRANSITION_STEPS,
        decay_rate=LR_DECAY_RATE,
        staircase=True
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.001)
    )
    opt_state = optimizer.init(params)
    
    @jax.jit
    def step(params, opt_state, key):
        loss, grads = jax.value_and_grad(loss_fn)(params, key)
        grads = jnp.nan_to_num(grads, nan=0.0)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, -MAX_MU, MAX_MU)
        return params, opt_state, loss, grads
    
    # Storage
    mu_c_history = [float(mu_c_init)]
    mu_s_history = [float(mu_s_init)]
    loss_history = []
    grad_norm_history = []
    
    # Compute initial loss
    key = random.PRNGKey(run_id * 10000)
    key, subkey = random.split(key)
    init_loss = float(loss_fn(params, subkey))
    loss_history.append(init_loss)
    
    # Track best
    best_loss = init_loss
    best_params = params
    best_mu_c = mu_c_init
    best_mu_s = mu_s_init
    best_iter = 0
    
    start_time = time.time()
    
    for it in range(n_iters):
        key, subkey = random.split(key)
        params, opt_state, loss_val, grads = step(params, opt_state, subkey)
        
        # Check for NaN
        if jnp.any(jnp.isnan(params)):
            if verbose:
                print(f"  [Run {run_id}] NaN at iter {it}, resetting")
            params = jnp.array([mu_c_history[-1], mu_s_history[-1]])
            opt_state = optimizer.init(params)
            continue
        
        # Record history
        mu_c_history.append(float(params[0]))
        mu_s_history.append(float(params[1]))
        loss_history.append(float(loss_val))
        grad_norm_history.append(float(jnp.linalg.norm(grads)))
        
        # Update best
        if float(loss_val) < best_loss:
            best_loss = float(loss_val)
            best_params = params
            best_mu_c = float(params[0])
            best_mu_s = float(params[1])
            best_iter = it + 1
        
        if verbose and it % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  [Run {run_id}] iter {it:03d} [{elapsed:.1f}s]: "
                  f"loss={float(loss_val):.6e}, mu_c={float(params[0]):.4f}, mu_s={float(params[1]):.4f}")
    
    elapsed_time = time.time() - start_time
    
    # Compute final omega using BEST params (not final params)
    key, subkey = random.split(key)
    keys = random.split(subkey, 20)
    
    def single_omega_best(k):
        return compute_single_omega(best_params[0], best_params[1], k)
    
    omega_bests = jax.vmap(single_omega_best)(keys)
    omega_best = np.array(jnp.mean(omega_bests, axis=0))
    omega_best_std = np.array(jnp.std(omega_bests, axis=0))
    
    # Also compute final omega (for comparison)
    def single_omega_final(k):
        return compute_single_omega(params[0], params[1], k)
    
    omega_finals = jax.vmap(single_omega_final)(keys)
    omega_final = np.array(jnp.mean(omega_finals, axis=0))
    omega_final_std = np.array(jnp.std(omega_finals, axis=0))
    
    return {
        'run_id': run_id,
        'mu_c_init': mu_c_init,
        'mu_s_init': mu_s_init,
        'mu_c_final': float(params[0]),
        'mu_s_final': float(params[1]),
        'mu_c_best': best_mu_c,
        'mu_s_best': best_mu_s,
        'loss_init': loss_history[0],
        'loss_final': loss_history[-1],
        'loss_best': best_loss,
        'best_iter': best_iter,
        'mu_c_history': np.array(mu_c_history),
        'mu_s_history': np.array(mu_s_history),
        'loss_history': np.array(loss_history),
        'grad_norm_history': np.array(grad_norm_history),
        'omega_final': omega_final,
        'omega_final_std': omega_final_std,
        'omega_best': omega_best,
        'omega_best_std': omega_best_std,
        'elapsed_time': elapsed_time,
    }


# ============================================================
# Main execution
# ============================================================
def main():
    print("\n" + "="*80)
    print("Starting parallel optimization using ω(r)")
    print(f"Running {N_PARALLEL_RUNS} optimizations in parallel")
    print(f"Target: μ_cen = {TARGET_PARAMS[0]:.4f}, μ_sat = {TARGET_PARAMS[1]:.4f}")
    print(f"Using DIFFERENT catalogs: target seed={SEED_TARGET}, opt seed={SEED_OPT}")
    print(f"Learning rate schedule: init={LEARNING_RATE}, decay={LR_DECAY_RATE} every {LR_TRANSITION_STEPS} steps")
    print("="*80)
    
    start_time = time.time()
    
    # Run optimizations in parallel
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_single_optimization)(
            run_id=i,
            mu_c_init=FIXED_INITIALIZATIONS[i][0],
            mu_s_init=FIXED_INITIALIZATIONS[i][1],
            n_iters=N_ITERS,
            lr=LEARNING_RATE,
            verbose=False
        )
        for i in range(N_PARALLEL_RUNS)
    )
    
    elapsed = time.time() - start_time
    print(f"\nCompleted {N_PARALLEL_RUNS} runs in {elapsed/60:.1f} minutes")
    print(f"Average time per run: {elapsed/N_PARALLEL_RUNS:.1f} seconds")
    
    # Analyze results
    mu_c_finals = [r['mu_c_final'] for r in results]
    mu_s_finals = [r['mu_s_final'] for r in results]
    mu_c_bests = [r['mu_c_best'] for r in results]
    mu_s_bests = [r['mu_s_best'] for r in results]
    loss_finals = [r['loss_final'] for r in results]
    loss_bests = [r['loss_best'] for r in results]
    
    # Find overall best
    best_idx = np.argmin(loss_bests)
    best_result = results[best_idx]
    
    print(f"\n{'='*80}")
    print("Results Summary")
    print(f"{'='*80}")
    print(f"\nTarget:     μ_cen = {TARGET_PARAMS[0]:.4f}, μ_sat = {TARGET_PARAMS[1]:.4f}")
    print(f"\nFinal values (mean ± std across {N_PARALLEL_RUNS} runs):")
    print(f"  μ_cen = {np.mean(mu_c_finals):.4f} ± {np.std(mu_c_finals):.4f}")
    print(f"  μ_sat = {np.mean(mu_s_finals):.4f} ± {np.std(mu_s_finals):.4f}")
    print(f"  loss  = {np.mean(loss_finals):.4e} ± {np.std(loss_finals):.4e}")
    
    print(f"\nBest values (mean ± std across {N_PARALLEL_RUNS} runs):")
    print(f"  μ_cen = {np.mean(mu_c_bests):.4f} ± {np.std(mu_c_bests):.4f}")
    print(f"  μ_sat = {np.mean(mu_s_bests):.4f} ± {np.std(mu_s_bests):.4f}")
    print(f"  loss  = {np.mean(loss_bests):.4e} ± {np.std(loss_bests):.4e}")
    
    print(f"\nOverall best run (Run {best_idx}):")
    print(f"  μ_cen = {best_result['mu_c_best']:.4f} (target: {TARGET_PARAMS[0]:.4f}, error: {abs(best_result['mu_c_best'] - TARGET_PARAMS[0]):.4f})")
    print(f"  μ_sat = {best_result['mu_s_best']:.4f} (target: {TARGET_PARAMS[1]:.4f}, error: {abs(best_result['mu_s_best'] - TARGET_PARAMS[1]):.4f})")
    print(f"  loss  = {best_result['loss_best']:.4e}")
    print(f"  best_iter = {best_result['best_iter']}")
    
    print(f"\n{'='*80}")
    print("Individual runs (sorted by best loss):")
    print(f"{'='*80}")
    print(f"{'Run':>4} {'Init (μc, μs)':>20} {'Best (μc, μs)':>20} {'Loss':>12} {'Iter':>6} {'Time':>8}")
    print("-" * 80)
    
    sorted_results = sorted(results, key=lambda x: x['loss_best'])
    for r in sorted_results[:20]:  # Show top 20
        print(
            f"{r['run_id']:4d} "
            f"({r['mu_c_init']:+.3f}, {r['mu_s_init']:+.3f})  ->  "
            f"({r['mu_c_best']:+.3f}, {r['mu_s_best']:+.3f})  "
            f"{r['loss_best']:12.4e} "
            f"{r['best_iter']:6d} "
            f"{r['elapsed_time']:7.1f}s"
        )
    
    if len(sorted_results) > 20:
        print(f"  ... ({len(sorted_results) - 20} more runs)")
    
    # Save results
    output_dir = Path('omega_optimization_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save all results
    results_file = output_dir / 'all_results.npy'
    np.save(results_file, results, allow_pickle=True)
    print(f"\nResults saved to {results_file}")
    
    # Save summary
    summary = {
        'target_params': TARGET_PARAMS,
        'target_mu_cen': TARGET_PARAMS[0],
        'target_mu_sat': TARGET_PARAMS[1],
        'n_runs': N_PARALLEL_RUNS,
        'n_iters': N_ITERS,
        'learning_rate': LEARNING_RATE,
        'lr_decay_rate': LR_DECAY_RATE,
        'lr_transition_steps': LR_TRANSITION_STEPS,
        'seed_target': SEED_TARGET,
        'seed_opt': SEED_OPT,
        'n_samples_per_step': N_SAMPLES_PER_STEP,
        'min_bin_idx': MIN_BIN_IDX,
        'omega_target': np.array(OMEGA_TARGET),
        'omega_target_std': np.array(OMEGA_TARGET_STD),
        'inv_var_weights': np.array(INV_VAR_WEIGHTS),
        'r_bins': np.array(r_bins),
        'r_mids': R_MIDS,
        'mu_c_finals_mean': np.mean(mu_c_finals),
        'mu_c_finals_std': np.std(mu_c_finals),
        'mu_s_finals_mean': np.mean(mu_s_finals),
        'mu_s_finals_std': np.std(mu_s_finals),
        'mu_c_bests_mean': np.mean(mu_c_bests),
        'mu_c_bests_std': np.std(mu_c_bests),
        'mu_s_bests_mean': np.mean(mu_s_bests),
        'mu_s_bests_std': np.std(mu_s_bests),
        'loss_finals_mean': np.mean(loss_finals),
        'loss_finals_std': np.std(loss_finals),
        'loss_bests_mean': np.mean(loss_bests),
        'loss_bests_std': np.std(loss_bests),
        'best_run_id': best_idx,
        'best_mu_c': best_result['mu_c_best'],
        'best_mu_s': best_result['mu_s_best'],
        'best_loss': best_result['loss_best'],
        'best_iter': best_result['best_iter'],
        'total_time_minutes': elapsed / 60,
    }
    summary_file = output_dir / 'summary.npy'
    np.save(summary_file, summary, allow_pickle=True)
    print(f"Summary saved to {summary_file}")
    
    # Save best result's omega for plotting
    best_omega_file = output_dir / 'best_omega.npy'
    np.save(best_omega_file, {
        'omega_best': best_result['omega_best'],
        'omega_best_std': best_result['omega_best_std'],
        'omega_final': best_result['omega_final'],
        'omega_final_std': best_result['omega_final_std'],
        'omega_target': np.array(OMEGA_TARGET),
        'omega_target_std': np.array(OMEGA_TARGET_STD),
        'r_mids': R_MIDS,
        'mu_c_best': best_result['mu_c_best'],
        'mu_s_best': best_result['mu_s_best'],
        'mu_c_final': best_result['mu_c_final'],
        'mu_s_final': best_result['mu_s_final'],
    }, allow_pickle=True)
    print(f"Best omega saved to {best_omega_file}")
    
    print("\n" + "="*80)
    print("All optimizations complete!")
    print("="*80)


if __name__ == "__main__":
    main()