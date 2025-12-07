#!/usr/bin/env python3
"""
Parallel optimization of intrinsic alignment parameters.
Runs multiple optimization trajectories in parallel using joblib.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from joblib import Parallel, delayed
import time
from pathlib import Path

# Your existing imports
from jax_diffhodIA import DiffHalotoolsIA
from diffhodIA_utils import mask_bad_halocat
from halotools.sim_manager import CachedHaloCatalog

print(f"JAX devices: {jax.devices()}")

# ============================================================
# Configuration
# ============================================================
TARGET_PARAMS = np.asarray([0.78, 0.33, 12.02, 0.26, 11.38, 13.31, 1.06])
BASE_PARAMS = jnp.array([0.0, 0.0, 12.02, 0.26, 11.38, 13.31, 1.0], dtype=jnp.float32)

N_PARALLEL_RUNS = 50  # Number of parallel optimizations
N_ITERS = 100
LEARNING_RATE = 5e-2
MAX_MU = 0.95

# Fixed random seed for reproducible initializations
MASTER_INIT_SEED = 42

## reorder seed strategies for 5 2 1
SEED_STRATEGIES = {
    '1_seed': [3],
    '3_seeds': [3, 34, 345],
}


# Loss weights
W_C_MEAN = 1.0
W_C_VAR = 0.5
W_S_MEAN = 1.0
W_S_VAR = 0.5
CENTRAL_BOOST = 2.0

# ============================================================
# Generate fixed random initializations (same for all strategies)
# ============================================================
print(f"\nGenerating {N_PARALLEL_RUNS} random initializations...")
np.random.seed(MASTER_INIT_SEED)
FIXED_INITIALIZATIONS = []
for i in range(N_PARALLEL_RUNS):
    mu_c_init = np.random.uniform(-0.95, 0.95)
    mu_s_init = np.random.uniform(-0.95, 0.95)
    FIXED_INITIALIZATIONS.append((mu_c_init, mu_s_init))
    print(f"  Init {i+1}: mu_c={mu_c_init:+.3f}, mu_s={mu_s_init:+.3f}")

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
@jax.jit
def mu_from_theta(theta_raw, max_mu=MAX_MU):
    """Map unconstrained theta to mu"""
    return max_mu * jnp.tanh(theta_raw)

@jax.jit
def theta_from_mu(mu, max_mu=MAX_MU, eps=1e-6):
    """Inverse: map mu to theta"""
    ratio = jnp.clip(mu / max_mu, -1.0 + eps, 1.0 - eps)
    return jnp.arctanh(ratio)

@jax.jit
def alignment_stats_from_catalog_jit(catalog, meta_host_idx, host_pos, host_axis):
    """JIT-able version using masks instead of boolean indexing"""
    pos = catalog[:, 0:3]
    n = catalog[:, 3:6]
    n = n / (jnp.linalg.norm(n, axis=1, keepdims=True) + 1e-10)

    host_pos_per_gal = host_pos[meta_host_idx]
    host_axis_per_gal = host_axis[meta_host_idx]

    disp = pos - host_pos_per_gal
    r = jnp.linalg.norm(disp, axis=1)

    is_central = r < 1e-6
    r_hat = disp / (r[:, None] + 1e-10)
    u = jnp.where(is_central[:, None], host_axis_per_gal, r_hat)
    u = u / (jnp.linalg.norm(u, axis=1, keepdims=True) + 1e-10)

    t = jnp.sum(n * u, axis=1)
    t2 = t * t

    # Masked operations
    cent_mask = is_central.astype(jnp.float32)
    sat_mask = (~is_central).astype(jnp.float32)
    
    n_c = jnp.sum(cent_mask)
    t2_c_mean = jnp.sum(t2 * cent_mask) / (n_c + 1e-10)
    t2_c_sq_mean = jnp.sum((t2 ** 2) * cent_mask) / (n_c + 1e-10)
    t2_c_var = t2_c_sq_mean - (t2_c_mean ** 2)

    n_s = jnp.sum(sat_mask)
    t2_s_mean = jnp.sum(t2 * sat_mask) / (n_s + 1e-10)
    t2_s_sq_mean = jnp.sum((t2 ** 2) * sat_mask) / (n_s + 1e-10)
    t2_s_var = t2_s_sq_mean - (t2_s_mean ** 2)

    return jnp.stack([t2_c_mean, t2_c_var, t2_s_mean, t2_s_var])

def alignment_stats_from_catalog(catalog, meta_host_idx, host_pos, host_axis):
    """Non-JIT version with boolean indexing"""
    pos = catalog[:, 0:3]
    n = catalog[:, 3:6]
    n = n / (jnp.linalg.norm(n, axis=1, keepdims=True) + 1e-10)

    host_pos_per_gal = host_pos[meta_host_idx]
    host_axis_per_gal = host_axis[meta_host_idx]

    disp = pos - host_pos_per_gal
    r = jnp.linalg.norm(disp, axis=1)
    is_central = r < 1e-6

    r_hat = disp / (r[:, None] + 1e-10)
    u = jnp.where(is_central[:, None], host_axis_per_gal, r_hat)
    u = u / (jnp.linalg.norm(u, axis=1, keepdims=True) + 1e-10)

    t = jnp.sum(n * u, axis=1)
    t2 = t * t

    t2_c = t2[is_central]
    t2_s = t2[~is_central]

    return jnp.stack([
        t2_c.mean(), ((t2_c - t2_c.mean()) ** 2).mean(),
        t2_s.mean(), ((t2_s - t2_s.mean()) ** 2).mean()
    ])

# ============================================================
# Build target catalog (do once globally)
# ============================================================
print("\nBuilding target catalog...")
data_builder = DiffHalotoolsIA(
    subcat=subcat,
    params=TARGET_PARAMS,
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

data_catalog, data_meta = data_builder.return_catalog_with_ids()
data_stats_vec = alignment_stats_from_catalog(
    jnp.asarray(data_catalog),
    jnp.asarray(data_meta["host_idx_per_gal"]),
    data_builder.host_pos,
    data_builder.host_axis
)

print("Target catalog statistics:")
print(f"  Centrals:  mean={float(data_stats_vec[0]):.4f}, var={float(data_stats_vec[1]):.4f}")
print(f"  Satellites: mean={float(data_stats_vec[2]):.4f}, var={float(data_stats_vec[3]):.4f}")

# ============================================================
# Simulator
# ============================================================
def simulator_alignment_stats(params_7d, seed):
    """Generate catalog and compute stats - NOT JIT-able"""
    builder = DiffHalotoolsIA(
        subcat=subcat,
        params=params_7d,
        do_discrete=True,
        do_nfw_fallback=True,
        seed=int(seed),
        alignment_model="radial",
        alignment_strength="constant",
        relaxed=True,
        tau=0.1,
        Nmax_sat=256,
        t_rank=0.5,
    )
    catalog, meta = builder.return_catalog_with_ids()
    return alignment_stats_from_catalog_jit(
        jnp.asarray(catalog),
        jnp.asarray(meta["host_idx_per_gal"]),
        builder.host_pos,
        builder.host_axis
    )

def loss_simple(theta_raw_2d, seeds):
    """Simple loss without caching - regenerates every time"""
    theta_c, theta_s = theta_raw_2d
    mu_c = mu_from_theta(theta_c)
    mu_s = mu_from_theta(theta_s)
    
    params_sim = BASE_PARAMS.at[0].set(mu_c).at[1].set(mu_s)
    
    total_loss = 0.0
    for seed in seeds:
        sim_stats = simulator_alignment_stats(params_sim, seed)
        
        diff = sim_stats - data_stats_vec
        # [c_mean, c_var, s_mean, s_var]
        loss_c = (W_C_MEAN * diff[0]**2) + (W_C_VAR * diff[1]**2)
        loss_s = (W_S_MEAN * diff[2]**2) + (W_S_VAR * diff[3]**2)
        total_loss += CENTRAL_BOOST * loss_c + loss_s
    
    return total_loss / len(seeds)

# ============================================================
# Single optimization run (to be parallelized)
# ============================================================
def run_single_optimization(run_id, mu_c_init, mu_s_init, seed_pool, n_iters=N_ITERS, lr=LEARNING_RATE, verbose=False):
    """
    Run one optimization trajectory.
    
    Args:
        run_id: unique identifier for this run
        mu_c_init: initial value for mu_c
        mu_s_init: initial value for mu_s
        seed_pool: list of HOD seeds to average over
        n_iters: number of optimization steps
        lr: learning rate
        verbose: whether to print progress
    
    Returns:
        dict with results
    """
    theta0 = jnp.array([
        theta_from_mu(mu_c_init),
        theta_from_mu(mu_s_init)
    ], dtype=jnp.float32)
    
    schedule = optax.exponential_decay(
        init_value=lr,
        transition_steps=20,
        decay_rate=0.95
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),  # Clip gradients with norm > 1.0
        optax.adam(learning_rate=schedule)
    )
    opt_state = optimizer.init(theta0)
    
    def step(theta, opt_state):
        loss, grads = jax.value_and_grad(loss_simple)(theta, seed_pool)
        
        # Check for NaN/Inf
        if jnp.isnan(loss) or jnp.isinf(loss):
            print(f"  [Run {run_id}] WARNING: NaN/Inf loss detected, stopping")
            return theta, opt_state, loss, grads
        
        if jnp.any(jnp.isnan(grads)) or jnp.any(jnp.isinf(grads)):
            print(f"  [Run {run_id}] WARNING: NaN/Inf gradients detected, stopping")
            return theta, opt_state, loss, grads
        
        updates, opt_state = optimizer.update(grads, opt_state, theta)
        theta_new = optax.apply_updates(theta, updates)
        return theta_new, opt_state, loss, grads
    
    # Storage
    theta_history = [np.asarray(theta0)]
    loss_history = [float(loss_simple(theta0, seed_pool))]
    grad_norm_history = []
    
    # Track best loss
    best_loss = loss_history[0]
    best_theta = theta0
    best_iter = 0
    
    theta = theta0
    start_time = time.time()
    
    for it in range(n_iters):
        theta, opt_state, loss_val, grads = step(theta, opt_state)
        
        theta_history.append(np.asarray(theta))
        loss_history.append(float(loss_val))
        grad_norm_history.append(float(jnp.linalg.norm(grads)))
        
        # Update best if current loss is lower
        if float(loss_val) < best_loss:
            best_loss = float(loss_val)
            best_theta = theta
            best_iter = it + 1
        
        if verbose and it % 10 == 0:
            elapsed = time.time() - start_time
            mu_c = float(mu_from_theta(theta[0]))
            mu_s = float(mu_from_theta(theta[1]))
            print(
                f"  [Run {run_id}] iter {it:03d} [{elapsed:.1f}s]: "
                f"loss={float(loss_val):.6e}, mu_c={mu_c:.4f}, mu_s={mu_s:.4f}"
            )
    
    elapsed_time = time.time() - start_time
    
    # Final values
    mu_c_final = float(mu_from_theta(theta[0]))
    mu_s_final = float(mu_from_theta(theta[1]))
    
    # Best values
    mu_c_best = float(mu_from_theta(best_theta[0]))
    mu_s_best = float(mu_from_theta(best_theta[1]))
    
    return {
        'run_id': run_id,
        'n_seeds': len(seed_pool),
        'seed_pool': seed_pool,
        'mu_c_init': mu_c_init,
        'mu_s_init': mu_s_init,
        'mu_c_final': mu_c_final,
        'mu_s_final': mu_s_final,
        'mu_c_best': mu_c_best,      # NEW: best mu_c
        'mu_s_best': mu_s_best,      # NEW: best mu_s
        'loss_init': loss_history[0],
        'loss_final': loss_history[-1],
        'loss_best': best_loss,       # NEW: best loss
        'best_iter': best_iter,       # NEW: iteration where best occurred
        'theta_history': np.array(theta_history),
        'loss_history': np.array(loss_history),
        'grad_norm_history': np.array(grad_norm_history),
        'elapsed_time': elapsed_time,
    }

# ============================================================
# Main execution
# ============================================================
def main():
    print("\n" + "="*80)
    print("Starting parallel optimization")
    print(f"Running {N_PARALLEL_RUNS} optimizations in parallel")
    print(f"Using SAME {N_PARALLEL_RUNS} random initializations for all strategies")
    print("="*80)
    
    all_results = {}
    
    for strategy_name, seed_pool in SEED_STRATEGIES.items():
        print(f"\n{'='*80}")
        print(f"Strategy: {strategy_name} (averaging over {len(seed_pool)} seeds)")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Run N_PARALLEL_RUNS optimizations in parallel
        # Each run uses the SAME initialization from FIXED_INITIALIZATIONS
        results = Parallel(n_jobs=-1, verbose=10)(  # n_jobs=-1 uses all CPU cores
            delayed(run_single_optimization)(
                run_id=i,
                mu_c_init=FIXED_INITIALIZATIONS[i][0],
                mu_s_init=FIXED_INITIALIZATIONS[i][1],
                seed_pool=seed_pool,
                n_iters=N_ITERS,
                lr=LEARNING_RATE,
                verbose=False
            )
            for i in range(N_PARALLEL_RUNS)
        )
        
        elapsed = time.time() - start_time
        print(f"\nCompleted {N_PARALLEL_RUNS} runs in {elapsed/60:.1f} minutes")
        print(f"Average time per run: {elapsed/N_PARALLEL_RUNS:.1f} seconds")
        
        all_results[strategy_name] = results
        
        # Analyze results
        mu_c_finals = [r['mu_c_final'] for r in results]
        mu_s_finals = [r['mu_s_final'] for r in results]
        loss_finals = [r['loss_final'] for r in results]
        
        print(f"\n{'='*60}")
        print(f"Results Summary for {strategy_name}:")
        print(f"{'='*60}")
        print(f"Target:     mu_c={TARGET_PARAMS[0]:.4f}, mu_s={TARGET_PARAMS[1]:.4f}")
        print(f"Recovered:  mu_c={np.mean(mu_c_finals):.4f} ± {np.std(mu_c_finals):.4f}")
        print(f"            mu_s={np.mean(mu_s_finals):.4f} ± {np.std(mu_s_finals):.4f}")
        print(f"Final loss: {np.mean(loss_finals):.4e} ± {np.std(loss_finals):.4e}")
        
        print(f"\nIndividual runs:")
        print(f"{'Run':>4} {'Init (μc, μs)':>20} {'Final (μc, μs)':>20} {'Loss':>12} {'Time(s)':>8}")
        print("-" * 70)
        for r in results:
            print(
                f"{r['run_id']:4d} "
                f"({r['mu_c_init']:+.3f}, {r['mu_s_init']:+.3f})  ->  "
                f"({r['mu_c_final']:+.3f}, {r['mu_s_final']:+.3f})  "
                f"{r['loss_final']:12.4e} "
                f"{r['elapsed_time']:8.1f}"
            )
        
        # Save results
        output_dir = Path('optimization_results')
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f'results_{strategy_name}.npy'
        np.save(output_file, results, allow_pickle=True)
        print(f"\nResults saved to {output_file}")
    
    # Save all results together
    all_results_file = Path('optimization_results') / 'all_results.npy'
    np.save(all_results_file, all_results, allow_pickle=True)
    print(f"\nAll results saved to {all_results_file}")
    
    print("\n" + "="*80)
    print("All optimizations complete!")
    print("="*80)

if __name__ == "__main__":
    main()