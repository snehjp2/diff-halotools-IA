"""
Timing script for diffHOD-IA catalog generation.
Measures wall clock time for return_catalog() only.

Usage:
    python timing_hod.py
"""

import time
import numpy as np
import jax
from halotools.sim_manager import CachedHaloCatalog

# Assuming diffHalotoolsIA.py is in the same directory or on PYTHONPATH
from diffHalotoolsIA import DiffHalotoolsIA

# Target HOD parameters: [mu_cen, mu_sat, logMmin, sigma_logM, logM0, logM1, alpha]
TARGET_PARAMS = np.asarray([0.79, 0.30, 12.54, 0.26, 12.68, 13.48, 1.0])


def mask_bad_halocat(halocat):
    """Remove halos with invalid axis ratios."""
    mask = (halocat.halo_table["halo_b_to_a"] > 0) & (halocat.halo_table["halo_b_to_a"] <= 1)
    halocat.halo_table = halocat.halo_table[mask]


def main():
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    
    # Load halo catalog (one-time setup, not timed)
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
        "halo_hostid", "halo_b_to_a",
    ]]
    print(f"Loaded {len(subcat)} halos")
    
    # Initialize builder (one-time setup, not timed for HOD)
    print("\nInitializing DiffHalotoolsIA builder...")
    builder = DiffHalotoolsIA(
        subcat=subcat,
        params=TARGET_PARAMS,
        do_discrete=True,
        do_nfw_fallback=True,
        seed=42,
        alignment_model="radial",
        alignment_strength="constant",
        relaxed=True,
        tau=0.1,
        Nmax_sat=256,
        t_rank=0.5,
    )
    
    # Warmup runs (JIT compilation)
    n_warmup = 3
    n_timed = 10
    
    print(f"\nWarmup ({n_warmup} runs)...")
    for i in range(n_warmup):
        builder.key = jax.random.PRNGKey(i)
        cat = builder.return_catalog()
        jax.block_until_ready(cat)
    
    # Timed runs
    print(f"\nTiming ({n_timed} runs)...")
    times = []
    sizes = []
    
    for i in range(n_timed):
        builder.key = jax.random.PRNGKey(1000 + i)
        
        t0 = time.perf_counter()
        cat = builder.return_catalog()
        jax.block_until_ready(cat)
        t1 = time.perf_counter()
        
        times.append(t1 - t0)
        sizes.append(cat.shape[0])
        print(f"  Run {i+1}: {times[-1]:.4f}s, {sizes[-1]:,} galaxies")
    
    # Results
    times = np.array(times)
    sizes = np.array(sizes)
    
    print("\n" + "="*50)
    print("RESULTS: return_catalog() timing")
    print("="*50)
    print(f"Mean:   {np.mean(times):.4f} ± {np.std(times):.4f} s")
    print(f"Median: {np.median(times):.4f} s")
    print(f"Min:    {np.min(times):.4f} s")
    print(f"Max:    {np.max(times):.4f} s")
    print(f"Galaxies: {np.mean(sizes):.0f} ± {np.std(sizes):.0f}")
    print("="*50)


if __name__ == "__main__":
    main()