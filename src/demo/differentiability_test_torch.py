import torch, numpy as np, csv, os, time, json
from torch_diffhodIA import DiffHalotoolsIA
from diffhodIA_utils import load_cleaned_catalogs_from_h5, mask_bad_halocat

# --- logger (unchanged except explicit .copy() when saving) ---
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
                csv.writer(f).writerow(["step","loss","N_true","N_hat","params_json","seed","notes"])

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def log_step(self, step, loss, N_true, N_hat, params, gal_cat, seed=12345, notes=""):
        # scalars
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                int(step),
                float(loss.detach().item() if isinstance(loss, torch.Tensor) else loss),
                int(N_true),
                float(N_hat.detach().item() if isinstance(N_hat, torch.Tensor) else N_hat),
                json.dumps(self._to_numpy(params).tolist()),
                int(seed),
                str(notes),
            ])

        # full catalog
        if (step % self.save_every) == 0:
            cat_np = self._to_numpy(gal_cat).copy()  # <-- explicit copy
            out_path = os.path.join(self.dir, f"catalog_step_{step:05d}.npz")

            sample = None
            if (self.sample_n is not None) and (cat_np.shape[0] > self.sample_n):
                idx = np.random.default_rng(seed + step).choice(cat_np.shape[0], size=self.sample_n, replace=False)
                sample = cat_np[idx]

            np.savez_compressed(
                out_path,
                gal_cat=cat_np,
                sample=(sample if sample is not None else np.empty((0, cat_np.shape[1]))),
                params=self._to_numpy(params).copy(),
                N_true=np.array([N_true], dtype=np.int64),
                N_hat=np.array([float(N_hat.detach().item())], dtype=np.float32),
                loss=np.array([float(loss.detach().item())], dtype=np.float32),
                seed=np.array([seed], dtype=np.int64),
                notes=np.array([notes]),
            )
            with open(os.path.join(self.dir, "latest.txt"), "w") as f:
                f.write(f"{step}\n")

    def path(self):
        return self.dir


if __name__ == "__main__":
    # --- params + opt ---
    params = torch.tensor([-0.9446, -0.5776, 15.8064, 0.4702, 11.1825, 12.7235, 0.7437],
                        dtype=torch.float32, requires_grad=True)
    
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

    opt = torch.optim.AdamW([params], lr=5e-2
                        )
    logger = CatalogLogger(root="runs", run_name="hod_maxN", save_every=1, sample_n=10000)

    def build_and_forward(params, tau, seed=12345):
        builder = DiffHalotoolsIA(
            subcat, params,
            do_discrete=True, do_nfw_fallback=True,
            seed=seed, satellite_alignment="radial",
            relaxed=True, tau=tau, Nmax_sat=48, t_rank=0.5,
        )
        
        gal_cat = builder.return_catalog()
        N_true = gal_cat.shape[0]
        Hc_st = builder._cache.get("Hc_st")
        Kh_st = builder._cache.get("Kh_st")
        if (Hc_st is not None) and (Kh_st is not None):
            N_hat = Hc_st.sum() + Kh_st.sum()
        else:
            N_hat = builder.cent_w.sum() + builder.sat_w.sum()
        return N_true, N_hat, gal_cat, builder

    target = 500_000.0
    base_seed = 12345
    K = 4
    tau0, tau_min, decay = 0.5, 0.03, 0.98
    num_steps = 50
    lr_min = 1e-1
    lr_max = 1.0
    warmup_steps = 10

    def get_lr(step):
        if step < warmup_steps:
            # cosine from 0→π/2
            return lr_min
        else:
            ## anneal from lr_min to lr_max
            return lr_max * (1 - (step - warmup_steps) / (num_steps - warmup_steps))

    for step in range(num_steps):
        opt.zero_grad()  
        
        lr = get_lr(step)
        for g in opt.param_groups:
            g["lr"] = lr

        tau = max(tau_min, tau0 * (decay ** step))

        N_true_avg = 0.0
        N_hat_avg  = 0.0
        gal_cat_to_save = None

        for k in range(K):
            seed_k = base_seed + 1000*step + k
            Nt, Nh, gal_cat_k, builder_k = build_and_forward(params, seed=seed_k, tau=tau)  # same as your build_and_forward but takes tau
            N_true_avg += Nt
            N_hat_avg  += Nh
            if k == 0:
                gal_cat_to_save = gal_cat_k  # save one actual catalog for visualization

        N_true_avg /= K
        N_hat_avg  /= K

        # loss on the averaged surrogate
        loss = (N_hat_avg - target) ** 2
        loss = loss 
        # + 1e-6 *(builder_k.cent_w.sum() + builder_k.sat_w.sum())

        loss.backward()
        torch.nn.utils.clip_grad_norm_([params], max_norm=5)
        opt.step()

        with torch.no_grad():
            params.clamp_(-5.0, 20.0)

        # log the averaged scalars and one representative catalog
        logger.log_step(step, loss, N_true_avg, N_hat_avg, params, gal_cat_to_save, seed=base_seed)

        if step % 2 == 0 or step == 299:
            print(f"[step {step:03d}] loss={loss.item():.6g}  N_true={N_true_avg:.0f}  N_hat={N_hat_avg.item():.1f}")