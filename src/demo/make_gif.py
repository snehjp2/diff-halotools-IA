import os, glob, re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------
# User settings
# ----------------------------
run_dir = "../scripts/runs/hod_maxN_jax_reparam"     # folder with catalog_step_*.npz
pattern = "catalog_step_*.npz"
out_gif = "scatter_evolution.gif"
fps = 12                      # smoother playback

ix, iy = 0, 1                 # which columns to plot (x,y)
lock_axes = True              # keep fixed bounds across frames
point_size = 1.0
alpha = 0.04                  # a bit softer for dense fields
color = "darkred"
figsize = (4.0, 4.0)
dpi = 100
# ----------------------------

# sort files by numeric step
step_re = re.compile(r".*step_(\d+)\.npz$")
def step_key(p):
    m = step_re.match(p)
    return int(m.group(1)) if m else p

files = sorted(glob.glob(os.path.join(run_dir, pattern)), key=step_key)
if not files:
    raise FileNotFoundError(f"No files matching {pattern} in {run_dir}")

def load_xy(path):
    with np.load(path) as D:
        X = D["gal_cat"]
    return X[:, ix].astype(np.float64), X[:, iy].astype(np.float64)

# fixed axes limits
if lock_axes:
    mins, maxs = [], []
    for f in files:
        x, y = load_xy(f)
        if x.size:
            mins.append((x.min(), y.min()))
            maxs.append((x.max(), y.max()))
    xmin = min(m[0] for m in mins)
    ymin = min(m[1] for m in mins)
    xmax = max(M[0] for M in maxs)
    ymax = max(M[1] for M in maxs)

# ----------------------------
# render frames
# ----------------------------
# ----------------------------
# render frames (HiDPI-safe)
# ----------------------------
frames = []
fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values(): spine.set_visible(False)
ax.set_aspect("equal", adjustable="box")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

for f in files:
    ax.clear()
    x, y = load_xy(f)
    ax.scatter(x, y, s=point_size, alpha=alpha, color=color, rasterized=True)
    if lock_axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())      # (H, W, 4)
    rgb  = rgba[..., :3].copy()                       # make it C-contiguous for PIL
    frames.append(Image.fromarray(rgb, mode="RGB"))

plt.close(fig)

# ----------------------------
# save GIF (shared palette not required when frames are RGB)
# ----------------------------
frames[0].save(
    out_gif,
    save_all=True,
    append_images=frames[1:],
    duration=int(1000 / fps),
    loop=0,
    disposal=2,
)
print(f"✅ GIF saved to: {out_gif}")

# ----------------------------
# also save first/last PNGs (HiDPI-safe)
# ----------------------------
def save_png(x, y, name):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(x, y, s=point_size, alpha=alpha, color=color, rasterized=True)
    if lock_axes:
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    Image.fromarray(rgba[..., :3], "RGB").save(name, optimize=True)
    plt.close(fig)

x0, y0 = load_xy(files[0])
xL, yL = load_xy(files[-1])
save_png(x0, y0, "frame_first.png")
save_png(xL, yL, "frame_last.png")
print("📸 Wrote PNG snapshots: frame_first.png, frame_last.png")
