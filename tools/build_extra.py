"""Extra analyses for the learning site (run after build_assets.py, from the project root):
PCA explained variance, t-SNE perplexity / UMAP parameter sweeps on the same samples, latent histograms
(AE vs VAE vs the N(0,1) prior), best/median/worst reconstructions, per-run sample sheets for the explorer,
and AE-vs-VAE interpolation on CelebA between two generated faces.

    python <repo>/tools/build_extra.py --out <repo>/assets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path.cwd()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).parent))
from build_assets import DIMS, MODELS, SPACE, enc, model, r3, run_path, save, sheet  # noqa: E402
from common.dataset import SliceDataset  # noqa: E402

torch.set_grad_enabled(False)


def folder(ds, mk, z):
    return SPACE / ds / f"{'ae' if mk == 'ae' else 'vae'}_{run_path(ds, mk, z).name}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out
    d = {}
    test = SliceDataset(PROJECT / "data/processed", "test")
    base = json.loads((out / "data.json").read_text())
    keep_rows = None

    # ---- PCA explained variance on the full codes ----
    from sklearn.decomposition import PCA
    d["pca_var"] = {}
    for ds, mk, z in [("ixi", "ae", 64), ("ixi", "vae01", 64), ("ixi", "vae1", 64), ("ixi", "ae", 16), ("ixi", "ae", 128),
                      ("celeba", "ae", 64), ("celeba", "vae01", 64), ("celeba", "vae1", 64)]:
        lat = np.load(folder(ds, mk, z) / "codes.npz")["latents"].astype(np.float32)
        r = PCA(n_components=min(z, 30)).fit(lat - lat.mean(0)).explained_variance_ratio_
        d["pca_var"][f"{ds}_{mk}_{z}"] = r3(r, 4)

    # ---- the same 1,200 IXI test slices for every sweep (identical to the explorer subset) ----
    rng = np.random.RandomState(7)
    c64 = np.load(folder("ixi", "ae", 64) / "codes.npz")
    keep = np.sort(rng.choice(len(c64["test_index"]), 1200, replace=False))
    lat = c64["latents"].astype(np.float32)[keep]
    sp = test.slice_pos[c64["test_index"][keep]]
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    import umap
    x = PCA(n_components=50, random_state=42).fit_transform(StandardScaler().fit_transform(lat))
    d["sweep"] = {"slice_pos": r3(sp, 3), "tsne": {}, "umap_nn": {}, "umap_md": {}}
    for p in (5, 30, 100):
        d["sweep"]["tsne"][str(p)] = r3(TSNE(2, perplexity=p, init="pca", learning_rate="auto", random_state=42).fit_transform(x), 2)
    for nn in (5, 15, 50):
        d["sweep"]["umap_nn"][str(nn)] = r3(umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=42).fit_transform(x), 2)
    for md in (0.0, 0.5, 0.99):
        d["sweep"]["umap_md"][str(md)] = r3(umap.UMAP(n_neighbors=15, min_dist=md, random_state=42).fit_transform(x), 2)
    print("sweeps done", flush=True)

    # ---- latent histograms: AE codes vs VAE means vs the prior ----
    bins = np.linspace(-12, 12, 97)
    d["hist"] = {"edges": r3(bins, 2)}
    for mk in MODELS:
        L = np.load(run_path("ixi", mk, 16) / "latents/test.npz")
        mu = L["latents"]
        var = mu.var(0)
        dims = np.argsort(-var)[:3]
        d["hist"][mk] = {"dims": dims.tolist(), "h": [np.histogram(mu[:, j], bins=bins, density=True)[0].round(4).tolist() for j in dims],
                         "mean": r3(mu.mean(0), 2), "std": r3(mu.std(0), 2)}
        if "logvar" in L.files:
            d["hist"][mk]["sigma_median"] = round(float(np.median(np.exp(0.5 * L["logvar"]))), 3)

    # ---- best / median / worst reconstructions (AE z=16 and VAE z=16, IXI test set) ----
    d["failures"] = {}
    for mk in ("ae", "vae01"):
        L = np.load(run_path("ixi", mk, 16) / "latents/test.npz")
        order = np.argsort(L["mse"])
        picks = {"best": order[:6], "median": order[len(order) // 2 - 3: len(order) // 2 + 3], "worst": order[-6:][::-1]}
        m, _, _ = model("ixi", mk, 16)
        rows, info = [], {}
        for name, idx in picks.items():
            ti = L["index"][idx]
            x = torch.stack([test[int(i)] for i in ti])
            r = m.decode(enc(m, x))
            rows += [list(x), list(r), list(((x - r).abs() * 4).clamp(0, 1))]
            info[name] = {"psnr": [round(float(10 * np.log10(1 / max(v, 1e-10))), 1) for v in L["mse"][idx]],
                          "pos": r3(test.slice_pos[ti], 2)}
        save(sheet(rows, 96), out / f"failures_ixi_{mk}_16.png")
        d["failures"][mk] = info

    # ---- per-run sample sheets for the explorer ----
    real_idx = np.random.RandomState(21).choice(len(test), 6, replace=False)
    real = torch.stack([test[int(i)] for i in real_idx])
    d["explorer_runs"] = []
    for ds in ("ixi", "celeba"):
        for mk in MODELS:
            for z in DIMS[ds]:
                if run_path(ds, mk, z) is None:
                    continue
                m, gm, gL = model(ds, mk, z)
                e = torch.randn(12 if ds == "celeba" else 6, z, generator=torch.Generator().manual_seed(2))
                gen = m.decode(gm + e @ gL.T) if mk == "ae" else m.decode(e)
                if ds == "ixi":
                    rows = [list(real), list(m.decode(enc(m, real))), list(gen)]
                else:
                    rows = [list(gen[:6]), list(gen[6:])]
                save(sheet(rows, 96), out / "runs" / f"{ds}_{mk}_{z}.jpg")
                d["explorer_runs"].append({"ds": ds, "mk": mk, "z": z})
    print("explorer sheets done", flush=True)

    # ---- CelebA interpolation between two generated faces, AE vs VAE ----
    vref, _, _ = model("celeba", "vae01", 128)
    ends = vref.decode(torch.randn(2, 128, generator=torch.Generator().manual_seed(31)))
    t = torch.linspace(0, 1, 8)[:, None]
    for z in (16, 128):
        rows = []
        for mk in MODELS:
            m, _, _ = model("celeba", mk, z)
            za, zb = enc(m, ends[0:1]), enc(m, ends[1:2])
            rows.append([ends[0]] + list(m.decode((1 - t) * za + t * zb)) + [ends[1]])
        save(sheet(rows, 96), out / f"interp_celeba_{z}.png")

    # ---- a latent "walk" along one number (traversal) for the AE z=2 and VAE z=2 on IXI ----
    for mk in MODELS:
        m, gm, gL = model("ixi", mk, 2)
        sd = torch.sqrt((gL @ gL.T).diagonal())
        steps = torch.linspace(-2, 2, 9)
        rows = []
        for j in range(2):
            zz = gm.repeat(9, 1); zz[:, j] = gm[j] + steps * sd[j]
            rows.append(list(m.decode(zz)))
        save(sheet(rows, 96), out / f"traverse_ixi_{mk}_2.png")

    (out / "data2.json").write_text(json.dumps(d, separators=(",", ":")))
    print("data2.json written")


if __name__ == "__main__":
    main()
