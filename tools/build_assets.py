"""Render every figure and data file used by index.html from the trained AE / VAE runs.

Run from the project root (the folder with autoencoder/, vae/, outputs/):
    python <this repo>/tools/build_assets.py --out <this repo>/assets

Everything shown on the page comes from the real runs: real IXI test slices (CC BY-SA 3.0), the trained
models, their test-set latent codes and training logs. CelebA appears only as model-generated images.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path.cwd()
sys.path.insert(0, str(PROJECT))
from autoencoder.model import Autoencoder  # noqa: E402
from common.augment import augment_batch  # noqa: E402
from common.dataset import SliceDataset  # noqa: E402
from vae.model import VAE  # noqa: E402

torch.set_grad_enabled(False)
OUT_ROOT = {"ixi": PROJECT / "outputs", "celeba": PROJECT / "outputs_celeba"}
HF = PROJECT / "hf" / "build"
SPACE = HF / "medlatent-studio" / "data" / "embeddings"
MODELS = {"ae": ("ae", None), "vae01": ("vae", 0.1), "vae1": ("vae", 1.0)}
DIMS = {"ixi": [2, 3, 4, 8, 16, 64, 128], "celeba": [2, 4, 8, 16, 64, 128]}
FONT = ImageFont.load_default(size=14)


def run_path(ds, mk, z):
    kind, beta = MODELS[mk]
    name = f"z{z}" if kind == "ae" else f"z{z}_beta{beta:g}"
    p = OUT_ROOT[ds] / kind / name
    return p if (p / "checkpoints/best.pt").exists() else None


_cache = {}


def model(ds, mk, z):
    key = (ds, mk, z)
    if key not in _cache:
        p = run_path(ds, mk, z)
        cls = Autoencoder if MODELS[mk][0] == "ae" else VAE
        m, _ = cls.from_checkpoint(p / "checkpoints/best.pt", "cpu")
        g = np.load(HF / ("medlatent-autoencoder" if mk == "ae" else "medlatent-vae") / ds / p.name / "latent_gaussian.npz")
        _cache[key] = (m.eval(), torch.tensor(g["mean"]).float(), torch.tensor(g["chol"]).float())
    return _cache[key]


def enc(m, x):
    o = m.encode(x)
    return o[0] if isinstance(o, tuple) else o


def to_img(t, size=None):
    a = t.detach().clamp(0, 1).mul(255).round().byte().cpu().numpy()
    im = Image.fromarray(a[0] if a.shape[0] == 1 else a.transpose(1, 2, 0)).convert("RGB")
    return im.resize((size, size), Image.LANCZOS) if size else im


def sheet(rows, size, pad=2, bg=(255, 255, 255)):
    """rows: list of lists of CxHxW tensors (None = empty cell) -> PIL image."""
    ncol = max(len(r) for r in rows)
    W, H = ncol * size + (ncol - 1) * pad, len(rows) * size + (len(rows) - 1) * pad
    im = Image.new("RGB", (W, H), bg)
    for i, r in enumerate(rows):
        for j, t in enumerate(r):
            if t is not None:
                im.paste(to_img(t, size), (j * (size + pad), i * (size + pad)))
    return im


def save(im, path, quality=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jpg":
        im.save(path, quality=quality or 88, optimize=True)
    else:
        im.save(path, optimize=True)


def r3(a, nd=3):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def psnr(x, y):
    return float(10 * torch.log10(1 / ((x - y) ** 2).mean().clamp_min(1e-10)))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    ixi_test = SliceDataset(PROJECT / "data/processed", "test")

    # ---------- the shared embedding subsample (same 4000 test slices for every IXI run) -------------
    codes0 = np.load(SPACE / "ixi" / "ae_z16" / "codes.npz")
    sub_index = codes0["test_index"]                       # test-set indices of the plotted points
    rng = np.random.RandomState(7)
    keep = np.sort(rng.choice(len(sub_index), 1200, replace=False))

    # ---------- picks: 8 real test slices across head height, all inside the plotted subset ----------
    pos = ixi_test.slice_pos[sub_index[keep]]
    targets = np.linspace(0.33, 0.78, 8)
    pick_rows = []
    for t in targets:
        cand = np.argsort(np.abs(pos - t))
        for c in cand:
            sid = ixi_test.subject_ids[sub_index[keep[c]]]
            if sid not in {ixi_test.subject_ids[sub_index[keep[p]]] for p in pick_rows}:
                pick_rows.append(int(c)); break
    pick_test = [int(sub_index[keep[c]]) for c in pick_rows]          # index in the test set
    X = torch.stack([ixi_test[i] for i in pick_test])
    picks = []
    for k, i in enumerate(pick_test):
        save(to_img(X[k], 256), out / f"pick_{k}.png")
        picks.append({"test_index": i, "subject": str(ixi_test.subject_ids[i]),
                      "slice_pos": round(float(ixi_test.slice_pos[i]), 3), "plot_row": int(pick_rows[k])})

    # ---------- per pick: reconstruction ladder (3 models x 7 sizes) + error maps ----------
    dims = DIMS["ixi"]
    psnr_tab = {}
    for k in range(len(picks)):
        rec_rows, err_rows = [], []
        for mk in MODELS:
            rr, er = [], []
            for z in dims:
                if run_path("ixi", mk, z) is None:
                    rr.append(None); er.append(None); continue
                m, _, _ = model("ixi", mk, z)
                x = X[k:k + 1]
                r = m.decode(enc(m, x))
                rr.append(r[0]); er.append(((x - r).abs() * 4).clamp(0, 1)[0])
                psnr_tab.setdefault(str(k), {}).setdefault(mk, {})[str(z)] = round(psnr(x, r), 2)
            rec_rows.append(rr); err_rows.append(er)
        save(sheet(rec_rows, 112), out / f"ladder_{k}.png")
        save(sheet(err_rows, 112), out / f"ladder_err_{k}.png")
    data["ladder"] = {"dims": dims, "models": list(MODELS), "psnr": psnr_tab}

    # ---------- per pick: encoder + decoder feature maps of the AE z=16 ----------
    m16, _, _ = model("ixi", "ae", 16)
    enc_idx, dec_idx = [2, 4, 6, 8, 10], [2, 5, 8, 11, 14]
    codes16 = {}
    for k in range(len(picks)):
        x = X[k:k + 1]
        h, rows = x, []
        for i, layer in enumerate(m16.encoder.features):
            h = layer(h)
            if i in enc_idx:
                rows.append(top_maps(h[0], 8))
        z = m16.encoder.to_latent(h.flatten(1))
        codes16[str(k)] = r3(z[0], 2)
        d = m16.decoder.from_latent(z).view(1, *m16.decoder.bottleneck)
        drows = []
        for i, layer in enumerate(m16.decoder.net):
            d = layer(d)
            if i in dec_idx:
                drows.append(top_maps(d[0], 8))
        save(sheet(rows, 72, pad=3), out / f"enc_{k}.png")
        save(sheet(drows, 72, pad=3), out / f"dec_{k}.png")
        save(to_img(m16.decode(z)[0], 256), out / f"rec16_{k}.png")
    # VAE z=16: mean and spread of the posterior for each pick
    v16, _, _ = model("ixi", "vae01", 16)
    mu, lv = v16.encode(X)
    data["codes16"] = {"ae": codes16, "vae_mu": r3(mu, 2), "vae_sigma": r3((0.5 * lv).exp(), 3)}
    data["picks"] = picks

    # ---------- 2-D latent planes: decoded sprite grids + scatter of test codes ----------
    data["planes"] = {}
    for ds, n, tile in (("ixi", 21, 56), ("celeba", 15, 56)):
        test = ixi_test if ds == "ixi" else None
        for mk in MODELS:
            m, gm, gL = model(ds, mk, 2)
            c = np.load(SPACE / ds / f"{'ae' if mk == 'ae' else 'vae'}_{run_path(ds, mk, 2).name}" / "codes.npz")
            lat = c["latents"].astype(np.float32)
            lo, hi = np.percentile(lat, 0.5, 0), np.percentile(lat, 99.5, 0)
            span = hi - lo
            lo, hi = lo - 0.15 * span, hi + 0.15 * span
            gx = np.linspace(lo[0], hi[0], n); gy = np.linspace(hi[1], lo[1], n)        # top row = high z2
            grid = torch.tensor(np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)).float()
            imgs = m.decode(grid)
            rows = [[imgs[i * n + j] for j in range(n)] for i in range(n)]
            save(sheet(rows, tile, pad=0), out / f"plane_{ds}_{mk}.jpg", quality=86)
            sel = np.sort(np.random.RandomState(1).choice(len(lat), 1500, replace=False))
            entry = {"n": n, "tile": tile, "lo": r3(lo), "hi": r3(hi), "x": r3(lat[sel, 0]), "y": r3(lat[sel, 1])}
            emb = np.load(SPACE / ds / f"{'ae' if mk == 'ae' else 'vae'}_{run_path(ds, mk, 2).name}" / "raw_2d.npz")
            if ds == "ixi":
                entry["c"] = r3(emb["label_slice_pos"][sel], 3)
                if mk != "ae":       # posterior clouds of the picks + their means
                    mu2, lv2 = m.encode(X)
                    entry["pick_mu"] = r3(mu2); entry["pick_sigma"] = r3((0.5 * lv2).exp())
                else:
                    entry["pick_mu"] = r3(enc(m, X))
            else:
                entry["c"] = r3(emb["label_attr_Male"][sel], 0)
                entry["c2"] = r3(emb["label_attr_Smiling"][sel], 0)
            data["planes"][f"{ds}_{mk}"] = entry

    # ---------- interactive latent-space explorer data (PCA / t-SNE / UMAP, 2-D / 3-D) ----------
    from sklearn.cluster import KMeans
    data["explorer"] = {}
    for ds, zs in (("ixi", [2, 3, 8, 16, 64, 128]), ("celeba", [2, 16, 64, 128])):
        for mk in MODELS:
            for z in zs:
                p = run_path(ds, mk, z)
                if p is None:
                    continue
                folder = SPACE / ds / f"{'ae' if mk == 'ae' else 'vae'}_{p.name}"
                c = np.load(folder / "codes.npz")
                lat = c["latents"].astype(np.float32)
                kk = keep if ds == "ixi" else np.sort(np.random.RandomState(7).choice(len(lat), 1200, replace=False))
                labels_km = KMeans(6, n_init=4, random_state=0).fit(lat).labels_
                order = np.argsort(-np.bincount(labels_km, minlength=6)); remap = np.empty(6, int); remap[order] = np.arange(6)
                e = {"cluster": remap[labels_km][kk].tolist()}
                for f in sorted(folder.glob("*_*d.npz")):
                    if f.name == "codes.npz":
                        continue
                    arr = np.load(f)
                    e[f.stem] = r3(arr["embedding"][kk], 2)
                    if "lab" not in e:
                        if ds == "ixi":
                            e["lab"] = {"slice_pos": r3(arr["label_slice_pos"][kk], 3), "mse": r3(arr["label_mse"][kk] * 1000, 2)}
                        else:
                            e["lab"] = {k2: r3(arr[f"label_attr_{k2}"][kk], 0) for k2 in ("Male", "Smiling", "Eyeglasses", "Blond_Hair")}
                data["explorer"][f"{ds}_{mk}_{z}"] = e

    # ---------- cluster galleries: decoded centre + nearest real slices (IXI), generated (CelebA) ----------
    for ds, mk, z in (("ixi", "ae", 16), ("ixi", "vae01", 16), ("celeba", "vae01", 128)):
        m, _, _ = model(ds, mk, z)
        folder = SPACE / ds / f"{'ae' if mk == 'ae' else 'vae'}_{run_path(ds, mk, z).name}"
        c = np.load(folder / "codes.npz"); lat = c["latents"].astype(np.float32)
        km = KMeans(6, n_init=4, random_state=0).fit(lat)
        order = np.argsort(-np.bincount(km.labels_, minlength=6))
        rows = []
        for ci in order:
            members = np.where(km.labels_ == ci)[0]
            centre = torch.tensor(km.cluster_centers_[ci]).float()[None]
            row = [m.decode(centre)[0], None]
            if ds == "ixi":
                near = members[np.argsort(((lat[members] - km.cluster_centers_[ci]) ** 2).sum(1))[:6]]
                row += [ixi_test[int(c["test_index"][j])] for j in near]
            else:
                sd = torch.tensor(lat[members].std(0)).float()
                g = torch.Generator().manual_seed(int(ci))
                row += list(m.decode(centre + 0.6 * sd * torch.randn(6, z, generator=g)))
            rows.append(row)
        save(sheet(rows, 88, pad=3), out / f"clusters_{ds}_{mk}_{z}.jpg", quality=88)

    # ---------- generation figures ----------
    g = torch.Generator().manual_seed(3)
    eps = torch.randn(8, 16, generator=g)
    ae16, mu_ae, L_ae = model("ixi", "ae", 16)
    v1_16, _, _ = model("ixi", "vae1", 16)
    rows = [list(ae16.decode(eps)),                        # AE fed N(0, I): wrong place
            list(ae16.decode(mu_ae + eps @ L_ae.T)),       # AE fed its own Gaussian fit
            list(v16.decode(eps)),                         # VAE beta=0.1 prior
            list(v1_16.decode(eps))]                       # VAE beta=1 prior
    save(sheet(rows, 112), out / "gen_ixi_z16.png")
    data["ae_code_stats"] = {"mean_abs": round(float(mu_ae.abs().mean()), 2),
                             "std": r3(np.sqrt((L_ae @ L_ae.T).diagonal().numpy()), 2)}
    # temperature sweep, same noise in every row
    temps = [0.5, 0.7, 1.0, 1.3]
    rows = [list(v16.decode(t * eps)) for t in temps]
    save(sheet(rows, 112), out / "temperature_ixi.png")
    # real -> real interpolation in three z=128 models
    a, b = X[1:2], X[6:7]
    tt = torch.linspace(0, 1, 8)[:, None]
    rows = []
    for mk in MODELS:
        m, _, _ = model("ixi", mk, 128)
        za, zb = enc(m, a), enc(m, b)
        rows.append([a[0]] + list(m.decode((1 - tt) * za + tt * zb)) + [b[0]])
    save(sheet(rows, 104), out / "interp_ixi.png")
    # real vs generated, side by side (IXI, z = 128)
    real = torch.stack([ixi_test[int(i)] for i in np.random.RandomState(11).choice(len(ixi_test), 8, replace=False)])
    m, gm, gL = model("ixi", "vae01", 128)
    fake_v = m.decode(torch.randn(8, 128, generator=torch.Generator().manual_seed(5)))
    ma, gma, gLa = model("ixi", "ae", 128)
    fake_a = ma.decode(gma + torch.randn(8, 128, generator=torch.Generator().manual_seed(5)) @ gLa.T)
    save(sheet([list(real), list(fake_a), list(fake_v)], 112), out / "real_vs_fake_ixi.png")
    # CelebA: generated only
    rows = []
    for z in (16, 128):
        for mk in MODELS:
            m, gm, gL = model("celeba", mk, z)
            e = torch.randn(8, z, generator=torch.Generator().manual_seed(9))
            rows.append(list(m.decode(gm + e @ gL.T) if mk == "ae" else m.decode(e)))
    save(sheet(rows, 96), out / "gen_celeba.png")
    rows = []
    for z in (2, 8, 16, 64, 128):
        m, gm, gL = model("celeba", "vae01", z)
        rows.append(list(m.decode(torch.randn(8, z, generator=torch.Generator().manual_seed(4)))))
    save(sheet(rows, 88), out / "celeba_by_z.png")

    # ---------- data pipeline figures ----------
    subj = ixi_test.subject_ids[pick_test[3]]
    idx = np.where(ixi_test.subject_ids == subj)[0]
    idx = idx[np.argsort(ixi_test.slice_pos[idx])]
    idx = idx[np.linspace(0, len(idx) - 1, 10).astype(int)]
    save(sheet([[ixi_test[int(i)] for i in idx]], 104), out / "subject_slices.png")
    data["subject_slices"] = {"subject": str(subj), "pos": r3(ixi_test.slice_pos[idx], 2)}
    torch.manual_seed(0)
    cfg = {"augment": True, "aug_flip_p": 0.5, "aug_rotate_deg": 10.0, "aug_scale": 0.05, "aug_translate": 0.05, "aug_gamma": 1.2}
    views = augment_batch(X[3:4].repeat(8, 1, 1, 1), cfg)
    save(sheet([[X[3]] + list(views)], 104), out / "augment.png")

    # ---------- numbers for the plots ----------
    def hist(p):
        return pd.read_csv(p).to_dict(orient="list")
    data["history"] = {
        "v1_ae16": hist(PROJECT / "outputs_v1_noaug/ae/z16/history.csv"),
        "v2_ae16": hist(OUT_ROOT["ixi"] / "ae/z16/history.csv"),
        "v2_vae16": hist(OUT_ROOT["ixi"] / "vae/z16_beta0.1/history.csv"),
    }
    data["results"] = {ds: pd.read_csv(OUT_ROOT[ds] / "analysis/results_table.csv").replace({np.nan: None}).to_dict(orient="records")
                       for ds in OUT_ROOT}
    data["kl_dims"] = {}
    for ds in OUT_ROOT:
        for mk in ("vae01", "vae1"):
            lat = np.load(run_path(ds, mk, 128) / "latents/test.npz")
            kl = 0.5 * (lat["latents"] ** 2 + np.exp(lat["logvar"]) - 1 - lat["logvar"]).mean(0)
            data["kl_dims"][f"{ds}_{mk}"] = r3(np.sort(kl)[::-1], 3)
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    print("assets written:", sorted(p.name for p in out.iterdir())[:8], "...", len(list(out.iterdir())), "files")


def top_maps(h, n):
    """n strongest channels of a C x H x W activation, each scaled to its own min..max (viridis-free gray)."""
    score = h.flatten(1).abs().mean(1)
    idx = torch.argsort(score, descending=True)[:n]
    tiles = []
    for c in idx:
        m = h[c]
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        tiles.append(colourise(m))
    return tiles


_VIR = None


def colourise(m):
    """Map a H x W map in [0, 1] to a 3 x H x W magma-like colour image."""
    global _VIR
    if _VIR is None:
        import matplotlib
        _VIR = torch.tensor(matplotlib.colormaps["magma"](np.linspace(0, 1, 256))[:, :3]).float()
    idx = (m * 255).long().clamp(0, 255)
    return _VIR[idx].permute(2, 0, 1)


if __name__ == "__main__":
    main()
