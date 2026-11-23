import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_manager import dataset_confs
from proxies import proxy_columns, reference_stats, make_spec, save_spec


def parse_args():
    p = argparse.ArgumentParser(description="Add twins / gaussian proxies of confounders to a real-world log.")
    p.add_argument("--dataset", type=str, required=True, help="dataset name in dataset_confs, e.g. bpic2017_accepted")
    p.add_argument("--confounders", type=str, required=True, help="comma-separated column names")
    p.add_argument("--proxy_type", type=str, default="gaussian", choices=["gaussian", "twins"])
    p.add_argument("--proxy_noise", type=str, default="0.3", help="comma-separated noise levels")
    p.add_argument("--proxy_bins", type=int, default=10)
    p.add_argument("--proxy_reps", type=int, default=3)
    p.add_argument("--proxy_seed", type=int, default=12345)
    p.add_argument("--sep", type=str, default=";")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def safe(name):
    return re.sub(r"[^0-9A-Za-z_.]+", "_", name)


def case_values(log, case_col, src):
    s = log[src].replace("", np.nan)
    return s.groupby(log[case_col], sort=False).apply(lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan)


def is_numeric(values):
    return pd.to_numeric(values.dropna(), errors="coerce").notna().all()


def draw(v, src, spec, info, rng):
    n_bins, n_reps, noise = spec["n_bins"], spec["n_reps"], spec["noise"]
    missing = pd.isna(v)
    if spec["type"] == "gaussian" and info["numeric"]:
        ref = spec["ref"][src]
        z = (pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(float) - ref["mean"]) / ref["sd"]
        out = (z + rng.normal(0.0, noise, len(z)))[:, None]
    elif spec["type"] == "gaussian":
        cats = info["categories"]
        onehot = np.zeros((len(v), len(cats)))
        for i, x in enumerate(v):
            if not pd.isna(x):
                onehot[i, cats.index(x) if x in cats else len(cats) - 1] = 1.0
        out = onehot + rng.normal(0.0, noise, onehot.shape)
    else:
        if info["numeric"]:
            edges = np.asarray(spec["ref"][src]["edges"])
            cat = np.digitize(np.nan_to_num(pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(float)), edges)
            k = len(edges) + 1
        else:
            cats = info["categories"]
            cat = np.array([cats.index(x) if x in cats else len(cats) - 1 for x in v])
            k = len(cats)
        onehot = np.zeros((len(v), n_bins))
        onehot[np.arange(len(v)), np.minimum(cat, n_bins - 1)] = 1.0
        onehot[:, k:] = 0.0
        bits = np.tile(onehot, (1, n_reps))
        out = np.where(rng.random(bits.shape) < noise, 1.0 - bits, bits)
    out[missing] = np.nan
    return out


def main():
    args = parse_args()
    if args.dataset not in dataset_confs.filename:
        raise KeyError(f"unknown dataset '{args.dataset}' in dataset_confs")
    path = dataset_confs.filename[args.dataset]
    case_col = dataset_confs.case_id_col[args.dataset]
    sources = [c.strip() for c in args.confounders.split(",") if c.strip()]
    noises = [float(x) for x in args.proxy_noise.split(",") if x.strip()]

    log = pd.read_csv(path, sep=args.sep, dtype=str, keep_default_na=False)
    missing_cols = [s for s in sources if s not in log.columns]
    if missing_cols:
        raise KeyError(f"{missing_cols} not in {path}; columns: {list(log.columns)}")
    case_ids = log[case_col].drop_duplicates().to_numpy()
    print(f"{args.dataset}: {len(log)} events, {len(case_ids)} cases, confounders {sources}")

    values, info, numeric_refs = {}, {}, {}
    for s in sources:
        v = case_values(log, case_col, s).reindex(case_ids)
        numeric = is_numeric(v)
        if numeric:
            v = pd.to_numeric(v, errors="coerce")
            numeric_refs[s] = v.to_numpy(float)
            info[s] = {"numeric": True}
        else:
            top = v.dropna().value_counts().index.tolist()
            cats = top[:args.proxy_bins - 1] + (["<other>"] if len(top) >= args.proxy_bins else [])
            v = v.where(v.isin(cats) | v.isna(), "<other>")
            info[s] = {"numeric": False, "categories": cats}
        values[s] = v.to_numpy(object if not numeric else float)
        kind = "numeric" if numeric else f"categorical ({len(info[s]['categories'])} categories)"
        print(f"  {s}: {kind}, missing in {pd.isna(v).mean():.1%} of the cases")
    ref = reference_stats(numeric_refs, args.proxy_bins)
    for s in sources:
        if not info[s]["numeric"]:
            ref[s] = {"categories": info[s]["categories"]}

    cols_tag = "-".join(safe(s) for s in sources)
    for noise in noises:
        name = f"{args.dataset}_proxy-{cols_tag}-{args.proxy_type}{noise:g}"
        out_path = os.path.join(os.path.dirname(path), f"{name}.csv")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"  exists, skipped (use --overwrite): {out_path}")
            continue
        spec = make_spec(sources, args.proxy_type, noise, args.proxy_bins, args.proxy_reps, ref)
        rng = np.random.default_rng(args.proxy_seed)
        table = pd.DataFrame({case_col: case_ids})
        for s in sources:
            if args.proxy_type == "gaussian" and not info[s]["numeric"]:
                cols = [f"proxy_{safe(s)}_c{k}" for k in range(len(info[s]["categories"]))]
            else:
                cols = [c.replace(f"proxy_{s}", f"proxy_{safe(s)}")
                        for c in proxy_columns(s, args.proxy_type, args.proxy_bins, args.proxy_reps)]
            p = draw(values[s], s, spec, info[s], rng)
            miss = np.isnan(p).any(1)
            table[cols] = np.nan_to_num(p, nan=0.0)
            if miss.any():
                table[f"proxy_{safe(s)}_missing"] = miss.astype(float)
        out = log.merge(table, on=case_col, how="left", validate="many_to_one")
        out.to_csv(out_path, sep=args.sep, index=False)
        save_spec(spec, out_path)
        n_proxy = table.shape[1] - 1
        print(f"  noise {noise:g}: {n_proxy} proxy columns -> {out_path}")
        if args.proxy_type == "gaussian":
            for s in [s for s in sources if info[s]["numeric"]]:
                ok = ~np.isnan(np.asarray(values[s], float))
                r = np.corrcoef(np.asarray(values[s], float)[ok], table.loc[ok, f"proxy_{safe(s)}"])[0, 1]
                print(f"    corr({s}, proxy) = {r:.3f}")


if __name__ == "__main__":
    main()
