import argparse
import math
import os
import pickle
from copy import deepcopy
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import SimBank.SimBank.simulation as simulation
import SimBank.SimBank.confounding_level as confounding_level
from dataset_manager.dataset_confs import simbank_dataset_name, logs_dir
from proxies import make_spec, save_spec


INTERVENTIONS = {
    "choose_procedure": dict(
        data_impact="direct", actions=["start_standard", "start_priority"], action_width=2,
        action_depth=1, activities=["start_standard", "start_priority"], column="activity",
        start_control_activity=["initiate_application"], end_control_activity=["initiate_application"]),
    "set_ir_3_levels": dict(
        data_impact="indirect", actions=[0.07, 0.08, 0.09], action_width=3,
        action_depth=1, activities=["calculate_offer"], column="interest_rate",
        start_control_activity=[], end_control_activity=[]),
    "time_contact_HQ": dict(
        data_impact="direct", actions=["do_nothing", "contact_headquarters"], action_width=2,
        action_depth=4, activities=["do_nothing", "contact_headquarters"], column="activity",
        start_control_activity=["start_standard"],
        end_control_activity=["start_standard", "email_customer", "call_customer"]),
}


def build_dataset_params(intervention_names: list, args) -> dict:
    p = {}
    p["train_size"] = args.train_size
    p["test_size"] = args.test_size
    p["val_share"] = .5
    p["train_val_size"] = args.train_val_size
    p["test_val_size"] = min(int(p["val_share"] * p["test_size"]), 1000)
    p["simulation_start"] = datetime(2024, 3, 20, 8, 0)
    p["random_seed_train"] = args.seed_train
    p["random_seed_test"] = args.seed_test
    p["log_cols"] = ["case_nr", "activity", "timestamp", "elapsed_time", "cum_cost", "est_quality",
                     "unc_quality", "amount", "interest_rate", "discount_factor", "outcome", "quality",
                     "noc", "nor", "min_interest_rate"]
    p["case_cols"] = ["amount"]
    p["event_cols"] = ["activity", "elapsed_time", "cum_cost", "est_quality", "unc_quality",
                       "interest_rate", "discount_factor"]
    p["cat_cols"] = ["activity"]
    p["scale_cols"] = ["amount", "elapsed_time", "cum_cost", "est_quality", "unc_quality",
                       "interest_rate", "discount_factor", "outcome"]
    info = {"name": list(intervention_names)}
    for key in ["data_impact", "actions", "action_width", "action_depth", "activities", "column",
                "start_control_activity", "end_control_activity"]:
        info[key] = [INTERVENTIONS[n][key] for n in intervention_names]
    info["retain_method"] = "precise"
    info["action_combinations"] = list(product(*info["actions"]))
    info["action_width_combinations"] = math.prod(info["action_width"])
    info["action_depth_combinations"] = math.prod(info["action_depth"])
    info["len"] = [w if w > 2 else 1 for w in info["action_width"]]
    info["RCT"] = False
    p["intervention_info"] = info
    p["filename"] = "loan_log_" + str(info["name"])
    p["policies_info"] = {
        "general": "real",
        "choose_procedure": {"amount": 50000, "est_quality": 5},
        "time_contact_HQ": "real",
        "min_quality": 2,
        "max_noc": 3,
        "max_nor": 1,
        "min_amount_contact_cust": 50000,
    }
    return p


def add_contact_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, act in [("skip", "skip_contact"), ("contact_hq", "contact_headquarters")]:
        df[col] = (df["activity"] == act).astype(int).groupby(df["case_nr"]).cummax()
    return df


PROXY_SOURCES = ["amount", "est_quality"]


def proxy_columns(source: str, proxy_type: str, n_bins: int = 10, n_reps: int = 3) -> list:
    if proxy_type == "gaussian":
        return [f"proxy_{source}"]
    return [f"proxy_{source}_r{r}_b{b}" for r in range(n_reps) for b in range(n_bins)]


def case_values(df: pd.DataFrame, source: str) -> pd.Series:
    return df.groupby("case_nr")[source].first()


def proxy_reference(frames: list, sources: list, n_bins: int) -> dict:
    ref = {}
    for s in sources:
        v = pd.concat([case_values(f, s) for f in frames]).dropna().to_numpy(float)
        edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1]))
        ref[s] = {"edges": edges, "mean": float(v.mean()), "sd": float(v.std()) or 1.0}
    return ref


def add_proxies(df: pd.DataFrame, sources: list, ref: dict, proxy_type: str, noise: float,
                n_bins: int, n_reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    case_ids = df["case_nr"].drop_duplicates().to_numpy()
    for s in sources:
        v = case_values(df, s).reindex(case_ids).to_numpy(float)
        cols = proxy_columns(s, proxy_type, n_bins, n_reps)
        if proxy_type == "gaussian":
            z = (v - ref[s]["mean"]) / ref[s]["sd"]
            proxy = (z + rng.normal(0.0, noise, len(z)))[:, None]
        else:
            k = len(ref[s]["edges"]) + 1
            cat = np.digitize(v, ref[s]["edges"])
            onehot = np.zeros((len(v), n_bins))
            onehot[np.arange(len(v)), cat] = 1.0
            onehot[:, k:] = 0.0
            bits = np.tile(onehot, (1, n_reps))
            flip = rng.random(bits.shape) < noise
            proxy = np.where(flip, 1.0 - bits, bits)
            proxy[np.isnan(v)] = np.nan
        df = df.merge(pd.DataFrame(proxy, columns=cols).assign(case_nr=case_ids), on="case_nr", how="left")
    return df


def finalize(df: pd.DataFrame, contact_flags: bool) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    cols_to_fill = [c for c in ["interest_rate", "discount_factor", "min_interest_rate"] if c in df]
    df[cols_to_fill] = df[cols_to_fill].fillna(0)
    return add_contact_flags(df) if contact_flags else df


def parse_args():
    p = argparse.ArgumentParser(description="SimBank logs with a given intervention and confounding level.")
    p.add_argument("--intervention", type=str, default="choose_procedure",
                   help="SimBank intervention(s), '+'-separated, e.g. choose_procedure, set_ir_3_levels, "
                        "time_contact_HQ, choose_procedure+set_ir_3_levels.")
    p.add_argument("--deltas", type=str, default="0.99",
                   help="Comma-separated confounding levels (share of bank-policy cases): "
                        "0 = pure RCT, 1 = pure bank policy.")
    p.add_argument("--logs_dir", type=str, default=os.path.normpath(logs_dir),
                   help="Where the CSVs go. Default: the folder DatasetManager reads from "
                        "(dataset_confs.logs_dir) -- only change it if you also change that.")
    p.add_argument("--params_dir", type=str, default="src",
                   help="Folder for the dataset_params pickles.")
    p.add_argument("--train_size", type=int, default=50000)
    p.add_argument("--train_val_size", type=int, default=10000)
    p.add_argument("--test_size", type=int, default=10000)
    p.add_argument("--seed_train", type=int, default=82)
    p.add_argument("--seed_test", type=int, default=130)
    p.add_argument("--save_val", action="store_true", help="Also save the validation log.")
    p.add_argument("--no_contact_flags", action="store_true",
                   help="Do not add the 'skip' / 'contact_hq' columns (the guide notebook adds them).")
    p.add_argument("--proxy_cols", type=str, default="",
                   help=f"Comma-separated confounders to add a noisy proxy for, from {PROXY_SOURCES}. "
                        "Empty = no proxy (file names unchanged).")
    p.add_argument("--proxy_type", type=str, default="twins", choices=["twins", "gaussian"],
                   help="twins: replicated one-hot of the binned confounder with bit flips (CEVAE TWINS); "
                        "gaussian: standardised confounder + Gaussian noise.")
    p.add_argument("--proxy_noise", type=str, default="0.1",
                   help="Comma-separated noise levels, one dataset per level (like --deltas). "
                        "twins: flip probability per bit (CEVAE: 0.05-0.5); gaussian: noise sd.")
    p.add_argument("--proxy_bins", type=int, default=10, help="twins: number of quantile bins (CEVAE: 10).")
    p.add_argument("--proxy_reps", type=int, default=3, help="twins: one-hot replications (CEVAE: 3).")
    p.add_argument("--proxy_seed", type=int, default=7)
    return p.parse_args()


def main():
    args = parse_args()
    names = [n.strip() for n in args.intervention.split("+") if n.strip()]
    unknown = [n for n in names if n not in INTERVENTIONS]
    if unknown:
        raise ValueError(f"unknown intervention(s) {unknown}; choose from {list(INTERVENTIONS)}")
    deltas = [float(d) for d in args.deltas.split(",")]
    if any(not 0.0 <= d <= 1.0 for d in deltas):
        raise ValueError("every delta must be in [0, 1]")
    intervention_tag = "+".join(names)

    data_folder = args.logs_dir
    params_folder = args.params_dir
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(params_folder, exist_ok=True)

    dataset_params = build_dataset_params(names, args)
    pkl_path = os.path.join(params_folder, f"dataset_params_{intervention_tag}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(deepcopy(dataset_params), f)
    print(f"dataset_params ({intervention_tag}) -> {pkl_path}")

    print(f"Simulating {args.train_size} + {args.train_val_size} bank-policy cases ({intervention_tag}) ...")
    gen_normal = simulation.PresProcessGenerator(deepcopy(dataset_params), None,
                                                 seed=dataset_params["random_seed_train"])
    train_normal = gen_normal.run_simulation_normal(dataset_params["train_size"])
    train_normal_val = gen_normal.run_simulation_normal(dataset_params["train_val_size"], seed_to_add=88)

    params_rct = deepcopy(dataset_params)
    params_rct["intervention_info"]["RCT"] = True
    params_rct["random_seed_train"] = dataset_params["random_seed_train"] * 10
    params_rct["simulation_start"] = deepcopy(gen_normal.simulation_end)
    print(f"Simulating {args.train_size} + {args.train_val_size} RCT cases ...")
    gen_rct = simulation.PresProcessGenerator(params_rct, None, seed=params_rct["random_seed_train"])
    train_rct = gen_rct.run_simulation_normal(params_rct["train_size"])
    train_rct_val = gen_rct.run_simulation_normal(params_rct["train_val_size"], seed_to_add=88)

    proxy_sources = [c.strip() for c in args.proxy_cols.split(",") if c.strip()]
    bad = [c for c in proxy_sources if c not in PROXY_SOURCES]
    if bad:
        raise ValueError(f"--proxy_cols {bad} not in {PROXY_SOURCES}")
    noises = [float(n) for n in args.proxy_noise.split(",") if n.strip()] if proxy_sources else [None]
    max_noise = 0.5 if args.proxy_type == "twins" else np.inf
    if proxy_sources and any(not 0.0 <= n <= max_noise for n in noises):
        raise ValueError("--proxy_noise: flip probabilities in [0, 0.5] (twins) or sds >= 0 (gaussian)")
    ref = proxy_reference([train_normal, train_rct], proxy_sources, args.proxy_bins) if proxy_sources else None
    base = (train_normal, train_rct, train_normal_val, train_rct_val)

    for noise in noises:
        proxy_tag = ""
        sets = base
        if proxy_sources:
            kw = dict(sources=proxy_sources, ref=ref, proxy_type=args.proxy_type, noise=noise,
                      n_bins=args.proxy_bins, n_reps=args.proxy_reps)
            sets = tuple(add_proxies(d, seed=args.proxy_seed + i, **kw) for i, d in enumerate(base))
            proxy_tag = f"_proxy-{'-'.join(proxy_sources)}-{args.proxy_type}{noise:g}"
            cols = sum((proxy_columns(s, args.proxy_type, args.proxy_bins, args.proxy_reps) for s in proxy_sources), [])
            print(f"proxies ({args.proxy_type}, noise {noise:g}): {len(cols)} columns, e.g. {cols[:3]}")
            for s in proxy_sources:
                z = case_values(sets[1], s)
                p = sets[1].groupby("case_nr")[proxy_columns(s, args.proxy_type, args.proxy_bins, args.proxy_reps)].first()
                score = p.to_numpy() @ np.tile(np.arange(args.proxy_bins), args.proxy_reps) / args.proxy_reps \
                    if args.proxy_type == "twins" else p.iloc[:, 0].to_numpy()
                ok = np.isfinite(score) & np.isfinite(z.reindex(p.index).to_numpy(float))
                print(f"  corr(proxy, {s}) on RCT cases: {np.corrcoef(score[ok], z.reindex(p.index)[ok])[0, 1]:.2f}")
        tr_normal, tr_rct, tr_normal_val, tr_rct_val = sets

        for delta in deltas:
            name = simbank_dataset_name(intervention_tag, delta) + proxy_tag
            np.random.seed(42)
            train = confounding_level.set_delta(data=tr_normal, data_RCT=tr_rct, delta=delta)
            train = finalize(train, not args.no_contact_flags)
            csv_path = os.path.join(data_folder, f"{name}.csv")
            train.to_csv(csv_path, sep=";", index=False)
            if proxy_sources:
                save_spec(make_spec(proxy_sources, args.proxy_type, noise, args.proxy_bins, args.proxy_reps,
                                    {s: {"edges": np.asarray(r["edges"]).tolist(), "mean": float(r["mean"]),
                                         "sd": float(r["sd"])} for s, r in ref.items()}), csv_path)

            msg = f"[{name}] {train['case_nr'].nunique()} cases -> {csv_path}"
            if args.save_val:
                np.random.seed(42)
                val = confounding_level.set_delta(data=tr_normal_val, data_RCT=tr_rct_val, delta=delta)
                val_path = os.path.join(data_folder, f"{name}_val.csv")
                finalize(val, not args.no_contact_flags).to_csv(val_path, sep=";", index=False)
                msg += f"; val -> {val_path}"
            print(msg)


if __name__ == "__main__":
    main()
