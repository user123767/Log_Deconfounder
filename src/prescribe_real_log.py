import argparse
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from MDP_functions import *
from dataset_manager import dataset_confs
from select_exclude_col_single import INTERVENTIONS
from seq_cevae_log import feature_layout, build_cases
from real_log_agents import make_agent
from fqe_eval import evaluate

AGENT_TAG = {"seq_vi": "seqvi", "causal_rl": "crl"}


def parse_args():
    p = argparse.ArgumentParser(description="PresPM agents on a real log, FQE evaluation (one intervention per case).")
    p.add_argument("--dataset", type=str, required=True, help="log name, e.g. bpic2017_accepted_proxy-CreditScore-gaussian0.3")
    p.add_argument("--exclude_cols", type=str, default=None, help="comma-separated hidden columns")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--agent", type=str, default="cql", choices=["cql", "seq_vi", "causal_rl"])
    p.add_argument("--train_data", type=str, default="logged", choices=["logged", "cevae"])
    p.add_argument("--cevae_dir", type=str, default=None, help="folder of the seq_cevae_log.py outputs (default: --output_dir)")
    p.add_argument("--cevae_prefix", type=str, default="cevae")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--order", type=int, default=1)
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--mdp_model", type=str, default="MDP")
    p.add_argument("--max_hist_len", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cql_alpha", type=float, default=1.0)
    p.add_argument("--cql_hidden", type=int, default=128)
    p.add_argument("--cql_updates", type=int, default=20_000)
    p.add_argument("--cql_batch", type=int, default=256)
    p.add_argument("--cql_lr", type=float, default=1e-3)
    p.add_argument("--seqvi_n_clusters", type=int, default=None)
    p.add_argument("--crl_alpha", type=float, default=0.1)
    p.add_argument("--crl_n_boot", type=int, default=10)
    p.add_argument("--crl_conformal_alpha", type=float, default=0.1)
    p.add_argument("--crl_effect_model", type=str, default="tlearner", choices=["tlearner", "forest"])
    p.add_argument("--crl_ppo_iters", type=int, default=200)
    p.add_argument("--crl_ppo_batch", type=int, default=2048)
    p.add_argument("--crl_hidden", type=int, default=64)
    p.add_argument("--fqe_state", type=str, default="full", choices=["full", "observed"],
                   help="full: evaluator also sees the hidden columns; observed: same attributes as the agents")
    p.add_argument("--fqe_folds", type=int, default=5)
    p.add_argument("--fqe_boot", type=int, default=1000)
    p.add_argument("--fqe_clip", type=float, default=0.05)
    return p.parse_args()


def case_features(x, hist, hlen, n_ids):
    n = len(x)
    freq, lastpos = np.zeros((n, n_ids)), np.zeros((n, n_ids))
    last = np.full(n, -1, np.int64)
    for i in range(n):
        h = hist[i, :hlen[i]]
        if len(h):
            freq[i] = np.bincount(h, minlength=n_ids)[:n_ids] / len(h)
            for j, act in enumerate(h):
                lastpos[i, act] = (j + 1) / len(h)
            last[i] = h[-1]
    Lmax = max(1, int(hlen.max()))
    cf = np.c_[freq, lastpos, hlen / Lmax].astype(np.float32)
    return {"x": np.asarray(x, np.float32), "cf": cf, "last": last, "L": hlen.astype(np.float64)}


def cases(env, df, spec, cols, max_hist_len):
    shim = SimpleNamespace(train_df=df, activity_index=env.activity_index, dataset_manager=env.dataset_manager)
    return build_cases(shim, spec, cols, max_hist_len)


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    hidden = [c.strip() for c in args.exclude_cols.split(",") if c.strip()] if args.exclude_cols else []
    run = args.run_name or ("+".join(hidden) if hidden else "no_exclusion")
    agent_tag = f"cql{args.cql_alpha:g}" if args.agent == "cql" else AGENT_TAG[args.agent]
    tag = f"{run}_{args.train_data}_treat_{agent_tag}"
    out = lambda kind, ext: os.path.join(args.output_dir, f"fqe_{kind}_{tag}.{ext}")
    base = dataset_confs.base_dataset(args.dataset)
    if base not in INTERVENTIONS:
        raise KeyError(f"no intervention defined for '{base}' (select_exclude_col_single.INTERVENTIONS)")
    spec = INTERVENTIONS[base]
    print(f"=== {args.agent} on {args.dataset} ({spec['name']}), hidden={hidden}, train data={args.train_data} ===")

    env = ProcessEnv(dataset=args.dataset, order=args.order, train_fraction=args.train_fraction,
                     MDP_model=args.mdp_model)
    cols, *_ = feature_layout(env, hidden)
    full_cols, *_ = feature_layout(env, [])
    x_tr, h_tr, l_tr, t_tr, y_tr, id_tr, _, n_ids = cases(env, env.train_df, spec, cols, args.max_hist_len)
    x_te, h_te, l_te, t_te, y_te, id_te, _, _ = cases(env, env.test_df, spec, cols, args.max_hist_len)
    F_tr, F_te = case_features(x_tr, h_tr, l_tr, n_ids), case_features(x_te, h_te, l_te, n_ids)
    print(f"train: {len(t_tr)} cases (treated {t_tr.mean():.3f}, positive {y_tr.mean():.3f}); "
          f"test: {len(t_te)} cases (treated {t_te.mean():.3f}, positive {y_te.mean():.3f}); "
          f"agent state: {len(cols)} attributes ({sum(c.startswith('proxy_') for c in cols)} proxies)")

    if args.train_data == "logged":
        F_fit, a_fit, r_fit = F_tr, t_tr.astype(int), y_tr.astype(float)
    else:
        cdir = args.cevae_dir or args.output_dir
        path = os.path.join(cdir, f"{args.cevae_prefix}_static_{run}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} -- run seq_cevae_log.py first (same dataset / hidden set / seed)")
        gen = pd.read_csv(path)
        row = {str(c): i for i, c in enumerate(id_tr)}
        keep = gen["case_id"].astype(str).map(row)
        if keep.isna().mean() > 0.01:
            raise ValueError(f"{keep.isna().mean():.1%} of the generated rows have no training case: "
                             "the CEVAE was trained on another split / log")
        gen, idx = gen[keep.notna()], keep.dropna().astype(int).to_numpy()
        F_fit = {k: v[idx] for k, v in F_tr.items()}
        a_fit, r_fit = gen["option"].to_numpy(int), gen["y"].to_numpy(float)
        print(f"deconfounded data: {len(gen)} rows from {path}; option shares {np.bincount(a_fit, minlength=2) / len(a_fit)}")

    agent = make_agent(args.agent, 2, args)
    history = agent.fit(F_fit, a_fit, r_fit)
    P = agent.probs(F_te)

    if args.fqe_state == "full":
        xf_te = cases(env, env.test_df, spec, full_cols, args.max_hist_len)[0]
        S_eval = np.c_[xf_te, F_te["cf"]]
    else:
        S_eval = np.c_[F_te["x"], F_te["cf"]]
    n = len(t_te)
    policies = {"learned": P, "always_0": np.tile([1.0, 0.0], (n, 1)), "always_1": np.tile([0.0, 1.0], (n, 1)),
                "random": np.full((n, 2), 0.5)}
    rows, diag = evaluate(policies, S_eval.astype(np.float64), t_te.astype(int), y_te.astype(float),
                          K=2, n_folds=args.fqe_folds, seed=args.seed, n_boot=args.fqe_boot, clip=args.fqe_clip)
    summary = pd.DataFrame(rows).assign(dataset=args.dataset, hidden=run, agent=args.agent, train_data=args.train_data,
                                        seed=args.seed, fqe_state=args.fqe_state)
    summary.to_csv(out("summary", "csv"), index=False)
    pd.DataFrame({"case_id": id_te, "logged_option": t_te, "logged_y": y_te,
                  "option": P.argmax(1)}).to_csv(out("policy", "csv"), index=False)
    with open(out("diagnostics", "json"), "w") as f:
        json.dump({"args": vars(args), "agent_diagnostics": getattr(agent, "diagnostics", {}),
                   "fallback_share": getattr(agent, "n_fallback", 0) / max(1, n),
                   "knn_share": getattr(agent, "knn_share", None), "evaluator": diag,
                   "agent_state_columns": cols, "evaluator_state_columns": full_cols if args.fqe_state == "full" else cols,
                   "train_history_tail": history[-3:] if isinstance(history, list) else None},
                  f, indent=2, default=str)
    print(summary[["policy", "fqe", "fqe_lo", "fqe_hi", "dr", "treat_share"]].round(4).to_string(index=False))
    print(f"-> {out('summary', 'csv')}")


if __name__ == "__main__":
    main()
