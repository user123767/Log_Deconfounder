import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch

from MDP_functions import *
from dataset_manager import dataset_confs
from cevae import generate_unconfounded
from seq_cevae import train_seq_cevae
from select_exclude_col_single import INTERVENTIONS, OUTCOME_FUNCS, first_position, occurs, ID, ACT


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def parse_args():
    p = argparse.ArgumentParser(description="Sequence CEVAE on a real-world log (one intervention per case).")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--exclude_cols", type=str, default=None, help="comma-separated hidden columns (original names)")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./results_realworld_seqcevae")
    p.add_argument("--file_prefix", type=str, default="cevae")
    p.add_argument("--order", type=int, default=1)
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--mdp_model", type=str, default="MDP")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--z_dim", type=int, default=20)
    p.add_argument("--hidden", type=int, default=200)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--hist_hidden", type=int, default=64)
    p.add_argument("--hist_layers", type=int, default=1)
    p.add_argument("--decode_hist", type=str2bool, default=True)
    p.add_argument("--hist_weight", type=float, default=1.0)
    p.add_argument("--max_hist_len", type=int, default=0, help="keep only the last N activities (0 = all)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--val_share", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--n_effect_samples", type=int, default=100)
    p.add_argument("--sampling", type=str, default="uniform", choices=["uniform", "all"])
    p.add_argument("--n_per_case", type=int, default=1)
    p.add_argument("--z_from", type=str, default="posterior", choices=["posterior", "x"])
    return p.parse_args()


def feature_layout(env, hidden):
    df = env.train_df
    cols = [c for c in env.state_cols if c in df.columns]

    def belongs(c, h):
        return c == h or c.startswith(f"{h}_") or c.startswith(f"{h}_lag")
    cols = [c for c in cols if not any(belongs(c, h) for h in hidden)]
    pos = {c: i for i, c in enumerate(cols)}
    onehot_groups, used = [], set()
    for o in env.onehot_cols:
        g = [pos[c] for c in cols if c.startswith(f"{o}_") and "_lag" not in c[len(o):]]
        if len(g) > 1:
            onehot_groups.append(g)
            used |= set(g)
    scalar_cat = [(pos[c], int(n)) for c, n in getattr(env, "embedding_cols", {}).items() if c in pos]
    used |= {i for i, _ in scalar_cat}
    continuous = [i for i in range(len(cols)) if i not in used]
    return cols, continuous, onehot_groups, scalar_cat


def build_cases(env, spec, cols, max_hist_len):
    act_index = {a: i for i, a in enumerate(env.activity_index)}
    n_act = len(act_index)
    X, H, L, T, Y, ids, k_pos = [], [], [], [], [], [], []
    for cid, case in env.train_df.groupby(ID, sort=False):
        case = case.reset_index(drop=True)
        k = first_position(case, spec["decision"])
        if k is None:
            continue
        stop = None
        if spec.get("treat_before"):
            later = [first_position(case.iloc[k + 1:].reset_index(drop=True), a) for a in spec["treat_before"]]
            later = [k + 1 + p for p in later if p is not None]
            stop = min(later) if later else None
        t = int(occurs(case, spec["treatment"], after=k, before=stop))
        kind, val = spec["outcome"]
        if kind == "activity_good":
            y = int(any(occurs(case, a) for a in val))
        elif kind == "activity_bad":
            y = int(not any(occurs(case, a) for a in val))
        elif kind == "func":
            y = int(OUTCOME_FUNCS[val](case))
        else:
            lc = env.dataset_manager.label_col
            y = int(case[lc].iloc[-1] == val)
        hist = [act_index.get(a, n_act) for a in case[ACT].iloc[:k]]
        if max_hist_len:
            hist = hist[-max_hist_len:]
        X.append(case.loc[k, cols].to_numpy(float))
        H.append(hist); L.append(len(hist)); T.append(t); Y.append(y); ids.append(cid); k_pos.append(k)
    Lmax = max(1, max(L))
    Hm = np.zeros((len(H), Lmax), np.int64)
    for i, h in enumerate(H):
        Hm[i, :len(h)] = h
    return (np.nan_to_num(np.asarray(X, np.float32)), Hm, np.asarray(L, np.int64), np.asarray(T),
            np.asarray(Y, np.float64), np.asarray(ids), np.asarray(k_pos), n_act + 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    hidden = [c.strip() for c in args.exclude_cols.split(",") if c.strip()] if args.exclude_cols else []
    run = args.run_name or ("+".join(hidden) if hidden else "no_exclusion")
    out = lambda kind, ext: os.path.join(args.output_dir, f"{args.file_prefix}_{kind}_{run}.{ext}")
    base = dataset_confs.base_dataset(args.dataset)
    if base not in INTERVENTIONS:
        raise KeyError(f"no intervention defined for '{base}' (select_exclude_col_single.INTERVENTIONS)")
    spec = INTERVENTIONS[base]
    print(f"=== sequence CEVAE on {args.dataset} (intervention: {spec['name']}), hidden={hidden}, run={run} ===")

    env = ProcessEnv(dataset=args.dataset, order=args.order, train_fraction=args.train_fraction,
                     MDP_model=args.mdp_model)
    missing = [h for h in hidden if h not in env.train_df.columns
               and not any(c.startswith(f"{h}_") for c in env.train_df.columns)]
    if missing:
        raise KeyError(f"hidden columns {missing} not in the log")
    cols, cont, onehot, scalar_cat = feature_layout(env, hidden)
    proxies = [c for c in cols if c.startswith("proxy_")]
    x_cur, hist, hlen, t, y, case_ids, k_pos, n_act = build_cases(env, spec, cols, args.max_hist_len)
    print(f"{len(x_cur)} cases at the decision point; x_cur dim {x_cur.shape[1]} ({len(proxies)} proxy columns); "
          f"history {hlen.mean():.1f} activities on average (max {hlen.max()}); treated share {t.mean():.3f}; "
          f"positive outcome {y.mean():.3f} (t=0: {y[t == 0].mean():.3f}, t=1: {y[t == 1].mean():.3f})")

    model, history = train_seq_cevae(
        x_cur, hist, hlen, t, y, cont, onehot, scalar_cat, 2, n_act, z_dim=args.z_dim, hidden=args.hidden,
        n_layers=args.n_layers, hist_hidden=args.hist_hidden, hist_layers=args.hist_layers,
        decode_hist=args.decode_hist, hist_weight=args.hist_weight, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, epochs=args.epochs, val_share=args.val_share, patience=args.patience,
        device=args.device, seed=args.seed)
    torch.save(model.state_dict(), out("model", "pt"))
    pd.DataFrame(history).to_csv(out("history", "csv"), index=False)
    with open(out("config", "json"), "w") as f:
        json.dump({**vars(args), "base_dataset": base, "intervention": spec, "hidden": hidden,
                   "features": cols, "proxy_columns": proxies, "n_activities": n_act,
                   "max_hist_len_seen": int(hlen.max())}, f, indent=2, default=str)

    x = model.all_features(x_cur, hist, hlen)
    X = torch.as_tensor(x, device=args.device)
    po = model.potential_outcomes(X, args.n_effect_samples).cpu().numpy() * float(model.y_std) + float(model.y_mean)
    diag = {"dataset": args.dataset, "intervention": spec["name"], "n_cases": int(len(x)),
            "treated_share": float(t.mean()),
            "naive_mean_y": [float(y[t == k].mean()) if (t == k).any() else None for k in (0, 1)],
            "cevae_mean_y_do": po.mean(0).tolist(), "best_epoch": int(history[-1]["best_epoch"])}
    diag["naive_effect"] = (diag["naive_mean_y"][1] or 0) - (diag["naive_mean_y"][0] or 0)
    diag["cevae_ate"] = float((po[:, 1] - po[:, 0]).mean())
    zx = model.sample_z_given_x(X, 20).mean(0).cpu().numpy()
    if hidden:
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import cross_val_score
        for h in hidden:
            if h in env.train_df.columns and pd.api.types.is_numeric_dtype(env.train_df[h]):
                vals = np.array([env.train_df.loc[env.train_df[ID] == c, h].iloc[k] for c, k in zip(case_ids, k_pos)],
                                float)
                ok = np.isfinite(vals)
                if ok.sum() > 50:
                    diag[f"r2_{h}_from_z"] = float(cross_val_score(RidgeCV(), zx[ok], vals[ok], cv=5,
                                                                   scoring="r2").mean())
    pd.DataFrame(zx, columns=[f"Z_{j}" for j in range(zx.shape[1])]).assign(
        case_id=case_ids, logged_option=t).to_csv(out("z", "csv"), index=False)
    print(json.dumps(diag, indent=2))
    with open(out("diagnostics", "json"), "w") as f:
        json.dump(diag, f, indent=2)

    unit, t_new, y_new = generate_unconfounded(model, x, t, y, n_per_unit=args.n_per_case, sampling=args.sampling,
                                               z_from=args.z_from, device=args.device, seed=args.seed)
    pd.DataFrame({"case_id": case_ids[unit], "logged_option": t[unit], "logged_y": y[unit], "option": t_new,
                  "y": y_new}).to_csv(out("static", "csv"), index=False)
    np.savez_compressed(out("cases", "npz"), x_cur=x_cur, hist=hist, hist_len=hlen, t=t, y=y, case_ids=case_ids,
                        decision_pos=k_pos, features=np.asarray(cols))
    print(f"generated {len(t_new)} rows; mean y' per option "
          f"{[round(float(y_new[t_new == k].mean()), 3) for k in (0, 1)]}")


if __name__ == "__main__":
    main()
