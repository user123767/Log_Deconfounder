import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch

from MDP_functions import *
from two_way_deconfounder import build_two_way_dataset
from prefix_deconfounder import build_prefix_dataset
from Log_deconfounder import TreatmentSpec, load_dataset_params, str2bool
from cevae import generate_unconfounded
from seq_cevae import train_seq_cevae


def parse_args():
    p = argparse.ArgumentParser(description="Sequence CEVAE on SimBank (one intervention per case).")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--dataset_params", type=str, required=True)
    p.add_argument("--exclude_cols", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./results_seq_cevae")
    p.add_argument("--file_prefix", type=str, default="seqcevae",
                   help="output names <prefix>_*_<run>; 'cevae' (with its own output root) for the agent script.")
    p.add_argument("--order", type=int, default=1)
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--mdp_model", type=str, default="MDP")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--z_dim", type=int, default=20)
    p.add_argument("--hidden", type=int, default=200)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--hist_hidden", type=int, default=64, help="size of E_hist(history), appended to x_cur")
    p.add_argument("--hist_layers", type=int, default=1)
    p.add_argument("--decode_hist", type=str2bool, default=True, help="false: encoder only, no p(history | z)")
    p.add_argument("--hist_weight", type=float, default=1.0, help="weight of log p(history | z) in the ELBO")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--val_share", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--n_effect_samples", type=int, default=100, help="z samples for the effect estimates.")
    p.add_argument("--sampling", type=str, default="uniform", choices=["uniform", "all"])
    p.add_argument("--n_per_case", type=int, default=1)
    p.add_argument("--z_from", type=str, default="posterior", choices=["posterior", "x"])
    p.add_argument("--make_agent_dataset", type=str2bool, default=True)
    return p.parse_args()


def activity_sequences(data):
    order = np.lexsort((data["time_idx"], data["case_idx"]))
    case, act = data["case_idx"][order], data["action"][order]
    bounds = np.flatnonzero(np.r_[True, case[1:] != case[:-1], True])
    first = int(data["n_actions"])
    return {int(case[b0]): np.r_[first, act[b0:b1]].astype(np.int64) for b0, b1 in zip(bounds[:-1], bounds[1:])}


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    hidden_cols = [c.strip() for c in args.exclude_cols.split(",") if c.strip()] if args.exclude_cols else []
    run = args.run_name or ("+".join(hidden_cols) if hidden_cols else "no_exclusion")
    out = lambda kind, ext: os.path.join(args.output_dir, f"{args.file_prefix}_{kind}_{run}.{ext}")
    print(f"=== sequence CEVAE on {args.dataset}, hidden={hidden_cols}, run={run} ===")

    env = ProcessEnv(dataset=args.dataset, order=args.order, train_fraction=args.train_fraction,
                     MDP_model=args.mdp_model)
    data = build_two_way_dataset(env, exclude_cols=hidden_cols or None, first_row=0)
    spec = TreatmentSpec(load_dataset_params(args.dataset_params)["intervention_info"], env.activity_index)
    if spec.n_stages != 1:
        raise ValueError(f"CEVAE is static: one intervention per case, got {spec.names}. "
                         f"Use a dataset with a single intervention (e.g. choose_procedure).")
    K = len(spec.stage_slots[0])

    pdata = build_prefix_dataset(data, spec)
    rows = np.flatnonzero(pdata["dec_pos"] == 0)
    c, p = pdata["c"][rows], pdata["p"][rows]
    case_idx = pdata["case_idx"][rows]
    x_cur = pdata["case_states"][c, p].astype(np.float32)
    acts = activity_sequences(data)
    n_activities = int(data["n_actions"]) + 1
    L = max(1, int(p.max()))
    hist = np.zeros((len(rows), L), np.int64)
    for i, (ci, pi) in enumerate(zip(case_idx, p)):
        hist[i, :pi] = acts[int(ci)][:pi]
    hlen = p.astype(np.int64)
    t = np.array([spec.stage_slots[0].index(int(s)) for s in pdata["slot"][rows]])
    y = pdata["reward"][rows].astype(np.float64)
    case_ids = np.asarray(data["case_ids"])[case_idx]
    print(f"{len(x_cur)} cases; x_cur dim {x_cur.shape[1]}; history: {hlen.mean():.1f} previous activities on "
          f"average (min {hlen.min()}, max {hlen.max()}; {n_activities} activity ids); "
          f"options {spec.options[0]} logged {np.bincount(t, minlength=K).tolist()}; y mean {y.mean():.1f} sd {y.std():.1f}")

    model, history = train_seq_cevae(
        x_cur, hist, hlen, t, y, data["continuous_idx"], data["onehot_groups"], data["scalar_cat_groups"], K,
        n_activities, z_dim=args.z_dim, hidden=args.hidden, n_layers=args.n_layers, hist_hidden=args.hist_hidden,
        hist_layers=args.hist_layers, decode_hist=args.decode_hist, hist_weight=args.hist_weight, lr=args.lr,
        weight_decay=args.weight_decay, batch_size=args.batch_size, epochs=args.epochs, val_share=args.val_share,
        patience=args.patience, device=args.device, seed=args.seed)
    torch.save(model.state_dict(), out("model", "pt"))
    pd.DataFrame(history).to_csv(out("history", "csv"), index=False)
    with open(out("config", "json"), "w") as f:
        json.dump({**vars(args), "n_activities": n_activities, "max_hist_len": L,
                   "mean_hist_len": float(hlen.mean())}, f, indent=2)

    x = model.all_features(x_cur, hist, hlen)
    X = torch.as_tensor(x, device=args.device)
    po = model.potential_outcomes(X, args.n_effect_samples).cpu().numpy() * float(model.y_std) + float(model.y_mean)
    diag = {"options": [str(o) for o in spec.options[0]], "n_cases": int(len(x)),
            "logged_share": (np.bincount(t, minlength=K) / len(t)).tolist(),
            "naive_mean_y_per_option": [float(y[t == k].mean()) if (t == k).any() else None for k in range(K)],
            "cevae_mean_y_do_option": po.mean(0).tolist(),
            "best_epoch": int(history[-1]["best_epoch"])}
    if K == 2:
        diag["naive_effect"] = diag["naive_mean_y_per_option"][1] - diag["naive_mean_y_per_option"][0]
        diag["cevae_ate"] = float((po[:, 1] - po[:, 0]).mean())
    zx = model.sample_z_given_x(X, 20).mean(0).cpu().numpy()
    if hidden_cols:
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import cross_val_score
        for col in hidden_cols:
            true = env.train_df.groupby("ID")[col].first().reindex(case_ids).to_numpy(np.float64)
            ok = np.isfinite(true)
            diag[f"r2_{col}_from_z"] = float(cross_val_score(RidgeCV(), zx[ok], true[ok], cv=5, scoring="r2").mean())
    pd.DataFrame(zx, columns=[f"Z_{j}" for j in range(zx.shape[1])]).assign(
        case_id=case_ids, logged_option=t).to_csv(out("z", "csv"), index=False)
    print(json.dumps(diag, indent=2))
    with open(out("diagnostics", "json"), "w") as f:
        json.dump(diag, f, indent=2)

    unit, t_new, y_new = generate_unconfounded(model, x, t, y, n_per_unit=args.n_per_case, sampling=args.sampling,
                                               z_from=args.z_from, device=args.device, seed=args.seed)
    pd.DataFrame({"case_id": case_ids[unit], "logged_option": t[unit], "logged_y": y[unit], "option": t_new,
                  "y": y_new}).to_csv(out("static", "csv"), index=False)
    print(f"generated {len(t_new)} rows; option shares {np.bincount(t_new, minlength=K) / len(t_new)}; "
          f"mean y per option {[round(float(y_new[t_new == k].mean()), 1) for k in range(K)]}")

    if args.make_agent_dataset:
        H, D = pdata["max_prefix_len"], pdata["token_dim"]
        n = len(unit)
        W = np.zeros((n, H, D), np.float32); Lw = np.zeros(n, np.int64)
        for i, u in enumerate(unit):
            h = pdata["case_tokens"][c[u], :p[u] + 1][-H:]
            W[i, :len(h)] = h; Lw[i] = len(h)
        ds = {"obs_window": W, "obs_window_len": Lw, "stage": np.zeros(n, np.int64),
              "slot": np.asarray([spec.stage_slots[0][k] for k in t_new], np.int64),
              "reward": y_new.astype(np.float32), "next_obs_window": W, "next_obs_window_len": Lw,
              "next_stage": np.zeros(n, np.int64), "done": np.ones(n, np.float32),
              "case_idx": case_idx[unit].astype(np.int64), "dec_pos": np.zeros(n, np.int64),
              "behavior_prob": np.full(n, 1.0 / K, np.float32), "returns": y_new.astype(np.float32),
              "n_decisions": np.ones(n, np.int64), "n_rollouts": n, "mode": "seq_cevae", "share_ended": 1.0}
        path = out("dataset", "npz")
        np.savez_compressed(path, **ds)
        print(f"agent dataset ({n} one-step episodes) -> {path}")


if __name__ == "__main__":
    main()
