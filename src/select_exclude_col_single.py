import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from MDP_functions import *


INTERVENTIONS = {
    "bpic2017_accepted": dict(
        name="additional offer",
        decision="O_Create Offer",
        treatment="O_Create Offer",
        outcome=("activity_good", ["A_Pending"]),
    ),
    "bpic2012_accepted": dict(
        name="additional offer",
        decision="O_CREATED*",
        treatment="O_CREATED*",
        outcome=("activity_good", ["A_APPROVED*"]),
    ),
    "traffic_fines_1": dict(
        name="add penalty",
        decision="Insert Fine Notification",
        treatment="Add penalty",
        outcome=("activity_bad", ["Send for Credit Collection"]),
    ),
    "sepsis_cases_1": dict(
        name="IV antibiotics before admission/release",
        decision="ER Triage",
        treatment="IV Antibiotics",
        treat_before=["Admission NC", "Admission IC", "Release*"],
        outcome=("func", "sepsis_no_return_28d"),
    ),
}

ID, ACT = "ID", "action"


def sepsis_no_return_28d(case):
    if "Return ER" not in case.columns:
        raise ValueError("sepsis outcome: no 'Return ER' column in the log")
    ret = case["Return ER"].astype(float).to_numpy()
    if not (ret > 0).any():
        return 1
    first = int(np.flatnonzero(ret > 0)[0])
    recent = float(case["recent_release"].iloc[first]) if "recent_release" in case.columns else 1.0
    return 0 if recent > 0 else 1


OUTCOME_FUNCS = {"sepsis_no_return_28d": sepsis_no_return_28d}


def parse_args():
    p = argparse.ArgumentParser(description="Rank confounder candidates for a single intervention.")
    p.add_argument("--dataset", type=str, required=True, choices=sorted(INTERVENTIONS))
    p.add_argument("--order", type=int, default=1)
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--mdp_model", type=str, default="MDP")
    p.add_argument("--exclude_from_candidates", type=str, nargs="*", default=[],
                   help="columns (original names) never to rank, e.g. ones the outcome definition reads")
    p.add_argument("--keep_control_flow", action="store_true",
                   help="also rank last_action / control-flow columns (excluded by default: they describe the "
                        "path, not a removable attribute)")
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--n_repeats", type=int, default=10)
    p.add_argument("--max_depth", type=int, default=12, help="0 = unbounded")
    p.add_argument("--min_samples_leaf", type=int, default=20)
    p.add_argument("--test_size", type=float, default=0.3)
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--list_activities", action="store_true", help="print the activities and stop")
    p.add_argument("--output_dir", type=str, default="./results")
    return p.parse_args()


def matches(series, pattern):
    s = series.astype(str)
    return s.str.startswith(pattern[:-1]) if pattern.endswith("*") else s.eq(pattern)


def flag_column(case, activity):
    name = activity[:-1] if activity.endswith("*") else activity
    return case[name] if name in case.columns else None


def occurs(case, pattern, after=None, before=None):
    lo = -1 if after is None else after
    hi = len(case) if before is None else before
    window = case.iloc[lo + 1:hi]
    if len(window) and matches(window[ACT], pattern).any():
        return True
    flag = flag_column(case, pattern)
    if flag is not None and len(window):
        base = flag.iloc[lo] if lo >= 0 else 0
        return bool((window[flag.name].astype(float) > float(base)).any())
    return False


def first_position(case, pattern):
    pos = np.flatnonzero(matches(case[ACT], pattern).to_numpy())
    if len(pos):
        return int(pos[0])
    flag = flag_column(case, pattern)
    if flag is not None:
        pos = np.flatnonzero(flag.astype(float).to_numpy() > 0)
        if len(pos):
            return int(pos[0])
    return None


def build_case_table(env, spec, feature_cols):
    df = env.train_df
    rows, n_cases, n_no_dp = [], 0, 0
    label_col = getattr(env.dataset_manager, "label_col", None)
    for cid, case in df.groupby(ID, sort=False):
        n_cases += 1
        case = case.reset_index(drop=True)
        k = first_position(case, spec["decision"])
        if k is None:
            n_no_dp += 1
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
        elif kind == "label":
            if label_col is None or label_col not in case.columns:
                raise ValueError("outcome ('label', ...) but the log has no label column")
            y = int(case[label_col].iloc[-1] == val)
        else:
            raise ValueError(f"unknown outcome kind {kind}")
        rows.append({ID: cid, "t": t, "y": y, **case.loc[k, feature_cols].to_dict()})
    print(f"{n_cases} cases; {n_no_dp} never reach the decision point '{spec['decision']}' (dropped); "
          f"{len(rows)} kept")
    return pd.DataFrame(rows)


def column_groups(env, args):
    state_cols = [c for c in env.state_cols if c in env.train_df.columns]
    groups = {}
    for col in state_cols:
        src = next((o for o in env.onehot_cols if col.startswith(f"{o}_")), col)
        groups.setdefault(src, []).append(col)
    drop = set(args.exclude_from_candidates)
    if not args.keep_control_flow:
        drop |= {"last_action"} | set(getattr(env.dataset_manager, "control_flow_var", []) or [])
    return {g: cols for g, cols in groups.items() if g not in drop}


def case_level(df, cols):
    return all(df.groupby(ID)[c].nunique(dropna=False).max() <= 1 for c in cols)


def grouped_permutation_importance(model, X, target, groups, n_repeats, rng, extra=None):
    def auc(Xm):
        Z = Xm if extra is None else np.c_[Xm, extra]
        return roc_auc_score(target, model.predict_proba(Z)[:, 1])
    base = auc(X)
    col_idx = {c: i for i, c in enumerate(X.columns)}
    out = {}
    Xv = X.to_numpy(copy=True)
    for g, cols in groups.items():
        idx = [col_idx[c] for c in cols]
        drops = []
        for _ in range(n_repeats):
            Xp = Xv.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, idx] = Xp[perm][:, idx]
            drops.append(base - auc(pd.DataFrame(Xp, columns=X.columns)))
        out[g] = float(np.mean(drops))
    return base, out


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    spec = INTERVENTIONS[args.dataset]
    print(f"=== Confounder candidates for {args.dataset}, intervention: {spec['name']} ===")
    env = ProcessEnv(dataset=args.dataset, order=args.order, train_fraction=args.train_fraction,
                     MDP_model=args.mdp_model)

    if args.list_activities:
        print(env.train_df[ACT].astype(str).value_counts().to_string())
        flags = [c for c in getattr(env.dataset_manager, "control_flow_var", []) or [] if c in env.train_df]
        print("activities turned into state flags:", flags)
        return
    for key in ["decision", "treatment"]:
        a = spec[key]
        if not matches(env.train_df[ACT], a).any() and flag_column(env.train_df, a) is None:
            raise ValueError(f"{key} activity '{a}' not found in {args.dataset} -- run with --list_activities")

    groups = column_groups(env, args)
    feature_cols = [c for cols in groups.values() for c in cols]
    table = build_case_table(env, spec, feature_cols)
    X = table[feature_cols].astype(float).fillna(0.0)
    t, y = table["t"].to_numpy(), table["y"].to_numpy()
    print(f"treated share {t.mean():.3f}; positive-outcome share {y.mean():.3f}; "
          f"y | t=0 {y[t == 0].mean():.3f}, y | t=1 {y[t == 1].mean():.3f}")
    if min(t.mean(), 1 - t.mean()) < 0.02 or min(y.mean(), 1 - y.mean()) < 0.02:
        print("WARNING: t or y is almost constant -- the ranking will be unreliable")

    idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=args.test_size, random_state=args.seed,
                                      stratify=t if 0 < t.mean() < 1 else None)
    rf = dict(n_estimators=args.n_estimators, max_depth=args.max_depth or None,
              min_samples_leaf=args.min_samples_leaf, n_jobs=args.n_jobs, random_state=args.seed,
              class_weight="balanced")
    rng = np.random.default_rng(args.seed)

    m_t = RandomForestClassifier(**rf).fit(X.iloc[idx_tr], t[idx_tr])
    auc_t, imp_t = grouped_permutation_importance(m_t, X.iloc[idx_te].reset_index(drop=True), t[idx_te],
                                                  groups, args.n_repeats, rng)
    m_y = RandomForestClassifier(**rf).fit(np.c_[X.iloc[idx_tr], t[idx_tr]], y[idx_tr])
    auc_y, imp_y = grouped_permutation_importance(m_y, X.iloc[idx_te].reset_index(drop=True), y[idx_te],
                                                  groups, args.n_repeats, rng, extra=t[idx_te])
    print(f"held-out ROC AUC: intervention model {auc_t:.3f} (overlap: closer to 0.5 is better), "
          f"outcome model {auc_y:.3f}")

    a = pd.Series(imp_t).clip(lower=0)
    o = pd.Series(imp_y).clip(lower=0)
    a_n = a / a.max() if a.max() > 0 else a
    o_n = o / o.max() if o.max() > 0 else o
    ranking = pd.DataFrame({
        "column": list(groups),
        "n_model_cols": [len(groups[g]) for g in groups],
        "case_level": [case_level(env.train_df, groups[g]) for g in groups],
        "action_importance": [imp_t[g] for g in groups],
        "outcome_importance": [imp_y[g] for g in groups],
    })
    ranking["confounder_score"] = np.sqrt(a_n.reindex(ranking["column"]).to_numpy()
                                          * o_n.reindex(ranking["column"]).to_numpy())
    ranking = ranking.sort_values("confounder_score", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(args.output_dir, f"confounder_candidates_{args.dataset}_single.csv")
    ranking.to_csv(csv_path, index=False)
    with open(csv_path.replace(".csv", "_info.txt"), "w") as f:
        f.write(f"intervention: {spec}\ncases kept: {len(table)}\ntreated share: {t.mean():.4f}\n"
                f"positive-outcome share: {y.mean():.4f}\nAUC t-model: {auc_t:.4f}\nAUC y-model: {auc_y:.4f}\n")
    print(f"Saved ranking ({len(ranking)} columns) to {csv_path}\n")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(ranking.head(args.top_k).to_string(index=False))
    if len(ranking):
        best = ranking.iloc[0]
        print(f"\nBest candidate: '{best['column']}' (score {best['confounder_score']:.3f}, "
              f"{'case-level' if best['case_level'] else 'event-level'}) -> --exclude_cols {best['column']}")


if __name__ == "__main__":
    main()
