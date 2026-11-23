import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

_REG = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=20)


def fit_nuisances(S, a, y, K=2, n_folds=5, seed=0, min_cases=20):
    n = len(S)
    Q = np.full((n, K), np.nan)
    E = np.full((n, K), np.nan)
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(S):
        for k in range(K):
            idx = tr[a[tr] == k]
            if len(idx) >= min_cases:
                Q[te, k] = HistGradientBoostingRegressor(random_state=seed, **_REG).fit(S[idx], y[idx]).predict(S[te])
            else:
                Q[te, k] = y[idx].mean() if len(idx) else y[tr].mean()
        if len(np.unique(a[tr])) > 1:
            clf = HistGradientBoostingClassifier(random_state=seed, **_REG).fit(S[tr], a[tr])
            pr = np.zeros((len(te), K))
            pr[:, clf.classes_] = clf.predict_proba(S[te])
            E[te] = pr
        else:
            E[te] = np.eye(K)[a[tr][0]]
    return Q, E


def evaluate(policies, S, a, y, K=2, n_folds=5, seed=0, n_boot=1000, clip=0.05):
    a = np.asarray(a, int); y = np.asarray(y, float)
    Q, E = fit_nuisances(S, a, y, K, n_folds, seed)
    Ec = np.clip(E, clip, 1 - clip)
    rng = np.random.default_rng(seed)
    boots = rng.integers(0, len(y), (n_boot, len(y)))
    rows = [{"policy": "logged", "fqe": float(y.mean()), "fqe_lo": float(np.percentile(y[boots].mean(1), 2.5)),
             "fqe_hi": float(np.percentile(y[boots].mean(1), 97.5)), "dr": float(y.mean()),
             "treat_share": float(np.mean(a > 0)), "n_test": int(len(y))}]
    for name, P in policies.items():
        P = np.asarray(P, float)
        v_fqe = (P * Q).sum(1)
        w = P[np.arange(len(a)), a] / Ec[np.arange(len(a)), a]
        v_dr = v_fqe + w * (y - Q[np.arange(len(a)), a])
        bf = v_fqe[boots].mean(1)
        rows.append({"policy": name, "fqe": float(v_fqe.mean()), "fqe_lo": float(np.percentile(bf, 2.5)),
                     "fqe_hi": float(np.percentile(bf, 97.5)), "dr": float(v_dr.mean()),
                     "treat_share": float(P[:, 1:].sum(1).mean()), "n_test": int(len(y)),
                     "match_logged": float(P[np.arange(len(a)), a].mean())})
    diag = {"q_mean_per_option": np.nanmean(Q, 0).tolist(), "propensity_mean": E.mean(0).tolist(),
            "share_clipped": float(np.mean((E < clip) | (E > 1 - clip)))}
    return rows, diag
