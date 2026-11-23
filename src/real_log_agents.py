from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from causal_rl_agent import CausalRLAgent


def greedy(q):
    p = np.zeros_like(q, dtype=np.float64)
    p[np.arange(len(q)), np.argmax(q, 1)] = 1.0
    return p


class _Standardiser:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-8] = 1.0
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


class OneStepCQL:
    def __init__(self, K=2, alpha=1.0, hidden=128, n_updates=20_000, batch=256, lr=1e-3, seed=0):
        self.K, self.alpha, self.hidden, self.n_updates, self.batch, self.lr, self.seed = \
            K, alpha, hidden, n_updates, batch, lr, seed

    @staticmethod
    def state(Fd):
        return np.c_[Fd["x"], Fd["cf"]].astype(np.float32)

    def fit(self, Fd, a, r):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        S = self.state(Fd)
        self.sx = _Standardiser().fit(S)
        self.r_mu, self.r_sd = float(r.mean()), float(r.std()) or 1.0
        St = torch.as_tensor(self.sx(S), dtype=torch.float32)
        At = torch.as_tensor(a, dtype=torch.long)
        Rt = torch.as_tensor((r - self.r_mu) / self.r_sd, dtype=torch.float32)
        self.net = nn.Sequential(nn.Linear(S.shape[1], self.hidden), nn.ReLU(),
                                 nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.K))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        hist = []
        for it in range(self.n_updates):
            i = torch.as_tensor(rng.integers(0, len(S), self.batch))
            q = self.net(St[i])
            qa = q.gather(1, At[i, None]).squeeze(1)
            td = 0.5 * F.mse_loss(qa, Rt[i])
            cons = (torch.logsumexp(q, 1) - qa).mean()
            loss = td + self.alpha * cons
            opt.zero_grad(); loss.backward(); opt.step()
            if it % max(1, self.n_updates // 10) == 0:
                hist.append({"update": it, "td": td.item(), "cql": cons.item()})
                print(f"[cql] update {it:6d}  td={td.item():.4f}  cql={cons.item():.4f}")
        return hist

    def q(self, Fd):
        with torch.no_grad():
            q = self.net(torch.as_tensor(self.sx(self.state(Fd)), dtype=torch.float32)).numpy()
        return q * self.r_sd + self.r_mu

    def probs(self, Fd):
        return greedy(self.q(Fd))


class OneStepSeqVI:
    def __init__(self, K=2, n_clusters=None, cluster_range=(2, 30), seed=0):
        self.K, self.n_clusters, self.cluster_range, self.seed = K, n_clusters, tuple(cluster_range), seed

    def _elbow(self, X):
        lo, hi = self.cluster_range
        hi = min(hi, len(np.unique(X.round(6), axis=0)))
        ks = list(range(lo, max(lo, hi) + 1))
        wss = [KMeans(k, n_init=3, random_state=self.seed).fit(X).inertia_ for k in ks]
        if len(ks) < 3:
            return ks[-1]
        x = (np.asarray(ks) - ks[0]) / (ks[-1] - ks[0])
        y = (np.asarray(wss) - wss[-1]) / max(1e-12, wss[0] - wss[-1])
        return ks[int(np.argmax(np.abs(1 - x - y)))]

    def fit(self, Fd, a, r):
        rng = np.random.default_rng(self.seed)
        X = Fd["cf"]
        Xs = X[rng.choice(len(X), min(len(X), 20_000), replace=False)]
        k = self.n_clusters or self._elbow(Xs)
        self.kmeans = KMeans(k, n_init=10, random_state=self.seed).fit(Xs)
        states = list(zip(Fd["last"].tolist(), self.kmeans.predict(X).tolist()))
        sums, counts = {}, {}
        for s, ai, ri in zip(states, a, r):
            sums[(s, ai)] = sums.get((s, ai), 0.0) + float(ri)
            counts[(s, ai)] = counts.get((s, ai), 0) + 1
        self.Q = {key: sums[key] / counts[key] for key in sums}
        self.default = int(np.bincount(a, minlength=self.K).argmax())
        self.diagnostics = {"n_clusters": int(k), "n_states": len(set(states)), "n_state_option_pairs": len(self.Q)}
        print(f"[seq-vi] k={k}, states={len(set(states))}, (state, option) pairs={len(self.Q)}")
        return [self.diagnostics]

    def probs(self, Fd):
        states = zip(Fd["last"].tolist(), self.kmeans.predict(Fd["cf"]).tolist())
        p = np.zeros((len(Fd["cf"]), self.K))
        self.n_fallback = 0
        for i, s in enumerate(states):
            vals = {k: self.Q[(s, k)] for k in range(self.K) if (s, k) in self.Q}
            if vals:
                p[i, max(vals, key=vals.get)] = 1.0
            else:
                p[i, self.default] = 1.0
                self.n_fallback += 1
        return p


class _FeatureCausalRL(CausalRLAgent):
    def _encode(self, W, L):
        return W[:, 0, :].astype(np.float64)


class OneStepCausalRL:
    def __init__(self, K=2, seed=0, **kw):
        spec = SimpleNamespace(n_stages=1, n_slots=K, stage_slots=[list(range(K))])
        self.K = K
        self.agent = _FeatureCausalRL(spec, layout=None, seed=seed, **kw)

    @staticmethod
    def state(Fd):
        return np.c_[Fd["x"], Fd["cf"], Fd["L"]].astype(np.float64)

    def fit(self, Fd, a, r):
        X = self.state(Fd)
        ds = {"obs_window": X[:, None, :], "obs_window_len": np.ones(len(X), np.int64),
              "slot": np.asarray(a, np.int64), "reward": np.asarray(r, np.float64), "dec_pos": np.zeros(len(X))}
        hist = self.agent.fit(ds)
        self.diagnostics = self.agent.diagnostics
        return hist

    def probs(self, Fd):
        S = self.agent._state(self.state(Fd))
        with torch.no_grad():
            logits = self.agent._logits(torch.as_tensor(S, dtype=torch.float32))[0].numpy()
        return greedy(logits)


def make_agent(name, K, args):
    if name == "cql":
        return OneStepCQL(K, alpha=args.cql_alpha, hidden=args.cql_hidden, n_updates=args.cql_updates,
                          batch=args.cql_batch, lr=args.cql_lr, seed=args.seed)
    if name == "seq_vi":
        return OneStepSeqVI(K, n_clusters=args.seqvi_n_clusters, seed=args.seed)
    if name == "causal_rl":
        return OneStepCausalRL(K, seed=args.seed, alpha=args.crl_alpha, n_boot=args.crl_n_boot,
                               conformal_alpha=args.crl_conformal_alpha, effect_model=args.crl_effect_model,
                               ppo_iters=args.crl_ppo_iters, ppo_batch=args.crl_ppo_batch, hidden=args.crl_hidden)
    raise ValueError(name)
