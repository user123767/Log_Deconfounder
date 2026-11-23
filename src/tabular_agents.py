import numpy as np
import torch
from sklearn.cluster import KMeans

UNSUPPORTED = -1e9


def activity_group(data: dict) -> list:
    cols = list(data["state_cols"])
    for g in data["onehot_groups"]:
        if all(cols[j].startswith("last_action_") for j in g):
            return list(g)
    raise ValueError("no one-hot 'last_action_*' block in data['state_cols'] -- needed by SEQ-VI")


class TokenLayout:
    def __init__(self, obs_dim, n_slots, n_stages, act_cols=None):
        self.obs_dim, self.n_slots, self.n_stages = obs_dim, n_slots, n_stages
        self.act_cols = act_cols

    def stage(self, tok) -> int:
        bits = tok[self.obs_dim + self.n_slots:]
        return int(np.argmax(bits)) if bits.max() > 0 else -1

    def obs(self, tok):
        return tok[:self.obs_dim]

    def activities(self, toks) -> np.ndarray:
        block = toks[:, self.act_cols]
        return np.where(block.max(1) > 0, block.argmax(1), -1)


class _TabularAgent:
    def __init__(self, spec, layout, reward_scale=1.0):
        self.spec, self.layout, self.reward_scale = spec, layout, float(reward_scale) or 1.0
        self.device = "cpu"

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, x, lengths) -> torch.Tensor:
        xs = x.detach().cpu().numpy()
        ls = lengths.detach().cpu().numpy() if torch.is_tensor(lengths) else np.asarray(lengths)
        q = np.stack([self.q_for(xs[i], int(ls[i])) for i in range(len(xs))])
        return torch.as_tensor(q, dtype=torch.float32, device=x.device)

    def _options_q(self, stage, values: dict) -> np.ndarray:
        q = np.full(self.spec.n_slots, UNSUPPORTED, np.float32)
        for s in self.spec.stage_slots[stage]:
            if s in values:
                q[s] = values[s]
        return q


class SeqVIAgent(_TabularAgent):
    def __init__(self, spec, layout, reward_scale=1.0, n_clusters=None, cluster_range=(2, 30), gamma=1.0,
                 vi_threshold=1e-5, vi_max_iters=10_000, n_init=10, seed=0, max_prefix_len=None):
        super().__init__(spec, layout, reward_scale)
        if layout.act_cols is None:
            raise ValueError("SEQ-VI needs the activity block of the event tokens")
        self.n_clusters, self.cluster_range, self.gamma = n_clusters, tuple(cluster_range), gamma
        self.vi_threshold, self.vi_max_iters, self.n_init, self.seed = vi_threshold, vi_max_iters, n_init, seed
        self.max_prefix_len = max_prefix_len
        self.n_act = len(layout.act_cols)
        self.n_fallback, self.n_lookups = 0, 0

    def encode(self, window, length):
        acts = self.layout.activities(window[:length])
        acts = acts[acts >= 0]
        n = max(1, len(acts))
        freq = np.bincount(acts, minlength=self.n_act) / n
        last = np.zeros(self.n_act)
        for j, a in enumerate(acts):
            last[a] = (j + 1) / n
        return np.r_[freq, last, length / (self.max_prefix_len or length)], (int(acts[-1]) if len(acts) else -1)

    def _elbow(self, X):
        lo, hi = self.cluster_range
        hi = min(hi, len(np.unique(X.round(6), axis=0)))
        ks = list(range(lo, max(lo, hi) + 1))
        wss = [KMeans(k, n_init=3, random_state=self.seed).fit(X).inertia_ for k in ks]
        if len(ks) < 3:
            return ks[-1], list(zip(ks, wss))
        x = (np.asarray(ks) - ks[0]) / (ks[-1] - ks[0])
        y = (np.asarray(wss) - wss[-1]) / max(1e-12, wss[0] - wss[-1])
        return ks[int(np.argmax(np.abs(1 - x - y)))], list(zip(ks, wss))

    def fit(self, ds):
        n = len(ds["slot"])
        enc_s = [self.encode(ds["obs_window"][i], int(ds["obs_window_len"][i])) for i in range(n)]
        enc_n = [None if ds["done"][i] else self.encode(ds["next_obs_window"][i], int(ds["next_obs_window_len"][i]))
                 for i in range(n)]
        X = np.stack([e[0] for e in enc_s] + [e[0] for e in enc_n if e is not None])
        rng = np.random.default_rng(self.seed)
        Xs = X[rng.choice(len(X), min(len(X), 20_000), replace=False)]
        if self.n_clusters:
            k, wss = self.n_clusters, []
        else:
            k, wss = self._elbow(Xs)
        self.kmeans = KMeans(k, n_init=self.n_init, random_state=self.seed).fit(Xs)
        lab_s = self.kmeans.predict(np.stack([e[0] for e in enc_s]))
        idx_n = [i for i in range(n) if enc_n[i] is not None]
        lab_n = dict(zip(idx_n, self.kmeans.predict(np.stack([enc_n[i][0] for i in idx_n])) if idx_n else []))

        stage_of = lambda w, l: self.layout.stage(w[l - 1])
        counts, rsum, nxt = {}, {}, {}
        opt_counts = {}
        for i in range(n):
            s = (int(ds["stage"][i]), enc_s[i][1], int(lab_s[i]))
            a = int(ds["slot"][i])
            counts[(s, a)] = counts.get((s, a), 0) + 1
            rsum[(s, a)] = rsum.get((s, a), 0.0) + float(ds["reward"][i]) / self.reward_scale
            s2 = None if enc_n[i] is None else (stage_of(ds["next_obs_window"][i], int(ds["next_obs_window_len"][i])),
                                               enc_n[i][1], int(lab_n[i]))
            d = nxt.setdefault((s, a), {})
            d[s2] = d.get(s2, 0) + 1
            opt_counts.setdefault(s[0], {}); opt_counts[s[0]][a] = opt_counts[s[0]].get(a, 0) + 1
        self.default_option = {k_: max(v, key=v.get) for k_, v in opt_counts.items()}
        states = {s for (s, _) in counts}
        V = {s: 0.0 for s in states}
        self.Q = {}
        for it in range(self.vi_max_iters):
            delta = 0.0
            for (s, a), c in counts.items():
                ev = sum(cnt * (V.get(s2, 0.0) if s2 is not None else 0.0) for s2, cnt in nxt[(s, a)].items()) / c
                self.Q[(s, a)] = rsum[(s, a)] / c + self.gamma * ev
            for s in states:
                v = max(self.Q[(s, a)] for a in self.spec.stage_slots[s[0]] if (s, a) in self.Q)
                delta = max(delta, abs(v - V[s])); V[s] = v
            if delta < self.vi_threshold:
                break
        self.diagnostics = {"n_clusters": int(k), "wss_curve": wss, "n_states": len(states),
                            "n_state_option_pairs": len(counts), "vi_iterations": it + 1, "vi_final_delta": delta,
                            "share_pairs_seen_once": float(np.mean([c == 1 for c in counts.values()]))}
        print("[seq-vi] " + ", ".join(f"{k_}={v}" for k_, v in self.diagnostics.items() if k_ != "wss_curve"))
        return [{"vi_iterations": it + 1, "vi_final_delta": delta}]

    def q_for(self, window, length):
        stage = self.layout.stage(window[length - 1])
        enc, last = self.encode(window, length)
        s = (stage, last, int(self.kmeans.predict(enc[None])[0]))
        vals = {a: self.Q[(s, a)] for a in self.spec.stage_slots[stage] if (s, a) in self.Q}
        self.n_lookups += 1
        if not vals:
            self.n_fallback += 1
            vals = {self.default_option.get(stage, self.spec.stage_slots[stage][0]): 0.0}
        return self._options_q(stage, vals)

    def reset_lookup_stats(self):
        self.n_fallback, self.n_lookups = 0, 0

    def lookup_summary(self):
        return {"n_lookups": self.n_lookups, "n_fallback": self.n_fallback,
                "fallback_share": self.n_fallback / max(1, self.n_lookups)}
