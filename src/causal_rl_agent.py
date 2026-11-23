import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

UNSUPPORTED = -1e9


class CausalRLAgent:
    def __init__(self, spec, layout, reward_scale=1.0, alpha=0.1, n_boot=10, conformal_alpha=0.1,
                 effect_model="tlearner", ppo_iters=200, ppo_batch=2048, ppo_epochs=4, ppo_clip=0.2,
                 ppo_lr=3e-4, entropy_coef=0.01, hidden=64, calib_share=0.2, seed=0):
        if spec.n_stages != 1:
            raise ValueError("causal_rl supports one intervention per case (n_stages == 1).")
        self.spec, self.layout, self.reward_scale = spec, layout, float(reward_scale) or 1.0
        self.options = list(spec.stage_slots[0])
        self.K = len(self.options)
        self.alpha, self.n_boot, self.conformal_alpha = alpha, n_boot, conformal_alpha
        self.effect_model = effect_model
        self.ppo = dict(iters=ppo_iters, batch=ppo_batch, epochs=ppo_epochs, clip=ppo_clip, lr=ppo_lr,
                        ent=entropy_coef, hidden=hidden)
        self.calib_share, self.seed = calib_share, seed
        self.device = "cpu"
        self.diagnostics = {}

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, x, lengths):
        xs = x.detach().cpu().numpy()
        ls = lengths.detach().cpu().numpy() if torch.is_tensor(lengths) else np.asarray(lengths)
        s = self._state(self._encode(xs, ls))
        with torch.no_grad():
            p = F.softmax(self._logits(torch.as_tensor(s, dtype=torch.float32))[0], -1).numpy()
        q = np.full((len(xs), self.spec.n_slots), UNSUPPORTED, np.float32)
        q[:, self.options] = p
        return torch.as_tensor(q, dtype=torch.float32, device=x.device)

    def _encode(self, W, L):
        obs = W[:, :, :self.layout.obs_dim].astype(np.float64)
        L = np.asarray(L, int)
        valid = (np.arange(W.shape[1])[None] < L[:, None])[..., None]
        last = obs[np.arange(len(W)), np.maximum(L - 1, 0)]
        mean = (obs * valid).sum(1) / np.maximum(L, 1)[:, None]
        return np.c_[last, mean, L]

    def _logits(self, S):
        logits, v = self.policy(S)
        return logits.masked_fill(~self.mask, -1e9), v

    def _effect_bounds(self, X):
        if self.effect_model == "forest" and self._forest is not None:
            lo, hi = self._forest.effect_interval(X, alpha=self.alpha)
            return np.c_[lo, hi]
        mu = np.stack([np.stack([m.predict(X) for m in self.models[k]]) for k in range(self.K)])
        eff = mu[1:] - mu[:1]
        lo = np.quantile(eff, self.alpha / 2, axis=1).T
        hi = np.quantile(eff, 1 - self.alpha / 2, axis=1).T
        return np.c_[lo, hi] / self.reward_scale

    def _rho(self, X):
        if self.clf is None:
            return np.full(len(X), 0.5)
        p1 = self.clf.predict_proba(X)[:, 1]
        in1, in0 = (1 - p1) <= self.qhat, p1 <= self.qhat
        return np.where(in1 & ~in0, 1.0, np.where(in0 & ~in1, 0.0, 0.5))

    def _state(self, X):
        return np.c_[self._effect_bounds(X), self._rho(X), X[:, -1] / self.len_scale].astype(np.float32)

    def fit(self, ds):
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        first = np.flatnonzero(np.asarray(ds.get("dec_pos", np.zeros(len(ds["slot"])))) == 0)
        X = self._encode(ds["obs_window"][first], ds["obs_window_len"][first])
        a = np.array([self.options.index(int(s)) for s in ds["slot"][first]])
        y = np.asarray(ds["reward"][first], float)
        self.len_scale = float(X[:, -1].max()) or 1.0
        print(f"[causal_rl] {len(X)} cases, options {np.bincount(a, minlength=self.K).tolist()}, "
              f"mean y per option {[round(float(y[a == k].mean()), 2) if (a == k).any() else None for k in range(self.K)]}")

        MIN_CASES = 20
        self.supported = np.array([(a == k).sum() >= MIN_CASES for k in range(self.K)])
        if not self.supported.any():
            raise ValueError(f"no option has {MIN_CASES} cases: no outcome model can be fitted")
        if not self.supported.all():
            print(f"[causal_rl] options {np.flatnonzero(~self.supported).tolist()} have < {MIN_CASES} cases "
                  f"({np.bincount(a, minlength=self.K).tolist()}): not supported, never recommended")
        self.mask = torch.as_tensor(self.supported)
        self.models, self.resid = [], []
        for k in range(self.K):
            idx = np.flatnonzero(a == k) if self.supported[k] else np.arange(len(X))
            ens = []
            for b in range(self.n_boot):
                bi = rng.choice(idx, len(idx), replace=True)
                ens.append(HistGradientBoostingRegressor(max_iter=200, random_state=self.seed + b)
                           .fit(X[bi], y[bi]))
            self.models.append(ens)
            mu = np.mean([m.predict(X[idx]) for m in ens], 0)
            self.resid.append(y[idx] - mu)
        if self.effect_model == "forest" and not self.supported.all():
            print("[causal_rl] an option is not supported -- using the bootstrap T-learner bounds instead of the forest")
        self._forest = None
        if self.effect_model == "forest" and self.K == 2 and self.supported.all():
            try:
                from econml.dml import CausalForestDML
                self._forest = CausalForestDML(discrete_treatment=True, random_state=self.seed).fit(
                    y / self.reward_scale, a, X=X)
            except ImportError:
                print("[causal_rl] econml not installed -- using the bootstrap T-learner bounds")

        pos = (y > 0).astype(int)
        self.clf, self.qhat = None, 1.0
        if 0 < pos.mean() < 1:
            perm = rng.permutation(len(X))
            n_cal = max(50, int(self.calib_share * len(X)))
            cal, tr = perm[:n_cal], perm[n_cal:]
            self.clf = HistGradientBoostingClassifier(max_iter=200, random_state=self.seed).fit(X[tr], pos[tr])
            p = self.clf.predict_proba(X[cal])[np.arange(n_cal), pos[cal]]
            scores = np.sort(1 - p)
            q_level = min(1.0, np.ceil((n_cal + 1) * (1 - self.conformal_alpha)) / n_cal)
            self.qhat = float(np.quantile(scores, q_level, method="higher"))

        S = self._state(X)
        mu_all = np.stack([np.mean([m.predict(X) for m in self.models[k]], 0) for k in range(self.K)], 1)
        H = self.ppo["hidden"]
        self.policy = _ActorCritic(S.shape[1], self.K, H)
        opt = torch.optim.Adam(self.policy.parameters(), lr=self.ppo["lr"])
        St = torch.as_tensor(S, dtype=torch.float32)
        history = []
        for it in range(self.ppo["iters"]):
            idx = rng.integers(0, len(S), self.ppo["batch"])
            with torch.no_grad():
                logits, v = self._logits(St[idx])
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                logp_old = dist.log_prob(act)
            act_np = act.numpy()
            r = mu_all[idx, act_np] + np.array([rng.choice(self.resid[k]) for k in act_np])
            r = torch.as_tensor(r / self.reward_scale, dtype=torch.float32)
            adv = r - v
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            for _ in range(self.ppo["epochs"]):
                logits, v_new = self._logits(St[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(act) - logp_old)
                l_clip = -torch.min(ratio * adv, torch.clamp(ratio, 1 - self.ppo["clip"], 1 + self.ppo["clip"]) * adv).mean()
                loss = l_clip + 0.5 * F.mse_loss(v_new, r) - self.ppo["ent"] * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                greedy = self._logits(St)[0].argmax(1).numpy()
            history.append({"iter": it, "mean_reward": float(r.mean()) * self.reward_scale,
                            "greedy_value": float(mu_all[np.arange(len(S)), greedy].mean()),
                            "greedy_share": np.bincount(greedy, minlength=self.K).tolist()})
            if it % max(1, self.ppo["iters"] // 10) == 0 or it == self.ppo["iters"] - 1:
                print(f"[causal_rl] PPO iter {it}: sampled reward {history[-1]['mean_reward']:.2f}, "
                      f"greedy value {history[-1]['greedy_value']:.2f}, greedy shares {history[-1]['greedy_share']}")
        self.diagnostics = {"n_cases": int(len(X)), "options": [int(o) for o in self.options],
                            "supported_options": [int(o) for o, ok in zip(self.options, self.supported) if ok],
                            "cases_per_option": np.bincount(a, minlength=self.K).tolist(),
                            "conformal_qhat": self.qhat,
                            "rho_shares": {str(v): float(np.mean(S[:, -2] == v)) for v in (0.0, 0.5, 1.0)},
                            "mean_effect_bounds": S[:, :-2].mean(0).tolist(),
                            "final_greedy_share": history[-1]["greedy_share"]}
        return history


class _ActorCritic(nn.Module):
    def __init__(self, d, K, H):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, H), nn.Tanh(), nn.Linear(H, H), nn.Tanh())
        self.pi, self.v = nn.Linear(H, K), nn.Linear(H, 1)

    def forward(self, s):
        h = self.body(s)
        return self.pi(h), self.v(h).squeeze(-1)
