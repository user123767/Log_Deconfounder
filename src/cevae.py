import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(d_in, d_hidden, n_layers):
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, d_hidden), nn.ELU()]
        d = d_hidden
    return nn.Sequential(*layers)


class CEVAE(nn.Module):
    def __init__(self, x_dim, continuous_idx, onehot_groups, scalar_cat_groups, n_treat,
                 z_dim=20, hidden=200, n_layers=3):
        super().__init__()
        self.x_dim, self.n_treat, self.z_dim = x_dim, n_treat, z_dim
        self.cont_idx = list(continuous_idx)
        self.onehot_groups = [list(g) for g in onehot_groups]
        self.scalar_cat_groups = [(int(c), int(n)) for c, n in scalar_cat_groups]
        n_cat = sum(len(g) for g in self.onehot_groups) + sum(n for _, n in self.scalar_cat_groups)
        self.px_trunk = _mlp(z_dim, hidden, n_layers)
        self.px_cont = nn.Linear(hidden, 2 * len(self.cont_idx)) if self.cont_idx else None
        self.px_cat = nn.Linear(hidden, n_cat) if n_cat else None
        self.pt = nn.Sequential(_mlp(z_dim, hidden, n_layers), nn.Linear(hidden, n_treat))
        self.py = nn.ModuleList([nn.Sequential(_mlp(z_dim, hidden, n_layers), nn.Linear(hidden, 1))
                                 for _ in range(n_treat)])
        self.qz_trunk = _mlp(x_dim + 1, hidden, n_layers)
        self.qz_heads = nn.ModuleList([nn.Linear(hidden, 2 * z_dim) for _ in range(n_treat)])
        self.qt = nn.Sequential(_mlp(x_dim, hidden, n_layers), nn.Linear(hidden, n_treat))
        self.qy_trunk = _mlp(x_dim, hidden, n_layers)
        self.qy_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_treat)])
        self.register_buffer("y_mean", torch.zeros(()))
        self.register_buffer("y_std", torch.ones(()))

    @staticmethod
    def _pick(heads, h, t):
        out = torch.stack([hd(h) for hd in heads], 1)
        return out[torch.arange(len(t), device=h.device), t]

    def q_z(self, x, t, y):
        mu, logvar = self._pick(self.qz_heads, self.qz_trunk(torch.cat([x, y[:, None]], -1)), t).chunk(2, -1)
        return mu, logvar.clamp(-10, 5)

    def q_y_mean(self, x, t):
        return self._pick(self.qy_heads, self.qy_trunk(x), t).squeeze(-1)

    def p_y_mean(self, z, t):
        return self._pick(self.py, z, t).squeeze(-1)

    def log_px(self, x, z):
        h = self.px_trunk(z)
        lp = torch.zeros(len(x), device=x.device)
        if self.px_cont is not None:
            mu, log_sig = self.px_cont(h).chunk(2, -1)
            sig = F.softplus(log_sig) + 1e-3
            xc = x[:, self.cont_idx]
            lp = lp - (0.5 * ((xc - mu) / sig) ** 2 + torch.log(sig) + 0.5 * np.log(2 * np.pi)).sum(-1)
        if self.px_cat is not None:
            logits, s = self.px_cat(h), 0
            for g in self.onehot_groups:
                lp = lp - F.cross_entropy(logits[:, s:s + len(g)], x[:, g].argmax(-1), reduction="none")
                s += len(g)
            for c, n in self.scalar_cat_groups:
                lp = lp - F.cross_entropy(logits[:, s:s + n], x[:, c].round().long().clamp(0, n - 1),
                                          reduction="none")
                s += n
        return lp

    def loss(self, x, t, y):
        mu, logvar = self.q_z(x, t, y)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        lpx = self.log_px(x, z)
        lpt = -F.cross_entropy(self.pt(z), t, reduction="none")
        lpy = -0.5 * (y - self.p_y_mean(z, t)) ** 2
        kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(-1)
        lqt = -F.cross_entropy(self.qt(x), t, reduction="none")
        lqy = -0.5 * (y - self.q_y_mean(x, t)) ** 2
        elbo = lpx + lpt + lpy - kl
        loss = -(elbo + lqt + lqy).mean()
        return loss, {"loss": loss.item(), "log_px": lpx.mean().item(), "log_pt": lpt.mean().item(),
                      "log_py": lpy.mean().item(), "KL": kl.mean().item(), "log_qt": lqt.mean().item(),
                      "log_qy": lqy.mean().item()}

    @torch.no_grad()
    def sample_z_given_x(self, x, n_samples=1):
        zs = []
        for _ in range(n_samples):
            t = torch.multinomial(F.softmax(self.qt(x), -1), 1)[:, 0]
            y = self.q_y_mean(x, t) + torch.randn(len(x), device=x.device)
            mu, logvar = self.q_z(x, t, y)
            zs.append(mu + (0.5 * logvar).exp() * torch.randn_like(mu))
        return torch.stack(zs)

    @torch.no_grad()
    def sample_z_posterior(self, x, t, y, n_samples=1):
        mu, logvar = self.q_z(x, t, y)
        return mu + (0.5 * logvar).exp() * torch.randn(n_samples, *mu.shape, device=mu.device)

    @torch.no_grad()
    def potential_outcomes(self, x, n_samples=100):
        z = self.sample_z_given_x(x, n_samples)
        S, B, _ = z.shape
        zf = z.reshape(S * B, -1)
        out = torch.stack([self.py[k](zf).reshape(S, B) for k in range(self.n_treat)], -1)
        return out.mean(0)


@torch.no_grad()
def generate_unconfounded(model, x, t_obs, y_obs, n_per_unit=1, sampling="uniform", z_from="posterior",
                          device="cpu", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.as_tensor(x, dtype=torch.float32, device=device)
    T = torch.as_tensor(t_obs, dtype=torch.long, device=device)
    Yn = torch.as_tensor((np.asarray(y_obs) - float(model.y_mean)) / float(model.y_std), dtype=torch.float32,
                         device=device)
    K, N = model.n_treat, len(X)
    reps = K if sampling == "all" else n_per_unit
    units, ts, ys = [], [], []
    for r in range(reps):
        z = (model.sample_z_posterior(X, T, Yn, 1)[0] if z_from == "posterior"
             else model.sample_z_given_x(X, 1)[0])
        t_new = (torch.full((N,), r, dtype=torch.long) if sampling == "all"
                 else torch.randint(0, K, (N,), generator=g)).to(device)
        y_std = model.p_y_mean(z, t_new) + torch.randn(N, generator=g).to(device)
        units.append(np.arange(N)); ts.append(t_new.cpu().numpy())
        ys.append((y_std * model.y_std + model.y_mean).cpu().numpy())
    return np.concatenate(units), np.concatenate(ts), np.concatenate(ys)
