import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from cevae import CEVAE


class SeqCEVAE(CEVAE):
    def __init__(self, obs_dim, continuous_idx, onehot_groups, scalar_cat_groups, n_treat, n_activities,
                 z_dim=20, hidden=200, n_layers=3, hist_hidden=64, hist_layers=1, decode_hist=True, hist_weight=1.0):
        super().__init__(obs_dim + hist_hidden, continuous_idx, onehot_groups, scalar_cat_groups, n_treat,
                         z_dim, hidden, n_layers)
        self.obs_dim, self.hist_hidden, self.hist_layers = obs_dim, hist_hidden, hist_layers
        self.decode_hist, self.hist_weight = bool(decode_hist), float(hist_weight)
        self.n_activities = int(n_activities)
        H = hist_hidden
        self.enc_start = nn.Parameter(torch.zeros(H))
        self.enc_emb = nn.Embedding(self.n_activities, H)
        self.enc = nn.GRU(H, H, hist_layers, batch_first=True)
        if self.decode_hist:
            self.dec_init = nn.Linear(z_dim, hist_layers * H)
            self.dec_start = nn.Parameter(torch.zeros(H))
            self.dec_emb = nn.Embedding(self.n_activities, H)
            self.dec = nn.GRU(H, H, hist_layers, batch_first=True)
            self.dec_head = nn.Linear(H, self.n_activities + 1)

    def encode_history(self, hist, hlen):
        B = len(hist)
        inp = torch.cat([self.enc_start.expand(B, 1, -1), self.enc_emb(hist)], 1)
        packed = pack_padded_sequence(inp, (hlen + 1).cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.enc(packed)
        return h[-1]

    def features(self, x_cur, hist, hlen):
        return torch.cat([x_cur, self.encode_history(hist, hlen)], -1)

    def log_phist(self, z, hist, hlen):
        B, L = hist.shape
        h0 = torch.tanh(self.dec_init(z)).view(B, self.hist_layers, self.hist_hidden).transpose(0, 1).contiguous()
        inp = torch.cat([self.dec_start.expand(B, 1, -1), self.dec_emb(hist)], 1)
        logits = self.dec_head(self.dec(inp, h0)[0])
        steps = torch.arange(L + 1, device=z.device)[None]
        tgt = torch.cat([hist, hist[:, :1]], 1).clone()
        tgt[steps.expand(B, -1) == hlen[:, None]] = self.n_activities
        valid = (steps <= hlen[:, None]).float()
        lp = -F.cross_entropy(logits.reshape(-1, self.n_activities + 1), tgt.reshape(-1),
                              reduction="none").view(B, L + 1)
        return (lp * valid).sum(-1)

    def seq_loss(self, x_cur, hist, hlen, t, y):
        x = self.features(x_cur, hist, hlen)
        mu, logvar = self.q_z(x, t, y)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        lpx = self.log_px(x_cur, z)
        lph = self.log_phist(z, hist, hlen) if self.decode_hist else torch.zeros_like(lpx)
        lpt = -F.cross_entropy(self.pt(z), t, reduction="none")
        lpy = -0.5 * (y - self.p_y_mean(z, t)) ** 2
        kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(-1)
        lqt = -F.cross_entropy(self.qt(x), t, reduction="none")
        lqy = -0.5 * (y - self.q_y_mean(x, t)) ** 2
        elbo = lpx + self.hist_weight * lph + lpt + lpy - kl
        loss = -(elbo + lqt + lqy).mean()
        return loss, {"loss": loss.item(), "log_px": lpx.mean().item(), "log_phist": lph.mean().item(),
                      "log_pt": lpt.mean().item(), "log_py": lpy.mean().item(), "KL": kl.mean().item(),
                      "log_qt": lqt.mean().item(), "log_qy": lqy.mean().item()}

    @torch.no_grad()
    def all_features(self, x_cur, hist, hlen, batch_size=2048):
        dev = next(self.parameters()).device
        out = []
        for s in range(0, len(x_cur), batch_size):
            sl = slice(s, s + batch_size)
            out.append(self.features(torch.as_tensor(x_cur[sl], device=dev),
                                     torch.as_tensor(hist[sl], dtype=torch.long, device=dev),
                                     torch.as_tensor(hlen[sl], device=dev)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def train_seq_cevae(x_cur, hist, hlen, t, y, continuous_idx, onehot_groups, scalar_cat_groups, n_treat, n_activities,
                    z_dim=20, hidden=200, n_layers=3, hist_hidden=64, hist_layers=1, decode_hist=True, hist_weight=1.0,
                    lr=3e-4, weight_decay=1e-4, batch_size=256, epochs=300, val_share=0.1, patience=30, device="cpu",
                    seed=0, verbose=True):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Xc = torch.as_tensor(x_cur, dtype=torch.float32, device=device)
    Hs = torch.as_tensor(hist, dtype=torch.long, device=device)
    Hl = torch.as_tensor(hlen, dtype=torch.long, device=device)
    T = torch.as_tensor(t, dtype=torch.long, device=device)
    y = np.asarray(y, np.float64)
    model = SeqCEVAE(Xc.shape[1], continuous_idx, onehot_groups, scalar_cat_groups, n_treat, n_activities, z_dim,
                     hidden, n_layers, hist_hidden, hist_layers, decode_hist, hist_weight).to(device)
    model.y_mean.fill_(float(y.mean())); model.y_std.fill_(float(y.std()) or 1.0)
    Y = torch.as_tensor((y - y.mean()) / (y.std() or 1.0), dtype=torch.float32, device=device)
    n = len(T)
    perm = rng.permutation(n)
    n_val = int(val_share * n)
    va, tr = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, best_ep, bad, history = np.inf, None, 0, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(tr)
        agg = {}
        for s in range(0, len(order), batch_size):
            i = torch.as_tensor(order[s:s + batch_size], device=device)
            loss, logs = model.seq_loss(Xc[i], Hs[i], Hl[i], T[i], Y[i])
            opt.zero_grad(); loss.backward(); opt.step()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v * len(i)
        row = {"epoch": ep, **{k: v / len(order) for k, v in agg.items()}}
        if n_val:
            model.eval()
            with torch.no_grad():
                vagg = {}
                for s in range(0, n_val, 4096):
                    i = torch.as_tensor(va[s:s + 4096], device=device)
                    _, vl = model.seq_loss(Xc[i], Hs[i], Hl[i], T[i], Y[i])
                    for k, v in vl.items():
                        vagg[k] = vagg.get(k, 0.0) + v * len(i)
            row.update({f"val_{k}": v / n_val for k, v in vagg.items()})
        history.append(row)
        monitor = row.get("val_loss", row["loss"])
        if monitor < best - 1e-4 * abs(best if np.isfinite(best) else monitor):
            best, best_state, best_ep, bad = monitor, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
        if verbose and (ep == 1 or ep % max(1, epochs // 20) == 0):
            print(f"[seq-CEVAE] epoch {ep:4d}  loss={row['loss']:.4f}  KL={row['KL']:.3f}  log_py={row['log_py']:.3f}"
                  f"  log_phist={row['log_phist']:.2f}" + (f" | val loss={row['val_loss']:.4f}" if n_val else ""))
        if patience and bad >= patience:
            if verbose:
                print(f"[seq-CEVAE] early stop at epoch {ep}; best epoch {best_ep}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    history[-1]["best_epoch"] = best_ep
    return model.eval(), history
