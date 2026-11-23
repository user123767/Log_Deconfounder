import argparse
import json
import os
import pickle
import random
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import inspect

from two_way_deconfounder import (
    build_two_way_dataset,
    QNetwork,
    RecurrentQNetwork,
    print_categorical_structure,
)
from stage_deconfounder import decision_token
from prefix_deconfounder import build_prefix_dataset
from tabular_agents import TokenLayout, SeqVIAgent, activity_group
from causal_rl_agent import CausalRLAgent

print("Starting script")

from MDP_functions import *
from SimBank.SimBank.simulation import PresProcessGenerator
from dataset_manager import dataset_confs
from proxies import ProxyInjector


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_dataset_params(path: str) -> dict:
    with open(path, "rb") as f:
        params = pickle.load(f)
    params["simulation_start"] = datetime(2024, 3, 20, 8, 0)
    return params


def _action_index(activity_index: dict, option):
    for key, idx in activity_index.items():
        if key == option or str(key) == str(option):
            return idx
        try:
            if abs(float(key) - float(option)) < 1e-9:
                return idx
        except (TypeError, ValueError):
            pass
    raise KeyError(f"treatment option {option!r} is not an action of the MDP; "
                   f"activity_index keys: {list(activity_index)}")


class TreatmentSpec:
    def __init__(self, intervention_info: dict, activity_index: dict):
        self.names = list(intervention_info["name"])
        self.options = [list(a) for a in intervention_info["actions"]]
        self.stage_slots, self.slot_action_idx, self.slot_label = [], [], []
        for k, opts in enumerate(self.options):
            if "do_nothing" in opts:
                raise ValueError(
                    f"intervention {self.names[k]!r} has a 'do_nothing' option: its "
                    f"timing is part of the decision, which this script does not "
                    f"model. Use choose_procedure / set_ir_3_levels.")
            slots = []
            for o in opts:
                slots.append(len(self.slot_action_idx))
                self.slot_action_idx.append(_action_index(activity_index, o))
                self.slot_label.append(f"{self.names[k]}={o}")
            self.stage_slots.append(slots)
        self.n_stages = len(self.names)
        self.n_slots = len(self.slot_action_idx)
        self.stage_action_idx = [[self.slot_action_idx[s] for s in sl] for sl in self.stage_slots]
        self.all_treatment_idx = sorted(set(self.slot_action_idx))
        self.mask = torch.zeros(self.n_stages, self.n_slots, dtype=torch.bool)
        for k, sl in enumerate(self.stage_slots):
            self.mask[k, sl] = True

    def describe(self):
        for k in range(self.n_stages):
            print(f"  intervention {k} {self.names[k]}: options {self.options[k]} "
                  f"-> MDP actions {self.stage_action_idx[k]}, Q slots {self.stage_slots[k]}")


def check_logged_decisions(data: dict, spec: TreatmentSpec) -> dict:
    counts = {}
    for k in range(spec.n_stages):
        acts = data["action"][np.isin(data["action"], spec.stage_action_idx[k])]
        counts[spec.names[k]] = {str(o): int((acts == a).sum())
                                 for o, a in zip(spec.options[k], spec.stage_action_idx[k])}
    print("Logged treatment decisions (as actions in the training data):")
    for name, c in counts.items():
        print(f"  {name}: {c}")
    missing = [n for n, c in counts.items() if sum(c.values()) == 0]
    if missing:
        raise ValueError(
            f"No logged decisions for {missing}: the deconfounder never sees these "
            f"treatments as actions, so it cannot simulate them. A likely cause: "
            f"build_two_way_dataset skips each case's first row (range(1, ...)); if "
            f"that row is the event BEFORE the decision (e.g. initiate_application "
            f"before choose_procedure), the decision is dropped -- start that loop at 0.")
    return counts


@torch.no_grad()
def _greedy_slot(q_net, spec, stage, window, length, recurrent, device):
    x = torch.as_tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
    q = q_net(x, torch.tensor([length])) if recurrent else q_net(x[:, length - 1])
    q = q.squeeze(0).masked_fill(~spec.mask[stage].to(device), float("-inf"))
    return int(q.argmax().item())


def treatment_dataset_report(ds: dict, spec: TreatmentSpec) -> pd.DataFrame:
    print(f"=== Decision dataset ({ds.get('mode', '')}): {ds['n_rollouts']} cases, {len(ds['slot'])} decisions, "
          f"{ds['n_decisions'].mean():.2f} decisions/case ===")
    hist = Counter(ds["n_decisions"].tolist())
    print(f"decisions per case: {dict(sorted(hist.items()))}")
    rows = []
    for k in range(spec.n_stages):
        sel = ds["stage"] == k
        for j, s in enumerate(spec.stage_slots[k]):
            m = sel & (ds["slot"] == s)
            rows.append({"intervention": spec.names[k], "option": spec.options[k][j],
                         "n": int(m.sum()), "share_in_stage": float(m.sum() / max(1, sel.sum())),
                         "mean_reward_to_next_decision": float(ds["reward"][m].mean()) if m.any() else np.nan})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return table


@dataclass
class TreatmentCQLConfig:
    gamma: float = 1.0
    lr: float = 1e-3
    batch_size: int = 256
    n_updates: int = 20_000
    target_update_every: int = 500
    cql_alpha: float = 1.0
    reward_scale: float = 1.0
    log_every: int = 500


def make_q_net(obs_dim, n_slots, recurrent, rnn_type, rnn_hidden, rnn_layers):
    if recurrent:
        return RecurrentQNetwork(obs_dim, n_slots, hidden=rnn_hidden, rnn_type=rnn_type,
                                 rnn_layers=rnn_layers)
    return QNetwork(obs_dim, n_slots)


def train_treatment_cql(ds: dict, obs_dim: int, spec: TreatmentSpec, cfg: TreatmentCQLConfig,
                        device="cpu", recurrent=True, rnn_type="lstm", rnn_hidden=128,
                        rnn_layers=1, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    q_net = make_q_net(obs_dim, spec.n_slots, recurrent, rnn_type, rnn_hidden, rnn_layers).to(device)
    target = make_q_net(obs_dim, spec.n_slots, recurrent, rnn_type, rnn_hidden, rnn_layers).to(device)
    target.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)

    X = torch.as_tensor(ds["obs_window"], device=device)
    XL = torch.as_tensor(ds["obs_window_len"], dtype=torch.long)
    Xn = torch.as_tensor(ds["next_obs_window"], device=device)
    XnL = torch.as_tensor(ds["next_obs_window_len"], dtype=torch.long)
    S = torch.as_tensor(ds["stage"], dtype=torch.long, device=device)
    Sn = torch.as_tensor(ds["next_stage"], dtype=torch.long, device=device)
    A = torch.as_tensor(ds["slot"], dtype=torch.long, device=device)
    R = torch.as_tensor(ds["reward"], dtype=torch.float32, device=device) / cfg.reward_scale
    D = torch.as_tensor(ds["done"], dtype=torch.float32, device=device)
    mask = spec.mask.to(device)
    n = len(A)

    def q_of(net, x, l):
        if recurrent:
            return net(x, l)
        rows = torch.arange(x.shape[0], device=x.device)
        return net(x[rows, (l - 1).to(x.device)])

    history, acc = [], dict.fromkeys(["loss", "td", "cql", "qd", "qm"], 0.0)
    for step in range(1, cfg.n_updates + 1):
        idx = torch.as_tensor(rng.integers(0, n, size=cfg.batch_size), device=device)
        cidx = idx.cpu()
        q_all = q_of(q_net, X[idx], XL[cidx])
        m = mask[S[idx]]
        q_sa = q_all.gather(1, A[idx, None]).squeeze(1)
        with torch.no_grad():
            m_next = mask[Sn[idx]]
            a_next = q_of(q_net, Xn[idx], XnL[cidx]).masked_fill(~m_next, float("-inf")).argmax(-1)
            q_next = q_of(target, Xn[idx], XnL[cidx]).gather(1, a_next[:, None]).squeeze(1)
            y = R[idx] + cfg.gamma * (1.0 - D[idx]) * q_next
        td = F.smooth_l1_loss(q_sa, y)
        q_masked = q_all.masked_fill(~m, float("-inf"))
        cql = (torch.logsumexp(q_masked, dim=-1) - q_sa).mean()
        loss = td + cfg.cql_alpha * cql
        opt.zero_grad(); loss.backward(); opt.step()
        if step % cfg.target_update_every == 0:
            target.load_state_dict(q_net.state_dict())

        acc["loss"] += loss.item(); acc["td"] += td.item(); acc["cql"] += cql.item()
        acc["qd"] += q_sa.mean().item(); acc["qm"] += q_masked.max(-1).values.mean().item()
        if step % cfg.log_every == 0 or step == cfg.n_updates:
            k = cfg.log_every if step % cfg.log_every == 0 else step % cfg.log_every
            row = {"step": step, "loss": acc["loss"] / k, "td_loss": acc["td"] / k,
                   "cql_penalty": acc["cql"] / k, "q_data": acc["qd"] / k, "q_max": acc["qm"] / k}
            history.append(row)
            acc = dict.fromkeys(acc, 0.0)
            print(f"[treat-CQL] step {step:6d}/{cfg.n_updates}  td={row['td_loss']:.4f}  "
                  f"cql={row['cql_penalty']:.4f}  Q(data)={row['q_data']:+.3f}  max Q={row['q_max']:+.3f}")
    q_net.eval()
    return q_net, history


def prefix_to_states(env, prefix: list) -> list:
    hist, states = [], []
    for ev in prefix:
        es = {k: (0 if (np.isscalar(v) and pd.isna(v)) else v) for k, v in ev.items()}
        es["last_action"] = es.get("interest_rate", 0) if es.get("activity") == "calculate_offer" \
            else es.get("activity")
        es.setdefault("receive_acceptance", 0)
        es.setdefault("receive_refusal", 0)
        for c in env.numeric_cols:
            es.setdefault(c, 0)
        df = pd.DataFrame([es])
        df[env.numeric_cols] = env.scaler.transform(df[env.numeric_cols])
        es = df.iloc[0].to_dict()
        state = np.asarray(env.build_state_with_lags(hist, es), dtype=np.float32).reshape(-1)
        if len(state) != len(env.state_cols):
            raise ValueError(f"build_state_with_lags gave {len(state)} values, "
                             f"expected len(env.state_cols)={len(env.state_cols)}")
        states.append(state)
        hist.append(es)
    return states


class PrefixTreatmentPolicy:
    def __init__(self, q_net, spec, env, keep_col_idx, recurrent, max_hist, device="cpu",
                 prepend_start=False, proxy=None):
        self.q_net, self.spec, self.env = q_net.eval(), spec, env
        self.keep, self.recurrent, self.max_hist, self.device = keep_col_idx, recurrent, max_hist, device
        self.prepend_start = prepend_start
        self.proxy = proxy
        self.reset()

    def reset(self, case_nr=None):
        self._dec = []
        self.first_positions = getattr(self, "first_positions", [])
        if self.proxy is not None:
            self.proxy.new_case(case_nr)

    def __call__(self, prefix: list, stage: int) -> int:
        events = list(prefix)
        if self.proxy is not None:
            events = self.proxy.add_raw(events)
        if self.prepend_start:
            start = dict(events[0]); start["activity"] = "start"
            events = [start] + events
        states = prefix_to_states(self.env, events)
        pos = len(states) - 1
        if not self._dec:
            self.first_positions.append(pos)
        stage_at = {p: k for (p, k, _) in self._dec}; stage_at[pos] = stage
        slot_at = {p + 1: s for (p, _, s) in self._dec}
        K, n = self.spec.n_stages, self.spec.n_slots
        toks = np.stack([decision_token(st[self.keep], slot_at.get(j, -1), stage_at.get(j, K), n, K)
                         for j, st in enumerate(states)])[-self.max_hist:]
        W = np.zeros((self.max_hist, toks.shape[1]), np.float32); W[:len(toks)] = toks
        slot = _greedy_slot(self.q_net, self.spec, stage, W, len(toks), self.recurrent, self.device)
        self._dec.append((pos, stage, slot))
        return self.spec.stage_slots[stage].index(slot)


def bank_policy(dataset_params):
    from SimBank.SimBank.activity_execution import ActivityExecutioner
    info, pol = dataset_params["intervention_info"], dataset_params["policies_info"]

    def decide(prefix, stage):
        prev = prefix[-1]
        name = info["name"][stage]
        if name == "choose_procedure":
            return int(prev["amount"] > pol["choose_procedure"]["amount"]
                       and prev["est_quality"] >= pol["choose_procedure"]["est_quality"])
        if name == "set_ir_3_levels":
            ir, _, _ = ActivityExecutioner().calculate_offer(prev_event=prev, intervention_info=info)
            return info["actions"][stage].index(ir)
        return 0
    return decide


def random_policy(dataset_params, seed):
    rng = random.Random(seed)
    widths = dataset_params["intervention_info"]["action_width"]
    return lambda prefix, stage: rng.choice(range(widths[stage]))


def run_simbank_inference(env, dataset_params, decide, n_cases: int, label: str) -> pd.DataFrame:
    params = deepcopy(dataset_params)
    gen = PresProcessGenerator(params, env, seed=params["random_seed_test"])
    names, opts = params["intervention_info"]["name"], params["intervention_info"]["actions"]
    rows = []
    t0 = time.time()
    for case_nr in range(n_cases):
        prefix_list = gen.start_simulation_inference(seed_to_add=case_nr)
        if hasattr(decide, "reset"):
            decide.reset(case_nr)
        chosen = []
        while gen.int_points_available:
            k = gen.current_int_index
            j = int(decide(prefix_list[0][:-1], k))
            chosen.append((names[k], opts[k][j]))
            prefix_list = gen.continue_simulation_inference(j)
        full = pd.DataFrame(gen.end_simulation_inference())
        rows.append({"case": case_nr, "outcome": float(full["outcome"].iloc[-1]),
                     "n_decisions": len(chosen), "decisions": chosen,
                     "trace": list(full["activity"])})
        if (case_nr + 1) % max(1, n_cases // 10) == 0:
            print(f"[online/{label}] {case_nr + 1}/{n_cases}  mean outcome "
                  f"{np.mean([r['outcome'] for r in rows]):.1f}  ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def summarize_online(results: dict, spec: TreatmentSpec) -> pd.DataFrame:
    summary = []
    for label, df in results.items():
        o = df["outcome"].to_numpy()
        row = {"policy": label, "n_cases": len(o), "total_outcome": o.sum(),
               "mean_outcome": o.mean(), "se": o.std(ddof=1) / np.sqrt(len(o))}
        flat = [d for ds in df["decisions"] for d in ds]
        for k, name in enumerate(spec.names):
            picks = [v for (n, v) in flat if n == name]
            for o_ in spec.options[k]:
                row[f"{name}={o_}"] = picks.count(o_) / max(1, len(picks))
        summary.append(row)
    table = pd.DataFrame(summary)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if "bank" in results and len(results) > 1:
        bank = results["bank"]["decisions"]
        for label, df in results.items():
            if label == "bank":
                continue
            agree = [a == b for x, y in zip(df["decisions"], bank) for a, b in zip(x, y)]
            print(f"agreement with the bank policy ({label}): {np.mean(agree):.3f} of decisions")
    return table


def build_logged_agent_dataset(sd: dict, spec: TreatmentSpec, max_hist: int) -> dict:
    K = spec.n_stages
    n = len(sd["slot"])
    D = sd["token_dim"]
    W = np.zeros((n, max_hist, D), np.float32); L = np.zeros(n, np.int64)
    Wn, Ln = np.zeros_like(W), np.zeros(n, np.int64)
    for i in range(n):
        c, p, m = sd["c"][i], sd["p"][i], sd["m"][i]
        h = sd["case_tokens"][c, :p + 1][-max_hist:]; W[i, :len(h)] = h; L[i] = len(h)
        hn = sd["case_tokens"][c, :p + m + 1][-max_hist:]; Wn[i, :len(hn)] = hn; Ln[i] = len(hn)
    done = sd["next_stage"] == K
    returns = pd.Series(sd["reward"]).groupby(sd["case_idx"]).sum()
    return {"obs_window": W, "obs_window_len": L, "stage": sd["stage"].astype(np.int64),
            "slot": sd["slot"].astype(np.int64), "reward": sd["reward"].astype(np.float32),
            "next_obs_window": Wn, "next_obs_window_len": Ln,
            "next_stage": np.where(done, 0, sd["next_stage"]).astype(np.int64),
            "done": done.astype(np.float32), "case_idx": sd["case_idx"].astype(np.int64),
            "dec_pos": sd["dec_pos"].astype(np.int64),
            "returns": returns.to_numpy(np.float32),
            "n_decisions": pd.Series(sd["case_idx"]).value_counts().to_numpy(np.int64),
            "n_rollouts": int(len(returns)), "mode": "logged",
            "share_ended": float(pd.Series(sd["ended"]).groupby(sd["case_idx"]).max().mean())}


def agent_tag(args) -> str:
    return {"cql": f"cql{args.cql_alpha:g}", "seq_vi": "seqvi", "causal_rl": "crl"}[args.agent]


def policy_tag_of(args, run_name) -> str:
    if args.train_data == "cevae":
        return f"{run_name}_cevae_treat_{agent_tag(args)}"
    return f"{run_name}_logged-prefix_treat_{agent_tag(args)}"


def parse_args():
    p = argparse.ArgumentParser(description="Treatment-based Confounding-Robust Process "
                                            "Pilot on SimBank.")
    p.add_argument("--exclude_cols", type=str, default=None,
                   help="Comma-separated state columns to hide (lags are hidden too).")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="./results_treatment")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset", type=str, default="SimBank")
    p.add_argument("--order", type=int, default=1)
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--mdp_model", type=str, default="MDP")
    p.add_argument("--dataset_params", type=str, default="src/dataset_params.pkl",
                   help="SimBank dataset_params (as used by online_evaluation); its "
                        "intervention_info defines the treatments.")
    p.add_argument("--train_data", type=str, default="logged", choices=["logged", "cevae"],
                   help="logged: the logged decisions; cevae: the unconfounded one-step dataset written by "
                        "seq_cevae_simbank.py (output_dir/cevae_dataset_<run>.npz).")
    p.add_argument("--agent", type=str, default="cql", choices=["cql", "seq_vi", "causal_rl"])

    p.add_argument("--max_segment_len", type=int, default=None,
                   help="max events between two decisions (default: twice the longest logged segment).")
    p.add_argument("--prepend_start", type=str2bool, default=False,
                   help="online: prepend the synthetic 'start' event that "
                        "DatasetManager adds to every logged case.")

    p.add_argument("--cql_alpha", type=float, default=1.0)
    p.add_argument("--cql_updates", type=int, default=20000)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--q_lr", type=float, default=1e-3)
    p.add_argument("--q_batch_size", type=int, default=256)
    p.add_argument("--target_update_every", type=int, default=500)
    p.add_argument("--use_recurrent", type=str2bool, default=True,
                   help="Q-network over the history of intervention points (else only the current one).")
    p.add_argument("--rnn_type", type=str, default="gru", choices=["gru", "lstm"])
    p.add_argument("--rnn_hidden", type=int, default=128)
    p.add_argument("--rnn_layers", type=int, default=1)
    p.add_argument("--reload_policy", type=str2bool, default=False)

    p.add_argument("--seqvi_n_clusters", type=int, default=None, help="Default: elbow of the WSS curve.")
    p.add_argument("--seqvi_cluster_range", type=int, nargs=2, default=[2, 30])
    p.add_argument("--seqvi_gamma", type=float, default=1.0)
    p.add_argument("--seqvi_vi_threshold", type=float, default=1e-5)
    p.add_argument("--seqvi_vi_max_iters", type=int, default=10000)
    p.add_argument("--seqvi_n_init", type=int, default=10)

    p.add_argument("--crl_alpha", type=float, default=0.1,
                   help="causal_rl: effect bounds = alpha/2 and 1-alpha/2 quantiles over the bootstrap models.")
    p.add_argument("--crl_n_boot", type=int, default=10, help="causal_rl: bootstrap outcome models per option.")
    p.add_argument("--crl_conformal_alpha", type=float, default=0.1, help="causal_rl: conformal miscoverage.")
    p.add_argument("--crl_effect_model", type=str, default="tlearner", choices=["tlearner", "forest"],
                   help="causal_rl: forest = econml CausalForestDML (binary intervention, needs econml).")
    p.add_argument("--crl_ppo_iters", type=int, default=200)
    p.add_argument("--crl_ppo_batch", type=int, default=2048)
    p.add_argument("--crl_hidden", type=int, default=64)

    p.add_argument("--skip_online_eval", type=str2bool, default=False)
    p.add_argument("--n_eval_cases", type=int, default=None,
                   help="Online cases (default: dataset_params['test_size']).")
    p.add_argument("--eval_baselines", type=str2bool, default=True,
                   help="Also run SimBank's bank policy and a random policy online.")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    hidden_cols = [c.strip() for c in args.exclude_cols.split(",") if c.strip()] if args.exclude_cols else []
    run_name = args.run_name or ("+".join(hidden_cols) if hidden_cols else "no_exclusion")
    policy_tag = policy_tag_of(args, run_name)
    print(f"=== Treatment-based run: hidden={hidden_cols} run={run_name} policy={policy_tag} ===")
    print(f"Args: {vars(args)}")

    env = ProcessEnv(dataset=args.dataset, order=args.order,
                     train_fraction=args.train_fraction, MDP_model=args.mdp_model)
    if "first_row" not in inspect.signature(build_two_way_dataset).parameters:
        raise RuntimeError("build_two_way_dataset needs the first_row argument (add "
                           "`first_row: int = 1` and use range(first_row, len_case - 1)): "
                           "the first decision is the action out of the first event.")
    data = build_two_way_dataset(env, exclude_cols=hidden_cols or None, first_row=0)
    print_categorical_structure(data)

    dataset_params = load_dataset_params(args.dataset_params)
    spec = TreatmentSpec(dataset_params["intervention_info"], env.activity_index)
    print("Treatments:")
    spec.describe()
    logged_counts = check_logged_decisions(data, spec)
    with open(os.path.join(args.output_dir, f"logged_treatments_{run_name}.json"), "w") as f:
        json.dump(logged_counts, f, indent=2)

    sd = build_prefix_dataset(data, spec, max_segment_len=args.max_segment_len)
    max_hist = sd["max_prefix_len"]
    n_cases_dec = len(np.unique(sd["case_idx"]))
    print(f"prefix data: {len(sd['slot'])} decisions from {n_cases_dec} cases, "
          f"up to {sd['max_decisions']} per case; next stage = END for "
          f"{(sd['next_stage'] == spec.n_stages).mean():.1%}")

    token_dim = data["obs_dim"] + spec.n_slots + spec.n_stages
    print(f"--train_data {args.train_data}: the agent trains on "
          f"{'the logged intervention points' if args.train_data == 'logged' else f'the {args.train_data} dataset'}.")

    layout = TokenLayout(data["obs_dim"], spec.n_slots, spec.n_stages,
                         activity_group(data) if args.agent == "seq_vi" else None)
    reward_scale = float(np.std(sd["reward"])) or 1.0
    recurrent = args.use_recurrent if args.agent == "cql" else True
    ext = "pt" if args.agent == "cql" else "pkl"
    agent_ckpt = os.path.join(args.output_dir, f"treat_policy_{policy_tag}.{ext}")
    if args.reload_policy and os.path.exists(agent_ckpt):
        if args.agent == "cql":
            agent = make_q_net(token_dim, spec.n_slots, args.use_recurrent, args.rnn_type,
                               args.rnn_hidden, args.rnn_layers)
            agent.load_state_dict(torch.load(agent_ckpt, map_location=args.device))
            agent = agent.to(args.device).eval()
        else:
            with open(agent_ckpt, "rb") as f:
                agent = pickle.load(f)
        print(f"Reloaded {args.agent} policy {agent_ckpt}")
    else:
        if args.train_data == "cevae":
            path = os.path.join(args.output_dir, f"cevae_dataset_{run_name}.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found -- run seq_cevae_simbank.py first.")
            ds = {k: (v.item() if v.ndim == 0 else v) for k, v in np.load(path, allow_pickle=True).items()}
            if ds["obs_window"].shape[1:] != (max_hist, token_dim):
                raise ValueError(f"{path} has windows {ds['obs_window'].shape[1:]}, expected {(max_hist, token_dim)}: "
                                 f"rebuild it with the same dataset / hidden columns.")
            print(f"Loaded {args.train_data} dataset {path} ({len(ds['slot'])} decisions)")
        else:
            ds = build_logged_agent_dataset(sd, spec, max_hist)
        treatment_dataset_report(ds, spec).to_csv(
            os.path.join(args.output_dir, f"treat_dataset_report_{policy_tag}.csv"), index=False)

        print(f"=== Training {args.agent} on {args.train_data} data for {policy_tag} ===")
        if args.agent == "cql":
            cfg = TreatmentCQLConfig(gamma=args.gamma, lr=args.q_lr, batch_size=args.q_batch_size,
                                     n_updates=args.cql_updates, target_update_every=args.target_update_every,
                                     cql_alpha=args.cql_alpha, reward_scale=reward_scale)
            print(asdict(cfg))
            agent, history = train_treatment_cql(
                ds, token_dim, spec, cfg, device=args.device, recurrent=args.use_recurrent,
                rnn_type=args.rnn_type, rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers, seed=args.seed)
            torch.save(agent.state_dict(), agent_ckpt)
            hist_name = f"cql_loss_history_{policy_tag}.csv"
        else:
            if args.agent == "causal_rl":
                agent = CausalRLAgent(
                    spec, layout, reward_scale, alpha=args.crl_alpha, n_boot=args.crl_n_boot,
                    conformal_alpha=args.crl_conformal_alpha, effect_model=args.crl_effect_model,
                    ppo_iters=args.crl_ppo_iters, ppo_batch=args.crl_ppo_batch, hidden=args.crl_hidden,
                    seed=args.seed)
                diag_name = "diagnostics"
            else:
                agent = SeqVIAgent(
                    spec, layout, reward_scale, n_clusters=args.seqvi_n_clusters,
                    cluster_range=args.seqvi_cluster_range, gamma=args.seqvi_gamma,
                    vi_threshold=args.seqvi_vi_threshold, vi_max_iters=args.seqvi_vi_max_iters,
                    n_init=args.seqvi_n_init, seed=args.seed, max_prefix_len=max_hist)
                diag_name = "diagnostics"
            history = agent.fit(ds)
            with open(agent_ckpt, "wb") as f:
                pickle.dump(agent, f)
            with open(os.path.join(args.output_dir, f"{args.agent}_{diag_name}_{policy_tag}.json"), "w") as f:
                json.dump(getattr(agent, diag_name), f, indent=2, default=float)
            hist_name = f"{args.agent}_history_{policy_tag}.csv"
        pd.DataFrame(history).to_csv(os.path.join(args.output_dir, hist_name), index=False)
        del ds

    if args.skip_online_eval:
        print("--skip_online_eval set -- done.")
    else:
        n_eval = args.n_eval_cases or dataset_params["test_size"]
        col_pos = {c: i for i, c in enumerate(env.state_cols)}
        keep_col_idx = [col_pos[c] for c in data["state_cols"]]
        if hasattr(agent, "reset_lookup_stats"):
            agent.reset_lookup_stats()
        proxy = ProxyInjector.for_dataset(args.dataset, dataset_confs.register_dataset(args.dataset))
        if proxy is not None:
            missing = [c for c in proxy.columns if c not in env.state_cols]
            if missing:
                raise ValueError(f"proxy columns {missing[:3]}... are not in the state -- check dataset_confs")
        policy = PrefixTreatmentPolicy(agent, spec, env, keep_col_idx, recurrent, max_hist,
                                       args.device, prepend_start=args.prepend_start, proxy=proxy)
        results = {"learned": run_simbank_inference(env, dataset_params, policy, n_eval, "learned")}
        if policy.first_positions:
            logged = np.bincount(sd["p"][sd["dec_pos"] == 0])
            online = np.bincount(policy.first_positions)
            print(f"First intervention point (event index): logged mode {logged.argmax()}, "
                  f"online mode {online.argmax()} -- these must match (check --prepend_start).")
        if hasattr(agent, "lookup_summary"):
            with open(os.path.join(args.output_dir, f"{args.agent}_online_lookups_{policy_tag}.json"), "w") as f:
                json.dump(agent.lookup_summary(), f, indent=2)
            print(f"{args.agent} state lookups online: {agent.lookup_summary()}")
        if args.eval_baselines:
            try:
                results["bank"] = run_simbank_inference(env, dataset_params, bank_policy(dataset_params),
                                                        n_eval, "bank")
            except ImportError as e:
                print(f"bank policy skipped ({e})")
            results["random"] = run_simbank_inference(env, dataset_params,
                                                      random_policy(dataset_params, args.seed), n_eval, "random")
        for label, df in results.items():
            df.to_pickle(os.path.join(args.output_dir, f"online_treat_{label}_{policy_tag}.pkl"))
        table = summarize_online(results, spec)
        table.to_csv(os.path.join(args.output_dir, f"online_treat_summary_{policy_tag}.csv"), index=False)

    print(f"=== Run {policy_tag} complete in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
