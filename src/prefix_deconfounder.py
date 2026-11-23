import numpy as np

from stage_deconfounder import decision_token

CONTINUE = 0


def _slot_maps(spec):
    action_to_slot, stage_of_slot = {}, {}
    for k, slots in enumerate(spec.stage_slots):
        for s in slots:
            a = spec.slot_action_idx[s]
            if a in action_to_slot and action_to_slot[a] != s:
                raise ValueError(f"MDP action {a} belongs to two interventions -- ambiguous stage")
            action_to_slot[a] = s
            stage_of_slot[s] = k
    return action_to_slot, stage_of_slot


def build_prefix_dataset(data: dict, spec, max_segment_len: int | None = None) -> dict:
    action_to_slot, stage_of_slot = _slot_maps(spec)
    K, n_slots = spec.n_stages, spec.n_slots
    order = np.lexsort((data["time_idx"], data["case_idx"]))
    case, act, rew, obs, nxt, done = (data["case_idx"][order], data["action"][order], data["reward"][order],
                                      data["obs"][order], data["next_obs"][order], data["done"][order])
    bounds = np.flatnonzero(np.r_[True, case[1:] != case[:-1], True])

    cases, rows = [], {k: [] for k in ["c", "case_idx", "dec_pos", "p", "m", "stage", "slot", "prev_slot",
                                       "reward", "next_stage", "ended"]}
    seg_rewards = []
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        dec = [r for r in range(b0, b1) if int(act[r]) in action_to_slot]
        if not dec:
            continue
        n_ev = b1 - b0 + 1
        states = np.concatenate([obs[b0:b1], nxt[b1 - 1:b1]]).astype(np.float32)
        ev_rew = np.r_[0.0, rew[b0:b1]].astype(np.float32)
        stage_at, slot_at = np.full(n_ev, K), np.full(n_ev, -1)
        for r in dec:
            s = action_to_slot[int(act[r])]
            stage_at[r - b0], slot_at[r - b0 + 1] = stage_of_slot[s], s
        labels = np.where(stage_at < K, stage_at + 1, CONTINUE)
        if labels[-1] == CONTINUE:
            labels[-1] = K + 1
        tokens = np.stack([decision_token(states[j], int(slot_at[j]), int(stage_at[j]), n_slots, K)
                           for j in range(n_ev)])
        ci = len(cases)
        ended = bool(done[b0:b1].any())
        cases.append((int(case[b0]), tokens, states, ev_rew, labels))
        ps = [r - b0 for r in dec]
        for j, p in enumerate(ps):
            q = ps[j + 1] if j + 1 < len(ps) else n_ev - 1
            rows["c"].append(ci); rows["case_idx"].append(int(case[b0])); rows["dec_pos"].append(j)
            rows["p"].append(p); rows["m"].append(q - p)
            rows["stage"].append(int(stage_at[p])); rows["slot"].append(int(slot_at[p + 1]))
            rows["prev_slot"].append(int(slot_at[ps[j - 1] + 1]) if j else -1)
            rows["reward"].append(float(ev_rew[p + 1:q + 1].sum()))
            rows["next_stage"].append(int(stage_at[q]) if j + 1 < len(ps) else K)
            rows["ended"].append(ended)
            seg_rewards.append(ev_rew[p + 1:q + 1])

    if not cases:
        raise ValueError("no logged decisions -- check TreatmentSpec / first_row=0")
    E = max(len(t[1]) for t in cases)
    C, D, d = len(cases), cases[0][1].shape[1], cases[0][2].shape[1]
    tok, st = np.zeros((C, E, D), np.float32), np.zeros((C, E, d), np.float32)
    rw, lab, ln = np.zeros((C, E), np.float32), np.zeros((C, E), np.int64), np.zeros(C, np.int64)
    cid = np.zeros(C, np.int64)
    for i, (c_, t, s, r, l) in enumerate(cases):
        n = len(t)
        tok[i, :n], st[i, :n], rw[i, :n], lab[i, :n], ln[i], cid[i] = t, s, r, l, n, c_
    pdata = {k: np.asarray(v) for k, v in rows.items()}
    pdata["reward"] = pdata["reward"].astype(np.float32)
    seg_max = int(pdata["m"].max())
    pdata.update({
        "case_tokens": tok, "case_states": st, "case_rew": rw, "case_label": lab, "case_len": ln,
        "case_ids_c": cid, "n_cases": data["n_cases"], "obs_dim": data["obs_dim"], "token_dim": D,
        "max_prefix_len": E, "max_decisions": int(pdata["dec_pos"].max() + 1),
        "max_segment_len": int(max_segment_len or 2 * seg_max),
        "observed_max_segment_len": seg_max,
        "event_reward_mean": float(np.concatenate(seg_rewards).mean()),
        "event_reward_std": float(np.concatenate(seg_rewards).std()) or 1.0,
    })
    print(f"[prefix data] {C} cases, {len(pdata['p'])} decisions; events per case <= {E}; "
          f"events per segment: mean {pdata['m'].mean():.1f}, max {seg_max}"
          + (f" (truncated to {pdata['max_segment_len']})" if pdata["max_segment_len"] < seg_max else "")
          + f"; first intervention point at event {np.bincount(pdata['p'][pdata['dec_pos'] == 0]).argmax()} (mode)")
    return pdata
