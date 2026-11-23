from __future__ import annotations

import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _build_case_window_indices(
    case_idx: np.ndarray, time_idx: np.ndarray, window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(case_idx)
    window_idx = np.full((n, window_size), -1, dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)

    case_to_rows: dict = {}
    for r in range(n):
        case_to_rows.setdefault(int(case_idx[r]), []).append(r)

    for rows in case_to_rows.values():
        rows.sort(key=lambda r: time_idx[r])
        for pos, r in enumerate(rows):
            length = min(pos + 1, window_size)
            window_idx[r, :length] = rows[pos - length + 1: pos + 1]
            lengths[r] = length

    return window_idx, lengths


def build_two_way_dataset(
    env, exclude_cols: list[str] | None = None, source_df: pd.DataFrame | None = None,
    on_unseen_category: str = "error", window_size: int | None = None,
    first_row: int = 1,
) -> dict:
    config = env.config
    full_state_cols = env.state_cols
    if exclude_cols:
        missing = [c for c in exclude_cols if c not in full_state_cols]
        if missing:
            raise ValueError(
                f"exclude_cols {missing} not found in env.state_cols -- "
                f"check spelling/case against env.state_cols exactly. This "
                f"used to be a soft warning, which is exactly how a typo'd "
                f"--exclude_col can silently defeat confounder-hiding "
                f"everywhere in the pipeline: build_two_way_dataset would "
                f"exclude nothing, and separately, online_evaluation's "
                f"keep_col_idx filtering (computed the same way from the "
                f"same typo'd name) would ALSO exclude nothing -- so the "
                f"existing `assert len(keep_col_idx) == data['obs_dim']` "
                f"safety check passes anyway, because both sides agree on "
                f"the same wrong (unreduced) size. The result is an "
                f"experiment that never actually hides anything, with no "
                f"error anywhere -- which looks exactly like suspiciously "
                f"good performance. Failing loudly here is the fix."
            )


    exclude_cols_expanded = list(exclude_cols) if exclude_cols else []
    if exclude_cols:
        _lag_re = re.compile(r"^_lag\d+$")
        newly_found = []
        for base in exclude_cols:
            prefix = base + "_lag"
            for c in full_state_cols:
                if c.startswith(prefix) and _lag_re.match(c[len(base):]) and c not in exclude_cols_expanded:
                    exclude_cols_expanded.append(c)
                    newly_found.append(c)
        if newly_found:
            print(f"[build_two_way_dataset] also excluding {sorted(newly_found)} "
                  f"-- lagged variant(s) of {exclude_cols} (env.order > 1 "
                  f"detected). Without this, the hidden confounder's past "
                  f"values would still be observable via these columns.")

    state_cols = [c for c in full_state_cols if c not in exclude_cols_expanded]
    costs_dic = env.costs_dic

    df = env.train_df.copy() if source_df is None else source_df.copy()

    if on_unseen_category not in ("error", "drop_case"):
        raise ValueError(
            f"on_unseen_category must be 'error' or 'drop_case', got "
            f"{on_unseen_category!r}."
        )

    if on_unseen_category == "drop_case":
        n_cases_before = df["ID"].nunique()
        bad_case_ids = set()
        per_column_counts = {}
        for orig_col in env.label_encoders:
            if orig_col not in df.columns:
                continue
            mask = df[orig_col] == -1
            n_bad = int(mask.sum())
            if n_bad > 0:
                per_column_counts[orig_col] = n_bad
                bad_case_ids.update(df.loc[mask, "ID"].unique())
        if bad_case_ids:
            df = df[~df["ID"].isin(bad_case_ids)].copy()
            n_cases_after = df["ID"].nunique()
            print(f"[build_two_way_dataset] on_unseen_category='drop_case': "
                  f"dropped {len(bad_case_ids)}/{n_cases_before} cases "
                  f"({100 * len(bad_case_ids) / n_cases_before:.1f}%) containing "
                  f"an unseen-category (-1) value in one or more label-"
                  f"encoded column(s): {per_column_counts} (column -> raw row "
                  f"count before case-level dedup). {n_cases_after} cases "
                  f"remain.")

    df["action_nr"] = df["action"].map(env.activity_index)
    df["reward"] = 0.0
    df["done"] = False

    for case_id, case_df in df.groupby("ID"):
        for idx, row in case_df.iterrows():
            state = row.to_dict()
            action_name = state["action"]
            cost_action = costs_dic.get(action_name, 0)
            df.loc[idx, "reward"] = -cost_action
            is_terminal, granted = config.is_terminal_successful_state(state)
            if is_terminal:
                r = config.outcome_function(
                    granted, row[full_state_cols], cost_action,
                    env.scaler, env.numeric_cols, full_state_cols,
                )
                df.loc[idx, "reward"] = r
                df.loc[idx, "done"] = True
                break

    case_ids = list(df["ID"].unique())
    case_to_idx = {cid: i for i, cid in enumerate(case_ids)}

    rows_case, rows_time, rows_obs, rows_act, rows_rew, rows_next, rows_done = (
        [], [], [], [], [], [], []
    )
    T_max = 0

    for case_id, case_df in df.groupby("ID"):
        case_df = case_df.reset_index(drop=True)
        len_case = len(case_df)
        T_max = max(T_max, len_case)
        i = case_to_idx[case_id]

        for idx in range(first_row, len_case - 1):
            o = case_df.iloc[idx][state_cols].to_numpy(dtype=np.float32)
            o_next = case_df.iloc[idx + 1][state_cols].to_numpy(dtype=np.float32)
            a = int(case_df.iloc[idx + 1]["action_nr"])
            r = float(case_df.iloc[idx + 1]["reward"])
            done = bool(case_df.iloc[idx + 1]["done"])

            rows_case.append(i)
            rows_time.append(idx)
            rows_obs.append(o)
            rows_act.append(a)
            rows_rew.append(r)
            rows_next.append(o_next)
            rows_done.append(done)

            if done:
                break

    onehot_groups, scalar_cat_groups, continuous_idx = _detect_categorical_structure(env, state_cols)
    idx_to_action = {v: k for k, v in env.activity_index.items()}

    if exclude_cols:
        expected_obs_dim = len(full_state_cols) - len(set(exclude_cols_expanded) & set(full_state_cols))
        assert len(state_cols) == expected_obs_dim, (
            f"internal inconsistency: expected obs_dim={expected_obs_dim} after "
            f"excluding {exclude_cols_expanded} (lag-expanded from {exclude_cols}), "
            f"got {len(state_cols)} -- this should be unreachable given the "
            f"check above; please report this."
        )

    obs_arr = np.stack(rows_obs).astype(np.float32)
    next_obs_arr = np.stack(rows_next).astype(np.float32)

    bad_cols = []
    for idx, n_classes in scalar_cat_groups:
        col_name = state_cols[idx]
        col_name_is_label_encoded = any(
            col_name == orig_col for orig_col in env.label_encoders
        )
        if not col_name_is_label_encoded:
            continue
        out_of_range = ((obs_arr[:, idx] < 0) | (obs_arr[:, idx] >= n_classes)).sum()
        out_of_range += ((next_obs_arr[:, idx] < 0) | (next_obs_arr[:, idx] >= n_classes)).sum()
        if out_of_range > 0:
            bad_cols.append((col_name, int(out_of_range)))
    if bad_cols:
        raise ValueError(
            f"Found out-of-range values (likely the -1 'unseen category' "
            f"sentinel ProcessEnv assigns to label-encoded categories not "
            f"present during training -- see env.label_encoders) in "
            f"{len(bad_cols)} column(s): {bad_cols} (column, count of "
            f"out-of-range values). This almost always means source_df "
            f"is a split (e.g. env.test_df) containing a category this "
            f"column's LabelEncoder never saw in env.train_df. Training "
            f"or fine-tuning a model on this data as-is would corrupt or "
            f"crash cross-entropy loss for that column. If this is "
            f"expected (a genuinely rare test-only category), you'll "
            f"need to decide how to handle it explicitly -- e.g. drop "
            f"those rows, or remap -1 to an added 'unknown' class -- "
            f"rather than have it silently proceed."
        )

    result = dict(
        case_idx=np.asarray(rows_case, dtype=np.int64),
        time_idx=np.asarray(rows_time, dtype=np.int64),
        obs=obs_arr,
        action=np.asarray(rows_act, dtype=np.int64),
        reward=np.asarray(rows_rew, dtype=np.float32),
        next_obs=next_obs_arr,
        done=np.asarray(rows_done, dtype=bool),
        n_cases=len(case_ids),
        T_max=T_max,
        obs_dim=len(state_cols),
        n_actions=env.n_actions,
        state_cols=state_cols,
        onehot_groups=onehot_groups,
        scalar_cat_groups=scalar_cat_groups,
        continuous_idx=continuous_idx,
        costs_dic=env.costs_dic,
        idx_to_action=idx_to_action,
        case_ids=case_ids,
    )

    resolved_window = int(T_max if window_size is None else window_size)
    if resolved_window < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    window_idx, window_lengths = _build_case_window_indices(
        result["case_idx"], result["time_idx"], resolved_window,
    )
    result["window_idx"] = window_idx
    result["window_lengths"] = window_lengths
    result["window_size"] = resolved_window

    return result


def _detect_categorical_structure(env, state_cols: list | None = None) -> tuple[list, list, list]:
    if state_cols is None:
        state_cols = env.state_cols
    col_to_idx = {c: i for i, c in enumerate(state_cols)}
    covered = set()

    _lag_re = re.compile(r"^_lag\d+$")

    def _lag_suffixes_present(base: str) -> list[str]:
        prefix = base + "_lag"
        found = []
        for c in state_cols:
            if c.startswith(prefix) and _lag_re.match(c[len(base):]):
                found.append(c[len(base):])
        return sorted(set(found), key=lambda s: int(s[4:]))

    onehot_groups = []
    for orig_col in env.onehot_cols:
        ohe = env.onehot_encoders[orig_col]
        categories = list(ohe.categories_[0])

        cols_lag0 = [f"{orig_col}_{cat}" for cat in categories]
        idxs_lag0 = [col_to_idx[c] for c in cols_lag0 if c in col_to_idx]
        if len(idxs_lag0) > 1:
            onehot_groups.append(sorted(idxs_lag0))
            covered.update(idxs_lag0)

        suffixes = _lag_suffixes_present(f"{orig_col}_{categories[0]}") if categories else []
        for suffix in suffixes:
            cols_k = [f"{orig_col}_{cat}{suffix}" for cat in categories]
            idxs_k = [col_to_idx[c] for c in cols_k if c in col_to_idx]
            if len(idxs_k) > 1:
                onehot_groups.append(sorted(idxs_k))
                covered.update(idxs_k)

    scalar_cat_groups = []

    for orig_col, le in env.label_encoders.items():
        n_classes = len(le.classes_)
        if orig_col in col_to_idx and col_to_idx[orig_col] not in covered:
            idx = col_to_idx[orig_col]
            scalar_cat_groups.append((idx, n_classes))
            covered.add(idx)
        for suffix in _lag_suffixes_present(orig_col):
            lagged_col = f"{orig_col}{suffix}"
            if lagged_col in col_to_idx and col_to_idx[lagged_col] not in covered:
                idx = col_to_idx[lagged_col]
                scalar_cat_groups.append((idx, n_classes))
                covered.add(idx)

    for col, idx in col_to_idx.items():
        if idx in covered or col not in env.train_df.columns:
            continue
        vals = pd.unique(env.train_df[col].dropna())
        if 0 < len(vals) <= 2 and set(np.round(vals.astype(float), 6)).issubset({0.0, 1.0}):
            scalar_cat_groups.append((idx, 2))
            covered.add(idx)

    continuous_idx = [i for i in range(len(state_cols)) if i not in covered]
    return onehot_groups, scalar_cat_groups, continuous_idx


def _onehot_group_label(first_col: str) -> str:
    m = re.match(r"^(.*)(_lag\d+)$", first_col)
    if m:
        base_before_lag, lag_suffix = m.group(1), m.group(2)
        return base_before_lag.rsplit("_", 1)[0] + lag_suffix
    return first_col.rsplit("_", 1)[0]


def print_categorical_structure(data: dict):
    state_cols = data["state_cols"]
    onehot_groups = data["onehot_groups"]
    scalar_cat_groups = data["scalar_cat_groups"]
    continuous_idx = data["continuous_idx"]

    print(f"{len(state_cols)} total columns -> "
          f"{len(onehot_groups)} one-hot groups, "
          f"{len(scalar_cat_groups)} scalar categoricals, "
          f"{len(continuous_idx)} continuous\n")

    print("ONE-HOT GROUPS (multi-column, exactly one 1 per group):")
    if not onehot_groups:
        print("  (none detected)")
    for g_idx in onehot_groups:
        cols = [state_cols[j] for j in g_idx]
        label = _onehot_group_label(cols[0])
        print(f"  [{label}]  {len(cols)} categories:")
        for c in cols:
            print(f"      {c}")
    print()

    print("SCALAR CATEGORICALS (single column, integer class code):")
    if not scalar_cat_groups:
        print("  (none detected)")
    for col_idx, n_classes in scalar_cat_groups:
        print(f"  {state_cols[col_idx]:<40s}  {n_classes} classes")
    print()

    print("CONTINUOUS (Gaussian-modeled):")
    if not continuous_idx:
        print("  (none)")
    for j in continuous_idx:
        print(f"  {state_cols[j]}")


_ENSEMBLE_TRAIN_DATA = None
_ENSEMBLE_INITIAL_MODELS = None


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, o):
        return self.net(o)


class RecurrentQNetwork(nn.Module):
    def __init__(
        self, obs_dim: int, n_actions: int, hidden: int = 128,
        rnn_type: str = "gru", rnn_layers: int = 1,
    ):
        super().__init__()
        assert rnn_type in ("gru", "lstm"), f"rnn_type must be 'gru' or 'lstm', got {rnn_type!r}"
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.rnn_type = rnn_type
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn_type]
        self.rnn = rnn_cls(input_size=obs_dim, hidden_size=hidden, num_layers=rnn_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs_seq: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        lengths_cpu = seq_lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            obs_seq, lengths_cpu, batch_first=True, enforce_sorted=False,
        )
        if self.rnn_type == "lstm":
            _, (h_n, _) = self.rnn(packed)
        else:
            _, h_n = self.rnn(packed)
        final_hidden = h_n[-1]
        return self.head(final_hidden)
