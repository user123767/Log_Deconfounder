import json
import os
import re

import numpy as np
import pandas as pd

PROXY_RE = re.compile(r"_proxy-(?P<cols>.+)-(?P<type>twins|gaussian)(?P<noise>[\d.]+)$")


def proxy_columns(source, proxy_type, n_bins=10, n_reps=3):
    if proxy_type == "gaussian":
        return [f"proxy_{source}"]
    return [f"proxy_{source}_r{r}_b{b}" for r in range(n_reps) for b in range(n_bins)]


def reference_stats(values: dict, n_bins: int) -> dict:
    ref = {}
    for s, v in values.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        ref[s] = {"edges": np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])).tolist(),
                  "mean": float(v.mean()), "sd": float(v.std()) or 1.0}
    return ref


def draw_proxies(v, source, spec, rng):
    v = np.asarray(v, float)
    ref = spec["ref"][source]
    if spec["type"] == "gaussian":
        return ((v - ref["mean"]) / ref["sd"] + rng.normal(0.0, spec["noise"], len(v)))[:, None]
    n_bins, n_reps = spec["n_bins"], spec["n_reps"]
    edges = np.asarray(ref["edges"])
    k = len(edges) + 1
    onehot = np.zeros((len(v), n_bins))
    onehot[np.arange(len(v)), np.digitize(np.nan_to_num(v), edges)] = 1.0
    onehot[:, k:] = 0.0
    bits = np.tile(onehot, (1, n_reps))
    flip = rng.random(bits.shape) < spec["noise"]
    out = np.where(flip, 1.0 - bits, bits)
    out[np.isnan(v)] = np.nan
    return out


def make_spec(sources, proxy_type, noise, n_bins, n_reps, ref):
    return {"sources": list(sources), "type": proxy_type, "noise": float(noise), "n_bins": int(n_bins),
            "n_reps": int(n_reps), "ref": ref}


def spec_path(log_csv_path):
    return log_csv_path[:-4] + "_proxy.json" if log_csv_path.endswith(".csv") else log_csv_path + "_proxy.json"


def save_spec(spec, log_csv_path):
    with open(spec_path(log_csv_path), "w") as f:
        json.dump(spec, f, indent=2)


class ProxyInjector:
    def __init__(self, spec, seed=12345):
        self.spec, self.seed = spec, seed
        self.cols = {s: proxy_columns(s, spec["type"], spec["n_bins"], spec["n_reps"]) for s in spec["sources"]}
        self.case_nr = -1
        self.new_case()

    @classmethod
    def for_dataset(cls, dataset, log_csv_path=None, seed=12345):
        m = PROXY_RE.search(dataset)
        if not m:
            return None
        if log_csv_path and os.path.exists(spec_path(log_csv_path)):
            with open(spec_path(log_csv_path)) as f:
                spec = json.load(f)
            print(f"[proxies] settings from {spec_path(log_csv_path)}")
        else:
            if not log_csv_path or not os.path.exists(log_csv_path):
                raise FileNotFoundError(f"proxy settings for {dataset}: neither {log_csv_path} nor its _proxy.json")
            sources = m["cols"].split("-")
            log = pd.read_csv(log_csv_path, sep=";", usecols=["case_nr"] + sources)
            first = log.groupby("case_nr")[sources].first()
            spec = make_spec(sources, m["type"], float(m["noise"]), 10, 3,
                             reference_stats({s: first[s].to_numpy() for s in sources}, 10))
            print(f"[proxies] settings rebuilt from the name and {log_csv_path} (no _proxy.json)")
        print(f"[proxies] {spec['type']} proxies of {spec['sources']}, noise {spec['noise']:g}")
        return cls(spec, seed)

    @property
    def columns(self):
        return [c for s in self.spec["sources"] for c in self.cols[s]]

    def new_case(self, case_nr=None):
        self.case_nr = self.case_nr + 1 if case_nr is None else case_nr
        self._rng = np.random.default_rng([self.seed, self.case_nr])
        self._draws = {s: None for s in self.spec["sources"]}

    def values(self, events):
        out = {}
        for s in self.spec["sources"]:
            if self._draws[s] is None:
                v = next((float(e[s]) for e in events if s in e and e[s] is not None and pd.notna(e[s])), np.nan)
                if np.isfinite(v):
                    self._draws[s] = draw_proxies([v], s, self.spec, self._rng)[0]
            vals = self._draws[s] if self._draws[s] is not None else np.zeros(len(self.cols[s]))
            out.update(dict(zip(self.cols[s], np.nan_to_num(vals).tolist())))
        return out

    def add_raw(self, events):
        vals = self.values(events)
        return [{**e, **vals} for e in events]
