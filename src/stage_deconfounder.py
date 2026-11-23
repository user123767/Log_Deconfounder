import math
import os

import numpy as np


def decision_token(obs: np.ndarray, prev_slot: int, stage: int, n_slots: int, n_stages: int) -> np.ndarray:
    t = np.zeros(len(obs) + n_slots + n_stages, dtype=np.float32)
    t[:len(obs)] = obs
    if prev_slot >= 0:
        t[len(obs) + prev_slot] = 1.0
    if 0 <= stage < n_stages:
        t[len(obs) + n_slots + stage] = 1.0
    return t
