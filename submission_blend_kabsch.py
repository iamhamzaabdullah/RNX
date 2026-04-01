import argparse
import itertools
from typing import List, Tuple

import numpy as np
import pandas as pd


def target_id_from_row_id(row_id: str) -> str:
    parts = str(row_id).split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else str(row_id)


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    Pc = P - cP
    Qc = Q - cQ
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.eye(3)
    if d < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    t = cQ - R @ cP
    return R, t


def align_to(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    R, t = kabsch_align(P, Q)
    return (P @ R.T) + t


def tm_score_proxy(P: np.ndarray, Q: np.ndarray) -> float:
    L = min(len(P), len(Q))
    if L < 5:
        return 0.0
    P2 = P[:L]
    Q2 = Q[:L]
    Pa = align_to(P2, Q2)
    d = np.linalg.norm(Pa - Q2, axis=1)
    d0 = max(0.3, 0.6 * max(L - 0.5, 1.0) ** (1.0 / 3.0) - 2.5)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))


def reshape_target(df_t: pd.DataFrame, n_sample: int = 5) -> np.ndarray:
    L = len(df_t)
    arr = np.zeros((n_sample, L, 3), dtype=np.float32)
    for i in range(1, n_sample + 1):
        arr[i - 1, :, 0] = df_t[f"x_{i}"].to_numpy(np.float32)
        arr[i - 1, :, 1] = df_t[f"y_{i}"].to_numpy(np.float32)
        arr[i - 1, :, 2] = df_t[f"z_{i}"].to_numpy(np.float32)
    return arr


def best_perm(A: np.ndarray, B: np.ndarray) -> Tuple[Tuple[int, ...], np.ndarray]:
    # A, B shape: (5, L, 3)
    m = np.zeros((5, 5), dtype=np.float32)
    for i in range(5):
        for j in range(5):
            m[i, j] = tm_score_proxy(B[j], A[i])
    best_p = tuple(range(5))
    best_s = -1e9
    for p in itertools.permutations(range(5)):
        s = sum(float(m[i, p[i]]) for i in range(5))
        if s > best_s:
            best_s = s
            best_p = p
    return best_p, m


def select_top_candidates(candidates: List[np.ndarray], source_weights: List[float], n_select: int = 5) -> List[np.ndarray]:
    if len(candidates) <= n_select:
        return candidates

    cons = []
    for i in range(len(candidates)):
        vals = []
        for j in range(len(candidates)):
            if i == j:
                continue
            vals.append(tm_score_proxy(candidates[i], candidates[j]))
        cons.append(float(np.mean(vals)) if vals else 0.0)

    base = [0.65 * source_weights[i] + 0.35 * cons[i] for i in range(len(candidates))]

    selected = []
    remaining = list(range(len(candidates)))
    first = max(remaining, key=lambda x: base[x])
    selected.append(first)
    remaining.remove(first)

    while len(selected) < n_select and remaining:
        best_idx = None
        best_val = -1e9
        for idx in remaining:
            div = min(tm_score_proxy(candidates[idx], candidates[s]) for s in selected)
            val = 0.75 * base[idx] + 0.25 * (1.0 - div)
            if val > best_val:
                best_val = val
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def build_output_rows(df_base_t: pd.DataFrame, final_samples: List[np.ndarray]) -> List[dict]:
    out = []
    for r_idx, (_, r) in enumerate(df_base_t.iterrows()):
        row = {"ID": r["ID"], "resname": r["resname"], "resid": int(r["resid"])}
        for s in range(5):
            row[f"x_{s + 1}"] = float(final_samples[s][r_idx, 0])
            row[f"y_{s + 1}"] = float(final_samples[s][r_idx, 1])
            row[f"z_{s + 1}"] = float(final_samples[s][r_idx, 2])
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Kabsch-aligned blend of two RNA submission CSVs.")
    parser.add_argument("--base", required=True, help="Path to base/best submission.csv")
    parser.add_argument("--aux", required=True, help="Path to second submission.csv")
    parser.add_argument("--out", required=True, help="Path to output blended submission.csv")
    parser.add_argument("--alpha", type=float, default=0.70, help="Blend weight for base: blend = alpha*base + (1-alpha)*aux")
    args = parser.parse_args()

    df_a = pd.read_csv(args.base)
    df_b = pd.read_csv(args.aux)

    base_cols = ["ID", "resname", "resid"] + [f"{c}_{i}" for i in range(1, 6) for c in ["x", "y", "z"]]
    for c in base_cols:
        if c not in df_a.columns or c not in df_b.columns:
            raise ValueError(f"Missing required column: {c}")

    # Normalize ordering by ID, resid for deterministic blending.
    df_a = df_a[base_cols].copy()
    df_b = df_b[base_cols].copy()
    df_a["target_id"] = df_a["ID"].map(target_id_from_row_id)
    df_b["target_id"] = df_b["ID"].map(target_id_from_row_id)

    out_rows = []
    targets_a = list(df_a["target_id"].drop_duplicates())

    for tid in targets_a:
        ta = df_a[df_a["target_id"] == tid].sort_values("resid").reset_index(drop=True)
        tb = df_b[df_b["target_id"] == tid].sort_values("resid").reset_index(drop=True)
        if len(ta) != len(tb):
            raise ValueError(f"Length mismatch for target {tid}: {len(ta)} vs {len(tb)}")

        A = reshape_target(ta, n_sample=5)
        B = reshape_target(tb, n_sample=5)

        perm, _ = best_perm(A, B)
        Bp = B[list(perm)]
        Baligned = np.stack([align_to(Bp[i], A[i]) for i in range(5)], axis=0)

        cands = []
        weights = []

        for i in range(5):
            cands.append(A[i])
            weights.append(1.00)
        for i in range(5):
            cands.append(Baligned[i])
            weights.append(0.85)
        for i in range(5):
            cands.append(args.alpha * A[i] + (1.0 - args.alpha) * Baligned[i])
            weights.append(1.10)

        selected = select_top_candidates(cands, weights, n_select=5)
        out_rows.extend(build_output_rows(ta, selected))

    out_df = pd.DataFrame(out_rows)
    out_cols = ["ID", "resname", "resid"] + [f"{c}_{i}" for i in range(1, 6) for c in ["x", "y", "z"]]
    out_df = out_df[out_cols]

    coord_cols = [c for c in out_df.columns if c.startswith(("x_", "y_", "z_"))]
    out_df[coord_cols] = out_df[coord_cols].clip(-999.999, 9999.999)
    out_df.to_csv(args.out, index=False)
    print(f"Saved blended submission: {args.out} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()

