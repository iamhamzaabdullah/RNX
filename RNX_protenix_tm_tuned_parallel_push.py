# ==== CELL 1 ====
!pip install --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl

!pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl

!pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl

# ==== CELL 2 ====
import os
import sys
import pandas as pd

# ── Local vs Kaggle mode ─────────────────────────────────────────────────────
# On Kaggle competition rerun, KAGGLE_IS_COMPETITION_RERUN is set to a truthy value.
# When running locally we do NOT exit — instead we cap the test set to a small
# number of samples so the notebook finishes quickly.

IS_KAGGLE = True #bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN", ""))

# How many test samples to use when running locally
LOCAL_N_SAMPLES = None

if IS_KAGGLE:
    print("Running in KAGGLE COMPETITION mode — all test targets will be processed.")
else:
    print(f"Running in LOCAL mode — only the first {LOCAL_N_SAMPLES} test targets "
          f"will be processed to save time.")

# ==== CELL 3 ====
import gc
import json
import os
import time
import warnings

os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import BiopythonDeprecationWarning
from Bio.Align import PairwiseAligner
from pandas.errors import DtypeWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=DtypeWarning)

# ==== CELL 4 ====
def get_c1_mask(data: dict, atom_array) -> torch.Tensor:
    # 1. Try atom_array attributes first
    if atom_array is not None:
        try:
            if hasattr(atom_array, "centre_atom_mask"):
                m = atom_array.centre_atom_mask == 1
                if hasattr(atom_array, "is_rna"):
                    m = m & atom_array.is_rna
                return torch.from_numpy(m).bool()
            
            if hasattr(atom_array, "atom_name"):
                base = atom_array.atom_name == "C1'"
                if hasattr(atom_array, "is_rna"):
                    base = base & atom_array.is_rna
                return torch.from_numpy(base).bool()
        except Exception:
            pass

    # 2. Fallback to feature dict
    f = data["input_feature_dict"]
    
    if "centre_atom_mask" in f:
        return (f["centre_atom_mask"] == 1).bool()
    if "center_atom_mask" in f:
        return (f["center_atom_mask"] == 1).bool()
        
    # Heuristic fallback: check which index gives us roughly N_token atoms
    n_tokens = data.get("N_token", torch.tensor(0)).item()
    mask11 = (f["atom_to_tokatom_idx"] == 11).bool()
    mask12 = (f["atom_to_tokatom_idx"] == 12).bool()
    
    c11 = mask11.sum().item()
    c12 = mask12.sum().item()
    
    # Return the one closer to N_tokens (likely one per residue)
    if abs(c11 - n_tokens) < abs(c12 - n_tokens):
        return mask11
    else:
        return mask12

# ==== CELL 6 ====
# ─────────────── Paths & Constants ───────────────────────────────────────────
DATA_BASE              = "/kaggle/input/competitions/stanford-rna-3d-folding-2"
DEFAULT_TEST_CSV       = f"{DATA_BASE}/test_sequences.csv"
DEFAULT_TRAIN_CSV      = f"{DATA_BASE}/train_sequences.csv"
DEFAULT_TRAIN_LBLS     = f"{DATA_BASE}/train_labels.csv"
DEFAULT_VAL_CSV        = f"{DATA_BASE}/validation_sequences.csv"
DEFAULT_VAL_LBLS       = f"{DATA_BASE}/validation_labels.csv"
DEFAULT_OUTPUT         = "/kaggle/working/submission.csv"

DEFAULT_CODE_DIR = (
    "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted"
    "/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
)
DEFAULT_ROOT_DIR = DEFAULT_CODE_DIR

MODEL_NAME    = "protenix_base_20250630_v1.0.0"
N_SAMPLE      = 5
SEED          = 42
MAX_SEQ_LEN   = int(os.environ.get("MAX_SEQ_LEN",   "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP",  "128"))
USE_TM_CONSENSUS_RANK = True

# TBM quality thresholds — sequences below these get routed to Protenix
MIN_SIMILARITY       = float(os.environ.get("MIN_SIMILARITY",       "0.0"))
MIN_PERCENT_IDENTITY = float(os.environ.get("MIN_PERCENT_IDENTITY", "50.0"))

# Set False to skip Protenix and use de-novo fallback instead
USE_PROTENIX = True
FORCE_PROTENIX_ALL_TARGETS = False
MIN_PROTENIX_CANDS_PER_TARGET = 1
EXTRA_PROTENIX_FOR_WEAK_TBM = True
WEAK_TBM_PCT_THRESHOLD = float(os.environ.get("WEAK_TBM_PCT_THRESHOLD", "86.0"))
EXTRA_PROTENIX_MAX_TARGETS = int(os.environ.get("EXTRA_PROTENIX_MAX_TARGETS", "12"))
FORCE_MEDIUM_PROTENIX = True
MEDIUM_PROTENIX_MIN_LEN = int(os.environ.get("MEDIUM_PROTENIX_MIN_LEN", "150"))
MEDIUM_PROTENIX_MAX_LEN = int(os.environ.get("MEDIUM_PROTENIX_MAX_LEN", "420"))
MEDIUM_PROTENIX_EXTRA_TARGETS = int(os.environ.get("MEDIUM_PROTENIX_EXTRA_TARGETS", "6"))
OPTIONAL_PROTENIX_MAX_TARGETS = int(os.environ.get("OPTIONAL_PROTENIX_MAX_TARGETS", "2"))
TARGET_TOTAL_RUNTIME_MIN = float(os.environ.get("TARGET_TOTAL_RUNTIME_MIN", "30"))
PHASE1_EXPECTED_SEC = float(os.environ.get("PHASE1_EXPECTED_SEC", "700"))
PROTENIX_TIME_BUDGET_SEC = float(
    os.environ.get(
        "PROTENIX_TIME_BUDGET_SEC",
        str(max(420.0, TARGET_TOTAL_RUNTIME_MIN * 60.0 - PHASE1_EXPECTED_SEC)),
    )
)
MAX_TBM_CANDS = int(os.environ.get("MAX_TBM_CANDS", "9"))
PROTENIX_MAX_SEQ_LEN = int(os.environ.get("PROTENIX_MAX_SEQ_LEN", "900"))
LONG_SEQ_RELAXED_MIN_PCT = float(os.environ.get("LONG_SEQ_RELAXED_MIN_PCT", "36.0"))
SHORT_SEQ_RELAXED_MIN_PCT = float(os.environ.get("SHORT_SEQ_RELAXED_MIN_PCT", "42.0"))
SHORT_SEQ_RELAXED_TOPUP = int(os.environ.get("SHORT_SEQ_RELAXED_TOPUP", "1"))


def parse_bool(value: str, default: bool = False) -> str:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return "true"
    if v in {"0", "false", "f", "no", "n", "off"}:
        return "false"
    return "true" if default else "false"


USE_MSA      = parse_bool(os.environ.get("USE_MSA",      "false"))
USE_TEMPLATE = parse_bool(os.environ.get("USE_TEMPLATE", "false"))
USE_RNA_MSA  = parse_bool(os.environ.get("USE_RNA_MSA",  "true"))

MODEL_N_SAMPLE = int(os.environ.get("MODEL_N_SAMPLE", str(N_SAMPLE)))


# ─────────────── General Utilities ───────────────────────────────────────────
def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True


def resolve_paths():
    test_csv   = os.environ.get("TEST_CSV",           DEFAULT_TEST_CSV)
    output_csv = os.environ.get("SUBMISSION_CSV",     DEFAULT_OUTPUT)
    code_dir   = os.environ.get("PROTENIX_CODE_DIR",  DEFAULT_CODE_DIR)
    root_dir   = os.environ.get("PROTENIX_ROOT_DIR",  DEFAULT_ROOT_DIR)
    return test_csv, output_csv, code_dir, root_dir


def ensure_required_files(root_dir: str) -> None:
    for p, name in [
        (Path(root_dir) / "checkpoint" / f"{MODEL_NAME}.pt",          "checkpoint"),
        (Path(root_dir) / "common" / "components.cif",                "CCD file"),
        (Path(root_dir) / "common" / "components.cif.rdkit_mol.pkl",  "CCD cache"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {name}: {p}")


# ─────────────── Protenix Input / Config Helpers ─────────────────────────────
def build_input_json(df: pd.DataFrame, json_path: str) -> None:
    data = [
        {
            "name": row["target_id"],
            "covalent_bonds": [],
            "sequences": [{"rnaSequence": {"sequence": row["sequence"], "count": 1}}],
        }
        for _, row in df.iterrows()
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def build_configs(input_json_path: str, dump_dir: str, model_name: str):
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs

    base = {**configs_base, **{"data": data_configs}, **inference_configs}

    def deep_update(t, p):
        for k, v in p.items():
            if isinstance(v, dict) and k in t and isinstance(t[k], dict):
                deep_update(t[k], v)
            else:
                t[k] = v

    deep_update(base, model_configs[model_name])
    arg_str = " ".join([
        f"--model_name {model_name}",
        f"--input_json_path {input_json_path}",
        f"--dump_dir {dump_dir}",
        f"--use_msa {USE_MSA}",
        f"--use_template {USE_TEMPLATE}",
        f"--use_rna_msa {USE_RNA_MSA}",
        f"--sample_diffusion.N_sample {MODEL_N_SAMPLE}",
        f"--seeds {SEED}",
    ])
    return parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)


def get_c1_mask(data: dict, atom_array) -> torch.Tensor:
    # 1. Try atom_array attributes first
    if atom_array is not None:
        try:
            if hasattr(atom_array, "centre_atom_mask"):
                m = atom_array.centre_atom_mask == 1
                if hasattr(atom_array, "is_rna"):
                    m = m & atom_array.is_rna
                return torch.from_numpy(m).bool()
            
            if hasattr(atom_array, "atom_name"):
                base = atom_array.atom_name == "C1'"
                if hasattr(atom_array, "is_rna"):
                    base = base & atom_array.is_rna
                return torch.from_numpy(base).bool()
        except Exception:
            pass

    # 2. Fallback to feature dict
    f = data["input_feature_dict"]
    
    # CASE A: center_atom_mask exists
    if "center_atom_mask" in f:
        return (f["center_atom_mask"] == 1).bool()
    if "centre_atom_mask" in f:
        return (f["centre_atom_mask"] == 1).bool()
        
    # CASE B: Use atom_name
    if "atom_name" in f:
        # Check against "C1'" (byte encoded or string?)
        # For now assume typical behavior is center_atom_mask is present.
        pass

    # CASE C: atom_to_tokatom_idx fallback
    # The index for C1' is typically 11 or 12 depending on featurizer.
    # Let's try to match exactly C1' if possible.
    # But usually 'centre_atom_mask' should be there.
    
    # If we fall through, assume standard mask
    return (f["atom_to_tokatom_idx"] == 11).bool()


def get_feature_c1_mask(data: dict) -> torch.Tensor:
    f = data["input_feature_dict"]
    if "centre_atom_mask" in f:
        return f["centre_atom_mask"].long() == 1
    return f["atom_to_tokatom_idx"].long() == 12


def coords_to_rows(target_id: str, seq: str, coords: np.ndarray) -> list:
    """coords shape: (N_SAMPLE, seq_len, 3)"""
    rows = []
    for i in range(len(seq)):
        row = {"ID": f"{target_id}_{i + 1}", "resname": seq[i], "resid": i + 1}
        for s in range(N_SAMPLE):
            if s < coords.shape[0] and i < coords.shape[1]:
                x, y, z = coords[s, i]
            else:
                x, y, z = 0.0, 0.0, 0.0
            row[f"x_{s + 1}"] = float(x)
            row[f"y_{s + 1}"] = float(y)
            row[f"z_{s + 1}"] = float(z)
        rows.append(row)
    return rows


def pad_samples(coords: np.ndarray, n: int) -> np.ndarray:
    if coords.shape[0] >= n:
        return coords[:n]
    if coords.shape[0] == 0:
        return np.zeros((n, coords.shape[1], 3), dtype=coords.dtype)
    extra = np.repeat(coords[:1], n - coords.shape[0], axis=0)
    return np.concatenate([coords, extra], axis=0)


def split_into_chunks(seq_len: int, max_len: int, overlap: int) -> list:
    """Split a sequence into overlapping (start, end) chunks."""
    if seq_len <= max_len:
        return [(0, seq_len)]
    chunks = []
    step = max_len - overlap
    pos = 0
    while pos < seq_len:
        end = min(pos + max_len, seq_len)
        chunks.append((pos, end))
        if end == seq_len:
            break
        pos += step
    return chunks


def kabsch_align(P: np.ndarray, Q: np.ndarray):
    """Compute optimal rotation R and translation t so that  R @ P + t ≈ Q."""
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    Pc = P - centroid_P
    Qc = Q - centroid_Q
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.eye(3)
    if d < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    t = centroid_Q - R @ centroid_P
    return R, t

def tm_score_proxy(P: np.ndarray, Q: np.ndarray) -> float:
    L = min(len(P), len(Q))
    if L < 5:
        return 0.0
    P2 = P[:L]
    Q2 = Q[:L]
    R, t = kabsch_align(P2, Q2)
    Pa = (P2 @ R.T) + t
    d = np.linalg.norm(Pa - Q2, axis=1)
    d0 = max(0.3, 0.6 * max(L - 0.5, 1.0) ** (1.0 / 3.0) - 2.5)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))

def rmsd_after_align(P: np.ndarray, Q: np.ndarray) -> float:
    L = min(len(P), len(Q))
    if L < 3:
        return 99.0
    P2 = P[:L]
    Q2 = Q[:L]
    R, t = kabsch_align(P2, Q2)
    Pa = (P2 @ R.T) + t
    return float(np.sqrt(np.mean(np.sum((Pa - Q2) ** 2, axis=1))))

def select_top_candidates(candidates: list, source_weights: list, n_select: int) -> list:
    if len(candidates) <= n_select:
        return candidates

    # Consensus TM proxy
    cons = []
    for i in range(len(candidates)):
        vals = []
        for j in range(len(candidates)):
            if i == j:
                continue
            vals.append(tm_score_proxy(candidates[i], candidates[j]))
        cons.append(float(np.mean(vals)) if vals else 0.0)

    base = [0.55 * source_weights[i] + 0.45 * cons[i] for i in range(len(candidates))]

    selected = []
    remaining = list(range(len(candidates)))
    first = max(remaining, key=lambda x: base[x])
    selected.append(first)
    remaining.remove(first)

    while len(selected) < n_select and remaining:
        best_idx = None
        best_val = -1e9
        for idx in remaining:
            min_div = min(rmsd_after_align(candidates[idx], candidates[s]) for s in selected)
            val = 0.70 * base[idx] + 0.30 * min(min_div / 12.0, 1.0)
            if val > best_val:
                best_val = val
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def stitch_chunk_coords(chunk_coords_list: list,
                        chunk_ranges: list,
                        seq_len: int) -> np.ndarray:
    """
    Merge overlapping chunk coordinates into a full sequence geometry.
    Applies Kabsch alignment on overlapping residues, and smoothly
    blends the coordinates using a linear weight ramp.
    """
    if len(chunk_coords_list) == 1:
        coords = chunk_coords_list[0]
        if coords.shape[0] >= seq_len:
            return coords[:seq_len]
        out = np.zeros((seq_len, 3), dtype=coords.dtype)
        out[:coords.shape[0]] = coords
        return out

    # Start with the first chunk aligned to itself (identity)
    aligned = [chunk_coords_list[0].copy()]

    for i in range(1, len(chunk_coords_list)):
        prev_start, prev_end = chunk_ranges[i - 1]
        cur_start, cur_end = chunk_ranges[i]

        ov_start = cur_start
        ov_end = min(prev_end, cur_end)
        ov_len = ov_end - ov_start

        if ov_len < 3:
            # Cannot align reliably, just trust the coordinates as-is
            aligned.append(chunk_coords_list[i].copy())
            continue

        prev_ov = aligned[i - 1][ov_start - prev_start: ov_end - prev_start]
        cur_ov = chunk_coords_list[i][ov_start - cur_start: ov_end - cur_start]

        # Ignore invalid residues (e.g. padding/blank)
        valid = ~(np.isnan(prev_ov).any(axis=1) | np.isnan(cur_ov).any(axis=1))
        if valid.sum() < 3:
            aligned.append(chunk_coords_list[i].copy())
            continue

        # Align current chunk to previous chunk using only the overlap region
        R, t = kabsch_align(cur_ov[valid], prev_ov[valid])
        transformed = (chunk_coords_list[i] @ R.T) + t
        aligned.append(transformed)

    # Blend them together
    full = np.zeros((seq_len, 3), dtype=np.float64)
    weights = np.zeros(seq_len, dtype=np.float64)

    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        chunk_len = coords.shape[0]
        actual_end = min(s + chunk_len, seq_len)
        used_len = actual_end - s

        w = np.ones(used_len, dtype=np.float64)

        if i > 0:
            ov_start = s
            ov_end = min(chunk_ranges[i - 1][1], e)
            ramp_len = ov_end - ov_start
            if ramp_len > 0:
                w[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)

        if i < len(chunk_ranges) - 1:
            next_s = chunk_ranges[i + 1][0]
            ramp_start = next_s - s
            ramp_len = actual_end - next_s
            if ramp_len > 0 and ramp_start < used_len:
                w[ramp_start:used_len] = np.linspace(1.0, 0.0, ramp_len)

        full[s:actual_end] += coords[:used_len] * w[:, None]
        weights[s:actual_end] += w

    mask = weights > 0
    full[mask] /= weights[mask, None]

    return full


# ─────────────── TBM Core Functions ──────────────────────────────────────────
def _make_aligner() -> PairwiseAligner:
    al = PairwiseAligner()
    al.mode                           = "global"
    al.match_score                    = 2
    al.mismatch_score                 = -1.5
    al.open_gap_score                 = -8
    al.extend_gap_score               = -0.4
    # Biopython >=1.86 names
    al.open_left_deletion_score       = -8
    al.extend_left_deletion_score     = -0.4
    al.open_right_deletion_score      = -8
    al.extend_right_deletion_score    = -0.4
    al.open_left_insertion_score      = -8
    al.extend_left_insertion_score    = -0.4
    al.open_right_insertion_score     = -8
    al.extend_right_insertion_score   = -0.4
    return al


_aligner = _make_aligner()


def parse_stoichiometry(stoich: str) -> list:
    if pd.isna(stoich) or str(stoich).strip() == "":
        return []
    return [(ch.strip(), int(cnt)) for part in str(stoich).split(";")
            for ch, cnt in [part.split(":")]]


def parse_fasta(fasta_content: str) -> dict:
    out, cur, parts = {}, None, []
    for line in str(fasta_content).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur is not None:
                out[cur] = "".join(parts)
            cur = line[1:].split()[0]
            parts = []
        else:
            parts.append(line.replace(" ", ""))
    if cur is not None:
        out[cur] = "".join(parts)
    return out


def get_chain_segments(row) -> list:
    seq    = row["sequence"]
    stoich = row.get("stoichiometry", "")
    all_sq = row.get("all_sequences", "")
    if (pd.isna(stoich) or pd.isna(all_sq)
            or str(stoich).strip() == "" or str(all_sq).strip() == ""):
        return [(0, len(seq))]
    try:
        chain_dict = parse_fasta(all_sq)
        order = parse_stoichiometry(stoich)
        segs, pos = [], 0
        for ch, cnt in order:
            base = chain_dict.get(ch)
            if base is None:
                return [(0, len(seq))]
            for _ in range(cnt):
                segs.append((pos, pos + len(base)))
                pos += len(base)
        return segs if pos == len(seq) else [(0, len(seq))]
    except Exception:
        return [(0, len(seq))]


def build_segments_map(df: pd.DataFrame) -> tuple:
    seg_map, stoich_map = {}, {}
    for _, r in df.iterrows():
        tid               = r["target_id"]
        seg_map[tid]      = get_chain_segments(r)
        raw_s             = r.get("stoichiometry", "")
        stoich_map[tid]   = "" if pd.isna(raw_s) else str(raw_s)
    return seg_map, stoich_map


def process_labels(labels_df: pd.DataFrame) -> dict:
    coords = {}
    prefixes = labels_df["ID"].str.rsplit("_", n=1).str[0]
    for prefix, grp in labels_df.groupby(prefixes):
        coords[prefix] = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return coords


def _build_aligned_strings(query_seq, template_seq, alignment):
    q_segs, t_segs = alignment.aligned
    aq, at, qi, ti = [], [], 0, 0
    for (qs, qe), (ts, te) in zip(q_segs, t_segs):
        while qi < qs: aq.append(query_seq[qi]);    at.append("-");              qi += 1
        while ti < ts: aq.append("-");              at.append(template_seq[ti]); ti += 1
        for qp, tp in zip(range(qs, qe), range(ts, te)):
            aq.append(query_seq[qp]); at.append(template_seq[tp])
        qi, ti = qe, te
    while qi < len(query_seq):    aq.append(query_seq[qi]);    at.append("-");              qi += 1
    while ti < len(template_seq): aq.append("-");              at.append(template_seq[ti]); ti += 1
    return "".join(aq), "".join(at)


def find_similar_sequences_detailed(query_seq, train_seqs_df, train_coords_dict, top_n=30):
    results = []
    for _, row in train_seqs_df.iterrows():
        tid, tseq = row["target_id"], row["sequence"]
        if tid not in train_coords_dict:
            continue
        if abs(len(tseq) - len(query_seq)) / max(len(tseq), len(query_seq)) > 0.3:
            continue
        aln       = next(iter(_aligner.align(query_seq, tseq)))
        norm_s    = aln.score / (2 * min(len(query_seq), len(tseq)))
        identical = sum(
            1 for (qs, qe), (ts, te) in zip(*aln.aligned)
            for qp, tp in zip(range(qs, qe), range(ts, te))
            if query_seq[qp] == tseq[tp]
        )
        pct_id = 100 * identical / len(query_seq)
        aq, at = _build_aligned_strings(query_seq, tseq, aln)
        results.append((tid, tseq, norm_s, train_coords_dict[tid], pct_id, aq, at))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]


def adapt_template_to_query(query_seq, template_seq, template_coords) -> np.ndarray:
    aln        = next(iter(_aligner.align(query_seq, template_seq)))
    new_coords = np.full((len(query_seq), 3), np.nan)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        chunk = template_coords[ts:te]
        if len(chunk) == (qe - qs):
            new_coords[qs:qe] = chunk
    for i in range(len(new_coords)):
        if np.isnan(new_coords[i, 0]):
            pv = next((j for j in range(i - 1, -1, -1) if not np.isnan(new_coords[j, 0])), -1)
            nv = next((j for j in range(i + 1, len(new_coords)) if not np.isnan(new_coords[j, 0])), -1)
            if pv >= 0 and nv >= 0:
                w = (i - pv) / (nv - pv)
                new_coords[i] = (1 - w) * new_coords[pv] + w * new_coords[nv]
            elif pv >= 0:
                new_coords[i] = new_coords[pv] + [3, 0, 0]
            elif nv >= 0:
                new_coords[i] = new_coords[nv] + [3, 0, 0]
            else:
                new_coords[i] = [i * 3, 0, 0]
    return np.nan_to_num(new_coords)


def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2) -> np.ndarray:
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3:
                continue
            # bond i–i+1  ~5.95 Å
            d    = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj  = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj; C[1:] += adj
            # soft i–i+2  ~10.2 Å
            d2   = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
            C[:-2] -= adj2; C[2:] += adj2
            # Laplacian smoothing
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            # self-avoidance
            if L >= 25:
                idx  = np.linspace(0, L - 1, min(L, 160)).astype(int) if L > 220 else np.arange(L)
                P    = C[idx]; diff = P[:, None, :] - P[None, :, :]
                dm   = np.linalg.norm(diff, axis=2) + 1e-6
                sep  = np.abs(idx[:, None] - idx[None, :])
                mask = (sep > 2) & (dm < 3.2)
                if np.any(mask):
                    vec = (diff * ((3.2 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
                    C[idx] += (0.015 * strength) * vec
            X[s:e] = C
    return X


def _rotmat(axis, ang):
    a = np.asarray(axis, float); a /= np.linalg.norm(a) + 1e-12
    x, y, z = a; c, s = np.cos(ang), np.sin(ang); CC = 1 - c
    return np.array([[c+x*x*CC, x*y*CC-z*s, x*z*CC+y*s],
                     [y*x*CC+z*s, c+y*y*CC, y*z*CC-x*s],
                     [z*x*CC-y*s, z*y*CC+x*s, c+z*z*CC]])


def apply_hinge(coords, seg, rng, deg=22):
    s, e = seg; L = e - s
    if L < 30: return coords
    pivot = s + int(rng.integers(10, L - 10))
    R = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg, deg))))
    X = coords.copy(); p0 = X[pivot].copy()
    X[pivot+1:e] = (X[pivot+1:e] - p0) @ R.T + p0
    return X


def jitter_chains(coords, segs, rng, deg=12, trans=1.5):
    X = coords.copy(); gc_ = X.mean(0, keepdims=True)
    for s, e in segs:
        R     = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg, deg))))
        shift = rng.normal(size=3); shift = shift / (np.linalg.norm(shift) + 1e-12) * float(rng.uniform(0, trans))
        c     = X[s:e].mean(0, keepdims=True)
        X[s:e] = (X[s:e] - c) @ R.T + c + shift
    X -= X.mean(0, keepdims=True) - gc_
    return X


def smooth_wiggle(coords, segs, rng, amp=0.8):
    X = coords.copy()
    for s, e in segs:
        L = e - s
        if L < 20: continue
        ctrl = np.linspace(0, L - 1, 6); disp = rng.normal(0, amp, (6, 3)); t = np.arange(L)
        X[s:e] += np.vstack([np.interp(t, ctrl, disp[:, k]) for k in range(3)]).T
    return X


def generate_rna_structure(sequence: str, seed=None) -> np.ndarray:
    """Idealized A-form RNA helix — last-resort de-novo fallback."""
    if seed is not None:
        np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    for i in range(n):
        ang = i * 0.6
        coords[i] = [10.0 * np.cos(ang), 10.0 * np.sin(ang), i * 2.5]
    return coords


def estimate_protenix_seconds(seq_len: int, n_needed: int) -> float:
    # Calibrated from observed Kaggle runs: nonlinear in length, weakly scaled by sample count.
    sample_factor = 0.55 + 0.15 * max(1, int(n_needed))
    return 12.0 + 0.0016 * float(seq_len * seq_len) * sample_factor


# ─────────────── TBM Phase ───────────────────────────────────────────────────
def tbm_phase(test_df, train_seqs_df, train_coords_dict, segments_map):
    """
    Phase 1 — Template-Based Modeling.

    Returns
    -------
    template_predictions : {target_id: [np.ndarray(seq_len, 3), ...]}
        0 to N_SAMPLE predictions per target, from real templates.
    protenix_queue : {target_id: (n_needed, full_sequence)}
        Targets that still need more predictions.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 1: Template-Based Modeling")
    print(f"  MIN_SIMILARITY = {MIN_SIMILARITY}  |  MIN_PCT_IDENTITY = {MIN_PERCENT_IDENTITY}")
    print(f"{'='*60}")
    t0 = time.time()

    template_predictions: dict = {}
    protenix_queue:       dict = {}

    weak_extra_count = 0
    medium_extra_count = 0
    optional_extra_count = 0
    estimated_protenix_sec = 0.0
    for _, row in test_df.iterrows():
        tid = row["target_id"]
        seq = row["sequence"]
        segs = segments_map.get(tid, [(0, len(seq))])

        similar = find_similar_sequences_detailed(seq, train_seqs_df, train_coords_dict, top_n=40)
        preds   = []
        used    = set()
        best_pct = 0.0

        for i, (tmpl_id, tmpl_seq, sim, tmpl_coords, pct_id, _, _) in enumerate(similar):
            if len(preds) >= MAX_TBM_CANDS:
                break
            if sim < MIN_SIMILARITY:
                break           # list is sorted by similarity, so no point continuing
            if pct_id < MIN_PERCENT_IDENTITY:
                continue
            if tmpl_id in used:
                continue

            rng     = np.random.default_rng((row.name * 10000000000 + i * 10007) % (2**32))
            adapted = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)

            # Diversity transforms (same strategy as the 0-409 TBM notebook)
            slot = len(preds)
            if slot == 0:
                X = adapted
            elif slot == 1:
                X = adapted + rng.normal(0, max(0.01, (0.40 - sim) * 0.06), adapted.shape)
            elif slot == 2:
                longest = max(segs, key=lambda se: se[1] - se[0])
                X = apply_hinge(adapted, longest, rng)
            elif slot == 3:
                X = jitter_chains(adapted, segs, rng)
            else:
                X = smooth_wiggle(adapted, segs, rng)

            refined = adaptive_rna_constraints(X, tid, segments_map, confidence=sim)
            preds.append(refined)
            used.add(tmpl_id)
            best_pct = max(best_pct, float(pct_id))

        strict_count = len(preds)
        if len(preds) < N_SAMPLE:
            if len(seq) > PROTENIX_MAX_SEQ_LEN:
                relaxed_min_pct = LONG_SEQ_RELAXED_MIN_PCT
                relaxed_goal = N_SAMPLE
            else:
                relaxed_min_pct = SHORT_SEQ_RELAXED_MIN_PCT
                relaxed_goal = min(N_SAMPLE, len(preds) + max(SHORT_SEQ_RELAXED_TOPUP, 0))

            if relaxed_goal > len(preds):
                for i, (tmpl_id, tmpl_seq, sim, tmpl_coords, pct_id, _, _) in enumerate(similar):
                    if len(preds) >= relaxed_goal or len(preds) >= MAX_TBM_CANDS:
                        break
                    if sim < MIN_SIMILARITY:
                        break
                    if tmpl_id in used:
                        continue
                    if pct_id < relaxed_min_pct:
                        continue

                    rng = np.random.default_rng((row.name * 1000003 + i * 20011 + 17) % (2**32))
                    adapted = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)
                    slot = len(preds)
                    if slot % 3 == 0:
                        X = adapted + rng.normal(0, 0.10, adapted.shape)
                    elif slot % 3 == 1:
                        X = smooth_wiggle(adapted, segs, rng, amp=0.45)
                    else:
                        X = jitter_chains(adapted, segs, rng)
                    refined = adaptive_rna_constraints(
                        X, tid, segments_map, confidence=max(0.35, 0.9 * sim)
                    )
                    preds.append(refined)
                    used.add(tmpl_id)
                    best_pct = max(best_pct, float(pct_id))

        relaxed_added = len(preds) - strict_count
        template_predictions[tid] = preds
        n_needed = N_SAMPLE - len(preds)
        long_seq_skip = False
        if len(seq) > PROTENIX_MAX_SEQ_LEN and n_needed > 0:
            n_needed = 0
            long_seq_skip = True
        if FORCE_PROTENIX_ALL_TARGETS and USE_PROTENIX and len(seq) <= PROTENIX_MAX_SEQ_LEN:
            n_needed = max(n_needed, MIN_PROTENIX_CANDS_PER_TARGET)

        # Optional extras are budget-gated so runtime stays near the 30-min target.
        if n_needed == 0 and USE_PROTENIX and len(preds) >= N_SAMPLE and len(seq) <= PROTENIX_MAX_SEQ_LEN:
            opt_reason = ""
            if (EXTRA_PROTENIX_FOR_WEAK_TBM
                    and best_pct < WEAK_TBM_PCT_THRESHOLD
                    and weak_extra_count < EXTRA_PROTENIX_MAX_TARGETS):
                opt_reason = "weak"
            elif (FORCE_MEDIUM_PROTENIX
                    and MEDIUM_PROTENIX_MIN_LEN <= len(seq) <= MEDIUM_PROTENIX_MAX_LEN
                    and medium_extra_count < MEDIUM_PROTENIX_EXTRA_TARGETS):
                opt_reason = "medium"

            if opt_reason and optional_extra_count < OPTIONAL_PROTENIX_MAX_TARGETS:
                est = estimate_protenix_seconds(len(seq), 1)
                if estimated_protenix_sec + est <= PROTENIX_TIME_BUDGET_SEC:
                    n_needed = 1
                    optional_extra_count += 1
                    if opt_reason == "weak":
                        weak_extra_count += 1
                    else:
                        medium_extra_count += 1

        if n_needed > 0:
            protenix_queue[tid] = (n_needed, seq)
            estimated_protenix_sec += estimate_protenix_seconds(len(seq), n_needed)
            msg = f"  {tid} ({len(seq)} nt): {len(preds)} TBM"
            if relaxed_added > 0:
                msg += f" (+{relaxed_added} relaxed)"
            msg += f" → need {n_needed} from Protenix"
            print(msg)
        else:
            if len(preds) >= N_SAMPLE:
                msg = f"  {tid} ({len(seq)} nt): all {N_SAMPLE} from TBM ✓"
                if relaxed_added > 0:
                    msg += f" (+{relaxed_added} relaxed)"
                print(msg)
            elif long_seq_skip:
                print(
                    f"  {tid} ({len(seq)} nt): {len(preds)} TBM"
                    f" (+{relaxed_added} relaxed); Protenix skipped (> {PROTENIX_MAX_SEQ_LEN} nt)"
                )
            else:
                print(
                    f"  {tid} ({len(seq)} nt): {len(preds)} TBM"
                    f" (+{relaxed_added} relaxed); local fallback will fill remaining"
                )

    elapsed = time.time() - t0
    n_full  = len(test_df) - len(protenix_queue)
    print(f"\nPhase 1 done in {elapsed:.1f}s")
    print(f"  Fully covered by TBM : {n_full}")
    print(f"  Need Protenix        : {len(protenix_queue)}")
    print(f"  Optional extras used : {optional_extra_count} / {OPTIONAL_PROTENIX_MAX_TARGETS}")
    print(f"  Protenix budget est  : {estimated_protenix_sec:.0f}s / {PROTENIX_TIME_BUDGET_SEC:.0f}s")
    return template_predictions, protenix_queue


# ─────────────── Main ────────────────────────────────────────────────────────
def main() -> None:
    test_csv, output_csv, code_dir, root_dir = resolve_paths()

    if not os.path.isdir(code_dir):
        raise FileNotFoundError(
            f"Missing PROTENIX_CODE_DIR: {code_dir}. "
            "Set PROTENIX_CODE_DIR to the repo path."
        )

    os.environ["PROTENIX_ROOT_DIR"] = root_dir
    sys.path.append(code_dir)
    ensure_required_files(root_dir)
    seed_everything(SEED)

    # ── Load test data ──────────────────────────────────────────────────────
    test_df_full = pd.read_csv(test_csv)
    test_df      = (test_df_full.head(LOCAL_N_SAMPLES) if not IS_KAGGLE
                    else test_df_full).reset_index(drop=True)
    print(f"Test targets : {len(test_df)}"
          + (" (LOCAL MODE)" if not IS_KAGGLE else ""))

    seq_by_id = dict(zip(test_df["target_id"], test_df["sequence"]))

    # Truncated copy for Protenix (Protenix has token limits)
    test_df_trunc = test_df.copy()
    test_df_trunc["sequence"] = test_df_trunc["sequence"].str[:MAX_SEQ_LEN]

    # ── Load training data for TBM ──────────────────────────────────────────
    print("\nLoading training data for TBM …")
    train_seqs   = pd.read_csv(DEFAULT_TRAIN_CSV)
    val_seqs     = pd.read_csv(DEFAULT_VAL_CSV)
    train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS, low_memory=False)
    val_labels   = pd.read_csv(DEFAULT_VAL_LBLS, low_memory=False)

    combined_seqs   = pd.concat([train_seqs,   val_seqs],    ignore_index=True)
    combined_labels = pd.concat([train_labels, val_labels],  ignore_index=True)
    train_coords    = process_labels(combined_labels)
    segments_map, _ = build_segments_map(test_df)

    print(f"Template pool: {len(combined_seqs)} sequences, {len(train_coords)} structures")

    # ─── PHASE 1: TBM ──────────────────────────────────────────────────────
    template_preds, protenix_queue = tbm_phase(
        test_df, combined_seqs, train_coords, segments_map
    )

    # ─── PHASE 2: Protenix (only for targets that need extra predictions) ──
    protenix_preds: dict = {}   # target_id -> np.ndarray (n_needed, seq_len, 3)

    if protenix_queue and USE_PROTENIX:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Protenix for {len(protenix_queue)} targets")
        print(f"{'='*60}")

        work_dir = Path("/kaggle/working")
        work_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Preparation: create tasks for all sequences/chunks ────────────
        tasks = []          # list of dict for input_json
        chunk_info = {}     # target_id -> list of {"name": chunk_name, "range": (s, e)}
        
        for target_id, (n_needed, full_seq) in protenix_queue.items():
            seq_len = len(full_seq)
            if seq_len <= MAX_SEQ_LEN:
                tasks.append({"target_id": target_id, "sequence": full_seq})
                chunk_info[target_id] = [{"name": target_id, "range": (0, seq_len)}]
                print(f"  {target_id} ({seq_len} nt): single pass queued")
            else:
                chunks = split_into_chunks(seq_len, MAX_SEQ_LEN, CHUNK_OVERLAP)
                print(f"  {target_id} ({seq_len} nt): {len(chunks)} chunks queued "
                      f"{[(s, e) for s, e in chunks]}")
                
                chunk_info[target_id] = []
                for ci, (cs, ce) in enumerate(chunks):
                    chunk_name = f"{target_id}_chunk{ci}"
                    sub_seq = full_seq[cs:ce]
                    tasks.append({"target_id": chunk_name, "sequence": sub_seq})
                    chunk_info[target_id].append({"name": chunk_name, "range": (cs, ce)})

        # Build combined input JSON
        tasks_df = pd.DataFrame(tasks)
        input_json_path = str(work_dir / "protenix_queue_input.json")
        build_input_json(tasks_df, input_json_path)

        from protenix.data.inference.infer_dataloader import InferenceDataset
        from runner.inference import (InferenceRunner,
                                      update_gpu_compatible_configs,
                                      update_inference_configs)

        # Initialize model exactly ONCE
        configs = build_configs(input_json_path, str(work_dir / "outputs"), MODEL_NAME)
        configs = update_gpu_compatible_configs(configs)
        runner  = InferenceRunner(configs)
        dataset = InferenceDataset(configs)

        # ── 2. Inference: process dataset and collect predictions ────────────
        raw_predictions = {}  # sample_name -> coords (np.ndarray or None)

        def _extract_c1_coords(prediction, feat, chunk_seq_len, raw_coords):
            if "centre_atom_mask" in feat:
                mask = (feat["centre_atom_mask"] == 1).to(raw_coords.device)
            elif "atom_to_tokatom_idx" in feat:
                m11 = (feat["atom_to_tokatom_idx"] == 11).to(raw_coords.device)
                m12 = (feat["atom_to_tokatom_idx"] == 12).to(raw_coords.device)
                c11, c12 = m11.sum(), m12.sum()
                mask = m11 if abs(c11 - chunk_seq_len) < abs(c12 - chunk_seq_len) else m12
            else:
                mask = torch.zeros(raw_coords.shape[1], dtype=torch.bool, device=raw_coords.device)
            
            coords = raw_coords[:, mask, :].detach().cpu().numpy()
            
            # Collapse check
            if coords.shape[1] > 1:
                diffs = np.linalg.norm(coords[0, 1:] - coords[0, :-1], axis=-1)
                if np.all(diffs < 1e-4):
                    print(f"    WARNING: Collapsed coordinates detected")
                    return None
            
            if coords.shape[1] != chunk_seq_len:
                if coords.shape[1] == 1 and chunk_seq_len > 1:
                    return None
                padded = np.zeros((coords.shape[0], chunk_seq_len, 3), dtype=np.float32)
                ml = min(coords.shape[1], chunk_seq_len)
                padded[:, :ml, :] = coords[:, :ml, :]
                coords = padded
            return coords

        for i in tqdm(range(len(dataset)), desc="Protenix Inference"):
            data, atom_array, err = dataset[i]
            sample_name = data.get("sample_name", f"sample_{i}")
            
            if err:
                print(f"  {sample_name} data error: {err}")
                raw_predictions[sample_name] = None
                del data, atom_array, err
                gc.collect(); torch.cuda.empty_cache(); gc.collect()
                continue
            
            # Find how many samples are needed for this specific query
            target_id = sample_name.split("_chunk")[0] if "_chunk" in sample_name else sample_name
            n_needed = protenix_queue.get(target_id, (N_SAMPLE, ""))[0]
            sub_seq_len = data["N_token"].item() # roughly correct
            
            try:
                new_cfg = update_inference_configs(configs, sub_seq_len)
                new_cfg.sample_diffusion.N_sample = n_needed
                runner.update_model_configs(new_cfg)
                
                pred = runner.predict(data)
                raw_coords = pred["coordinate"]
                
                coords = _extract_c1_coords(pred, data["input_feature_dict"], 
                                            sub_seq_len, raw_coords)
                raw_predictions[sample_name] = coords
            except Exception as exc:
                print(f"  {sample_name} inference failed: {exc}")
                import traceback; traceback.print_exc()
                raw_predictions[sample_name] = None
            finally:
                try: del pred, data, atom_array, raw_coords
                except: pass
                gc.collect(); torch.cuda.empty_cache(); gc.collect()

        # ── 3. Post-processing: Stitching and final formatting ───────────────
        for target_id, (n_needed, full_seq) in protenix_queue.items():
            seq_len = len(full_seq)
            chunks = chunk_info.get(target_id, [])
            
            if not chunks:
                continue

            if len(chunks) == 1:
                # Single pass
                coords = raw_predictions.get(target_id)
                protenix_preds[target_id] = coords
                if coords is not None:
                    print(f"  {target_id}: {coords.shape[0]} predictions generated")
                else:
                    print(f"  {target_id}: FAILED")
            else:
                # Stitch chunks together
                chunk_results_per_sample = {s: [] for s in range(n_needed)}
                all_ok = True
                
                for ci, cinfo in enumerate(chunks):
                    cname = cinfo["name"]
                    crange = cinfo["range"]
                    ccoords = raw_predictions.get(cname)
                    
                    if ccoords is None:
                        all_ok = False
                        break
                    
                    for s_idx in range(n_needed):
                        if s_idx < ccoords.shape[0]:
                            chunk_results_per_sample[s_idx].append((ccoords[s_idx], crange))
                        else:
                            chunk_results_per_sample[s_idx].append((ccoords[-1], crange))
                
                if not all_ok:
                    print(f"  {target_id}: chunked inference incomplete, using fallback")
                    protenix_preds[target_id] = None
                    continue
                
                stitched_samples = []
                for s_idx in range(n_needed):
                    items = chunk_results_per_sample[s_idx]
                    coords_list = [c for c, _ in items]
                    ranges_list = [r for _, r in items]
                    full_coords = stitch_chunk_coords(coords_list, ranges_list, seq_len)
                    stitched_samples.append(full_coords)
                
                result = np.stack(stitched_samples, axis=0)
                protenix_preds[target_id] = result
                print(f"  {target_id}: {result.shape[0]} stitched predictions generated")
# ...existing code...

    elif protenix_queue and not USE_PROTENIX:
        print(f"\nPHASE 2 skipped (USE_PROTENIX=False). "
              f"De-novo fallback will cover {len(protenix_queue)} targets.")

    # ─── PHASE 3: Combine everything ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 3: Combine TBM + Protenix + de-novo fallback")
    print(f"{'='*60}")

    all_rows = []

    for _, row in test_df.iterrows():
        tid = row["target_id"]
        seq = row["sequence"]

        combined: list = []
        source_weights: list = []
        tbm_list = list(template_preds.get(tid, []))
        for i, pred in enumerate(tbm_list):
            combined.append(pred)
            source_weights.append(max(0.30, 0.92 - 0.06 * i))

        # Append all available Protenix candidates; selection is done afterward.
        ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                refined_ptx = adaptive_rna_constraints(ptx[j], tid, segments_map, confidence=0.85, passes=1)
                combined.append(refined_ptx)
                source_weights.append(max(0.45, 1.00 - 0.02 * j))

        # De-novo fallback for any still-empty slots
        n_denovo = 0
        while len(combined) < N_SAMPLE:
            seed_val = row.name * 1000000 + len(combined) * 1000
            rng = np.random.default_rng(seed_val % (2**32))
            if combined:
                base = combined[len(combined) % len(combined)].copy()
                noise_sigma = 0.20 if len(seq) > PROTENIX_MAX_SEQ_LEN else 0.30
                noise = rng.normal(0, noise_sigma, base.shape).astype(np.float32)
                dn = adaptive_rna_constraints(base + noise, tid, segments_map, confidence=0.35)
            else:
                dn = generate_rna_structure(seq, seed=seed_val)
                dn = adaptive_rna_constraints(dn, tid, segments_map, confidence=0.2)
            combined.append(dn)
            source_weights.append(0.06)
            n_denovo += 1

        if n_denovo:
            print(f"  {tid}: {n_denovo} slot(s) filled with de-novo fallback")

        if USE_TM_CONSENSUS_RANK:
            combined = select_top_candidates(combined, source_weights, N_SAMPLE)

        # Stack to (N_SAMPLE, seq_len, 3) and write rows
        final_sel = [
            adaptive_rna_constraints(c, tid, segments_map, confidence=0.75, passes=1)
            for c in combined[:N_SAMPLE]
        ]
        stacked = np.stack(final_sel, axis=0)
        all_rows.extend(coords_to_rows(tid, seq, stacked))

    # ── Save ───────────────────────────────────────────────────────────────
    sub = pd.DataFrame(all_rows)
    cols = ["ID", "resname", "resid"] + [
        f"{c}_{i}" for i in range(1, N_SAMPLE + 1) for c in ["x", "y", "z"]
    ]
    coord_cols = [c for c in cols if c.startswith(("x_", "y_", "z_"))]
    sub[coord_cols] = sub[coord_cols].clip(-999.999, 9999.999)
    sub[cols].to_csv(output_csv, index=False)

    print(f"\n✓ Saved submission to {output_csv}  ({len(sub):,} rows)")

# ==== CELL 8 ====
if __name__ == "__main__":
    main()


