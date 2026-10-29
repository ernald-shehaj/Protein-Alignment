# src/easy_compare.py
# A simpler, readable PDB comparison with features + Kabsch + Hungarian.

from __future__ import annotations
import argparse, csv, json, math, os
from typing import Dict, List, Tuple
import numpy as np

# ---------------------------
# Hungarian Algorithm Implementation
# ---------------------------

def hungarian_assignment(cost_matrix):
    """
    Hungarian algorithm (min-cost assignment, O(n^2 m) with potentials).
    Returns (row_indices, col_indices) like scipy.optimize.linear_sum_assignment
    """
    cost = np.array(cost_matrix, dtype=float)
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Algorithm needs rows <= cols; transpose otherwise
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape

    # 1-indexed potentials u (rows), v (cols); p[j] = row matched to column j
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            # Reduced costs from row i0 to all free columns
            cur = cost[i0 - 1] - u[i0] - v[1:]
            free = ~used[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            j1 = 1 + int(np.argmin(np.where(free, minv[1:], np.inf)))
            delta = minv[j1]
            # Update potentials
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # Augment along the alternating path
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    cols = np.nonzero(p[1:])[0]
    rows = p[1:][cols] - 1
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order].astype(int), cols[order].astype(int)

# For drop-in replacement
def linear_sum_assignment(cost_matrix):
    return hungarian_assignment(cost_matrix)

hungarian = linear_sum_assignment

# ---------------------------
# Small PDB parser (Cα only)
# ---------------------------
class Residue:
    __slots__ = ("chain","resseq","icode","resname","ca","sg")
    def __init__(self, chain, resseq, icode, resname):
        self.chain   = chain
        self.resseq  = resseq
        self.icode   = icode
        self.resname = resname
        self.ca = None           # np.array([x,y,z])
        self.sg = None           # CYS SG if present (for simple SS-bridge detection)

def load_pdb(path: str) -> List[Residue]:
    rmap: Dict[Tuple[str,int,str], Residue] = {}
    metals: List[Tuple[str,np.ndarray]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM","HETATM"): continue
            name    = line[12:16].strip()
            resname = line[17:20].strip()
            chain   = line[21].strip() or "_"
            try:
                resseq = int(line[22:26])
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except: continue
            icode   = line[26].strip()
            element = (line[76:78].strip() or name[0]).upper()

            if rec == "ATOM":
                key = (chain, resseq, icode)
                if key not in rmap: rmap[key] = Residue(chain, resseq, icode, resname)
                r = rmap[key]
                if name == "CA": r.ca = np.array([x,y,z], float)
                if resname == "CYS" and name == "SG": r.sg = np.array([x,y,z], float)
            else:
                if element in ("ZN","MG","MN","FE","CO","NI","CU","CA","NA","K","CD","HG"):
                    metals.append((element, np.array([x,y,z], float)))

    residues = [r for r in rmap.values() if r.ca is not None]
    residues.sort(key=lambda r:(r.chain, r.resseq, r.icode))
    return residues

# ---------------------------
# Geometry helpers
# ---------------------------
def unit(v): 
    n = np.linalg.norm(v); 
    return v/n if n>1e-8 else v*0

def angle(a,b,c):
    u = unit(a-b); v = unit(c-b)
    return math.degrees(math.acos(np.clip(np.dot(u,v), -1.0, 1.0)))

def dihedral(p1,p2,p3,p4):
    b0, b1, b2 = p2-p1, p3-p2, p4-p3
    b1u = unit(b1)
    v = b0 - np.dot(b0,b1u)*b1u
    w = b2 - np.dot(b2,b1u)*b1u
    x = np.dot(unit(v), unit(w))
    y = np.dot(np.cross(b1u, unit(v)), unit(w))
    return math.degrees(math.atan2(y, x))

# ---------------------------
# Features (simple + robust)
# ---------------------------
AA3to1 = {"ALA":"A","VAL":"V","LEU":"L","ILE":"I","MET":"M","PHE":"F","TYR":"Y","TRP":"W",
          "SER":"S","THR":"T","ASN":"N","GLN":"Q","CYS":"C","GLY":"G","PRO":"P",
          "ASP":"D","GLU":"E","LYS":"K","ARG":"R","HIS":"H"}

def chem_group(res3: str) -> str:
    one = AA3to1.get(res3, "?")
    if one in ("A","V","L","I","M"): return "H"   # hydrophobic
    if one in ("F","Y","W"):         return "AR"  # aromatic
    if one in ("S","T","N","Q","C"): return "P"   # polar
    if one in ("D","E"):             return "N-"  # negative
    if one in ("K","R","H"):         return "P+"  # positive
    if one == "G":                   return "G"
    if one == "P":                   return "P*"
    return "?"

def chain_index(res: List[Residue]) -> Dict[str, List[int]]:
    idx: Dict[str,List[int]] = {}
    for i,r in enumerate(res):
        idx.setdefault(r.chain, []).append(i)
    return idx

def knn_fingerprint(coords: np.ndarray, k:int) -> np.ndarray:
    N = len(coords)
    if N <= 1: 
        return np.zeros((N, 0))
    d = np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, :min(k, N-1)]

def simple_ss_from_angles(coords: np.ndarray, chains: Dict[str,List[int]]) -> np.ndarray:
    N = len(coords)
    A = np.full(N, np.nan); D = np.full(N, np.nan)
    for ch, idxs in chains.items():
        for t in range(1, len(idxs)-1):
            i0,i1,i2 = idxs[t-1], idxs[t], idxs[t+1]
            A[i1] = angle(coords[i0], coords[i1], coords[i2])
        for t in range(2, len(idxs)-1):
            i0,i1,i2,i3 = idxs[t-2], idxs[t-1], idxs[t], idxs[t+1]
            D[i2] = dihedral(coords[i0], coords[i1], coords[i2], coords[i3])
    lab = np.full(N, "C", dtype=object)
    for i in range(N):
        a, d = A[i], D[i]
        if np.isfinite(a) and np.isfinite(d):
            if 80<=a<=120 and -90<=d<=-30: lab[i] = "H"
            elif 100<=a<=140 and (d>=160 or d<=-160): lab[i] = "E"
    return lab

def extract_features(res: List[Residue], k:int=10) -> Dict[str, np.ndarray]:
    coords = np.array([r.ca for r in res], float)
    chains = chain_index(res)
    ss  = simple_ss_from_angles(coords, chains)
    knn = knn_fingerprint(coords, k)
    chem = np.array([chem_group(r.resname) for r in res], dtype=object)
    return dict(coords=coords, ss=ss, knn=knn, chem=chem, chains=chains)

# ---------------------------
# Alignment (Kabsch) + utils
# ---------------------------
def kabsch(P: np.ndarray, Q: np.ndarray):
    Pc, Qc = P.mean(0), Q.mean(0)
    P0, Q0 = P-Pc, Q-Qc
    C = Q0.T @ P0
    V,S,Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1,1,d]) @ Wt
    t = Pc - R @ Qc
    return R, t

def apply_rt(R,t,X): 
    return (R @ X.T).T + t

# ---------------------------
# Cost matrix
# ---------------------------
def chemistry_cost(a: str, b: str) -> float:
    if a == b: return 0.0
    a2 = "P" if a=="P*" else a
    b2 = "P" if b=="P*" else b
    return 0.4 if a2==b2 else 1.0

def ss_cost(sa: str, sb: str) -> float:
    return 0.0 if sa==sb else 0.5

def feature_cost(Af:dict, Bf:dict, w=dict(knn=1.0, angle=0.2, dihed=0.2, ss=0.3, chem=0.5)) -> np.ndarray:
    # Note: we don't explicitly include angle/dihedral scalars here; the SS proxy captures them.
    # Keep it simple: knn + ss + chem. (SS already derived from angles/dihedrals.)
    k = min(Af["knn"].shape[1], Bf["knn"].shape[1])
    Aknn = Af["knn"][:, :k]
    Bknn = Bf["knn"][:, :k]
    NA, NB = len(Af["coords"]), len(Bf["coords"])
    C = np.zeros((NA, NB), float)

    # kNN term (vectorized)
    if k > 0:
        a = Aknn[:,None,:]      # (NA,1,k)
        b = Bknn[None,:,:]      # (1,NB,k)
        C += w["knn"] * (np.linalg.norm(a-b, axis=2) / (k*10.0))

    # SS + chemistry terms (simple loops for clarity)
    for i in range(NA):
        for j in range(NB):
            C[i,j] += w["ss"]   * ss_cost(Af["ss"][i],   Bf["ss"][j])
            C[i,j] += w["chem"] * chemistry_cost(Af["chem"][i], Bf["chem"][j])
    return C

def add_local_rmsd(C: np.ndarray, Af:dict, Bf:dict, R:np.ndarray, t:np.ndarray,
                   window:int=2, w_rmsd:float=0.4, rmsd_cutoff:float=4.0) -> np.ndarray:
    """Add local RMSD around each residue after aligning B->A."""
    A = Af["coords"]; B = apply_rt(R,t,Bf["coords"])
    Ach = Af["chains"]; Bch = Bf["chains"]

    # index -> chain-local window
    def windows(chmap):
        win = {}
        for ch, idxs in chmap.items():
            for pos, idx in enumerate(idxs):
                lo = max(0, pos-window); hi = min(len(idxs), pos+window+1)
                win[idx] = idxs[lo:hi]
        return win

    Aw, Bw = windows(Ach), windows(Bch)
    NA, NB = C.shape
    for i in range(NA):
        Ai = Aw.get(i, [i])
        Apts = A[Ai]
        for j in range(NB):
            Bj = Bw.get(j, [j])
            Bpts = B[Bj]
            m = min(len(Apts), len(Bpts))
            if m == 0: 
                continue
            # center-align small slices
            ac = len(Apts)//2; bc = len(Bpts)//2
            AA = Apts[max(0,ac-m//2):max(0,ac-m//2)+m]
            BB = Bpts[max(0,bc-m//2):max(0,bc-m//2)+m]
            if len(AA) != len(BB) or len(AA)==0: 
                continue
            rmsd = float(np.sqrt(np.mean(np.sum((AA-BB)**2, axis=1))))
            C[i,j] += w_rmsd * min(rmsd / rmsd_cutoff, 1.0)
    return C

# ---------------------------
# Matching
# ---------------------------
def solve_hungarian(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if hungarian is not None:          # handles rectangular matrices
        return hungarian(C)
    # very small fallback: greedy rows → cols
    NA, NB = C.shape
    used, rows, cols = set(), [], []
    for i in range(NA):
        j = int(np.argmin([C[i,k] if k not in used else 1e9 for k in range(NB)]))
        used.add(j); rows.append(i); cols.append(j)
        if len(used) == NB: break
    return np.array(rows), np.array(cols)

# ---------------------------
# Main pipeline
# ---------------------------
def run(pdbA: str, pdbB: str, out_csv: str,
        iters:int=2, k:int=10, use_local_rmsd:bool=False,
        window:int=2, w_rmsd:float=0.4, rmsd_cutoff:float=4.0,
        prune_thresh:float=3.5, score_alpha:float=1.0):

    # Parse + features
    Ares = load_pdb(pdbA); Bres = load_pdb(pdbB)
    Af = extract_features(Ares, k=k)
    Bf = extract_features(Bres, k=k)

    # Initial match
    C = feature_cost(Af, Bf)
    ri, ci = solve_hungarian(C)

    # Iterate: align → (optional local RMSD term) → re-match
    R = np.eye(3); t = np.zeros(3)
    for _ in range(max(1, iters)):
        P = Af["coords"][ri]; Q = Bf["coords"][ci]
        if len(P) >= 3 and len(Q) >= 3:
            R, t = kabsch(P, Q)
        C = feature_cost(Af, Bf)
        if use_local_rmsd and len(P) >= 3:
            C = add_local_rmsd(C, Af, Bf, R, t, window, w_rmsd, rmsd_cutoff)
        ri, ci = solve_hungarian(C)

    # Final alignment
    P = Af["coords"][ri]; Q = Bf["coords"][ci]
    if len(P) >= 3 and len(Q) >= 3:
        R, t = kabsch(P, Q)
    Q_aln = apply_rt(R, t, Q)

    # Prune outliers and compute metrics
    d = np.sqrt(np.sum((P - Q_aln)**2, axis=1))
    keep = d <= prune_thresh
    kept_pairs = int(np.sum(keep))
    nA, nB = len(Ares), len(Bres)
    coverage = kept_pairs / max(1, min(nA, nB))
    global_rmsd = float(np.sqrt(np.mean((d[keep]**2))) if kept_pairs>0 else np.inf)
    fit_term = 1.0 / (1.0 + score_alpha * (global_rmsd if np.isfinite(global_rmsd) else 1e9))
    similarity = 100.0 * coverage * fit_term

    # Write CSV
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["A_idx","A_chain","A_resseq","A_resname",
                        "B_idx","B_chain","B_resseq","B_resname",
                        "cost","kept"])
            for pos,(ia,ib) in enumerate(zip(ri.tolist(), ci.tolist())):
                ra, rb = Ares[ia], Bres[ib]
                kept_flag = int(keep[pos]) if pos < len(keep) else 1
                w.writerow([ia, ra.chain, ra.resseq, ra.resname,
                            ib, rb.chain, rb.resseq, rb.resname,
                            f"{C[ia,ib]:.4f}", kept_flag])

    # Print summary JSON
    summary = dict(
        nA=nA, nB=nB, pairs=int(min(len(ri), len(ci))),
        kept_pairs=kept_pairs, coverage=round(coverage,3),
        global_rmsd=None if not np.isfinite(global_rmsd) else round(global_rmsd,3),
        similarity_score=round(similarity,1),
        iters=iters, use_local_rmsd=bool(use_local_rmsd),
        window=window, w_rmsd=w_rmsd, rmsd_cutoff=rmsd_cutoff,
        prune_thresh=prune_thresh, score_alpha=score_alpha,
        out_csv=out_csv
    )
    print(json.dumps(summary, indent=2))

def main():
    ap = argparse.ArgumentParser(description="Simple feature-based protein alignment (Cα).")
    ap.add_argument("pdbA"); ap.add_argument("pdbB")
    ap.add_argument("--out", default="mapping.csv", help="Output CSV for residue mapping")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--k", type=int, default=10, help="k for kNN Cα fingerprint")
    ap.add_argument("--use-local-rmsd", action="store_true", help="Refine cost with local RMSD after first alignment")
    ap.add_argument("--window", type=int, default=2, help="Half-window for local RMSD (i±window)")
    ap.add_argument("--w-rmsd", type=float, default=0.4, help="Weight for local RMSD term")
    ap.add_argument("--rmsd-cutoff", type=float, default=4.0, help="Normalize local RMSD (Å)")
    ap.add_argument("--prune-thresh", type=float, default=3.5, help="Pair distance cutoff after alignment (Å)")
    ap.add_argument("--score-alpha", type=float, default=1.0, help="RMSD scaling in similarity score")
    args = ap.parse_args()

    run(args.pdbA, args.pdbB, args.out, args.iters, args.k, args.use_local_rmsd,
        args.window, args.w_rmsd, args.rmsd_cutoff, args.prune_thresh, args.score_alpha)

if __name__ == "__main__":
    main()