"""Tier-2 refine: energy-field alignment of two session point clouds.

Kernel-correlation ("energy" / EBM) registration. The reference cloud P1 induces a smooth
scalar potential  E1(x) = -sum_i exp(-||x - p_i||^2 / sigma^2)  -- deeply negative near
P1's points, ~0 far away. Aligning source P2 = {q_j} onto P1 minimises  L(T) = sum_j
E1(T(q_j)),  a Gaussian kernel correlation, inherently robust to *partial* overlap (a
source point with no P1 neighbour sits on the flat E~=0 tail: ~0 energy, ~0 gradient).

The transform is a `Sim3` (herald.scene.common.geometry). Both session clouds come out of
Tier-1 in the **gravity-aligned NED world frame**, so "up" is a known common axis (world
Z) and the relative pose reduces to **yaw about Z + translation** (+ optional scale) -- no
per-cloud ground fitting needed. (`level=True` re-enables a RANSAC plane fit for clouds
that are *not* gravity-aligned, e.g. a raw VGGT gauge.)

Efficiency: E1 is **precomputed once** as a dense voxel grid -- splat P1's points, then
Gaussian-blur (blur std = sigma/sqrt(2) emulates the kernel) -> a 3D field. Each iteration
only *samples* the field at the transformed source points via differentiable trilinear
interpolation (F.grid_sample): O(N2) per step, no O(N1 x N2) pairwise rebuild. One field
per sigma level (coarse->fine annealing).

Search: a coarse global (yaw, x, y) grid (robust init under partial overlap), then
gradient refinement of the top candidates. Scale is fixed to 1 by default (metric clouds;
a free scale makes the energy degenerate -- it shrinks P2 onto one dense P1 blob);
`optimize_scale=True` re-enables it with a soft prior toward 1.

`align(...) -> AlignResult` carries the winning `Sim3`, the per-step `Sim3` history (for
animating convergence), the final energy and the inlier fraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from herald.scene.common.geometry import Sim3


@dataclass
class AlignResult:
    """Outcome of `align`: the recovered Sim3, its optimisation history, and quality."""

    transform: Sim3                 # source -> target similarity  (x' = s R x + t)
    history: list                   # list[Sim3], the transform at each recorded step
    energy: float                   # final mean per-source-point energy (lower = better)
    inlier_frac: float              # frac. of source points landing on target geometry
    yaw: float                      # recovered yaw about +Z, rad

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return self.transform.apply(pts).astype(np.float32)


# --------------------------------------------------------------------------- ground

def fit_ground_plane(pts: np.ndarray, thresh: float = 0.1, iters: int = 300,
                     seed: int = 0) -> tuple[np.ndarray, float]:
    """RANSAC-fit the dominant plane (n . x + d = 0, ||n|| = 1). Only needed for clouds
    that are NOT already gravity-aligned; the GT/NED recon path does not use this."""
    pts = np.asarray(pts, np.float32).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    best_n, best_d, best_inl = np.array([0, 0, 1.0], np.float32), 0.0, -1
    for _ in range(iters):
        tri = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nn = np.linalg.norm(n)
        if nn < 1e-8:
            continue
        n = n / nn
        d = -float(n @ tri[0])
        inl = int((np.abs(pts @ n + d) < thresh).sum())
        if inl > best_inl:
            best_n, best_d, best_inl = n, d, inl
    if float(pts.mean(0) @ best_n + best_d) < 0:
        best_n, best_d = -best_n, -best_d
    return best_n.astype(np.float32), float(best_d)


def _rot_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation taking unit vector `a` to unit vector `b` (Rodrigues)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c, s = float(a @ b), float(np.linalg.norm(v))
    if s < 1e-8:
        return np.eye(3, dtype=np.float32) if c > 0 else np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], np.float32) / s
    return (np.eye(3) + s * k + (1 - c) * k @ k).astype(np.float32)


def _subsample(pts: np.ndarray, n_max: int, rng) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(-1, 3)
    if len(pts) <= n_max:
        return pts
    return pts[rng.choice(len(pts), n_max, replace=False)]


# ------------------------------------------------------------------- potential field

class _Field:
    """Precomputed voxel potential  E1(x) = -(P splatted, then Gaussian-blurred).

    Built ONCE from the TARGET points; `energy(pts)` samples it by differentiable
    trilinear interpolation. This is a grid approximation of the exact sum-of-Gaussians
    (splat + blur), traded for O(1) queries. Grid is [D=z, H=y, W=x] for grid_sample."""

    def __init__(self, P: np.ndarray, sigma: float, vox: float, device):
        import torch
        import torch.nn.functional as F

        pad = 3.0 * sigma + vox
        lo, hi = P.min(0) - pad, P.max(0) + pad
        dims = (np.ceil((hi - lo) / vox).astype(int) + 1)          # (Dx, Dy, Dz)
        idx = np.clip(np.floor((P - lo) / vox).astype(int), 0, dims - 1)
        grid = torch.zeros((int(dims[2]), int(dims[1]), int(dims[0])), device=device)
        lin = (idx[:, 2] * dims[1] + idx[:, 1]) * dims[0] + idx[:, 0]
        grid.view(-1).index_add_(0, torch.as_tensor(lin, device=device),
                                 torch.ones(len(lin), device=device))
        sc = max(0.6, (sigma / np.sqrt(2.0)) / vox)                 # blur std, cells
        r = max(1, int(round(3 * sc)))
        x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
        k = torch.exp(-(x ** 2) / (2 * sc * sc)); k = k / k.sum()
        g = grid[None, None]
        g = F.conv3d(g, k.view(1, 1, -1, 1, 1), padding=(r, 0, 0))
        g = F.conv3d(g, k.view(1, 1, 1, -1, 1), padding=(0, r, 0))
        g = F.conv3d(g, k.view(1, 1, 1, 1, -1), padding=(0, 0, r))
        self.E = -g                                                # (1,1,Dz,Dy,Dx)
        self.lo = torch.as_tensor(lo, dtype=torch.float32, device=device)
        self.dims = torch.as_tensor(dims, dtype=torch.float32, device=device)   # x,y,z
        self.vox = vox

    def query(self, pts):
        import torch.nn.functional as F
        gc = 2.0 * (pts - self.lo) / self.vox / (self.dims - 1) - 1.0   # -> [-1,1] (x,y,z)
        out = F.grid_sample(self.E, gc.view(1, -1, 1, 1, 3),
                            align_corners=True, padding_mode="border")
        return out.view(-1)

    def energy(self, pts):
        return self.query(pts).mean()


# --------------------------------------------------------------------------- align

def align(source: np.ndarray, target: np.ndarray, *,
          sigmas: tuple[float, ...] = (2.0, 1.0, 0.5, 0.25),
          yaw_seeds: int = 24, iters: int = 80, lr: float = 0.05,
          src_points: int = 8000, field_points: int = 200000, field_vox: float = 0.12,
          optimize_scale: bool = False, scale_prior: float = 1.0, topk: int = 3,
          record: bool = True, rec_stride: int = 4, level: bool = False,
          ground_thresh: float = 0.1, device: str | None = None, seed: int = 0,
          verbose: bool = False) -> AlignResult:
    """Align `source` onto `target` (both (N,3), metric, gravity-aligned) and return the
    Sim3 plus its optimisation history. Precomputes a potential field per sigma from the
    target, does a coarse global (yaw, x, y) search, then gradient-refines the top-K."""
    import torch

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(seed)

    # up axis: world Z for gravity-aligned NED clouds (no per-cloud fit). level=True only
    # for non-gravity-aligned clouds (e.g. VGGT gauge).
    up = np.array([0, 0, 1.0], np.float32)
    if level:
        n_s, _ = fit_ground_plane(source, ground_thresh, seed=seed)
        n_t, _ = fit_ground_plane(target, ground_thresh, seed=seed)
        Rs, Rt = _rot_align(n_s, up), _rot_align(n_t, up)
    else:
        Rs = Rt = np.eye(3, dtype=np.float32)
    src_L, tgt_L = source @ Rs.T, target @ Rt.T

    Qnp = _subsample(src_L, src_points, rng)
    Pnp = _subsample(tgt_L, field_points, rng)
    Q = torch.as_tensor(Qnp, dtype=torch.float32, device=dev)

    if verbose:
        print(f"  building {len(sigmas)} potential fields from {len(Pnp)} target pts "
              f"(vox {field_vox}m)...", flush=True)
    fields = {sig: _Field(Pnp, sig, field_vox, dev) for sig in sigmas}

    def rotate(psi):                                               # Rz(psi) @ Q about +Z
        cz, sz = torch.cos(psi), torch.sin(psi)
        x, y, z = Q[:, 0], Q[:, 1], Q[:, 2]
        return torch.stack([cz * x - sz * y, sz * x + cz * y, z], 1)

    def compose(psi, t_vec, s) -> Sim3:                           # leveled -> original frame
        cz, sz = np.cos(psi), np.sin(psi)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float32)
        return Sim3(float(s), (Rt.T @ Rz @ Rs).astype(np.float64), (Rt.T @ t_vec).astype(np.float64))

    # coarse global (yaw, x, y) search on the coarsest field; z init aligns floor heights
    coarse = fields[max(sigmas)]
    tz0 = float(np.median(Pnp[:, 2]) - np.median(Qnp[:, 2]))
    cq = Qnp[:, :2].mean(0)
    lo, hi = Pnp[:, :2].min(0), Pnp[:, :2].max(0)
    res = 0.8
    gx = np.linspace(lo[0], hi[0], max(2, int((hi[0] - lo[0]) / res) + 1))
    gy = np.linspace(lo[1], hi[1], max(2, int((hi[1] - lo[1]) / res) + 1))
    cand = []
    with torch.no_grad():
        for psi0 in np.linspace(0.0, 2 * np.pi, yaw_seeds, endpoint=False):
            cz, sz = np.cos(psi0), np.sin(psi0)
            rc = np.array([cz * cq[0] - sz * cq[1], sz * cq[0] + cz * cq[1]])
            psi = torch.tensor(float(psi0), device=dev)
            rot = rotate(psi)
            for gxx in gx:
                for gyy in gy:
                    t = torch.tensor([gxx - rc[0], gyy - rc[1], tz0], dtype=torch.float32, device=dev)
                    cand.append((float(coarse.energy(rot + t)), float(psi0),
                                 np.array([gxx - rc[0], gyy - rc[1], tz0], np.float32)))
    cand.sort(key=lambda c: c[0])
    if verbose:
        print(f"  grid: {len(cand)} placements, best coarse energy={cand[0][0]:.4f} "
              f"(yaw {np.degrees(cand[0][1]):.1f}deg)", flush=True)

    def refine(psi0, t0):
        psi = torch.tensor(float(psi0), device=dev, requires_grad=True)
        t = torch.tensor(t0, dtype=torch.float32, device=dev, requires_grad=True)
        logs = torch.zeros((), device=dev, requires_grad=True)
        opt = torch.optim.Adam([psi, t] + ([logs] if optimize_scale else []), lr=lr)
        hist, step = [], 0
        for sig in sigmas:
            fld = fields[sig]
            for _ in range(iters):
                opt.zero_grad()
                s = torch.exp(logs) if optimize_scale else 1.0
                loss = fld.energy(s * rotate(psi) + t)
                if optimize_scale:
                    loss = loss + 0.5 * ((logs - np.log(scale_prior)) ** 2)
                loss.backward()
                opt.step()
                if record and step % rec_stride == 0:
                    sc = float(torch.exp(logs.detach())) if optimize_scale else 1.0
                    hist.append(compose(float(psi.item()), t.detach().cpu().numpy(), sc))
                step += 1
        with torch.no_grad():
            sc = float(torch.exp(logs)) if optimize_scale else 1.0
            e = float(fields[min(sigmas)].energy(sc * rotate(psi) + t))
        return e, float(psi.item()), t.detach().cpu().numpy(), sc, hist

    best = None
    for e0, psi0, t0 in cand[:topk]:
        e, psi, t_vec, s, hist = refine(psi0, t0)
        if verbose:
            print(f"  refine yaw0={np.degrees(psi0):6.1f} -> yaw={np.degrees(psi):7.1f} "
                  f"s={s:.3f} energy={e:.4f}", flush=True)
        if best is None or e < best[0]:
            best = (e, psi, t_vec, s, hist)
    e, psi, t_vec, s, hist = best
    T = compose(psi, t_vec, s)
    if record and (not hist or not np.allclose(hist[-1].t, T.t)):
        hist.append(T)

    # inlier fraction: aligned source landing where the finest field is strongly negative
    with torch.no_grad():
        al = _subsample(source, src_points, rng)
        al_L = torch.as_tensor(T.apply(al) @ Rt.T, dtype=torch.float32, device=dev)   # field frame
        e_fine = fields[min(sigmas)].query(al_L)
        peak = float(-fields[min(sigmas)].E.min())
        inl = float((e_fine < -0.1 * peak).float().mean()) if peak > 0 else 0.0

    return AlignResult(transform=T, history=hist, energy=float(e), inlier_frac=inl, yaw=float(psi))


def align_sessions(src_result, tgt_result, **kw) -> AlignResult:
    """Convenience: align two SessionResult clouds (`.points`)."""
    return align(src_result.points, tgt_result.points, **kw)
