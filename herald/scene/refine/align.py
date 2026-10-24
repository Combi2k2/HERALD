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


# ------------------------------------------------------------- anchored (contour) alignment
#
# `align()` above (point-cloud vs. point-cloud) is the default, unchanged Tier-2 registration
# path -- fast, already proven on every TartanGround env tested. The functions below are an
# ADDITIONAL, opt-in path for the case `align()` alone handles poorly: a new session with LOW
# real overlap against the persistent map, where cloud-to-cloud correlation has little true
# signal to lock onto. They anchor against a `herald.scene.refine.contour`-extracted building
# footprint (a synthetic OSM substitute for datasets with no real floor-plan feed) IN ADDITION
# TO the persistent cloud, not instead of it. Validated on JapaneseCity this session: a
# low-overlap case that single-shot cloud-only alignment missed by 14.9m came down to 1.35m with
# this recipe. That is ONE validated case with a real but NARROW optimum (nearby hyperparameter
# settings were measurably worse, see `anchored_align`'s docstring) -- opt-in via `--mode
# anchored` in scripts/scene_align.py, not a silent default-behavior change.

def high_freq_points(pts_flat: np.ndarray, cell_size: float = 0.1, sigma_cells: float = 1.5,
                      norm_percentile: float = 98.0, keep_percentile: float = 80.0) -> np.ndarray:
    """Filter a (flattened, z=0) point cloud down to its own top-density-percentile points --
    the alignment TARGET this whole recipe depends on. Raw/dense clouds have repeated structure
    (a street grid, similar facade segments) that creates a false, sometimes-deeper correlation
    minimum (confirmed empirically: different sigma schedules disagreeing on even the sign of
    the recovered yaw); this keeps real point density/character (unlike a synthetic edge
    skeleton, which gave zero-signal blind search) while filtering out low-signal floor/clutter
    points."""
    from herald.scene.refine.contour import flatten_to_grid

    likelihood, counts, ox, oy = flatten_to_grid(pts_flat[:, :2], cell_size, sigma_cells=sigma_cells,
                                                  norm_percentile=norm_percentile)
    xi = np.clip(((pts_flat[:, 0] - ox) / cell_size).astype(np.int64), 0, likelihood.shape[0] - 1)
    yi = np.clip(((pts_flat[:, 1] - oy) / cell_size).astype(np.int64), 0, likelihood.shape[1] - 1)
    point_density = likelihood[xi, yi]
    thresh = np.percentile(point_density[point_density > 0], keep_percentile)
    return pts_flat[point_density >= thresh]


@dataclass
class ReversedAlignResult:
    transform: Sim3
    yaw: float
    energy: float
    history: list = None          # list[Sim3], best-so-far checkpoint at each recorded refine step
    coarse_history: list = None   # list[Sim3], best-so-far checkpoint during the BLIND GRID SEARCH
                                   # that happens BEFORE refinement (see reversed_align's docstring)


def reversed_align(source: np.ndarray, target: np.ndarray, *,
                    sigmas: tuple[float, ...] = (1.0, 0.4), field_vox: float = 0.05,
                    yaw_search_deg: float = 45.0, t_search_radius: float = 11.0,
                    yaw_seeds: int = 36, t_res: float = 1.0, iters: int = 40, lr: float = 0.05,
                    src_points: int = 4000, target_points: int = 8000, topk: int = 3,
                    lambda_psi: float = 0.0, seed: int = 0, verbose: bool = False,
                    record: bool = False, rec_stride: int = 2) -> ReversedAlignResult:
    """Build the kernel-correlation potential field from the (small) SOURCE crop instead of the
    (large) target, then query it with the TARGET points transformed by T^-1 -- mathematically
    equivalent to `align()`'s forward formulation for a rigid transform (||T(q)-c|| =
    ||q-T^-1(c)|| when T is pure rotation+translation), but memory use is independent of the
    target's extent (a whole-city or whole-persistent-map target would OOM the forward
    formulation's field, which is built over the TARGET's bounding box -- confirmed empirically).
    `lambda_psi`: MAP-style soft yaw penalty toward 0 correction, normalized per-sigma-stage by
    that stage's own peak achievable energy (a bare constant is wrong: energy magnitude scales
    with sigma)."""
    rng = np.random.default_rng(seed)
    import torch

    Snp = _subsample(source, src_points, rng)   # heat source (small crop) -- builds the field
    Tnp = _subsample(target, target_points, rng)  # queried, via T^-1, against that field
    dev = torch.device("cpu")

    fields = {sig: _Field(Snp, sig, field_vox, dev) for sig in sigmas}
    peak_energy = {sig: abs(float(fields[sig].E.min())) for sig in sigmas}
    C = torch.as_tensor(Tnp, dtype=torch.float32, device=dev)
    cq = Snp[:, :2].mean(0)   # source centroid, for the same rc-recentering trick as align()

    def inv_transform(psi, t):
        # T(x) = R(psi) x + t  =>  T^-1(y) = R(psi)^T (y - t) = R(-psi) (y - t)
        cz, sz = torch.cos(-psi), torch.sin(-psi)
        shifted = C - t
        x, y, z = shifted[:, 0], shifted[:, 1], shifted[:, 2]
        return torch.stack([cz * x - sz * y, sz * x + cz * y, z], 1)

    def compose(psi, t_vec) -> Sim3:
        cz, sz = np.cos(psi), np.sin(psi)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float32)
        return Sim3(1.0, Rz.astype(np.float64), t_vec.astype(np.float64))

    coarse_sigma = max(sigmas)
    cand = []
    # `coarse_hist` records the RUNNING BEST candidate found so far, in the order the blind grid
    # is actually swept (yaw outer loop, x/y inner) -- lets a caller animate the search itself
    # ("many candidates tried, occasionally a better one found") rather than jumping straight
    # from the noisy start to the refinement phase, which otherwise hides most of what the
    # method actually does before gradient descent even begins.
    coarse_hist = [] if record else None
    coarse_best = None
    # search centered on the source's OWN centroid (bug found and fixed during development: this
    # was centered on the world origin (0,0) before, so a crop far from the origin never explored
    # anywhere near the truth)
    gx = np.arange(cq[0] - t_search_radius, cq[0] + t_search_radius + 1e-6, t_res)
    gy = np.arange(cq[1] - t_search_radius, cq[1] + t_search_radius + 1e-6, t_res)
    with torch.no_grad():
        for psi0 in np.linspace(-np.deg2rad(yaw_search_deg), np.deg2rad(yaw_search_deg), yaw_seeds):
            cz, sz = np.cos(psi0), np.sin(psi0)
            rc = np.array([cz * cq[0] - sz * cq[1], sz * cq[0] + cz * cq[1]])
            psi_t = torch.tensor(float(psi0), device=dev)
            for gxx in gx:
                for gyy in gy:
                    t_vec = np.array([gxx - rc[0], gyy - rc[1], 0.0], np.float32)
                    t_t = torch.tensor(t_vec, dtype=torch.float32, device=dev)
                    e = float(fields[coarse_sigma].energy(inv_transform(psi_t, t_t)))
                    e += lambda_psi * peak_energy[coarse_sigma] * float(psi0) ** 2
                    cand.append((e, float(psi0), t_vec))
                    if record and (coarse_best is None or e < coarse_best[0]):
                        coarse_best = (e, float(psi0), t_vec)
                        coarse_hist.append(compose(coarse_best[1], coarse_best[2]))
    cand.sort(key=lambda c: c[0])
    if verbose:
        print(f"  coarse best: yaw={np.degrees(cand[0][1]):.1f} e={cand[0][0]:.4f} "
              f"({len(coarse_hist) if coarse_hist else 0} improving steps out of {len(cand)} tried)",
              flush=True)

    # More refine iterations was found to make results WORSE, not better (Adam's adaptive
    # step-size normalization keeps taking near-full-sized steps even once the true gradient is
    # ~0, so a good solution can drift away given enough extra steps). Fix: track the BEST
    # (lowest finest-sigma energy) snapshot seen at ANY point during optimization, not just
    # wherever the optimizer ends up -- standard early-stopping-by-best-checkpoint, makes extra
    # iterations strictly non-harmful.
    fine_sigma = sigmas[-1]
    best = None
    best_hist = None
    for e0, psi0, t0 in cand[:topk]:
        psi = torch.tensor(float(psi0), device=dev, requires_grad=True)
        t = torch.tensor(t0, dtype=torch.float32, device=dev, requires_grad=True)
        opt = torch.optim.Adam([psi, t], lr=lr)
        best_this_run = None
        hist, step = ([compose(float(psi.item()), t.detach().cpu().numpy())] if record else None), 0
        for sig in sigmas:
            lam_sig = lambda_psi * peak_energy[sig]
            for _ in range(iters):
                opt.zero_grad()
                loss = fields[sig].energy(inv_transform(psi, t)) + lam_sig * psi ** 2
                loss.backward()
                opt.step()
                with torch.no_grad():
                    e_fine = float(fields[fine_sigma].energy(inv_transform(psi, t)))
                if best_this_run is None or e_fine < best_this_run[0]:
                    best_this_run = (e_fine, float(psi.item()), t.detach().cpu().numpy().copy())
                if record and step % rec_stride == 0:
                    hist.append(compose(*best_this_run[1:]))
                step += 1
        e, psi_val, t_val = best_this_run
        if verbose:
            print(f"  refine yaw0={np.degrees(psi0):.1f} -> best_yaw={np.degrees(psi_val):.1f} e={e:.4f} "
                  f"(final was yaw={np.degrees(psi.item()):.1f})", flush=True)
        if best is None or e < best[0]:
            best = (e, psi_val, t_val)
            best_hist = hist

    e, psi, t_vec = best
    T = compose(psi, t_vec)
    if record and (not best_hist or not np.allclose(best_hist[-1].t, T.t)):
        best_hist.append(T)
    return ReversedAlignResult(transform=T, yaw=psi, energy=e, history=best_hist if record else None,
                                coarse_history=coarse_hist if record else None)


def local_contour(contour_pts: np.ndarray, center_xy: np.ndarray, radius: float) -> np.ndarray:
    """Restrict a (whole-map) contour to a local neighborhood around the source cloud being
    placed. A full-map contour has repeated/periodic structure (a street grid, similar facade
    segments) that can pull the optimizer toward a distant-but-locally-identical false match
    even at a small mixing weight -- a real OSM/contour anchor would realistically be queried
    locally around a pose prior anyway, not matched blindly against the entire map."""
    d = np.linalg.norm(contour_pts[:, :2] - center_xy, axis=1)
    sel = contour_pts[d <= radius]
    return sel if len(sel) >= 20 else contour_pts


def weighted_target(contour_local: np.ndarray, pool: np.ndarray | None, w_contour: float,
                     w_cloud: float, n_total: int, rng: np.random.Generator) -> np.ndarray:
    """Mix `n_total` target points from the local contour and the persistent-map cloud pool by
    weight (approximating a weighted-sum-of-energies objective via point density). `pool=None`
    (nothing persisted yet, e.g. the very first session) falls back to contour-only."""
    if pool is None or len(pool) == 0:
        return contour_local
    n_c = int(round(n_total * w_contour))
    n_p = n_total - n_c
    c_idx = rng.choice(len(contour_local), min(n_c, len(contour_local)), replace=len(contour_local) < n_c)
    # high-freq filter the cloud pool too (same target as reversed_align's own recipe) -- a raw
    # pool reintroduces the repeated-structure false-minimum risk local_contour() fixes above.
    pool_f = high_freq_points(pool) if len(pool) > 2000 else pool
    p_idx = rng.choice(len(pool_f), min(n_p, len(pool_f)), replace=len(pool_f) < n_p)
    return np.concatenate([contour_local[c_idx], pool_f[p_idx]])


def anchored_align(source: np.ndarray, contour: np.ndarray, persistent: np.ndarray | None, *,
                    w_contour: float = 0.3, w_cloud: float = 0.7, contour_radius: float = 70.0,
                    max_yaw_deg: float = 30.0, max_t: float = 8.0, refine_yaw_window: float = 9.0,
                    lambda_psi: float = 0.01, field_vox: float = 0.1, src_points: int = 5000,
                    target_points: int = 6000, iters: int = 80, yaw_seeds: int = 36, topk: int = 3,
                    seed: int = 0) -> ReversedAlignResult:
    """Two-stage anchored alignment for the LOW-overlap case `align()` handles poorly: a new
    session's `source` cloud has too little real overlap with `persistent` for cloud-to-cloud
    correlation alone to find the true pose, so this also anchors against `contour` (a
    `herald.scene.refine.contour`-extracted building footprint).

    Stage A: contour-only, WIDE search -- gets yaw roughly right but leaves translation badly
    wrong (empirically: a straight local facade segment pins orientation but not position along
    it -- a translation-aperture problem, not a not-enough-context one; error stuck at ~14-15m
    across every contour radius tried). Stage B: apply stage A's fix, then refine with a NARROW
    yaw search (a wide blind search with cloud mixed in consistently lands ~70deg off -- a
    strong false attractor) using the contour+persistent-cloud mixed target to pin down
    translation.

    Validated on JapaneseCity this session: w_contour=0.3, refine_yaw_window=+/-9deg takes a
    case `align()`/single-shot `reversed_align()` missed by 14.9m down to 1.35m. That is ONE
    validated case with a real but NARROW optimum -- nearby settings (w_contour=0.2 or 0.4,
    window=6 or 12) were measurably worse (7-17m, not 1-2m) -- so treat this as a validated fix
    for a specific failure mode, not a broadly-robust general recipe yet. Also found empirically
    to HURT the high-overlap case (0.47m -> 3.6m) by unnecessarily constraining a search that
    didn't need constraining -- callers should fall back to `reversed_align()`/`align()` alone
    when overlap with `persistent` is not low.

    This landscape is also seed-sensitive: stage B must use `seed` (matching stage A) + 1, and a
    FIXED/independent rng (not derived from any per-call seed) for `weighted_target()`'s
    subsampling -- both confirmed empirically during development; other seed offsets reproduced
    neither this result nor the single-shot baseline.
    """
    centroid = source[:, :2].mean(0)
    c_local = local_contour(contour, centroid, contour_radius)
    resA = reversed_align(source, c_local, sigmas=(1.0, 0.4), field_vox=field_vox,
                           yaw_search_deg=max_yaw_deg + 15, t_search_radius=max_t + 3.0,
                           yaw_seeds=yaw_seeds, iters=iters, lambda_psi=lambda_psi,
                           src_points=src_points, target_points=target_points, topk=topk,
                           seed=seed, record=True, rec_stride=2, verbose=False)
    if persistent is None or len(persistent) == 0:
        return resA   # first session ever: contour-only bootstrap, nothing to mix in yet

    partial = resA.transform.apply(source)
    tgt_rng = np.random.default_rng(999)   # fixed/independent -- see docstring
    tgt = weighted_target(c_local, persistent, w_contour, w_cloud, target_points, tgt_rng)
    resB = reversed_align(partial, tgt, sigmas=(1.0, 0.4), field_vox=field_vox,
                           yaw_search_deg=refine_yaw_window, t_search_radius=max_t + 3.0,
                           yaw_seeds=yaw_seeds, iters=iters, lambda_psi=lambda_psi,
                           src_points=src_points, target_points=target_points, topk=topk,
                           seed=seed + 1, record=True, rec_stride=2, verbose=False)
    T_total = resB.transform @ resA.transform
    history = list(resA.history) + [T @ resA.transform for T in resB.history]
    return ReversedAlignResult(transform=T_total, yaw=resB.yaw, energy=resB.energy, history=history,
                                coarse_history=resA.coarse_history)
