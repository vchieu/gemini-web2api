"""Unified skinned hand mesh (Tier B).

One smooth-unioned mesh per hand, built once from a signed-distance field
(SDF) in the wrist-local rest pose and cached in the user cache dir::

    origin : wrist centre
    +y     : toward the fingers
    +x     : toward the pinky (right hand; mirrored for the left)
    +z     : up (dorsal side)

The rest pose (F1) is a slight curl + slight spread solved with the existing
``solve_chain`` IK -- no rig required.  Primitives are forearm/palm
ellipsoids from the ``_ARM`` stations, per-finger tapered capsules on the
rest chains plus joint spheres, and a thumb CMC capsule.  Combination uses
``smin`` (``-k*logaddexp``) inside a finger / finger<->palm / CMC<->palm and
plain ``min()`` finger<->finger so neighbours touch but never fuse into a
sheet (F2).

Per-frame skinning (F3, in ``hand3d``) reuses the already-solved IK ``q``
chains: per bone ``R_b = [X T Z] @ [X0 T0 Z0].T``,
``v' = P_b + R_b @ (v - P0_b)`` (linear blend skinning, vectorised).

No new dependencies: numpy + PIL + vtk/vedo only (no scipy).
"""

from __future__ import annotations

import os
import time

import numpy as np

HANDMESH_VERSION = "v1"
from pianoplayer.handspec import GEOMETRY_HASH
# Deviation (H6, plan-sanctioned fallback): GRID_STEP 0.12 -> 0.15. Timing
# evidence (size M, same machine): 0.12 SDF alone 5.4-8.1 s (total build
# 7-10 s incl. decimation); 0.15 SDF ~3.0-3.4 s, total ~4.5 s. Decimation to
# <=6000 tris dominates the final surface detail either way (a 3k-vert hand
# cannot resolve below ~2 mm), so 0.15 loses nothing visible.
GRID_STEP = 0.15
GRID_Y0 = -24.0   # fixed proximal border: the stub pole (~-23.5) and the
                  # widest station reach (XXL idx2: -23.2) both close inside
                  # it (app frame bottom is -18, so the cap never shows)
GRID_X = (-9.0, 9.0)
GRID_Y = (-20.0, 22.0)   # Deviation (H6): plan says y max 12, which clips the
GRID_Z = (-3.0, 4.0)     # rest fingertips (F1 chains reach y~14-21, z~3.3);
                         # extended so the iso surface stays closed.
TEX_NAME = "handtex_v2.png"
TRIS_MAX = 6000

# F2: true-anatomy finger widths.  The tubes (~0.75 cm mid-finger radius at
# size M) looked skinny; the SDF uses wider capsules so neighbours touch in
# the playing pose (gap == 1) while the F1 rest spread keeps small positive
# gaps.  Values are multipliers on the _SPECS radii (which already include
# the Tier-A _FING_R_SCALE); mid-finger half-width lands at 0.85-1.00 cm
# for size M (hf=0.82).  Deviation (H10): increased 2-5 scales to close gap
# in playing pose; max mid-width stays ≤1.00 cm.
# F2: finger widths from handspec (BUG-3 removed: use SPECS radii directly)
# No additional _SDF_R_SCALE multiplier; FING_R_SCALE from SPECS is applied

_K_INTRA = 0.05    # within one finger: near-hard union
_K_PALM = 0.47     # finger <-> palm/thenar: webbing bridges
_K_CMC = 0.40      # thumb CMC <-> palm
_K_BODY = 0.30     # forearm/palm ellipsoid chain


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def cache_dir():
    """User cache dir for meshes + textures (created on demand)."""
    d = os.path.join(os.path.expanduser("~"), ".cache", "pianoplayer")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def cache_path(hf, side):
    """Per-(hf, side, geometry-hash) cache filename: R1.4.

    The mesh must be rebuilt whenever the geometry constants change
    (geometry hash), when hand size changes, or for the opposite side.
    """
    s = "left" if str(side).lower().startswith("l") else "right"
    return os.path.join(cache_dir(),
                        "handmesh_%s_%.2f_%s_%s.npz" % (s, float(hf),
                                                        HANDMESH_VERSION,
                                                        GEOMETRY_HASH))


def texture_path(hf, side):
    """Per-(hf, side) skin texture: R1.5, per-hand-size + handedness."""
    s = "left" if str(side).lower().startswith("l") else "right"
    return os.path.join(cache_dir(), "handtex_%s_%.2f_%s.png" % (s, float(hf),
                                                                 HANDMESH_VERSION))


# --------------------------------------------------------------------------- #
# SDF helpers (pure numpy)
# --------------------------------------------------------------------------- #
def _smin(a, b, k):
    a = np.asarray(a)
    b = np.asarray(b)
    return -k * np.logaddexp(-a / k, -b / k)


def _seg_proj(p, a, b):
    """Distance of points `p` (N,3) to segment a->b; returns (d, t, closest)."""
    dt = np.asarray(p).dtype
    a = np.asarray(a, dtype=dt)
    b = np.asarray(b, dtype=dt)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        d = np.linalg.norm(p - a[None, :], axis=1)
        return d, np.zeros(p.shape[0]), np.repeat(a[None, :], p.shape[0],
                                                  axis=0)
    t = np.clip((p - a[None, :]) @ ab / denom, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * ab[None, :]
    d = np.linalg.norm(p - closest, axis=1)
    return d, t, closest


def _capsule(p, a, b, r0, r1):
    d, t, _ = _seg_proj(p, a, b)
    return d - (float(r0) + (float(r1) - float(r0)) * t)


def _ellipsoid(p, c, r):
    """Approximate SDF of an axis-aligned ellipsoid (stable, no scipy)."""
    dt = np.asarray(p).dtype
    c = np.asarray(c, dtype=dt)
    r = np.asarray(r, dtype=dt)
    r = np.maximum(r, 1e-6)
    q = (p - c[None, :]) / r[None, :]
    k0 = np.linalg.norm(q, axis=1)
    denom = np.linalg.norm((p - c[None, :]) / (r[None, :] ** 2), axis=1)
    denom = np.maximum(denom, 1e-9)
    out = k0 * (k0 - 1.0) / denom
    # deep inside: fall back to the scaled radius so the field stays finite
    out = np.where(k0 < 1e-9, -float(np.min(r)), out)
    return out


# --------------------------------------------------------------------------- #
# rest pose (F1) + bones
# --------------------------------------------------------------------------- #
def _thumb_by_for(hf):
    from pianoplayer.hand3d import _HF_XXS, _THUMB_BASE_Y0, _THUMB_FWD
    return max(_THUMB_BASE_Y0,
               _THUMB_BASE_Y0 + _THUMB_FWD * (float(hf) - _HF_XXS))


def rest_chains(hf):
    """F1 rest joints: {finger: q (4,3)} + thumb CMC start, right-hand local.

    target = base + (spread_x, L*0.96, +0.3), spread_x = (f-3)*0.35,
    solved with the existing solve_chain (slight curl + slight spread).
    """
    from pianoplayer.hand3d import _SPECS, solve_chain
    hf = float(hf)
    chains = {}
    for f in (1, 2, 3, 4, 5):
        spec = _SPECS[f]
        bx, by, bz = spec["base"]
        if f == 1:
            by = _thumb_by_for(hf) / hf  # _L multiplies y by hf below
        base = np.array([bx * hf, by * hf, bz * hf])
        lens = [l * hf for l in spec["lens"]]
        L = sum(lens)
        spread = (f - 3) * 0.35 * hf
        target = base + np.array([spread, L * 0.96, 0.3 * hf])
        pole = np.array(spec["pole"], float)
        pts, _ = solve_chain(base, target, lens, pole)
        chains[f] = np.array(pts, float)
    cmc_a = np.array([-3.0 * hf, -0.5 * hf, 0.3 * hf])
    return chains, cmc_a


def rest_bones(hf, side="right"):
    """Bone segments (a, b) in side-local coords.

    Order (B=18): forearm, palm, thumb-CMC, then 3 per finger (f1..f5).
    The first three are rigid with the palm (same root/R transform); the
    15 finger bones articulate with the IK chains.
    """
    from pianoplayer.hand3d import _SPECS  # noqa: F401 (keeps specs in sync)
    hf = float(hf)
    s = -1.0 if str(side).lower().startswith("l") else 1.0
    chains, cmc_a = rest_chains(hf)
    if s < 0:
        cmc_a = cmc_a * np.array([-1.0, 1.0, 1.0])
    bones = []
    names = []
    # forearm: far arm (out of frame) -> wrist
    bones.append((np.array([0.0, -20.0, 0.0]),
                  np.array([0.0, -1.8 * hf, 0.2 * hf])))
    names.append("forearm")
    # palm: wrist -> knuckle line
    bones.append((np.array([0.0, -1.8 * hf, 0.2 * hf]),
                  np.array([1.0 * s * hf, 8.2 * hf, 1.0 * hf])))
    names.append("palm")
    # thumb CMC: thenar -> thumb MCP
    q1 = chains[1][0].copy()
    if s < 0:
        q1 = q1 * np.array([-1.0, 1.0, 1.0])
    bones.append((cmc_a, q1))
    names.append("f1_cmc")
    for f in (1, 2, 3, 4, 5):
        q = chains[f]
        if s < 0:
            q = q * np.array([-1.0, 1.0, 1.0])
        for i in range(3):
            bones.append((np.array(q[i]), np.array(q[i + 1])))
            names.append("f%d_%d" % (f, i))
    return bones, names


def _bone_frames(dirs):
    """Rest/current bone rotation columns: R[b] = [X T Z] (B,3,3).

    Same construction as hand3d._frames (X ~ world +x, Z = X x T).
    """
    from pianoplayer.hand3d import _frames
    T = np.array(dirs, float)
    n = np.linalg.norm(T, axis=1, keepdims=True)
    T = T / np.maximum(n, 1e-9)
    X, Z = _frames(T)
    R = np.stack([X, T, Z], axis=2)
    return R


def rest_frames(hf, side="right"):
    """Per-bone rest (P0, R0): rigid bones get P0=0/R0=I (F3)."""
    bones, names = rest_bones(hf, side)
    B = len(bones)
    P0 = np.zeros((B, 3))
    R0 = np.repeat(np.eye(3)[None], B, axis=0)
    dirs = []
    for (a, b) in bones:
        d = np.array(b, float) - np.array(a, float)
        n = np.linalg.norm(d)
        dirs.append(d / max(n, 1e-9))
    Rf = _bone_frames(np.array(dirs))
    for i, nm in enumerate(names):
        if nm in ("forearm", "palm", "f1_cmc"):
            continue  # rigid: P0=0, R0=I
        P0[i] = np.array(bones[i][0], float)
        R0[i] = Rf[i]
    return bones, names, P0, R0


# --------------------------------------------------------------------------- #
# SDF field
# --------------------------------------------------------------------------- #
def _body_ellipsoids(hf):
    """Forearm->knuckle ellipsoid chain from the _ARM stations + eminences.

    The two farthest _ARM stations (y=-46,-30) are replaced by five unscaled
    forearm-stub ellipsoids (y -21.5..-7.5): the gap-based ry of the -30
    station (up to ~19 cm at small sizes) would otherwise poke through any
    reasonable grid border and leave the mesh open. The stub chain ends in
    a rounded pole at y~-23.5, safely inside the grid (app frame bottom is
    -18, so the cap never shows). Station ry is capped at 4.0 for the same
    reason; distal gaps (~2-3 cm) never hit the cap.
    """
    from pianoplayer.hand3d import _ARM, _ARM_PIN
    hf = float(hf)
    n = len(_ARM)
    ys = np.array([(y if k < _ARM_PIN else y * hf) for k, (y, *_r) in
                   enumerate(_ARM)])
    ell = []
    for k, (y, c, a1, a2, _p) in enumerate(_ARM):
        if k < _ARM_PIN:
            continue  # far out-of-frame stations: replaced by stubs below
        gaps = []
        if k > 0:
            gaps.append(ys[k] - ys[k - 1])
        if k + 1 < n:
            gaps.append(ys[k + 1] - ys[k])
        ry = min(0.75 * max(gaps) if gaps else 3.0, 4.0)
        ell.append((np.array([c * hf, ys[k], 0.0]),
                    np.array([a1 * hf, max(float(ry), 1.2 * hf),
                              a2 * hf])))
    # unscaled forearm stubs: constant slim tube ending rounded at y ~ -23.5
    for sy in (-21.5, -18.0, -14.5, -11.0, -7.5):
        ell.append((np.array([0.0, sy, 0.0]),
                    np.array([3.3 * hf, 2.0, 2.3 * hf])))
    # thenar / hypothenar eminences (same y neighbourhood as _arm_surface)
    ell.append((np.array([-3.4 * hf, 2.2 * hf, 0.4 * hf]),
                np.array([1.6 * hf, 2.8 * hf, 1.2 * hf])))
    ell.append((np.array([4.1 * hf, 4.2 * hf, 0.3 * hf]),
                np.array([1.3 * hf, 2.2 * hf, 1.0 * hf])))
    return ell


def _finger_capsules(hf, chains):
    """Per finger: 2 tapered capsules + PIP joint sphere (right-hand local).

    The distal capsule runs PIP->tip straight: the rest DIP bend is only
    ~0.14 rad (sagitta ~1.6 mm, subsumed by the ~20 mm width), and the
    capsule-capsule smin joint reads as a knuckle crease under the H9
    texture. Two evals + one sphere per finger instead of six.
    """
    from pianoplayer.hand3d import _SPECS
    hf = float(hf)
    out = {}
    for f in (1, 2, 3, 4, 5):
        spec = _SPECS[f]
        q = chains[f]
        base_r = [r * hf for r in spec["radii"]]
        caps = [(q[0], q[1], base_r[0], base_r[1]),
                (q[1], q[3], base_r[1], base_r[3] * 0.92)]
        spheres = [(q[1], max(base_r[1], base_r[2]) * 1.02)]
        out[f] = (caps, spheres)
    return out


def _sdf_slice(xy, y, chains, cmc_a, body_ell, fing_caps, hf):
    """SDF over one y slice (points (M,3) with constant y)."""
    p = np.asarray(xy, np.float32)
    INF = np.full(p.shape[0], np.inf, np.float32)
    # body: smin chain of ellipsoids
    d_body = INF
    for (c, r) in body_ell:
        if abs(y - c[1]) > (r[1] + 2.5):
            continue  # y-culling: no influence on this slice
        d_body = _smin(d_body, _ellipsoid(p, c, r), _K_BODY)
    # fingers
    d_fings = {}
    for f, (caps, spheres) in fing_caps.items():
        d_f = INF
        for (a, b, r0, r1) in caps:
            rm = max(float(r0), float(r1))
            if y < min(a[1], b[1]) - rm - 0.6:
                continue
            if y > max(a[1], b[1]) + rm + 0.6:
                continue
            d_f = _smin(d_f, _capsule(p, a, b, r0, r1), _K_INTRA)
        for (c, r) in spheres:
            if abs(y - c[1]) > r + 0.6:
                continue
            d = np.linalg.norm(p - c[None, :], axis=1).astype(np.float32) \
                - np.float32(r)
            d_f = _smin(d_f, d, _K_INTRA)
        d_fings[f] = d_f
    # thumb CMC capsule
    q1 = chains[1][0]
    d_cmc = _capsule(p, cmc_a, q1, 1.30 * hf, 1.06 * hf)
    # combine: palm/fingers with smin, fingers among themselves with min
    d_hand = _smin(d_body, d_cmc, _K_CMC)
    d_2345 = INF
    for f in (2, 3, 4, 5):
        d_2345 = np.minimum(d_2345, d_fings[f])
    d_hand = _smin(d_hand, d_2345, _K_PALM)
    d_hand = _smin(d_hand, d_fings[1], _K_PALM)
    return d_hand


def _build_sdf(hf):
    chains, cmc_a = rest_chains(hf)
    f32 = lambda a: np.asarray(a, np.float32)
    chains = {f: f32(q) for f, q in chains.items()}
    cmc_a = f32(cmc_a)
    body_ell = [(f32(c), f32(r)) for (c, r) in _body_ellipsoids(hf)]
    raw_caps = _finger_capsules(hf, {f: np.asarray(q, float) for f, q in
                                     chains.items()})
    fing_caps = {}
    for f, (caps, spheres) in raw_caps.items():
        fing_caps[f] = ([(f32(a), f32(b), r0, r1) for (a, b, r0, r1) in
                         caps],
                        [(f32(c), float(r)) for (c, r) in spheres])
    # Deviation (H6): adaptive y extent. The plan's fixed y[-20,12] clips the
    # F1 rest fingertips (chains reach y~14 at M, ~21 at XXL). Per-hf extent
    # (max rest tip y + 2.0 margin for radii + smoothing) is exact and cheap.
    # The proximal border is fixed (GRID_Y0): the stub pole (~-23.5) always
    # closes inside it. x/z likewise adapt to the primitive extents (+1.2
    # margin + max finger radius), so small hands build small grids.
    tipmax = max(float(q[:, 1].max()) for q in chains.values())
    y1 = max(tipmax + 2.0, GRID_Y0 + 4.0)
    ys = np.arange(GRID_Y0, y1 + 1e-9, GRID_STEP)
    xlo = min([float(np.min([q[:, 0].min() for q in chains.values()]))] +
              [float(c[0] - r[0]) for (c, r) in body_ell]) - 1.0
    xhi = max([float(np.max([q[:, 0].max() for q in chains.values()]))] +
              [float(c[0] + r[0]) for (c, r) in body_ell]) + 1.0
    zlo = min([float(c[2] - r[2]) for (c, r) in body_ell]) - 1.0
    zhi = max([float(np.max([q[:, 2].max() for q in chains.values()]))] +
              [float(c[2] + r[2]) for (c, r) in body_ell]) + 1.0
    # finger radii extend past the joint positions themselves
    from pianoplayer.hand3d import _SPECS as _SP
    rmax = max(max(_SP[f]["radii"]) * float(hf)
               for f in (1, 2, 3, 4, 5))
    xlo -= rmax
    xhi += rmax
    zhi += rmax
    xs = np.arange(xlo, xhi + 1e-9, GRID_STEP)
    zs = np.arange(zlo, zhi + 1e-9, GRID_STEP)
    nx, ny, nz = xs.size, ys.size, zs.size
    # Slice-major flat coordinates (idx = (j*nx+i)*nz+k): one boolean mask
    # per primitive selects exactly the slices the per-slice culling would
    # keep, so the field is bit-identical to the slice loop while each
    # primitive is evaluated in a single big numpy call.
    X = np.tile(np.repeat(xs, nz), ny).astype(np.float32)
    Y = np.repeat(ys, nx * nz).astype(np.float32)
    Z = np.tile(zs, ny * nx).astype(np.float32)
    INF = np.full(nx * ny * nz, np.inf, np.float32)

    def _block(mask):
        return np.stack([X[mask], Y[mask], Z[mask]], axis=1)

    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        acc_body = INF.copy()
        for (c, r) in body_ell:
            m = (Y >= c[1] - r[1] - 2.5) & (Y <= c[1] + r[1] + 2.5)
            if not m.any():
                continue
            d = _ellipsoid(_block(m), c, r)
            acc_body[m] = _smin(acc_body[m], d, _K_BODY)
        acc_f = {}
        for f, (caps, spheres) in fing_caps.items():
            acc = INF.copy()
            for (a, b, r0, r1) in caps:
                rm = max(float(r0), float(r1))
                m = (Y >= min(a[1], b[1]) - rm - 0.6) & \
                    (Y <= max(a[1], b[1]) + rm + 0.6)
                if not m.any():
                    continue
                acc[m] = _smin(acc[m], _capsule(_block(m), a, b, r0, r1),
                               _K_INTRA)
            for (c, r) in spheres:
                m = (Y >= c[1] - r - 0.6) & (Y <= c[1] + r + 0.6)
                if not m.any():
                    continue
                d = np.linalg.norm(_block(m) - c[None, :], axis=1).astype(
                    np.float32) - np.float32(r)
                acc[m] = _smin(acc[m], d, _K_INTRA)
            acc_f[f] = acc
        q1 = chains[1][0]
        # CMC: evaluated everywhere (as in the slice loop: no culling).
        acc_cmc = _capsule(np.stack([X, Y, Z], axis=1), cmc_a, q1,
                           1.30 * hf, 1.06 * hf)
        d_2345 = np.minimum(np.minimum(acc_f[2], acc_f[3]),
                            np.minimum(acc_f[4], acc_f[5]))
        d_hand = _smin(_smin(_smin(acc_body, acc_cmc, _K_CMC),
                             d_2345, _K_PALM), acc_f[1], _K_PALM)
    vol = d_hand.reshape(ny, nx, nz).transpose(1, 0, 2).astype(np.float32)
    return vol, (xs, ys, zs)


# --------------------------------------------------------------------------- #
# mesh extraction + cache
# --------------------------------------------------------------------------- #
def _pd_to_tris(pd):
    import vtkmodules.util.numpy_support as _ns
    polys = pd.GetPolys()
    data = _ns.vtk_to_numpy(polys.GetData())
    if len(data) and data[0] == 3 and (data[0::4] == 3).all():
        return np.ascontiguousarray(data.reshape(-1, 4)[:, 1:].astype(
            np.int64))
    tris = []
    i = 0
    while i < len(data):
        m = int(data[i])
        cell = [int(v) for v in data[i + 1:i + 1 + m]]
        if m == 3:
            tris.append(cell)
        elif m == 4:
            tris.extend(([cell[0], cell[1], cell[2]],
                         [cell[0], cell[2], cell[3]]))
        elif m > 4:
            for k in range(1, m - 1):
                tris.append([cell[0], cell[k], cell[k + 1]])
        i += 1 + m
    return np.array(tris, np.int64)


def _extract_mesh(vol, grid):
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkFiltersCore import (
        vtkPolyDataNormals,
        vtkQuadricDecimation,
        vtkWindowedSincPolyDataFilter,
    )
    try:
        from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D as _FE
    except Exception:  # pragma: no cover - older VTK keeps the serial filter
        from vtkmodules.vtkFiltersCore import vtkMarchingCubes as _FE
    xs, ys, zs = grid
    img = vtkImageData()
    img.SetDimensions(xs.size, ys.size, zs.size)
    img.SetOrigin(xs[0], ys[0], zs[0])
    img.SetSpacing(GRID_STEP, GRID_STEP, GRID_STEP)
    import vtkmodules.util.numpy_support as _ns
    arr = _ns.numpy_to_vtk(vol.ravel(order="F"), deep=1)
    arr.SetName("sdf")
    img.GetPointData().SetScalars(arr)
    fe = _FE()
    fe.SetInputData(img)
    fe.SetValue(0, 0.0)
    fe.ComputeNormalsOff()
    fe.Update()
    sm = vtkWindowedSincPolyDataFilter()
    sm.SetInputConnection(fe.GetOutputPort())
    sm.SetNumberOfIterations(15)
    sm.NormalizeCoordinatesOn()
    sm.Update()
    ntri = sm.GetOutput().GetNumberOfPolys()
    if ntri > TRIS_MAX:
        dec = vtkQuadricDecimation()
        dec.SetInputConnection(sm.GetOutputPort())
        dec.SetTargetReduction(1.0 - TRIS_MAX / float(ntri))
        dec.Update()
        pd = dec.GetOutput()
    else:
        pd = sm.GetOutput()
    nr = vtkPolyDataNormals()
    nr.SetInputData(pd)
    nr.ComputePointNormalsOn()
    nr.ComputeCellNormalsOff()
    nr.SplittingOff()
    nr.Update()
    pd = nr.GetOutput()
    import vtkmodules.util.numpy_support as _ns2
    pts = np.array(_ns2.vtk_to_numpy(pd.GetPoints().GetData()), float)
    nrm = np.array(_ns2.vtk_to_numpy(
        pd.GetPointData().GetNormals()), float)
    tris = _pd_to_tris(pd)
    return pts, tris, nrm


def build_rest_mesh(hf, side="right"):
    """Build (or load) the rest mesh; returns (pts, faces, nrm).

    Right-hand local SDF; mirrored x + reversed faces for the left hand
    (same convention as hand3d._arm_surface).  Cached as
    ``handmesh_{side}_{hf:.2f}_v1.npz`` (rest_pts, faces, rest_nrm).
    """
    s = "left" if str(side).lower().startswith("l") else "right"
    path = cache_path(hf, s)
    try:
        if os.path.isfile(path):
            z = np.load(path, allow_pickle=False)
            if {"rest_pts", "faces", "rest_nrm"} <= set(z.files):
                return (np.array(z["rest_pts"], float),
                        [tuple(int(v) for v in row)
                         for row in np.array(z["faces"], np.int64)],
                        np.array(z["rest_nrm"], float))
    except Exception:
        pass
    vol, grid = _build_sdf(float(hf))
    pts, tris, nrm = _extract_mesh(vol, grid)
    faces = [tuple(int(v) for v in row) for row in tris]
    if s == "left":
        pts = pts * np.array([-1.0, 1.0, 1.0])
        nrm = nrm * np.array([-1.0, 1.0, 1.0])
        faces = [tuple(reversed(f)) for f in faces]
    try:
        np.savez_compressed(path, rest_pts=np.asarray(pts, np.float32),
                            faces=np.asarray(faces, np.int64),
                            rest_nrm=np.asarray(nrm, np.float32))
    except Exception:
        pass
    return pts, faces, nrm


# --------------------------------------------------------------------------- #
# skinning weights (H7)
# --------------------------------------------------------------------------- #
def compute_weights(rest_pts, bones, rigid_names=("forearm", "palm"),
                    names=None):
    """Skinning weights: w = exp(-((d-d_min)/0.45)**2), top-4 renormalised.

    Verts whose nearest bone is a rigid (forearm/palm) bone and whose
    runner-up is >1.0 cm further away get a hard 1.0 (exact rigid motion).
    Returns (idx (N,4) int16, w (N,4) float32).
    """
    pts = np.asarray(rest_pts, float)
    N = pts.shape[0]
    B = len(bones)
    dmat = np.empty((N, B), np.float32)
    for b, (a, bb) in enumerate(bones):
        d, _, _ = _seg_proj(pts, np.asarray(a, float), np.asarray(bb, float))
        dmat[:, b] = d.astype(np.float32)
    dmin = dmat.min(axis=1)
    w = np.exp(-((dmat - dmin[:, None]) / 0.45) ** 2).astype(np.float32)
    idx = np.argsort(-w, axis=1)[:, :4]
    w4 = np.take_along_axis(w, idx, axis=1)
    w4 /= np.maximum(w4.sum(axis=1, keepdims=True), 1e-12)
    rigid = set(rigid_names or ())
    if rigid and names is not None and len(names) == B:
        order = np.argsort(dmat, axis=1)
        best = order[:, 0]
        second = dmat[np.arange(N), order[:, 1]]
        is_rigid = np.array([nm in rigid for nm in names])
        hard = is_rigid[best] & ((second - dmin) > 1.0)
        w4[hard] = np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)
        # keep the best bone first so the hard row is exact
        idx[hard, 1:] = idx[hard, 0:1]
    return idx.astype(np.int16), w4.astype(np.float32)


def get_rest_data(hf, side="right"):
    """Full cached rest set: pts/faces/nrm/idx/w/bones/names/P0/R0."""
    s = "left" if str(side).lower().startswith("l") else "right"
    path = cache_path(hf, s)
    bones, names = rest_bones(hf, s)
    _, _, P0, R0 = rest_frames(hf, s)
    try:
        if os.path.isfile(path):
            z = np.load(path, allow_pickle=False)
            if {"rest_pts", "faces", "rest_nrm", "idx", "w"} <= set(z.files):
                return dict(
                    pts=np.array(z["rest_pts"], float),
                    faces=[tuple(int(v) for v in row)
                           for row in np.array(z["faces"], np.int64)],
                    nrm=np.array(z["rest_nrm"], float),
                    idx=np.array(z["idx"], np.int16),
                    w=np.array(z["w"], np.float32),
                    bones=bones, names=names, P0=P0, R0=R0)
    except Exception:
        pass
    pts, faces, nrm = build_rest_mesh(hf, s)
    idx, w = compute_weights(np.asarray(pts, float), bones, names=names)
    try:
        np.savez_compressed(path, rest_pts=np.asarray(pts, np.float32),
                            faces=np.array(faces, np.int64),
                            rest_nrm=np.asarray(nrm, np.float32),
                            idx=np.asarray(idx, np.int16),
                            w=np.asarray(w, np.float32))
    except Exception:
        pass
    return dict(pts=np.array(pts, float), faces=faces,
                nrm=np.array(nrm, float), idx=idx, w=w,
                bones=bones, names=names, P0=P0, R0=R0)


# --------------------------------------------------------------------------- #
# planar UV + procedural skin texture (H9)
# --------------------------------------------------------------------------- #
def rest_uv(rest_pts):
    """Planar top-down UV from the rest bounds (static through skinning)."""
    p = np.asarray(rest_pts, float)
    xmin, xmax = p[:, 0].min(), p[:, 0].max()
    ymin, ymax = p[:, 1].min(), p[:, 1].max()
    W = max(float(xmax - xmin), 1e-9)
    H = max(float(ymax - ymin), 1e-9)
    uv = np.stack([(p[:, 0] - xmin) / W, (p[:, 1] - ymin) / H], axis=1)
    return uv.astype(np.float32), (xmin, xmax, ymin, ymax)


def _value_noise(size, cells, seed):
    """Smooth value noise: small random grid upscaled with PIL (no scipy)."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    small = rng.random((cells, cells)).astype(np.float32)
    img = Image.fromarray((small * 255.0 + 0.5).astype(np.uint8), "L")
    return np.asarray(img.resize((size, size),
                                 Image.BILINEAR)).astype(np.float32) / 255.0


def build_skin_texture(rest_pts, bones, out_png, names=None, hf=0.82):
    """Procedural 512^2 skin PNG: tone noise, joint/palm creases, nails."""
    from PIL import Image, ImageDraw
    S = 512
    base = np.array([0.95, 0.80, 0.70], float)
    n1 = _value_noise(S, 24, 7)
    n2 = _value_noise(S, 96, 13)
    tone = 1.0 + 0.03 * (0.65 * (n1 - 0.5) * 2.0 + 0.35 * (n2 - 0.5) * 2.0)
    img = np.empty((S, S, 3), float)
    for i in range(3):
        img[..., i] = base[i] * tone
    pil = Image.fromarray((np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    dr = ImageDraw.Draw(pil, "RGBA")
    p = np.asarray(rest_pts, float)
    xmin, xmax = p[:, 0].min(), p[:, 0].max()
    ymin, ymax = p[:, 1].min(), p[:, 1].max()
    W = max(float(xmax - xmin), 1e-9)
    H = max(float(ymax - ymin), 1e-9)

    def to_px(x, y):
        return ((float(x) - xmin) / W * (S - 1),
                (float(y) - ymin) / H * (S - 1))

    # finger joints + tips from the bone segments
    joints = {}
    tips = {}
    if names is not None:
        by_name = {nm: (np.asarray(a, float), np.asarray(b, float))
                   for nm, (a, b) in zip(names, bones)}
        for f in (1, 2, 3, 4, 5):
            try:
                a0, b0 = by_name["f%d_0" % f]
                _a1, b1 = by_name["f%d_1" % f]
                _a2, b2 = by_name["f%d_2" % f]
            except KeyError:
                continue
            joints[f] = (a0, b0, b1)
            tips[f] = b2
    else:
        for f in (1, 2, 3, 4, 5):
            try:
                a, b = bones[(f - 1) * 3]
                _a, b1 = bones[(f - 1) * 3 + 1]
                _a, b2 = bones[(f - 1) * 3 + 2]
            except Exception:
                continue
            joints[f] = (np.asarray(a, float), np.asarray(b, float),
                         np.asarray(b1, float))
            tips[f] = np.asarray(b2, float)
    # joint-crease bands (~12% dark + light ridge) across each finger u-span
    for f, (qm, qp, qd) in joints.items():
        r = 0.95 * float(hf)
        for j in (qm, qp, qd):
            x0, y0 = to_px(j[0] - r, j[1])
            x1, y1 = to_px(j[0] + r, j[1])
            for dy, col in ((-3, (120, 70, 60, 60)), (-1, (255, 235, 225, 70)),
                            (1, (120, 70, 60, 46))):
                dr.line([(x0, y0 + dy), (x1, y1 + dy)], fill=col, width=3)
    # palm-crease arcs
    for (x0, y0, x1, y1) in ((-2.5, 3.0, 3.5, 5.5), (-3.0, 1.0, 2.5, 3.0),
                             (0.5, 0.0, 4.0, 4.0)):
        try:
            ax0, ay0 = to_px(x0 * float(hf), y0 * float(hf))
            ax1, ay1 = to_px(x1 * float(hf), y1 * float(hf))
            dr.arc([ax0, ay0 - 30, ax1, ay1 + 30], 200, 340,
                   fill=(130, 75, 65, 55), width=3)
        except Exception:
            pass
    # nails: pinkish ellipse with a whiter tip + cuticle arc
    for f, t in tips.items():
        try:
            r = 0.62 * float(hf)
            x0, y0 = to_px(t[0] - r, t[1] - 1.15 * float(hf))
            x1, y1 = to_px(t[0] + r, t[1] - 0.10 * float(hf))
            dr.ellipse([x0, y0, x1, y1], fill=(247, 209, 189, 255))
            dr.ellipse([x0 + 3, y0 + 2, x1 - 3, y0 + (y1 - y0) * 0.45],
                       fill=(252, 238, 230, 255))
            dr.arc([x0, y0, x1, y1], 100, 260, fill=(150, 90, 80, 120),
                   width=2)
        except Exception:
            pass
    arr = np.asarray(pil).astype(np.float32) / 255.0
    # fingertip / palm-edge blush (warm, subtle)
    yy, xx = np.mgrid[0:S, 0:S]
    blush = np.zeros((S, S), float)
    for f, t in tips.items():
        cx, cy = to_px(t[0], t[1])
        blush += np.exp(-(((xx - cx) / 26.0) ** 2 + ((yy - cy) / 30.0) ** 2))
    blush = np.clip(blush, 0, 1) * 0.5
    arr[..., 0] = np.clip(arr[..., 0] + 0.015 * blush, 0, 1)
    arr[..., 1] = np.clip(arr[..., 1] - 0.010 * blush, 0, 1)
    arr[..., 2] = np.clip(arr[..., 2] - 0.008 * blush, 0, 1)
    # pore speckle
    rng = np.random.default_rng(29)
    arr *= (1.0 + (rng.random((S, S)) - 0.5) * 0.018)[..., None]
    out = Image.fromarray((np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
        out.save(out_png)
    except Exception:
        pass
    return out_png
