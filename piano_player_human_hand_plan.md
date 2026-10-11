# Phase R: Real Human Hand + Finger Numbering Algorithm Review

> Designed for small models, executing sequentially R0 → R1 → ... Each task has **Tasks** and **Acceptance**.
> Source: reading full code + Tier A/B diff + screenshots provided. **Not yet run tests** (review environment has no repo/vedo).
> Geometric numbers below are calculated directly from constants in code at size M (`hf = 0.82`).
> Items marked *(needs verification)* must be measured/tested before fixing.
> Add `PIANOPLAYER_HUMAN_HAND_PLAN.md` to `.gitignore` like the previous two plans.

---

## 1. Review of Updates per `PIANOPLAYER_HAND_LOOK_PLAN.md`

### 1.1 Task-by-Task Comparison

| Task | Status | Notes |
|---|---|---|
| H0 Measurement tool | ⚠ | Has `diag_hand_look.py` + `test_hand_look.py`. **Missing `docs/expected1.png`**, so `compare.png` is only "mid vs ends", not compared to real photo. `proj_ratio` is read from IK strings rather than measured on displayed mesh. Post-fix measurements not yet recorded in plan's Deviations. |
| H1.1 Reduce yaw | ✓ | `_YAW_ARM_MAX=0.10`, gain 0.4, `YAW_MAX=0.05`. |
| H1.2 `held` + lerp 0.75 | ✓ | Passes `held` from `_frame`. See BUG-7 (fingers not smoothed). |
| H1.3 Thumb abduction | ⚠ | Uses 0.5 instead of 1.0 (comment says 1.0 breaks 2 reach cells). Accepted. |
| H1.4 Finger extension | ⚠ | Replaced `style="flat"` with `_IDLE_FWD`. Accepted, but `aim_finger` **is still not passed** from `_frame`, so branch D5 "finger about to press lowers early" never executes. |
| H2.1 Finger width | ✗ (skinned) | `_FING_R_SCALE` only applies to tubes. Solid mesh uses separate `_SDF_R_SCALE` 1.30–1.58 → BUG-3. |
| H2.2 Tapering | ✗ (skinned) | Joint mounds, `_FING_TIP_FLAT`, taper only exist in tubes; SDF capsule is round, straight. |
| H2.3 Forearm | ✗ | **Out-of-scope fix**: plan only required y ∈ [-16, -4.5], diff also widened wrist station y = -1.8 (2.55 → 3.90) → BUG-1 (main cause of "ears" at the wrist in the photo). |
| H3.1 Press color | ⚠ | Exists only in tubes. In skinned, pressed fingers **do not change color** (does not call `_tint`/`_paint`). |
| H3.2 Nails | ⚠ | Nails are texture, color `(247,209,189)` is almost identical to skin `(242,204,178)` → almost invisible at image resolution (hand width ~150 px). |
| H3.3 Full hand shadow | ⚠ | Switched to opaque (for a reason, accepted), but blob uses **tube-arm hidden dataset**, not skinned mesh, no fingers → BUG-6. |
| H4 FXAA/Lighting | ✓ | `UseFXAAOn`, ambient/diffuse. FPS values not yet recorded. Edge-darkening only in tubes. |
| H5 Lock Tier A | ✗ | Cannot lock because expected image is missing. |
| H6 SDF mesh | ⚠ | `GRID_STEP` 0.15 (recorded). Added `_SDF_R_SCALE` (not in plan) → BUG-3. Plan requires 0.10 cm gap between fingers at rest pose, in reality fingers **overlap**. |
| H7 Weights | ⚠ | Leakage criterion ≤5% deemed "vacuous" (noted in `diag_handmesh.py`) — a sign that fingers overlap too much, making it untestable. |
| H8 Skinning | ⚠ | Accurate, but **2.7 ms/hand** vs plan budget < 1.5 ms. Precise optimization method exists (R6). |
| H9 Texture | ⚠ | See BUG-5 (a single texture file shared across all sizes/sides, colors multiplied). |
| H10 Integration | ⚠ | Switched default to `skinned` **before** acceptance (plan: switch only when passing). 2 hands × 2.7 ms = 5.4 ms > budget increase 4 ms *(needs measurement via `diag_frame.py`, because tubes also consume time)*. Version 2.5.0 synchronized in both files ✓. |

### 1.2 Newly Found Bugs / Risks

**BUG-1 (High) — Wrist becomes a "bulb", hourglass-shaped palm.**
Width (cm, size M) per current `_ARM` stations: y=-9: 5.9 · y=-4.5: 6.3 · **y=-1.8 (wrist): 6.4** · y=0.8: **5.2** · y=3.2: 6.7 · y=5.8: 7.5 · y=8.2: 7.1.
Before fix: 4.3 / 4.1 / 4.2. Wrist is now widest then pinches at y=0.8 → two protruding "ears" on both sides of the palm base in photos. Real anatomy: wrist is the **narrowest** spot (≈ 0.68–0.80 of palm width), forearm widens toward elbow, palm widens toward finger joints. Profile must be **monotonically increasing/decreasing in both directions** from the wrist.

**BUG-2 (High) — Thumb is longer than index finger.**
`_SPECS[1].lens = (4.2, 3.0, 2.5)` = 9.7 (hf=1): from MCP to thumb tip is **115% of index finger** (7.95 vs 6.89 cm at size M). Real humans ≈ 65–80%. Reason for length: CMC joint is rigid (bone `f1_cmc` fixed with palm), so the 3-segment chain compensates via length. Visible consequence: thumb looks like a long, thin "sixth finger" growing parallel to other fingers. Fix at root in R2.5 + R5.3, **do not patch by shrinking `lens`** (breaks reach tests).

**BUG-3 (High) — Finger overlap; "achieving metric by swelling fingers".**
SDF diameter at PIP: 2.05–2.16 cm; spacing of finger bases (MCP) only 1.72–1.84 cm → overlap ~0.3 cm. Comment in `handmesh.py` stating `_SPECS` "already includes `_FING_R_SCALE`" is **false** (that coefficient only applies in `_build` of hand3d). Plan option 2 (shrink MCP spacing 8–10%) was not tested; instead radius inflated ×1.45 so `gap_max == 1`. Consequence: mitten effect, skinning weights between adjacent fingers blend into each other (overlapping volumes → LBS distorts skin when fingers move independently).

**BUG-4 (High for dev loop) — Cache does not auto-invalidate.**
`handmesh_{side}_{hf}_v1.npz` and `handtex_v2.png` are locked only by `HANDMESH_VERSION`. After modifying `_SPECS`, `_ARM`, `_SDF_R_SCALE`... old mesh is still loaded → you think code changes did nothing. This phase modifies shapes heavily, so must lock by parameter hash (R1.4).

**BUG-5 (Medium) — Texture.**
(a) A single `handtex_v2.png` file used for all sizes and both sides, but rendered from the first built mesh (usually M, right hand): joint creases/nails match UV of that mesh → wrong position on left hand (UV flipped on x) and different sizes. (b) VTK multiplies texture color with actor color; mesh currently has `color=SKIN` **and** background texture is also SKIN → color is squared (≈ (0.90, 0.64, 0.49) before lighting) *(needs verification by setting actor color to white and comparing images)*.

**BUG-6 (Medium) — Shadows.**
`_add_blob(self.arm.dataset)` uses palm/arm of the **tubes** version (actors hidden). Skinned mesh is already in world coordinates so arm's matrix `R` cannot be used. Shadows lack finger shapes.

**BUG-7 (Medium) — Finger movement "jumping".**
Only `root` is smoothed (`TAU=0.065`). x of each fingertip is solved via direct IK every frame, so each time `p.aim` changes (or finger transitions from idle → `held`) the fingertip **teleports instantly** to 0.75×(fan_x − natural_x). Real humans don't teleport. Also `aim_finger` is not passed (see H1.4).

**BUG-8 (High process) — No metric measuring "human likeness".**
H0 measuring tilt/gap/ratio/bend: necessary conditions only. They do not detect hourglass shapes, overly long thumbs, or overlapping fingers. Need anatomical proportion metrics **on the mesh** and silhouette overlap with real hand photos (R0).

**BUG-9 (Medium) — Skinning slower than necessary.**
LBS is linear in matrices so `Rb[idx]` in shape `(M,4,3,3)` is unnecessary; matrix blending via a single multiplication `W(N×B) @ Rb(B×9)` then applying once (R6).

**BUG-10 (Low) — Code hygiene.**
`HandRig.__init__` assigns `side/s/hf/skin` twice; `handmesh.py`: `GRID_X/GRID_Y/GRID_Z`, `time`, `_sdf_slice` unused; `diag_handmesh.py::rep` has redundant `nonlocal_ok` variable; README mentions "fingertip blush" for skinned mesh while blush only exists in tubes. Deviations section of plan not updated.

---

## 2. Why Images Still Don't Look Like Human Hands (Read from Photos)

1. **Silhouette**: two "ears" at the wrist (BUG-1); rigid trapezoidal palm without smooth dorsal and thenar/hypothenar mounds; outer edge (hypothenar) is not a smooth curve from wrist to pinky.
2. **Fingers**: nearly uniform width from base to tip, sausage-like rounded tips; no subtle PIP/DIP joint bulges; no **webbing at the base** between fingers (real webbed spaces extend ~1/3 of the proximal phalanx); fanned out exposing gaps.
3. **Thumb**: too long, thin, growing parallel and base lacks muscular mass (BUG-2).
4. **Color/Lighting**: flat color patch, no crevice darkening, no gradient at joints/fingertips; nails nearly invisible. At this scale (~150 px per hand) **silhouette + cast shadows + nails** dictate 90% of the "real" feel, texture details are barely readable.
5. **Top-down orthographic camera** makes every hand look a bit "flat"; acceptable (Synthesia style) but mesh must still look good when pressing `v` to 55° angle (flat projected UV will stretch on side faces).

---

## 3. Strategy

```
R0 Measure (reference + anatomy)  →  R1 Bug fix / regression  →  R2 Rebuild anatomical mesh  ──► Pass?  → R4, R5, R6, R7
                                                                                           │ Fail
                                                                                           └► R3 (Hand-crafted / CC0 mesh) 
```

* **Principle 1:** Single source of truth for hand geometry (`handspec.py`), IK, mesh, texture all read from it (currently duplicated across `_SPECS`, `_ARM`, `_SDF_R_SCALE`, ... and misaligned).
* **Principle 2:** Do not "achieve metrics via shape tweaks" (like BUG-3). New metrics must measure anatomical proportions and real photo comparison.
* **Principle 3:** All changes must still pass `test_hand_reach.py` (tip_err < 0.05 cm, max_bend < 1.3), `test_regression.py`, `test_play.py`, `test_sync.py`; held fingers must lie **precisely** on key x-coordinates.
* **Decision Gate R3:** Only do R3 if after R2, IoU < 0.85 *or* user scores ≥ 2/5 R0.5 checklist items as "fail".

---

## R0. Reference and Anatomical Measurement Tools (First Step)

**Tasks**
1. **Your own reference photo** (copyright-free): shoot straight down, right hand naturally placed on keyboard (or flat on table with ruler beside). Save `docs/ref_top.jpg`. Create black-and-white mask `docs/ref_mask.png` (hand = white) using any software; can add `tools/make_ref_mask.py` (Pillow, skin color threshold) if desired.
2. **Real hand measurements** in `docs/hand_spec.json` (cm): hand length (wrist crease → middle fingertip), palm width at MCP joints, wrist width, individual finger lengths (base crease → tip), individual finger width at PIP, thumb length (MCP → tip), thumb natural abduction angle (top view). Reference ranges for sanity-check (replace with your numbers):

   | Proportion | Reference Range | Current (Size M) |
   |---|---|---|
   | palm width / hand length | 0.42–0.50 | 0.47 ✓ |
   | wrist width / palm width (MCP) | 0.68–0.80 | **0.85** ✗ |
   | wrist is narrowest point (monotonic profile both ways) | mandatory | **violated** (BUG-1) |
   | thumb (MCP→tip) / index finger | 0.65–0.80 | **1.15** ✗ |
   | index : ring : pinky relative to middle | ~0.90–0.95 : 0.92–0.97 : 0.72–0.80 | 0.92 : 0.92 : 0.78 ✓ |
   | finger diameter at PIP / adjacent MCP spacing | 0.85–1.00 | **1.16–1.20** ✗ |
   | natural thumb abduction angle | 20–40° | needs measurement |
   | DIP / PIP (flexion ratio) | 0.60–0.75 | 0.40 (`COUPLING`) |

3. Create `pianoplayer/handspec.py` (R0.2 below) and `diag_hand_anatomy.py` + `test_hand_anatomy.py` measuring **on mesh top** (rest and play poses), printing a table of:
   - width profile along y (is it monotonic both ways from wrist; contour defects > 0.15 cm along outer edge);
   - finger diameter / MCP spacing (no overlap; gap at 50% proximal phalanx in [0, 0.15] cm);
   - thumb/index ratio, wrist/palm width ratio;
   - number of connected components at 50% proximal phalanx of 4 fingers (=1) **and** at 85% finger length (≥ 1 allowing small fingertip gaps).
4. **Silhouette IoU** `diag_ref_iou.py`: render right hand in rest pose (fingers close, slightly curved), capture mask (like `diag_hand_look.py`), align via similarity transform (translation + rotation + scale via PCA/bbox), compute IoU with `ref_mask.png` (same for left hand using flipped image).
5. **Visual checklist** (user scores 1–5, printed in `compare.png`): silhouette · fingers · thumb · palm/wrist · color & shadows.

**Thresholds**

| Metric | Post R2 | Phase End |
|---|---|---|
| anatomical proportions within ±10% of real measurements | mandatory | mandatory |
| silhouette IoU vs real photo | ≥ 0.85 | ≥ 0.90 |
| monotonic width profile in both directions from wrist | mandatory | mandatory |
| visual checklist | no item < 3 | average ≥ 4 |

**Acceptance:** runs on current tree; **baseline must FAIL on BUG-1/2/3** (proof that measurement tools work). Record baseline in Deviations.

### R0.2 `handspec.py` (Single Source of Truth)
Contains (scaled by `hf` when used): joint positions (MCP/PIP/DIP/tip and thumb CMC), `lens` per segment, **width/thickness table per joint** (no shared circular radius), MCP spacing, palm/wrist/forearm profile `PALM_PROFILE` (y, half_width, half_thickness) with monotonicity check, joint rest angles, thumb parameters. `hand3d._SPECS`, `_ARM`, `handmesh` all import from here; `_SDF_R_SCALE` removed.

---

## R1. Bug Fixes and Regression (Cheap, Immediate Post-R0)

**R1.1 Wrist (BUG-1)** — `hand3d._ARM`. Restore monotonic profile, e.g., half-width `a1` (hf=1): y=-16: 3.5 · -9: 3.6 · -4.5: 3.45 · **-1.8 (wrist): 3.30** · 0.8: 3.55 · 3.2: 4.05 · 5.8: 4.50 · 8.2: 4.30 (starting point, adjust to real measurements in R0). Reduce protrusion of thenar/hypothenar ellipsoids in `handmesh._body_ellipsoids` so outer edge doesn't indent > 0.15 cm (test R0.3).
*Acceptance:* width profile in `test_hand_anatomy.py` is monotonic; wrist width / palm width ∈ [0.68, 0.80].

**R1.2 Finger Radius / Spacing (BUG-3)** — remove `_SDF_R_SCALE` (R0.2). Use spec radius × `_FING_R_SCALE` (like tubes). If `gap` at 50% proximal phalanx is still > 0.15 cm, **shrink MCP spacing 5–8%** (option 2 in old plan), **do not inflate fingers**. Fix incorrect comment in `handmesh.py`.
*Acceptance:* PIP diameter / MCP spacing ∈ [0.85, 1.00]; gap at 50% proximal phalanx ∈ [0, 0.15] cm; `H7` leakage ≤ 5% computable (no longer "vacuous").

**R1.3 Thumb:** *do not patch here* (see R2.5/R5.3). Log in Deviations.

**R1.4 Cache (BUG-4)** — filename `handmesh_{side}_{hf:.2f}_{hash8}.npz` and `handtex_{side}_{hf:.2f}_{hash8}.png`, `hash8 = sha1(repr(geometry params + SDF constants + GRID_STEP + HANDMESH_VERSION))[:8]`. Add `python -m pianoplayer.handmesh --clear-cache`.
*Acceptance:* changing a `handspec` constant ⇒ next run automatically rebuilds; no old files loaded.

**R1.5 Texture / Color (BUG-5)** — texture per `(hf, side)` (or flip u for left hand); when texture present, set `mesh.color((1,1,1))` to avoid color squaring; compare images before/after.
*Acceptance:* skin color in image ≈ `SKIN` (error ≤ 5% per channel on palm area), creases/nails align correctly on both hands and ≥ 3 sizes.

**R1.6 Lost Effects in Skinned (BUG-10 / H3.1 / H4.3)** — use **vertex color** (multiplied by texture; *needs verification* VTK multiplies scalar × texture) for: fingertip blush when pressing (via existing `snorm` in tubes), slight edge darkening on fingers, baked AO (R4.1). Requires `snorm`/`edge` array for skinned mesh: computed from bone weights (vertices close to bone `f?_2` ⇒ snorm→1).
*Acceptance:* pressed fingers blush only at tips; `test_hand_reach` still passes.

**R1.7 Shadows (BUG-6)** — blob uses **skinned mesh** dataset; shadow actor uses pure projection matrix (without multiplying `R`) since vertices are already in world coordinates. Try adding a blurred version (alpha 0.22) in a separate render layer drawn before hand (no depth-write) and measure budget; keep opaque version if > +2 ms.
*Acceptance:* shadow has full finger shapes; hides in perspective view; FPS drops < 3%.

**R1.8 Finger Smoothing + `aim_finger` (BUG-7)**
- In `vkeyboard._pump`, compute set `aim_fingers` = fingers of notes with `t_on - now < AIM_LEAD` that are not yet `fired`; pass `aim_fingers` into `rig.update(...)` (change `aim_finger` parameter to set, maintain compatibility).
- Fingers in `aim_fingers` or `held`: x **precisely** matches key x (no lerp toward natural) ⇒ no jumping when pressing.
- Idle fingers: critically damped spring on x, `tau ≈ 0.05 s` (does not slow down `held` fingers: mandatory `tip_err ≤ 0.05`).
*Acceptance:* `test_regression::test_finger_stays_on_held_key` passes; add test `max |Δx_tip|/frame ≤ 1.2 cm` (at 60 fps) for scaling scenario; `test_play.py` hold `tip_err` p95 ≤ 1.5 cm still achieved.

**R1.9 Hygiene (BUG-10)** — remove duplicate lines in `HandRig.__init__`, unused constants/imports/functions in `handmesh.py`, redundant variable in `diag_handmesh.py`; update README (blush); update Deviations with post-fix measurements.

---

## R2. Rebuilding Mesh per Anatomy (SDF v2)

File: `pianoplayer/handmesh.py` (+ `handspec.py`). Keep old pipeline (SDF → FlyingEdges → WindowedSinc → Quadric ≤ 8000 triangles) and hash caching.

**R2.1 Forearm – Wrist.** Elliptical cross-section (width : thickness ≈ 1.3–1.5), monotonic width profile per R1.1, **no** overlapping bulbous ellipsoids. Out-of-frame segment can be a simple truncated cone.

**R2.2 Palm.** Replace ellipsoid chain with a "rounded trapezoidal block with arch": linear half-width along y from wrist to MCP joints (widening), thickness per `PALM_PROFILE`, slight transverse arch: `z(x) = z0 + arch·(1 − (x/w)²)`, dorsal side curved, finger-base region slightly protruding at 4 MCP joint mounds. Outer edge (hypothenar) is **a single** smooth curve, large `k` smin (≥ 0.6), no protruding isolated ellipsoids.

**R2.3 Fingers 2–5.** Each segment is an **elliptical cross-section** capsule (a_x per width table, a_z = 0.80–0.88·a_x on dorsal side, fuller palmar side), radii interpolated between joints per table R0.2; slight mound at PIP (and very slight at DIP); fingertip is a flattened ellipsoid (pad) rather than a hemisphere. MCP radius ≤ 0.5 × MCP spacing (no overlap).

**R2.4 Interdigital Webbing.** Between adjacent fingers use `smin` with **y-dependent k**: `k(y) = k_web · (1 − smoothstep(y_mcp, y_mcp + 0.35*proximal_phalanx_len, y))`; outside that zone use `min` (no sticking). Thumb ↔ index web is wider (larger k, extending to ~40% of thumb proximal phalanx).

**R2.5 Thumb.** 3-bone chain **CMC→MCP (metacarpal, ~4.4 hf), MCP→IP (proximal phalanx, ~3.3 hf), IP→tip (distal phalanx, ~2.8 hf)**; MCP→tip ≈ 6.1 hf (≈ 0.72 index finger). Thenar mound is **a single** ellipsoid inclined ~35° along metacarpal, seamlessly connecting palm (k ≈ 0.6) and metacarpal (k ≈ 0.5). Left hand mirrored as before. Thumb nail rotated ~60–90° around axis (tilted outward).

**R2.6 Rest Pose = Average Playing Pose.** Replace "F1 slightly curved + splayed" with natural piano pose: fingers closed, MCP ~20°, PIP ~30–40°, DIP ≈ 0.65×PIP, thumb abducted ~25°. Rest pose close to most frequent playing pose minimizes LBS distortion.

**R2.7 Finger-wise Skinning Weights.** During SDF evaluation, record **nearest primitive label** for each vertex (palm / f1..f5 / cmc). Weights: vertex belonging to finger `f` only blends bones of finger `f` itself (along axial position, smooth transition 0.6 cm around each joint) and palm bone in finger base region; web vertices blend 50/50 between two adjacent fingers in web zone. Rigid 1.0 for palm/forearm as before.
*Acceptance:* vertex near finger i receives ≤ 5% weight from finger j ≠ i **except in web zones** (original H7 criterion, now meaningful).

**R2.8 Mesh Quality and Packaging.** `GRID_STEP` 0.12 (if time permits to build) or keep 0.15 + smoothing; triangles ≤ 8000/hand. Pre-build meshes for `XXS..XXL × 2 sides` (`tools/build_meshes.py`) stored in `pianoplayer/data/*.npz` (package_data + `MANIFEST.in`), loaded at runtime; rebuild only when hash mismatches. Eliminates ~4.5s first-run delay and old cache risks.

**R2.8 Acceptance:** `test_hand_anatomy.py` passes with real measurements; IoU ≥ 0.85; watertight mesh, 1 component; `test_hand_reach.py` passes (fingertip mesh matches `q[3]` ≤ 0.05 cm); visual checklist no item < 3; image `compare.png` attached to Deviations.

---

## R3. (Decision Gate) Hand-Crafted or CC0 Mesh Assets

Only if post-R2 fails (see Section 3). Reason: artist-crafted meshes (topology following muscles/tendons, wrinkles, nails) look much more "human" than mathematical SDF.

**Tasks**
1. Choose **CC0 or self-created** source (Blender sculpt/Skin modifier, or commissioned). Can consider MakeHuman base mesh (exported under CC0 — *re-verify terms of your used version before integrating*) then crop hand part. **Do not use MANO/SMPL-X** (non-commercial) and **do not use Sketchfab/TurboSquid "free to use" models** if terms forbid redistribution of assets — MIT repo will distribute it. Record source + license in `docs/ASSET_LICENSE.md`.
2. Define 21 landmarks (MediaPipe Hands style: wrist + 4 landmarks per finger) in `handspec.py`; `tools/import_hand_asset.py` reads OBJ/GLB via VTK (`vtkOBJReader` / `vtkGLTFReader`), registers asset landmarks with our bone chain (non-uniform scaling per bone to match `hand_spec.json`), outputs `.npz` matching R2.8 format (`rest_pts, faces, rest_nrm, uv, idx, w`). Weights taken from asset if available, otherwise R2.7.
3. `HandRig(mesh="asset")` loads this `.npz` via same `_build_skin`/`_update_skin` (skinning pipeline unchanged). Left hand = mirrored.

**Acceptance:** Same as R2, plus: visual checklist average ≥ 4; clear license; repo size increase < 3 MB.

---

## R4. Rendering (Post-Shape Accuracy)

- **R4.1 Baked AO** from SDF: at each rest vertex sample SDF at points offset along normal → `ao ∈ [0.75, 1]` (crevices, wrist creases darker), multiplied into vertex color. Runtime cost = 0.
- **R4.2 Skin Shading:** pinker at finger joints and tips, lighter on back of hand; slight noise (±3%). No veins.
- **R4.3 Nails:** higher contrast in texture (bright rim + small lunula), color deviation from skin ≤ 8% (avoid white spots); optional thin nail mesh rigidly attached to bone `f?_2` with mild specular (0.3).
- **R4.4 Soft Shadows:** see R1.7; try two slightly offset shadow maps with different alphas if budget permits.
- **R4.5 `--hq` (optional, default off):** try `vtkSSAOPass`; enable only if measured ≤ 3 ms/frame on target machine.
- **R4.6 Perspective View (`v`):** flat projected UV stretches on sides ⇒ use tri-planar projection *or* bake key details into vertex color so 55° angle still looks good.

**Acceptance:** standard deviation of brightness within hand mask ≥ 0.06 (no more flat color patch); FPS drop < 5% vs pre-R4.

---

## R5. Human-Like Movement

- **R5.1** Finger smoothing (R1.8) + natural acceleration: finger lifts before striking (`AIM_LEAD` exists), lowers to key per `TAU_PRESS`.
- **R5.2** Idle pose per R2.6; maintain `finger_proj_ratio ≥ 0.80` and `idle max_bend ≤ 0.9` (H0 threshold), but measured on **mesh**.
- **R5.3 Thumb with CMC:** 2-step IK solver — (i) metacarpal orientation: interpolate between rest orientation and target orientation, clamped in ±35° cone; (ii) remaining 2 segments solved flat (law of cosines) with `pole` pointing away from palm. Thumb MCP base now **moves** with CMC; update `rest_bones` (`f1_cmc` no longer rigid, becomes an articulated bone) and `_update_skin`. Must maintain `tip_err ≤ 0.05 cm` and `max_bend < 1.3` for all 70 reach cells.
- **R5.4** `COUPLING` 0.4 → try 0.6–0.7 (DIP ≈ 0.65×PIP like real humans). Remeasure reach; if any cell breaks, log in Deviations, **do not relax threshold**.
- **R5.5** Wrist: slight lift when hitting black keys (`BLACK_SHIFT` along y exists; add ±3° wrist pitch), hand pitch based on velocity.
- **R5.6** Tendon linking (subtle, visual only): when one finger presses, adjacent two fingers lift slightly by 0.1–0.2 cm, ring finger lifts slightly more than index. `held` fingers unaffected.

**Acceptance:** `test_hand_reach.py` (70 cells), `test_regression.py`, `test_play.py`, `test_sync.py` pass; no finger shifts > 1.2 cm/frame; visual inspection rotating 10s C major scale video.

---

## R6. Skinning Performance (BUG-9)

Linear LBS: `v' = Σ_b w_b (R_b (v − P0_b) + P_b) = (Σ w_b R_b) v + Σ w_b (P_b − R_b P0_b)`.

1. Build dense weight matrix `W (N×B, float32)` once (B = bone count, sparse non-zeros but B is small so dense is still cheap).
2. Every frame: `A = W @ Rb.reshape(B, 9)` → `(N,3,3)`; `t = W @ (Pb − einsum('bij,bj->bi', Rb, P0))`; `v' = einsum('nij,nj->ni', A, rest) + t`; normals `n' = normalize(einsum('nij,nj->ni', A, nrm0))`.
3. Keep rigid branch for vertices with weight 1.0 (palm/forearm) if measured faster.

*Acceptance:* max error vs old version ≤ 1e-4 cm across 100 random poses; ≤ **0.8 ms/hand** (size M, 3000–8000 vertices); total `_frame()+render()` for 2 hands increases by no more than 4 ms vs tubes (`diag_frame.py`); `diag_load.py` 60 s `underruns = 0`.

---

## R7. Testing, Documentation, Versioning

- New tests: `test_hand_anatomy.py`, `diag_ref_iou.py` (prints IoU, doesn't assert if reference image missing), finger smoothing test (R1.8), cache hash test (R1.4).
- `AGENTS.md` (test list, module map: `handspec.py`; hand3d/handmesh sections) and `README.md` updated simultaneously; version 2.6.0 in **both** `setup.py` and `pianoplayer/__init__.py`; add `pianoplayer/data/*.npz` to `package_data`/`MANIFEST.in`.
- `.gitignore`: `PIANOPLAYER_HUMAN_HAND_PLAN.md`, `diag_handmesh_out/`, `docs/ref_*` if hand photos shouldn't be pushed to repo.
- Update Deviations section (template at bottom).

---

# Part F: Finger Numbering Algorithm

## F-A. Review of Findings by Others

| Statement | Conclusion | Explanation |
|---|---|---|
| Description: Viterbi on finger pairs, `pair_cost`, `_skip_penalty`, `_repair_held` | **True** | Matches `hand.py`. |
| Global optimization, 500 notes ~0.28s | **True, conditionally** | Optimized **per cost model** with 2nd-order memory, not for "correct fingering". 0.28s is from `AGENTS.md`, not remeasured here. |
| Hard-soft `HARD=200` always has a solution | **True** | Note 200 ≫ typical velocity costs, so practically nearly hard. |
| `_repair_held` fixes long-held notes | **True, but greedy** | No re-running Viterbi after fix; total cost increase not guaranteed. |
| Lacks biomechanical model (tendons connecting fingers 3–4) | **True but vague, low priority** | Indeed no "independent finger 4" constraint. However, pair tension limits for (3,4), (4,5) in chords exist (`axba>5`). No specific failing scenario pointed out. |
| Thumb-under rule uses "fixed thresholds not reflecting wrist rotation" | **False on details** | No threshold for thumb-under exists. Only two boolean rules (do not switch to thumb on black keys when ascending; do not change thumb → other finger descending if thumb on black key and short note). **Thumb-under cost comes only from velocity** `reach_err/dt`; no separate cost by interval/preceding finger. Actual problem is *lack* of cost model for crossing over, and **complete ban** on finger crossing between non-thumb fingers (`(fb-fa)*xba<0` ⇒ HARD). |
| Large chords lack span limit with total | **True, and more specific** | Rules only check **adjacent** pairs in chords. 5-note chord, each adjacent pair valid but total span exceeds hand span passes. >5-note chords always violate. |
| Proposal: smooth muscle tension function instead of `if axba>…` chain | **Reasonable, medium priority** | Should have, including application to held notes (not just in chords). |
| Proposal: wrist rotation penalty | **Low priority** | Requires state beyond fingers; benefit hard to measure. Last resort if time permits. |

### Points Missed in Findings (Code Review; *Run tests to confirm in F0*)

1. **Time measured in "quarter lengths", not seconds.** `nb.time` is beat offset; `dt = |Δtime| + 0.1` adds 0.1 *beats*. Same piece at 60 and 200 bpm yields the **exact same** fingering, while `test_fingering.py` tests by seconds (0.25s). Tempo map is available (`score_tempo_map`) but not passed to `Hand`.
2. **Fast repeated notes always use the same finger.** Same finger + same key ⇒ `reach_err = 0` ⇒ 0 cost; changing fingers (3-2-1) always costs > 0. No "rapid note repetition penalty" ⇒ never generates 3-2-1 / 4-3-2-1 for rapid repeated notes.
3. **Absolute ban on same finger on different keys** (unless preceding note duration ≥ 4 beats): `fa==fb and xba!=0 and na.duration<4 ⇒ HARD`, regardless of 2-second rests or octave jumps. In practice that's valid technique (octave jumps with pinky 5 in bass, thumb sliding between adjacent keys, finger substitution). Correct constraint is "finger still holding preceding note" (already in B2) and "too fast" (measured in seconds).
4. **Cost = 0 when not moving ⇒ no static preference.** `weights`/`bfactor` only **multiply velocity**; when velocity is 0, thumb on black key and pinky on white key cost the same ⇒ tie ⇒ "keep first option" becomes arbitrary decision in slow passages.
5. **Thumb-under cost is just finger travel distance.** With `frest`, thumb-under after index (finger 2) has `reach_err` ≈ 5.8 cm, after middle (finger 3) ≈ 8.1 cm, after ring (finger 4) ≈ 10.4 cm (size M, white key interval) ⇒ model **favors thumb-under after finger 2**, opposite to pedagogy (after finger 3 or 4 in scales). Total cost might still pick correct `1 2 3 1 2 3 4 5` for scales due to summation; **must verify with gold set in F0**, don't assume beforehand.
6. **No external metric.** No ground truth so all adjustments to `HARD`, `WEAK_FINGER_*`, `weights` are guesswork. Repo already supports **PIG** format (PianoFingeringDataset, expert fingerings) via `PIG2Stream`; use it to measure match rate.

## F0. Evaluation Dataset (Before Algorithm Changes)

1. `eval_fingering.py`: (a) **Small hand-written gold standard**, hard test (no relaxed rules): C and G major scales 2 octaves — right hand ascending `1 2 3 1 2 3 4 5`, left hand ascending `5 4 3 2 1 3 2 1`; F major scale right hand `1 2 3 4 1 2 3 4`; C major arpeggio right hand `1 2 3 5 1 2 3 5`. (b) **PIG** (if dataset available, placed outside repo, path via argument): prints overall `match rate` per piece (comparing note-by-note with expert numbering; if multiple answers take best match) and average; do not redistribute dataset.
2. Record current **baseline** (gold pass/fail, PIG match rate) in Deviations before modifying anything.
3. Every F task below may only be merged if **it does not decrease** average PIG match rate and doesn't break gold; changes increasing ≥ 1 percentage point should be recorded.

## F1. Time in Seconds
- `reader` (or `Hand`) receives `tempo_map`; computes `t_sec` for each `INote` (`vkeyboard._beat_seconds` can be extracted into shared utility). `pair_cost` uses `dt_sec`; `0.1` becomes second-based constant; `WEAK_FINGER_MAX_DT` = 0.15s.
- CLI/GUI passes `score_tempo_map(sf)` (already available where `-v` is invoked).
- *Acceptance:* new test: same score, bpm 60 vs 200 yield **different cost sets**, gold still passes in both; `test_fingering` A1.1–A1.5 pass; 500 notes < 2s.

## F2. Repeated Notes
- Add `REPEAT_PENALTY`: same finger + same key + `dt_sec < 0.25` ⇒ add `c·(0.25 − dt)/0.25` (c ≈ 4–8, tuned via PIG); allow finger switching on repeated notes by reducing `reach_err` of repetition (same key ⇒ treat `reach_err` = 0 for all fingers if repeated note).
- *Acceptance:* sequence of 8 sixteenth-note C4s at 120 bpm does not use the same finger; at 40 bpm still allows same finger.

## F3. Same Finger Different Keys Over Time
- Replace unconditional `HARD` with: if preceding note still held (B2) ⇒ `HARD`; if `dt_sec < 0.25` ⇒ `HARD`; otherwise soft cost `SAME_FINGER_BASE + 0.15·|Δx|` (octave leaps cost more than 1-key slide). Preserve tests A1.2/A1.3.
- *Acceptance:* new test: two notes 2 seconds apart, octave interval ⇒ allowed to use same finger 5 if cost is lower; A1.2/A1.3 still 0 violations.

## F4. Chords and Spans
- **F4a (P1):** replace `if axba > …` chain with tables `MAX_SPAN[fa][fb]`, `COMFORT_SPAN[fa][fb]` (cm at hf=1, e.g., derived from hand specs) and smooth cost: `0` below COMFORT, quadratic increase up to MAX, `HARD` beyond MAX. Apply to pairs within the same chord **and** overlapping held note pairs.
- **F4b (P2):** for chords ≤ 5 notes, pre-generate valid **monotonic finger sets** (up to C(5,k) ≤ 10 for k = 2..4, 1 for k = 5), filter by total span, keep top-K (K=6) by static cost, then let Viterbi choose among those candidates (chord state = candidate). Chords > 5 notes: print single-line warning and split into two pitch groups (or report "unplayable with one hand").
- *Acceptance:* test: 5-note chord C-E-G-C-E (decima) ⇒ no solution with total span exceeding pair `MAX`; 7-note chord prints warning; 500 notes time still < 2s.

## F5. Finger Crossing
- Remove unconditional `HARD` for finger crossing between non-thumb fingers; replace with `CROSS_BASE` cost (3 over 4, 2 over 3, ...) increasing with interval and decreasing with time; thumb-under adds low `UNDER[fa]` cost after finger 3/4, high after 2/5 and high when interval ≥ 5 semitones unless note is slow.
- *Acceptance:* scale gold still passes; at least 3 PIG pieces (if available) do not drop match rate; README Limitations updated ("finger 3 over 4" no longer strictly banned).

## F6. Static Preference and Determinism
- Add small static cost per (finger, black/white key) **not multiplying velocity**: thumb on black key +0.5, pinky on black key +0.3, finger 4/5 on accented notes ... (constants tuned via PIG). Ties broken by picking finger with higher `weights`.
- *Acceptance:* running twice produces identical results; PIG does not drop.

## F7. `_repair_held`
- After repair, recompute total cost; if increased vs pre-repair ⇒ print warning; count repairs/piece. If > 1% notes repaired, log in Deviations to consider upgrading Viterbi memory to 3rd order (125 states).

## F8. Performance and Documentation
- Vectorize 5×5 `pair_cost` table with numpy per note pair (computed once per step), target 500 notes ≤ 0.6s after adding F1–F6.
- Update `AGENTS.md` section "The fingering algorithm" and README algorithmic/Limitations sections simultaneously.

---

## Priority Order

| Level | Task | Reason |
|---|---|---|
| **P0** | R0, R1.1, R1.2, R1.4, R1.5 | Accurate measurements, fix regressions causing current visual bugs (wrist bulb, overlapping fingers), clear old cache |
| **P0** | F0 | Without measurement tools, algorithm tuning is guesswork |
| **P1** | R1.6–R1.9, R2, F1–F3 | Anatomical shape; tempo/repeated notes/same finger are real playing bugs |
| **P1** | R5.1–R5.4 | Thumb with CMC goes with R2.5 |
| **P2** | R4, R5.5–R5.6, R6, F4–F8 | Polish |
| **Gate** | R3 | Only if R2 insufficient |

## Completion Criteria

1. `test_hand_anatomy.py`, `test_hand_look.py`, `test_hand_reach.py`, `test_regression.py`, `test_play.py`, `test_sync.py`, `test_fingering.py`, `test_audio.py` all pass.
2. IoU ≥ 0.90 vs your real hand photo; visual checklist average ≥ 4, no item < 3.
3. `eval_fingering.py`: gold passes 100%, PIG match rate not lower than baseline.
4. Budget: `_frame()+render()` for 2 hands increases by no more than 4 ms vs tubes, skinning ≤ 0.8 ms/hand, `diag_load.py` 60s `underruns = 0`.
5. You visually confirm on real machine: `pianoplayer scores/bach_invention4.xml -v`.

## Deviations (Record During Execution)

- Baseline R0 (pre-fix): _(fill: anatomical proportion table, IoU, tilt/gap/ratio/bend)_
- Baseline F0: _(fill: gold pass/fail, PIG match rate)_
- Thresholds adjusted and reasons: _(fill)_
- Changes from this plan and reasons: _(fill)_