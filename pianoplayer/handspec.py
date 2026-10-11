"""Single source of truth for hand geometry (R0.2).

All hand geometry parameters live here, scaled by hand factor (hf) when used.
hand3d.py, handmesh.py, and any tools import from this module.
"""

from __future__ import annotations
import math
import hashlib

# --------------------------------------------------------------------------- #
# Hand size factors (from utils.py, duplicated here for independence)
# --------------------------------------------------------------------------- #
_HAND_SIZE_FACTORS = {
    "XXS": 0.33,
    "XS": 0.50,
    "S": 0.65,
    "M": 0.82,
    "L": 1.00,
    "XL": 1.20,
    "XXL": 1.40,
}


def hand_size_factor(size: str) -> float:
    """Return the hand factor for a given size string."""
    return _HAND_SIZE_FACTORS.get(str(size).upper(), 0.82)


# --------------------------------------------------------------------------- #
# Finger specifications (per finger, for hf=1.0 right hand)
# base: MCP joint position in local hand frame (x, y, z) [cm]
# lens: segment lengths (MCP->PIP, PIP->DIP, DIP->tip) [cm]
# widths: half-widths at MCP, PIP, DIP, tip [cm] (for elliptical cross-section)
# thicknesses: half-thicknesses at MCP, PIP, DIP, tip [cm]
# pole: preferred side the finger arches toward (unit vector)
# --------------------------------------------------------------------------- #
SPECS = {
    1: dict(
        base=(-5.6, 2.4, 0.45),
        lens=(4.2, 3.0, 2.5),
        widths=(0.95, 0.86, 0.74, 0.62),
        thicknesses=(1.10, 1.00, 0.87, 0.73),
        pole=(0.15, 0.55, 1.00),
    ),
    2: dict(
        base=(-2.1, 8.2, 1.85),
        lens=(3.9, 2.5, 2.0),
        widths=(0.86, 0.80, 0.69, 0.58),
        thicknesses=(0.98, 0.91, 0.79, 0.67),
        pole=(0.00, 0.35, 1.00),
    ),
    3: dict(
        base=(0.15, 8.6, 1.95),
        lens=(4.3, 2.7, 2.1),
        widths=(0.88, 0.82, 0.71, 0.60),
        thicknesses=(1.00, 0.93, 0.81, 0.68),
        pole=(0.00, 0.35, 1.00),
    ),
    4: dict(
        base=(2.4, 8.3, 1.80),
        lens=(3.9, 2.5, 2.0),
        widths=(0.83, 0.77, 0.67, 0.56),
        thicknesses=(0.95, 0.88, 0.76, 0.64),
        pole=(0.00, 0.35, 1.00),
    ),
    5: dict(
        base=(4.5, 7.7, 1.60),
        lens=(3.2, 2.1, 1.8),
        widths=(0.77, 0.71, 0.61, 0.51),
        thicknesses=(0.88, 0.81, 0.70, 0.59),
        pole=(0.00, 0.30, 1.00),
    ),
}

# Backward compatibility: radii = widths (circular cross-section for tubes)
for f in SPECS:
    SPECS[f]["radii"] = SPECS[f]["widths"]


# --------------------------------------------------------------------------- #
# Arm + palm loft profile (y, centre_x, half_width, half_thickness, superellipse_p)
# y in cm, relative to wrist centre (y=0). Stations before ARM_PIN keep their y
# (out-of-frame forearm). Stations at/after ARM_PIN scale with hf.
# Monotonic width profile: wrist is the narrowest point (BUG-1 fix).
# --------------------------------------------------------------------------- #
ARM = (
    (-46.0, 0.00, 3.50, 2.70, 2.00),
    (-30.0, 0.00, 3.30, 2.60, 2.00),
    (-16.0, 0.00, 3.30, 2.30, 2.05),
    (-9.0, 0.00, 3.50, 1.90, 2.15),
    (-4.5, 0.00, 3.35, 1.55, 2.35),
    (-1.8, 0.00, 3.30, 1.40, 2.55),      # wrist (narrowest)
    (0.8, -0.35, 3.55, 1.36, 2.80),
    (3.2, -0.70, 4.00, 1.48, 3.00),
    (5.8, -0.10, 4.50, 1.46, 3.10),
    (8.2, 1.00, 4.30, 1.32, 3.05),      # knuckle line
    (9.2, 1.00, 3.90, 0.95, 2.80),
    (10.0, 1.00, 0.60, 0.55, 2.20),     # closed cap
)
ARM_PIN = 2         # first stations keep their y (out-of-frame arm)

# Thenar/hypothenar eminences (bulges on the palm section)
THENAR = 0.50       # thumb-side bulge
HYPO = 0.12         # pinky-side bulge
ARM_M = 26          # points around an arm section


# --------------------------------------------------------------------------- #
# Thumb geometry (R2.5): CMC joint is now articulated
# thumb_cmc_len: metacarpal length (CMC->MCP) [cm] at hf=1
# thumb_mcp_len: proximal phalanx (MCP->IP) [cm]
# thumb_ip_len: distal phalanx (IP->tip) [cm]
# thumb_abduction_rest: rest abduction angle [rad] from palm plane
# --------------------------------------------------------------------------- #
THUMB_CMC_LEN = 4.4
THUMB_MCP_LEN = 3.3
THUMB_IP_LEN = 2.8
THUMB_ABDUCTION_REST = math.radians(25.0)  # ~25 degrees


# --------------------------------------------------------------------------- #
# Palm profile for SDF (y, half_width, half_thickness)
# Used by handmesh.py for the "rounded trapezoidal block with arch"
# --------------------------------------------------------------------------- #
PALM_PROFILE = (
    (-1.8, 3.30, 1.40),   # wrist (matches ARM wrist station)
    (0.8, 3.55, 1.36),
    (3.2, 4.00, 1.48),
    (5.8, 4.50, 1.46),
    (8.2, 4.30, 1.32),
    (9.2, 3.90, 0.95),
)


# --------------------------------------------------------------------------- #
# Webbing parameters (R2.4)
# k_web: base smooth-min strength for finger-finger webbing
# web_y_extent: how far up the proximal phalanx the web extends (fraction)
# --------------------------------------------------------------------------- #
K_WEB = 0.55
WEB_Y_EXTENT = 0.35  # fraction of proximal phalanx length


# --------------------------------------------------------------------------- #
# Finger visual scaling (for tubes only; SDF uses exact widths from SPECS)
# --------------------------------------------------------------------------- #
FING_R_SCALE = {1: 1.0, 2: 1.10, 3: 1.10, 4: 1.10, 5: 1.05}
FING_TIP_TAPER = 0.92
FING_FLAT = 0.88        # z-radius scale of section ellipse (dorsal flatter)
FING_TIP_FLAT = 0.80    # flatter still over distal segment
FING_BELLY = 0.85       # extra z scale on pad side (sin < 0)


# --------------------------------------------------------------------------- #
# Rest pose (R2.6): natural piano pose - fingers slightly curled
# MCP ~20°, PIP ~35°, DIP ~0.65*PIP, thumb abducted ~25°
# --------------------------------------------------------------------------- #
REST_MCP_ANGLE = math.radians(20.0)
REST_PIP_ANGLE = math.radians(35.0)
REST_DIP_RATIO = 0.65
REST_THUMB_ABDUCTION = math.radians(25.0)


# --------------------------------------------------------------------------- #
# Rest pose offsets (world cm, tuned at M hf=0.82, scale with hf)
# --------------------------------------------------------------------------- #
REST_LIFT = {1: 0.4, 2: 0.9, 3: 1.1, 4: 1.0, 5: 0.8}
REST_PULL_Y = {1: 1.0, 2: 0.35, 3: 0.35, 4: 0.35, 5: 0.5}
REST_PULL_Z = 0.6
REST_REF_HF = 0.82
IDLE_FWD = {1: 1.2, 2: 1.2, 3: 1.7, 4: 1.2, 5: 1.2}
THUMB_SPLAY = 0.5       # idle thumb splays outward (x hf)


# --------------------------------------------------------------------------- #
# Breathing animation
# --------------------------------------------------------------------------- #
BREATHE_AMP = 0.08
BREATHE_FREQ = 0.35
BREATHE_PHASE = {1: 0.0, 2: 1.3, 3: 2.1, 4: 3.4, 5: 4.2}


# --------------------------------------------------------------------------- #
# Root placement (D1): scales with hand size so every size reaches keys
# --------------------------------------------------------------------------- #
ROOT_CONTACT_Y = -3.0
ROOT_WRIST_GAP = 8.6
ROOT_REACH_FRAC = 0.69
ROOT_Z_BASE = 2.0
ROOT_Z_C0 = 0.50
ROOT_Z_HF_K = 0.60

# Thumb MCP forward shift (D1b): progressive with hand size
THUMB_BASE_Y0 = 2.4
THUMB_FWD = 3.6
HF_XXS = 0.33

# Pinky contact retreat (D1c)
PINKY_CONTACT_DY = 0.80


# --------------------------------------------------------------------------- #
# Arm yaw (D3)
# --------------------------------------------------------------------------- #
KB_CENTER = 4.0 * 1.312 + 3.0 * 0.875  # from utils.KEYBSIZE, KEY_WB
SHOULDER_OFF = 15.0
YAW_ARM_LEN = 45.0
YAW_ARM_MAX = 0.10
YAW_ARM_GAIN = 0.4


# --------------------------------------------------------------------------- #
# Look/feel constants
# --------------------------------------------------------------------------- #
SKIN = (0.95, 0.80, 0.70)
SKIN_PRESS = (0.96, 0.72, 0.64)
NAIL_PATCH = (0.97, 0.82, 0.74)
NAIL_MIX = 0.6

JOINT_SHADE = (0.93, 0.895, 0.905)
JOINT_WIDTH = 0.18
PRESS_AO = 0.20

SKIN_AMBIENT = 0.40
SKIN_DIFFUSE = 0.70
SKIN_SPECULAR = 0.08

# Shadow
SHADOW_OFF = (0.35, -0.35)
SHADOW_COL = (0.04, 0.025, 0.025)
SHADOW_Z = 1.6

# Web ellipsoid (D4)
WEB_RADII = (1.20, 2.40, 1.10)
WEB_POS_X = -3.90
WEB_POS_Z = 0.70

# Chase (C4)
CHASE_TAU = 0.03
CHASE_ERR = 0.1

# IK
BEND_MAX = 1.75
COUPLING = 0.4

# Root spring
TAU = 0.065
YAW_GAIN = 0.006
YAW_MAX = 0.05
PITCH = math.radians(-5.0)
ROLL = math.radians(5.0)

# Key contact
BLACK_SHIFT = 1.15
HOVER = 0.55
KEY_SINK = 0.10

# Finger tube resolution
FING_M = 14
TIP_U = (0.0, 0.30, 0.55, 0.75, 0.89, 0.965)

# SDF grid
GRID_STEP = 0.15
GRID_Y0 = -24.0
GRID_X = (-9.0, 9.0)
GRID_Y = (-20.0, 22.0)
GRID_Z = (-3.0, 4.0)

# Skinning weights
K_INTRA = 0.05
K_PALM = 0.47
K_CMC = 0.40
K_BODY = 0.30

# Texture
TEX_NAME = "handtex_v2.png"
TRIS_MAX = 6000

# Cache versioning (R1.4): hash of geometry parameters
def _geometry_hash() -> str:
    """Compute an 8-char hash of all geometry parameters that affect the mesh."""
    import json
    data = {
        "SPECS": {k: {kk: vv for kk, vv in v.items() if kk != "pole"} for k, v in SPECS.items()},
        "ARM": ARM,
        "THENAR": THENAR,
        "HYPO": HYPO,
        "THUMB_CMC_LEN": THUMB_CMC_LEN,
        "THUMB_MCP_LEN": THUMB_MCP_LEN,
        "THUMB_IP_LEN": THUMB_IP_LEN,
        "THUMB_ABDUCTION_REST": THUMB_ABDUCTION_REST,
        "PALM_PROFILE": PALM_PROFILE,
        "K_WEB": K_WEB,
        "WEB_Y_EXTENT": WEB_Y_EXTENT,
        "GRID_STEP": GRID_STEP,
        "GRID_Y0": GRID_Y0,
        "GRID_X": GRID_X,
        "GRID_Y": GRID_Y,
        "GRID_Z": GRID_Z,
    }
    s = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode()).hexdigest()[:8]

HANDMESH_VERSION = "v1"
GEOMETRY_HASH = _geometry_hash()


# --------------------------------------------------------------------------- #
# Anatomical proportion targets (for test_hand_anatomy.py)
# All ratios are dimensionless; lengths in cm at hf=1.0
# --------------------------------------------------------------------------- #
ANATOMICAL_TARGETS = {
    "palm_width_to_hand_length": (0.42, 0.50),
    "wrist_width_to_palm_width": (0.68, 0.80),
    "thumb_to_index_ratio": (0.65, 0.80),
    "index_to_middle_ratio": (0.90, 0.95),
    "ring_to_middle_ratio": (0.92, 0.97),
    "pinky_to_middle_ratio": (0.72, 0.80),
    "pip_diameter_to_mcp_spacing": (0.85, 1.00),
    "gap_at_50_proximal": (0.0, 0.15),
    "dip_pip_ratio": (0.60, 0.75),
    "thumb_abduction_angle_deg": (20.0, 40.0),
}

FINGER_NAMES = {1: "thumb", 2: "index", 3: "middle", 4: "ring", 5: "little"}
FINGERS = (1, 2, 3, 4, 5)


# --------------------------------------------------------------------------- #
# Helper: get spec for a finger at a given hand factor
# --------------------------------------------------------------------------- #
def get_spec(finger: int, hf: float = 1.0) -> dict:
    """Return scaled spec for a finger at hand factor hf."""
    s = SPECS[finger]
    return {
        "base": (s["base"][0] * hf, s["base"][1] * hf, s["base"][2] * hf),
        "lens": tuple(l * hf for l in s["lens"]),
        "widths": tuple(w * hf for w in s["widths"]),
        "thicknesses": tuple(t * hf for t in s["thicknesses"]),
        "radii": tuple(r * hf for r in s["radii"]),
        "pole": s["pole"],
    }


def get_arm_profile(hf: float = 1.0):
    """Return ARM stations scaled by hf (stations before ARM_PIN keep y)."""
    out = []
    for i, (y, cx, hw, ht, p) in enumerate(ARM):
        if i < ARM_PIN:
            out.append((y, cx * hf, hw * hf, ht * hf, p))
        else:
            out.append((y * hf, cx * hf, hw * hf, ht * hf, p))
    return out


def thumb_by_for(hf: float) -> float:
    """Thumb MCP y position (world cm) for a given hand factor."""
    return max(THUMB_BASE_Y0, THUMB_BASE_Y0 + THUMB_FWD * (hf - HF_XXS))


def web_local(hf: float, thumb_by: float) -> tuple:
    """Thenar-web centre in palm frame (local coords)."""
    return (WEB_POS_X * hf, 0.5 * (thumb_by + 8.2 * hf), WEB_POS_Z * hf)


def root_position(hf: float) -> tuple:
    """Wrist root position (x, y, z) for a given hand factor."""
    finger_len = sum(SPECS[3]["lens"]) * hf
    root_y = ROOT_CONTACT_Y - ROOT_WRIST_GAP * hf - ROOT_REACH_FRAC * finger_len
    root_z = ROOT_Z_BASE * (ROOT_Z_C0 + ROOT_Z_HF_K * hf)
    return (0.0, root_y, root_z)