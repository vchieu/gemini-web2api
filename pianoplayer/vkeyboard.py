#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Name:         VirtualKeyboard
# Purpose:      Animated 3D piano player: procedural hands + piano synthesiser
# URL:          https://github.com/marcomusy/pianoplayer
# Author:       Marco Musy
# -----------------------------------------------------------------------------
"""
`VirtualKeyboard` renders the keyboard and plays a score with animated,
anatomically posed hands while a built-in additive piano synth renders the
sound (no external sample files are needed).

The scene is driven by a vedo timer: every frame the play head advances,
pending note events are fired (key dip, finger press, scheduled sound) and
both hands are re-posed with their analytic IK rigs.

Typical use::

    vk = VirtualKeyboard(songname="bach.xml")
    vk.build_RH(rh)          # a pianoplayer.hand.Hand with .generate() run
    vk.build_LH(lh)
    vk.speedfactor = 1.5
    vk.play()                # blocks until the window is closed
"""
from __future__ import annotations

import math
import os
import time as _time

try:
    from vedo import Box, Plotter, Text2D, Text3D, settings as _vsettings
except Exception:  # pragma: no cover - vedo is mandatory at runtime
    pass

try:
    from vtkmodules.vtkCommonMath import vtkMatrix4x4
    from vtkmodules.vtkRenderingCore import vtkLight
except Exception:  # pragma: no cover
    vtkMatrix4x4 = None
    vtkLight = None

from pianoplayer import __version__
from pianoplayer.audio import SoundEngine, midi_from_name, midi_of
from pianoplayer.hand3d import FINGERS, HandRig
from pianoplayer.utils import (
    BLACK_KEY_PRESS_ANGLE,
    KEYBSIZE,
    KEY_PIVOT_Y,
    KEY_PRESS_ANGLE,
    KEY_WB,
    key_is_black,
    nameof,
)

__all__ = ["VirtualKeyboard"]


# --------------------------------------------------------------------------- #
# tunables
# --------------------------------------------------------------------------- #
NR_OCTAVES = 7
KEY_TOL = 0.12
FPS_MS = 16               # animation period [ms]
LEAD_IN = 0.8             # s before the first note: hands settle in place
AIM_LEAD = 0.25           # s ahead at which the hand aims at the next note
AUDIO_LOOKAHEAD = 1.0      # s of audio events handed to the synth in advance.
                          # Must cover device period + mixer (0.08), the primed
                          # device buffer (4 x 60 ms), the ring cushion (0.40)
                          # and a stalled frame (0.15) = 0.87; any less and the
                          # note reaches the synth after it has already
                          # rendered past it and gets clamped to "now".
BLACK_AIM = 0.45          # s before a black-key note the hand slides forward
TAU_PRESS = 0.012         # key/finger easing while pressing [s]
TAU_RELEASE = 0.032       # ... while releasing [s]
HUD_PERIOD = 0.25         # s between HUD refreshes
SUSTAIN_LEGATO = 0.15    # s a note rings longer when the next one overlaps
                         # it (simple sustain pedal, C8)

KEY_WHITE = (0.94, 0.94, 0.91)
KEY_BLACK = (0.055, 0.055, 0.065)
ACCENT_R = (1.00, 0.45, 0.18)    # flash of a key played by the right hand
ACCENT_L = (0.72, 0.40, 1.00)    # ... left hand
WOOD = (0.46, 0.29, 0.17)
WOOD_DARK = (0.33, 0.20, 0.12)
DESK = (0.88, 0.85, 0.77)
BAR_COLOR = (1.00, 0.74, 0.20)
HUD_COLOR = (0.92, 0.92, 0.95)

# ---- Synthesia-style flat layout ------------------------------------------
# The keyboard is shown top-down (orthographic, world +y = screen up) and is
# pinned to the bottom of the window; the whole region above it is left empty
# so that falling notes can be drawn there later.
BG_COLOR = (0.05, 0.05, 0.07)     # near-black backdrop for the notes area
FALL_LINE = (0.25, 0.62, 1.00)    # bright divider glowing above the keys
KB_MIDI_LO = 21                   # A0, the lowest key of a real 88-key piano
KB_MIDI_HI = 108                  # C8, the highest key
KB_MARGIN_X = 2.0                 # world cm of empty space left/right of keys
KB_BOTTOM_Y = -18.0               # world y at the bottom edge of the window
                                  # (a little behind the wrist, so the forearms
                                  # enter from the bottom of the screen)
KB_SCREEN_FRAC = 0.36             # fraction of the window height given to the
                                  # keyboard + hands band (rest = notes area)
KB_ASPECT = 1.75                  # window width/height; bigger = more room
                                  # above the keys for the falling notes
MAX_W = 1800                      # never build a window wider than this
_NOTE_STEP = {"C": 0.5, "C#": 1.0, "D": 1.5, "D#": 2.0, "E": 2.5, "F": 3.5,
              "F#": 4.0, "G": 4.5, "G#": 5.0, "A": 5.5, "A#": 6.0, "B": 6.5}
_LETTER = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#",
           7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}


def _key_x(midi):
    """Keyboard x [cm] of a MIDI note, matching `utils.keypos`."""
    octave = midi // 12 - 1
    letter = _LETTER[midi % 12]
    return octave * KEYBSIZE + _NOTE_STEP[letter] * KEY_WB


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _fmt_time(t):
    t = max(0.0, float(t))
    return "%d:%02d" % (int(t // 60), int(t % 60))


def _lerp3(a, b, t):
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def _wood_texture(w=512, h=160, base=WOOD):
    """Procedural plank texture, no image files required."""
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return None
    rng = np.random.default_rng(11)
    X, Y = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    n = rng.normal(0, 1, (h, w))
    k = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    k /= k.sum()
    for _ in range(3):
        n = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, n)
    rings = np.sin(X * 0.085 + 5.0 * np.sin(Y * 0.045 + 0.02 * n) + 0.30 * n)
    streak = np.sin(Y * 0.6 + 0.8 * n)
    shade = 0.86 + 0.12 * rings + 0.05 * streak
    img = np.empty((h, w, 3), dtype=float)
    for i in range(3):
        img[..., i] = base[i] * shade
    return (np.clip(img, 0.0, 1.0) * 255.0).astype("uint8")


def _write_key_matrix(m4, amount, black):
    """Matrix hinging a key around its back edge (line y = KEY_PIVOT_Y)."""
    ang = math.radians((BLACK_KEY_PRESS_ANGLE if black else KEY_PRESS_ANGLE)
                       * amount)
    c, s = math.cos(ang), math.sin(ang)
    py = KEY_PIVOT_Y
    pz = 1.0 if black else 0.0
    rows = ((1.0, 0.0, 0.0, 0.0),
            (0.0, c, -s, py - c * py + s * pz),
            (0.0, s, c, pz - s * py - c * pz),
            (0.0, 0.0, 0.0, 1.0))
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            m4.SetElement(i, j, float(v))
    m4.Modified()


# --------------------------------------------------------------------------- #
# per note / per hand bookkeeping
# --------------------------------------------------------------------------- #
class _Ev:
    __slots__ = ("t_on", "t_off", "midi", "vel", "finger", "black",
                 "measure", "name", "fx", "key", "fired", "wx", "_sent_when")

    def __init__(self, t_on, t_off, midi, vel, finger, black, measure,
                 name, fx, wx=None):
        self.t_on = t_on
        self.t_off = t_off
        self.midi = midi
        self.vel = vel
        self.finger = finger        # 1..5, 0 = unknown/unused
        self.black = black
        self.measure = measure
        self.name = name            # e.g. "C#3"
        self.fx = fx                # {finger: world x} fan, or None
        self.wx = wx                # world x of the key this note presses
        self.key = None             # 3D key mesh name, filled on press
        self.fired = False
        self._sent_when = None      # engine time handed to note_on (F10 fix)


class _Player:
    def __init__(self, side, sign, pan, hand, rig):
        self.side = side            # "R" / "L"
        self.sign = sign            # +1 right, -1 left
        self.pan = pan              # stereo pan of the sound engine
        self.hand = hand
        self.rig = rig
        self.notes = []
        self.fire = 0               # next note to be pressed
        self.aim = 0                # note the hand is heading towards
        self.sched = 0              # next note to be handed to the synth
        self.held = []              # fired notes still sounding
        self.press = {f: 0.0 for f in FINGERS}
        self.active = {f: 0 for f in FINGERS}
        self.finger_x = {}


# --------------------------------------------------------------------------- #
# the keyboard
# --------------------------------------------------------------------------- #
class VirtualKeyboard:
    """3D keyboard with animated hands and synthesised piano sound."""

    def __init__(self, songname='', tempo=72.0, offscreen=False, shadows=False):
        self.KB = {}                # note name -> key mesh
        self.offscreen = bool(offscreen)
        # experimental: vtkShadowMapPass (hands cast shadows on the keys).
        # The shadow frustum is fitted to the scene, on wide screens some
        # surfaces fall outside it and turn black, so it is off by default.
        self.shadows = bool(shadows)
        self.vp = None
        self.rightHand = None
        self.leftHand = None
        self.RH = None              # HandRig
        self.LH = None
        self.vpRH = None            # legacy aliases
        self.vpLH = None
        self.playsounds = True
        self.verbose = True
        self.songname = songname
        self.tempo = float(tempo)   # beats per minute of the score
        self.tempo_map = None       # [(offset_quarters, bpm)], None = constant
        self.speedfactor = 1.0      # playback rate (image + sound)
        self.volume = 0.8
        self.t0 = 0                 # legacy attributes
        self.dt = 0.1
        self.engagedfingersR = [False] * 6
        self.engagedfingersL = [False] * 6
        self.engagedkeysR = []
        self.engagedkeysL = []

        self.engine = None
        self._audio_epoch = -LEAD_IN   # score time of engine time 0
        self._engine_wall = 0.0        # wall time when engine time was 0
        self._latency = 0.0            # largest output latency seen so far
        try:
            self.av_offset = float(os.environ.get("PIANOPLAYER_AV_MS", 0.0) or 0.0) / 1000.0
        except (TypeError, ValueError):
            self.av_offset = 0.0       # s; >0 means the sound is heard later
                                       # than the image (see _prepare)
        self._underrun_seen = 0        # buffer holes already reported
        self._players = {}
        self._key_black = {}
        self._key_by_midi = {}
        self._key_amt = {}
        self._key_tgt = {}
        self._key_cnt = {}
        self._key_col = {}
        self._key_acc = {}
        self._key_mats = {}
        self._touched = set()
        self._playing = False
        self._ended = False
        self._cb_id = None
        self._total = 1.0
        self._prog = -1.0
        self._hud = None
        self._hud_t = -1e9
        self._err = False
        self._camera = None
        self._span = 0.0
        self._bar = None
        self._bar_m4 = None

        self.build_keyboard()

    # ------------------------------------------------------------------ screen
    @staticmethod
    def _screen_size():
        """Screen resolution in pixels (best effort)."""
        try:
            from vtkmodules.vtkRenderingCore import vtkRenderWindow
            s = vtkRenderWindow.GetScreenSize()
            if int(s[0]) > 0 and int(s[1]) > 0:
                return int(s[0]), int(s[1])
        except Exception:
            pass
        try:
            import tkinter as tk
            root = getattr(tk, "_default_root", None)
            own = False
            if root is None:
                root = tk.Tk()
                root.withdraw()
                own = True
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            if own:
                root.destroy()
            return int(sw), int(sh)
        except Exception:
            return 1920, 1080

    def _window_size(self):
        """A full-width window; the height is derived from `KB_ASPECT`.

        The keyboard is drawn to span the whole width, so the height decides
        how much room is left above it for the falling notes (Synthesia style).
        """
        sw, sh = self._screen_size()
        w = int(min(sw - 20, MAX_W))
        h = int(round(w / KB_ASPECT))
        if h > sh - 60:                 # never taller than the screen
            h = int(sh - 60)
            w = int(round(h * KB_ASPECT))
        return max(w, 480), max(h, 260)

    @staticmethod
    def _kb_bounds():
        """(left, right, center, width) of the 88-key span in world x."""
        half = (KEY_WB - KEY_TOL) / 2.0
        left = _key_x(KB_MIDI_LO) - half
        right = _key_x(KB_MIDI_HI) + half
        return left, right, 0.5 * (left + right), right - left

    def _lock_camera(self):
        """Freeze the view: dragging the mouse must not move the camera.

        The interactor keeps its own KeyPress observers (so 'q'/Esc still quit
        and the animation timer still runs); only the trackball style that
        rotates/pans/zooms is swapped out.
        """
        if self.vp is None or getattr(self.vp, "interactor", None) is None:
            return
        try:
            from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser
            self.vp.interactor.SetInteractorStyle(vtkInteractorStyleUser())
        except Exception:
            pass

    def _bind_view_key(self):
        """D7: bind `v` to flip ortho top view <-> ~55° perspective view."""
        try:
            iren = getattr(self.vp, "interactor", None)
            if iren is None:
                return                      # headless: no keys to press
            iren.AddObserver("KeyPressEvent", self._on_keypress)
        except Exception:
            pass

    def _on_keypress(self, obj, _evt=None):
        """D7: `v` flips the camera; every other key keeps its behaviour."""
        try:
            sym = obj.GetKeySym()
        except Exception:
            return
        if sym in ("v", "V"):
            self.toggle_view()

    def toggle_view(self):
        """D7: flip between orthographic top-down and ~55° perspective."""
        self._view_persp = not getattr(self, "_view_persp", False)
        cam = self._camera_side if self._view_persp else self._camera_top
        try:
            self.vp.camera = dict(cam)
            if self._view_persp:
                self.vp.camera.SetParallelProjection(0)
            else:
                self.vp.camera.SetParallelProjection(1)
                self.vp.camera.SetParallelScale(cam.get("parallel_scale"))
        except Exception:
            pass
        # H3: blob shadows only make sense in the top-down ortho view.
        try:
            for rig in (getattr(self, "RH", None), getattr(self, "LH", None)):
                if rig is not None:
                    rig.set_shadows_visible(not self._view_persp)
        except Exception:
            pass
        return self._view_persp

    # ------------------------------------------------------------------ scene
    def build_keyboard(self):
        """Create the plotter and the 88 keys (non blocking).

        The view is a locked, orthographic top-down look: world +y points up
        the screen (notes area), world +x runs along the keyboard, the keys are
        pinned to the bottom edge and the whole area above them is left empty
        for the falling notes.
        """
        left, right, cx, kb_w = self._kb_bounds()
        self._span = kb_w
        self._cx = cx
        self._kb_left = left

        w, h = self._window_size()
        aspect = float(w) / float(max(h, 1))

        # Orthographic top-down: `parallel_scale` is half of the world height
        # on screen. Derive it from the width so the 88 keys span the window.
        pscale = (kb_w / 2.0 + KB_MARGIN_X) / aspect
        cy = KB_BOTTOM_Y + pscale         # pin the keys to the bottom edge
        cam_z = 400.0

        # 8x MSAA on a full-width window costs ~12 ms per frame, which is the
        # difference between ~6 fps and ~30 fps -- and a starved frame is
        # heard as an audio stutter.  The scene is axis-aligned boxes, so it
        # stays clean without it.
        try:
            _vsettings.multi_samples = 0
        except Exception:
            pass
        self.vp = Plotter(title='PianoPlayer ' + __version__, axes=0,
                          size=(w, h), bg=BG_COLOR, offscreen=self.offscreen)

        # H4: cheap anti-aliasing (multi_samples stays 0: 8x MSAA costs
        # ~12 ms/frame).  Guarded: old VTK has no FXAA entry point.
        try:
            self.vp.renderer.UseFXAAOn()
            self._fxaa = True
        except Exception:
            self._fxaa = False

        # dark slab under the keys: black gaps between the white keys, and the
        # strip below the keys (player side) stays black like the reference
        back_y = 6.0                       # a little behind the key backs
        base = Box(pos=(cx, 0.5 * (KB_BOTTOM_Y + back_y), -1.0),
                   length=kb_w + 2 * KB_MARGIN_X, height=1.0,
                   width=back_y - KB_BOTTOM_Y, c=(0.03, 0.03, 0.04))
        self.vp += base

        # the bright line the falling notes will land on --------------------
        fall = Box(pos=(cx, 4.6, 0.55), length=kb_w + 1.0,
                   height=0.5, width=0.5, c=FALL_LINE)
        try:
            fall.actor.GetProperty().LightingOff()
        except Exception:
            pass
        self.vp += fall

        # the 88 keys, A0 .. C8 --------------------------------------------
        black_pc = (1, 3, 6, 8, 10)       # C# D# F# G# A#
        for midi in range(KB_MIDI_LO, KB_MIDI_HI + 1):
            pc = midi % 12
            name = _LETTER[pc] + str(midi // 12 - 1)
            x = _key_x(midi)
            if pc in black_pc:
                k = Box(pos=(x, 0, 1), length=KEY_WB * 0.6,
                        height=1, width=8, c=KEY_BLACK)
                self._key_black[name] = True
            else:
                k = Box(pos=(x, -2, 0), length=KEY_WB - KEY_TOL,
                        height=1, width=12, c=KEY_WHITE)
                self._key_black[name] = False
            self.KB[name] = k
            self._key_by_midi[midi] = name
            self.vp += k

        # progress bar: a thin strip filling up along the fall line ---------
        if vtkMatrix4x4 is not None:
            self._bar = Box(pos=(0, 0, 0), length=1, height=0.10, width=0.8,
                            c=BAR_COLOR)
            m4 = vtkMatrix4x4()
            m4.Identity()
            self._bar.actor.SetUserMatrix(m4)
            self._bar_m4 = m4
            self.vp += self._bar

        # Camera: straight down at 90 deg, orthographic (a flat plane, no
        # perspective trapezoid), keyboard pinned to the bottom of the frame.
        self._camera = dict(pos=(cx, cy, cam_z), focalPoint=(cx, cy, 0.0),
                            viewup=(0.0, 1.0, 0.0),
                            clipping_range=(1.0, 2.0 * cam_z),
                            parallel_scale=pscale)
        self.vp.camera = self._camera
        try:
            self.vp.camera.SetParallelProjection(1)
            self.vp.camera.SetParallelScale(pscale)
        except Exception:
            pass
        self._add_lights()
        if self.shadows:
            self._enable_shadows()
        self._lock_camera()
        # D7: keep the orthographic top view and a ~55° perspective view;
        # the `v` key flips between them (see _bind_view_key).
        self._view_persp = False
        self._camera_top = dict(self._camera)
        _d = 90.0
        self._camera_side = dict(
            pos=(cx, -2.0 - _d * math.cos(math.radians(55.0)),
                 1.0 + _d * math.sin(math.radians(55.0))),
            focalPoint=(cx, -2.0, 1.0),
            viewup=(0.0, 1.0, 0.0))
        self._bind_view_key()
        self.vp.render()
        return self.vp

    def _enable_shadows(self):
        """Let the hands and the desk cast shadows on the keys (best effort)."""
        try:
            from vtkmodules.vtkRenderingOpenGL2 import (
                vtkCameraPass,
                vtkRenderPassCollection,
                vtkSequencePass,
                vtkShadowMapBakerPass,
                vtkShadowMapPass,
            )
            shadows = vtkShadowMapPass()
            baker = shadows.GetShadowMapBakerPass()
            if baker is None:
                baker = vtkShadowMapBakerPass()
                shadows.SetShadowMapBakerPass(baker)
            passes = vtkRenderPassCollection()
            passes.AddItem(baker)
            passes.AddItem(shadows)
            seq = vtkSequencePass()
            seq.SetPasses(passes)
            cam = vtkCameraPass()
            cam.SetDelegatePass(seq)
            self.vp.renderer.SetPass(cam)
        except Exception as exc:
            self.shadows = False
            if self.verbose:
                print("shadows unavailable:", exc)

    def _register_midi(self, keyname, pitchname, octave):
        try:
            self._key_by_midi[int(midi_from_name(pitchname, octave))] = keyname
        except Exception:
            pass

    def _add_lights(self):
        if vtkLight is None or self.vp is None or self.vp.renderer is None:
            return
        cx = getattr(self, "_cx", self._span / 2 + KEYBSIZE)
        # top-down view: keep the lights high so the key tops and the backs of
        # the hands are lit, with a slight offset for readable shading.
        # The total is kept just above 1: the skin tone is already bright and
        # anything more blows the hands out to white.
        specs = (
            ((cx - 30, -55, 95), (cx, -4, 0), 0.62, (1.00, 0.97, 0.92)),
            ((cx + 45, -25, 80), (cx, 0, 0), 0.30, (0.86, 0.92, 1.00)),
            ((cx, 55, 75), (cx, 4, 0), 0.22, (1.00, 1.00, 1.00)),
        )
        # with the shadow map pass only one light can be used: every light
        # must own a shadow map, otherwise its contribution is dropped.
        if self.shadows:
            specs = specs[:1]
        try:
            for pos, foc, inten, col in specs:
                lt = vtkLight()
                lt.SetLightTypeToSceneLight()
                lt.SetPosition(*[float(v) for v in pos])
                lt.SetFocalPoint(*[float(v) for v in foc])
                lt.SetIntensity(float(inten))
                lt.SetColor(*[float(v) for v in col])
                # the shadow map is built inside the light cone: widen it to
                # cover the whole keyboard (default is a narrow 30 deg)
                if self.shadows:
                    lt.SetConeAngle(95.0)
                self.vp.renderer.AddLight(lt)
            self.vp.renderer.LightFollowCameraOff()
        except Exception:
            pass

    # ------------------------------------------------------------------ hands
    def build_RH(self, hand):
        """Attach a `pianoplayer.hand.Hand` (side='right') to the scene."""
        self.rightHand = hand
        self.RH = HandRig(side="right", size=getattr(hand, "size", "M"))
        self.RH.add_to(self.vp)
        self.vpRH = self.RH
        return self

    def build_LH(self, hand):
        """Attach a `pianoplayer.hand.Hand` (side='left') to the scene."""
        self.leftHand = hand
        self.LH = HandRig(side="left", size=getattr(hand, "size", "M"))
        self.LH.add_to(self.vp)
        self.vpLH = self.LH
        return self

    # ------------------------------------------------------------------ score
    def _beat_seconds(self, beat):
        """Seconds (already divided by speedfactor) at quarter-length `beat`,
        integrating the tempo map piece by piece."""
        tm = self.tempo_map or [(0.0, float(self.tempo))]
        t = 0.0
        for k, (off, bpm) in enumerate(tm):
            if beat <= off:
                break
            end = tm[k + 1][0] if k + 1 < len(tm) else float("inf")
            t += (min(beat, end) - off) * 60.0 / max(bpm, 1e-3)
        return t / max(float(self.speedfactor), 1e-6)

    def _events_for(self, hand, sign, pan):
        """Turn a Hand noteseq into a time ordered list of _Ev."""
        sc = 60.0 / (max(float(self.tempo), 1e-3)
                     * max(float(self.speedfactor), 1e-6))
        nseq = list(getattr(hand, "noteseq", []) or [])
        fseq = list(getattr(hand, "fingerseq", []) or [])
        # map noteseq index -> fan row (generate() may have processed only a
        # subset of the notes, e.g. when a measure range was requested)
        fan_of = {}
        # fingerseq_index stays None until generate() runs (and is an
        # empty list when the -n / -s range held no note): only a
        # never-generated hand falls back to playing every note.
        idxs = getattr(hand, "fingerseq_index", None)
        played = set(idxs) if idxs is not None else None
        if idxs and fseq and len(idxs) == len(fseq):
            fan_of = {i2: row for i2, row in zip(idxs, fseq)}
        elif len(fseq) == len(nseq) and fseq:
            fan_of = dict(enumerate(fseq))
        weights = getattr(hand, "weights", [None, 1.1, 1.0, 1.1, 0.9, 0.8])
        frest = getattr(hand, "frest", [None, -7.0, -2.8, 0.0, 2.8, 5.6])

        evs = []
        for i, n in enumerate(nseq):
            if played is not None and i not in played:
                continue                       # outside -n / -s range
            try:
                wx = sign * float(n.x)
            except Exception:
                continue
            try:
                fing = int(n.fingering)
            except Exception:
                fing = 0
            if fing < 1 or fing > 5:
                fing = 0

            name = nameof(n)
            midi = None
            try:
                midi = int(midi_of(n))
            except Exception:
                try:
                    midi = int(midi_from_name(n.name, n.octave))
                except Exception:
                    midi = None

            try:
                q_on = float(getattr(n, "time_real", n.time))
                t_on = self._beat_seconds(q_on)
            except Exception:
                q_on, t_on = 0.0, 0.0
            try:
                q_len = float(getattr(n, "dur_real", n.duration))
                dur = self._beat_seconds(q_on + q_len) - t_on
            except Exception:
                dur = 0.0
            if dur <= 1e-4:
                dur = 0.25 * sc

            w = 1.0
            if fing:
                try:
                    w = float(weights[fing])
                except Exception:
                    w = 1.0

            # musical dynamics (C8): honour the score's own velocity when the
            # score carries one (music21 volume.velocity 0..127, as set by
            # Dynamic marks / MIDI import); otherwise the melody is louder
            # than the accompaniment, down beats are accented, the thumb is
            # stronger than the little finger, plus deterministic humanisation.
            vel = None
            try:
                vol = getattr(getattr(n, "note21", None), "volume", None)
                vv = getattr(vol, "velocity", None)
                if vv is None:
                    vs = getattr(vol, "velocityScalar", None)
                    if vs is not None:
                        vel = float(vs)
                else:
                    vel = float(vv) / 127.0
            except (TypeError, ValueError, AttributeError):
                vel = None
            if vel is None:
                try:
                    beatpos = float(getattr(n, "time_real", n.time))
                except Exception:
                    beatpos = 0.0
                onbeat = 1.0 - min(abs(beatpos - round(beatpos)) * 3.0, 1.0)
                base = 0.58 if sign > 0 else 0.50
                human = (((i * 2654435761) % 97) / 97.0) - 0.5
                vel = base + 0.18 * onbeat + 0.10 * (w - 1.0) + 0.06 * human
            vel = min(1.0, max(0.20, vel))

            fx = None
            row = fan_of.get(i)
            if row is not None:
                try:
                    fx = {j: sign * float(row[j]) for j in (1, 2, 3, 4, 5)}
                except Exception:
                    fx = None
            if fx is None and fing:
                fx = {j: wx + sign * (float(frest[j]) - float(frest[fing]))
                      for j in (1, 2, 3, 4, 5)}

            evs.append(_Ev(t_on, t_on + dur, midi, vel, fing,
                           key_is_black(wx), int(getattr(n, "measure", 0) or 0),
                           name, fx, wx))

        evs.sort(key=lambda e: e.t_on)

        # simple sustain pedal: a note the next one overlaps (legato
        # connection) rings a little longer.  Only a strict overlap counts:
        # chord mates (same onset) and exactly abutting notes keep their
        # tested durations.
        for a, b in zip(evs, evs[1:]):
            if a.t_on + 1e-9 < b.t_on < a.t_off - 1e-9:
                a.t_off = a.t_off + SUSTAIN_LEGATO

        # never let one pitch overlap itself (one synth voice per midi)
        nxt = {}
        for e in reversed(evs):
            if e.midi is not None:
                coming = nxt.get(e.midi)
                if coming is not None and coming < e.t_off:
                    e.t_off = coming
                nxt[e.midi] = e.t_on
        for e in evs:
            e.t_off = max(e.t_off, e.t_on + 0.06)
        return evs

    # -------------------------------------------------------------------- play
    def play(self):
        """
        Animate the score with sound.

        Blocks until the rendering window is closed (or the piece is over and
        the window is closed by the user).
        """
        if self.vp is None:
            self.build_keyboard()
        try:
            # Map the window and pay the cold-start cost (shader compile,
            # geometry upload) *before* the synth clock starts: otherwise a
            # half-second stall drains the audio ring before note one (F7).
            self.vp.initialize_interactor()
            self.vp.window.Render()
            for _ in range(3):
                self.vp.render()
        except Exception:
            pass
        self._prepare()

        if self._cb_id is not None:
            try:
                self.vp.remove_callback(self._cb_id)
            except Exception:
                pass
            self._cb_id = None

        self._playing = True
        try:
            self._lock_camera()          # dragging must never move the view
            self._run_frames()
        finally:
            self._playing = False
            self._teardown()
        return self

    def _run_frames(self):
        """
        Drive the animation ourselves instead of `interactor.Start()`.

        vedo's interactive loop parks a blocking message wait between the VTK
        timer callback and the next render.  Measured on this scene that costs
        ~30 ms per frame -- 11 fps instead of 30 -- and because the frames come
        out that slow the audio ring drains and playback stutters.  Polling the
        events here costs nothing and still runs every observer vedo attached
        to the interactor, so 'q'/Esc and the window's close button behave as
        before (vedo turns them into `BreakInteraction`/`TerminateApp`).
        """
        vp = self.vp
        # The window was already mapped in play(), before the synth clock
        # started; nothing to initialise here.
        period = FPS_MS / 1000.0
        nxt = _time.perf_counter()
        frame = 0
        while self._playing:
            if vp.interactor is None:    # headless: there is nothing to pump
                break
            try:
                vp.interactor.ProcessEvents()
                if vp.interactor.GetDone() or vp._must_close_now:
                    break
                # the window is gone (close button): stop before Render() on a
                # dead interactor turns into a traceback in _on_timer
                if frame >= 5 and not vp.window.GetMapped():
                    break
            except Exception:
                break
            self._on_timer()
            frame += 1
            nxt += period
            slack = nxt - _time.perf_counter()
            if slack > 0:
                _time.sleep(slack)
            else:
                nxt = _time.perf_counter()

    def _prepare(self):
        self._players = {}
        for side, sign, pan, attr, rigname in (
                ("R", 1.0, +0.30, "rightHand", "RH"),
                ("L", -1.0, -0.30, "leftHand", "LH")):
            hand = getattr(self, attr, None)
            rig = getattr(self, rigname, None)
            if hand is None or rig is None:
                continue
            p = _Player(side, sign, pan, hand, rig)
            p.notes = self._events_for(hand, sign, pan)
            if not p.notes:
                continue
            self._players[side] = p
            if p.notes[0].fx:
                rig.settle(p.notes[0].fx, {})

        self._stop_audio()
        engine_wall = _time.perf_counter()
        if self.playsounds:
            try:
                self.engine = SoundEngine(volume=self.volume, enabled=True)
                self.engine.start()
                # right after start(): the device is already pulling, so this
                # is when sample 0 of the synth clock reaches the speakers
                # (start() may spend a while pre-rendering the audio cushion)
                engine_wall = _time.perf_counter()
                if self.verbose and self.engine.backend_name != "none":
                    print("Audio backend:", self.engine.backend_name)
            except Exception as exc:
                print("Audio unavailable:", exc)
                self.engine = None

        for name in self._key_tgt:
            self._key_tgt[name] = 0.0
            self._key_amt[name] = 0.0
        self._key_cnt = {}
        self._touched = set()

        tot = 1.0
        for p in self._players.values():
            tot = max(tot, max(e.t_off for e in p.notes))
        self._total = tot
        self._prog = -1.0
        self._hud_t = -1e9
        self._ended = False
        self._underrun_seen = 0
        self._late_notes = 0       # note_ons that reached the synth past due
        self._prev_now = -LEAD_IN  # previous frame's score time (F10 fix)
        self._t0 = _time.perf_counter() + LEAD_IN
        # The synth counts seconds from the engine start, the score counts
        # them from _t0, and a sample rendered at engine time t only reaches
        # the speakers output_latency seconds later -- which the backend
        # *measures* (any silence the device had to pad in while priming its
        # buffer, plus the OS mixer).  Fold both offsets in, so a note written
        # at score time T is *heard* exactly when the hands play it:
        #   when = T - _audio_epoch = T + t0 - engine_wall - latency + av_offset
        # `av_offset` (>0 = sound later than image) lets one machine be
        # calibrated: raise it when the picture runs ahead of the sound
        # (--av-offset-ms / PIANOPLAYER_AV_MS).
        latency = getattr(self.engine, "output_latency", 0.0) if self.engine else 0.0
        self._engine_wall = engine_wall
        self._latency = latency
        self._audio_epoch = (engine_wall + latency - self._t0
                             - getattr(self, "av_offset", 0.0))
        self._last = _time.perf_counter()

    def _teardown(self):
        self._stop_audio()
        if self.verbose and self._ended:
            print("End of note sequence reached.")
        if self.verbose and getattr(self, "_late_notes", 0):
            print("audio: %d note(s) reached the synth late (clamped to now)"
                  % self._late_notes)

    def _stop_audio(self):
        if self.engine is not None:
            try:
                if self.verbose and self.engine.underruns:
                    print("audio: %d buffer underrun(s)" % self.engine.underruns)
            except Exception:
                pass
            try:
                self.engine.all_notes_off()
            except Exception:
                pass
            try:
                self.engine.stop()
            except Exception:
                pass
            self.engine = None

    # ------------------------------------------------------------------ events
    def _key_for(self, ev):
        if ev.midi is not None:
            k = self._key_by_midi.get(ev.midi)
            if k is not None:
                return k
        return self.KB.get(ev.name)

    def _press(self, p, ev):
        if ev.finger:
            p.active[ev.finger] = p.active.get(ev.finger, 0) + 1
        key = self._key_for(ev)
        if key is None:
            return
        ev.key = key
        self._key_cnt[key] = self._key_cnt.get(key, 0) + 1
        self._key_tgt[key] = 1.0
        self._key_acc[key] = ACCENT_R if p.side == "R" else ACCENT_L
        self._touched.add(key)

    def _release(self, p, ev):
        if ev.finger and p.active.get(ev.finger):
            p.active[ev.finger] -= 1
        key = ev.key
        if key is None:
            return
        c = self._key_cnt.get(key, 0) - 1
        self._key_cnt[key] = c
        if c <= 0:
            self._key_tgt[key] = 0.0

    def _pump(self, p, now, engine):
        notes = p.notes

        # hand the coming notes to the synthesiser --------------------------
        if engine is not None:
            eng_now = getattr(engine, "now", None)
            while p.sched < len(notes) and notes[p.sched].t_on - now <= AUDIO_LOOKAHEAD:
                e = notes[p.sched]
                if e.midi is not None:
                    t_on = max(0.0, e.t_on - self._audio_epoch)
                    t_off = max(t_on + 0.02, e.t_off - self._audio_epoch)
                    e._sent_when = t_on
                    if eng_now is not None and t_on < eng_now - 1e-3:
                        # already past the synth clock: SynthCore clamps it to
                        # "now" and the sound lags behind the hands (C7)
                        self._late_notes = getattr(self, "_late_notes", 0) + 1
                    engine.note_on(e.midi, e.vel, when=t_on, pan=p.pan)
                    engine.note_off(e.midi, when=t_off)
                p.sched += 1
        else:
            p.sched = len(notes)

        # what the hand is aiming at ----------------------------------------
        while p.aim + 1 < len(notes) and notes[p.aim + 1].t_on - AIM_LEAD <= now:
            p.aim += 1
        target = notes[p.aim] if p.aim < len(notes) else None

        # presses due --------------------------------------------------------
        while p.fire < len(notes) and notes[p.fire].t_on <= now:
            e = notes[p.fire]
            p.fire += 1
            if not e.fired:
                e.fired = True
                p.held.append(e)
                self._press(p, e)

        # releases due -------------------------------------------------------
        i = 0
        while i < len(p.held):
            e = p.held[i]
            if e.t_off <= now:
                self._release(p, e)
                p.held.pop(i)
            else:
                i += 1

        # Finger targets: fan of the note we are heading to, but every finger
        # that is currently holding a key stays exactly on that key.
        fx = (dict(target.fx) if (target is not None and target.fx)
              else dict(p.finger_x))
        for e in p.held:                       # oldest first, newest wins
            if e.finger and e.wx is not None:
                fx[e.finger] = e.wx
        p.finger_x = fx

        # is a black key coming up soon? -------------------------------------
        advance = bool(target is not None and target.black
                       and (target.t_on - now) < BLACK_AIM)
        return advance

    # ------------------------------------------------------------------ frame
    def _retime_pending(self, engine, shift, prev_now):
        """Move already-scheduled future audio events by `shift` seconds.

        Called when the measured output latency grows mid-playback (F10):
        every sample still inside the synth is heard later than planned, so
        pending note on/off times move earlier by the same amount and stay
        glued to the hands.  Must run *before* `_pump` schedules new notes
        in the same frame, so everything still pending was timed with the
        old epoch.  Only notes that have not started yet (`t_on` past the
        previous frame) are moved: a note already sounding is committed to
        the stream and cannot be pulled back.
        """
        if not shift:
            return
        fn = getattr(engine, "shift_pending", None)
        if callable(fn):
            try:
                fn(shift)
            except Exception:
                pass
            return
        # Engines without a synth clock of their own only record the times
        # we handed them (test doubles exposing `.sent`): rewrite the stale
        # entries in place so the measured heard-vs-visual gap stays true.
        sent = getattr(engine, "sent", None)
        if not isinstance(sent, list):
            return
        for p in self._players.values():
            for e in p.notes[:p.sched]:
                old = e._sent_when
                if old is None or e.midi is None:
                    continue
                if e.t_on <= prev_now:
                    continue              # already sounding: stream committed
                want = max(0.0, e.t_on - self._audio_epoch)
                if abs(want - old) < 1e-12:
                    continue
                for i, rec in enumerate(sent):
                    if (isinstance(rec, (tuple, list)) and len(rec) == 2
                            and rec[0] == e.midi and rec[1] == old):
                        sent[i] = (rec[0], want)
                        e._sent_when = want
                        break
                    if isinstance(rec, (int, float)) and rec == old:
                        sent[i] = want
                        e._sent_when = want
                        break

    def _on_timer(self, evt=None):
        if not self._playing or self.vp is None:
            return
        try:
            t = _time.perf_counter()
            dt = t - self._last
            self._last = t
            if dt <= 0.0 or dt > 0.25:
                dt = FPS_MS / 1000.0
            now = t - self._t0
            self._frame(now, dt)
            self.vp.render()
        except Exception:
            if not self._err:
                self._err = True
                import traceback
                traceback.print_exc()

    def _frame(self, now, dt):
        engine = self.engine      # the synth clock runs even without a device
        prev_now = getattr(self, "_prev_now", now)
        self._prev_now = now
        if engine is not None:
            # Render the audio *here*, on the thread that owns the window:
            # a background thread cannot get enough of the interpreter lock
            # while VTK renders and the playback ring would drain (heard as
            # stuttering).  Costs a few ms only when the ring is low.
            engine.pump()
            # Every underrun pads the stream with silence, which delays all
            # later samples for good.  Re-measure each frame (the figure only
            # grows) so notes scheduled from now on compensate for it.
            lat = engine.output_latency
            if lat > self._latency:
                shift = -(lat - self._latency)
                self._latency = lat
                self._audio_epoch = (self._engine_wall + lat - self._t0
                                     - getattr(self, "av_offset", 0.0))
                # F10: notes handed over *before* the jump would now be heard
                # `shift` late -- move the pending ones with the stream.
                self._retime_pending(engine, shift, prev_now)
            seen = engine.underruns
            if seen != self._underrun_seen:      # report each hole the moment
                self._underrun_seen = seen       # it happens, with its position
                if self.verbose:
                    print("audio: buffer underrun #%d at %+.2f s of score"
                          % (seen, now))
        for p in self._players.values():
            advance = self._pump(p, now, engine)

            for f in FINGERS:                       # ease the fingers
                tgt = 1.0 if p.active.get(f) else 0.0
                cur = p.press[f]
                if cur != tgt:
                    tau = TAU_PRESS if tgt > cur else TAU_RELEASE
                    newv = tgt + (cur - tgt) * math.exp(-dt / max(tau, 1e-4))
                    if abs(newv - tgt) < 0.004:
                        newv = tgt
                    p.press[f] = newv

            p.rig.update(dt, p.finger_x, p.press, advance_black=advance,
                         held={f for f in FINGERS if p.active.get(f)})

        self._sync_keys(dt)
        self._hud_update(now)
        self._progress(now)

        if not self._ended and now >= self._total:
            self._ended = True
            if self.verbose:
                print("End of note sequence reached.")

    # -------------------------------------------------------------- key motion
    def _sync_keys(self, dt):
        for name in self._touched:
            tgt = self._key_tgt.get(name, 0.0)
            cur = self._key_amt.get(name, 0.0)
            if cur == tgt:
                continue
            tau = TAU_PRESS if tgt > cur else TAU_RELEASE
            a = tgt + (cur - tgt) * math.exp(-dt / max(tau, 1e-4))
            if abs(a - tgt) < 0.004:
                a = tgt
            self._key_amt[name] = a

            mesh = self.KB[name]
            black = self._key_black[name]
            m4 = self._key_mats.get(name)
            if m4 is None and vtkMatrix4x4 is not None:
                m4 = vtkMatrix4x4()
                m4.Identity()
                mesh.actor.SetUserMatrix(m4)
                self._key_mats[name] = m4
            if m4 is not None:
                _write_key_matrix(m4, a, black)
                mesh.actor.Modified()

            base = KEY_BLACK if black else KEY_WHITE
            acc = self._key_acc.get(name) or base
            col = _lerp3(base, acc, a)
            prev = self._key_col.get(name)
            if prev is None or abs(prev[0] - col[0]) + abs(prev[1] - col[1]) \
                    + abs(prev[2] - col[2]) > 0.012:
                mesh.color(col)
                self._key_col[name] = col

    # --------------------------------------------------------------------- HUD
    def _hud_update(self, now):
        if now < self._hud_t + HUD_PERIOD or self.vp is None:
            return
        self._hud_t = now

        measure = "-"
        for p in self._players.values():
            if p.aim < len(p.notes):
                measure = str(p.notes[p.aim].measure)
        txt = "  %s / %s      measure %s      %d bpm  (speed x%g)  " % (
            _fmt_time(now), _fmt_time(self._total), measure,
            int(round(self.tempo)), float(self.speedfactor))

        if self._hud is None:
            try:
                # top-left: the empty notes area, so it never covers the keys
                self._hud = Text2D(txt, pos='top-left', s=0.75,
                                   c=HUD_COLOR, bg=(0.12, 0.12, 0.15))
                self.vp.add(self._hud)
            except Exception:
                self._hud = False
        elif self._hud:
            try:
                self._hud.text(txt)
            except Exception:
                pass

        if self.verbose and self.engagedfingersR is not None:
            rp = self._players.get("R")
            if rp is not None:
                for f in FINGERS:
                    self.engagedfingersR[f - 1] = rp.active.get(f, 0) > 0
                self.engagedkeysR = [n for n in self._key_tgt
                                     if self._key_tgt[n] > 0]

    def _progress(self, now):
        if self._bar is None or self._bar_m4 is None:
            return
        prog = 0.0 if self._total <= 0 else min(1.0, max(0.0, now / self._total))
        if abs(prog - self._prog) < 0.0005:
            return
        self._prog = prog
        m4 = self._bar_m4
        length = max(prog * self._span, 1e-3)
        for i in range(4):
            for j in range(4):
                m4.SetElement(i, j, 0.0)
        m4.SetElement(0, 0, length)
        m4.SetElement(0, 3, self._kb_left + length / 2.0)
        m4.SetElement(1, 1, 1.0)
        m4.SetElement(1, 3, 4.6)       # strip sits on the fall line
        m4.SetElement(2, 2, 1.0)
        m4.SetElement(2, 3, 0.90)
        m4.SetElement(3, 3, 1.0)
        m4.Modified()
        self._bar.actor.Modified()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    vk = VirtualKeyboard(songname="pianoplayer demo")
    vk.vp.show(interactive=1, camera=vk._camera, resetcam=0)
