#
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:22:20 2015

@author: marco musy
"""


def nameof(n):
    a = n.name + str(n.octave)
    if "--" in a:                 # order matters
        b = a.replace("B--", "A")
        b = b.replace("A--", "G") # chain b.replace not a.replace
        b = b.replace("G--", "F")
        b = b.replace("E--", "D")
        b = b.replace("D--", "C")
        return b
    elif "-" in a:
        b = a.replace("C-", "B")
        b = b.replace("D-", "C#")  # chain b.replace not a.replace
        b = b.replace("E-", "D#")
        b = b.replace("F-", "E")
        b = b.replace("G-", "F#")
        b = b.replace("A-", "G#")
        b = b.replace("B-", "A#")
        return b
    elif "##" in a:
        b = a.replace("C##", "D")
        b = b.replace("D##", "E") # chain b.replace not a.replace
        b = b.replace("F##", "G")
        b = b.replace("G##", "A")
        b = b.replace("A##", "B")
        return b
    elif "E#" in a:
        b = a.replace("E#", "F")
        return b
    elif "B#" in a:
        b = a.replace("B#", "C")
        return b
    else:
        return a


def fpress(f, color):
    f.rotate(-20, axis=(1, 0, 0), point=f.pos())
    f.addPos([0, 0, -1])
    f.color(color)


def frelease(f):
    f.addPos([0, 0, 1])
    f.rotate(20, axis=(1, 0, 0), point=f.pos())
    f.color((0.7, 0.3, 0.3))


def kpress(f, color):
    f.rotate(4, axis=(1, 0, 0), point=f.pos())
    f.addPos([0, 0, -0.4])
    f.color(color)


def krelease(f):
    f.addPos([0, 0, 0.4])
    p = f.pos()
    f.rotate(-4, axis=(1, 0, 0), point=p)
    if p[2] > 0.5:
        f.color("k")
    else:
        f.color("w")


_PC_STEP = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.5,
            6: 4.0, 7: 4.5, 8: 5.0, 9: 5.5, 10: 6.0, 11: 6.5}


def keypos(n):
    """Keyboard x [cm] of a music21 Note or Pitch.

    Computed from the MIDI number, so enharmonic spellings (B#3 = C4,
    Cb4 = B3, E#, Fb, C##, B##, Dbb ...) land on the right physical key
    and always agree with the sound and with vkeyboard._key_x().
    """
    p = getattr(n, "pitch", n)
    midi = int(round(p.ps))
    return 16.5 * (midi // 12 - 1) + _PC_STEP[midi % 12] * (16.5 / 7.0)


def handSizeFactor(s):
    f = 0.82
    if s == "XXS":
        f = 0.33
    elif s == "XS":
        f = 0.46
    elif s == "S":
        f = 0.64
    elif s == "M":
        f = 0.82
    elif s == "L":
        f = 1.0
    elif s == "XL":
        f = 1.1
    elif s == "XXL":
        f = 1.2
    return f


#-------------------------------------------------------------------------------
# keyboard geometry shared by the 3D keyboard and by the hand rig
#-------------------------------------------------------------------------------
KEYBSIZE = 16.5          # cm, span of one octave (7 white keys)
KEY_WB = KEYBSIZE / 7.0  # cm, width of a white key
WHITE_TOP = 0.5          # z of the top surface of a white key
BLACK_TOP = 1.5          # z of the top surface of a black key
KEY_PIVOT_Y = 4.0        # y of the back edge: keys hinge around it
KEY_PRESS_ANGLE = 5.0    # deg, white key dip when fully pressed
BLACK_KEY_PRESS_ANGLE = 7.0
KEY_PRESS_DEPTH = 0.05   # sink of the fingertip pad into the key [cm]

# contact point of a fingertip on the key surface (absolute keyboard coords)
CONTACT_Y_WHITE = {"thumb": -5.2, "index": -3.0, "middle": -3.0,
                   "ring": -3.0, "little": -3.4}
CONTACT_Y_BLACK = {"thumb": -3.0, "index": -1.5, "middle": -1.5,
                   "ring": -1.5, "little": -1.9}


def key_is_black(x) -> bool:
    """True if the keyboard position x (cm) falls on a black key."""
    import math
    u = (x % KEYBSIZE) / KEY_WB      # 0..7 inside the octave
    m = int(math.floor(u - 0.7))
    if m < 0:
        return False
    return m in (0, 1, 3, 4, 5) and u < m + 1.3


def key_top(x) -> float:
    """z of the (unpressed) key surface at keyboard position x."""
    return BLACK_TOP if key_is_black(x) else WHITE_TOP


def key_angle(x) -> float:
    """Degrees the key tilts when fully pressed."""
    return BLACK_KEY_PRESS_ANGLE if key_is_black(x) else KEY_PRESS_ANGLE


def key_drop(x, y, amount=1.0) -> float:
    """
    How much the key surface sank at depth `y` when pressed by `amount`
    (0..1).  Keys hinge on their back edge, so the drop grows linearly
    from the pivot towards the player.
    """
    import math
    a = key_angle(x) * float(min(max(amount, 0.0), 1.0))
    return max(KEY_PIVOT_Y - y, 0.0) * math.sin(math.radians(a))


def score_tempo_map(sf, default=72.0):
    """
    [(offset in quarter lengths, quarter-note bpm), ...], sorted, first at 0.

    Uses MetronomeMark.getQuarterBPM(), which accounts for the referent
    (eighth = 120 is 60 bpm of quarters, dotted quarter = 60 is 90, ...).
    """
    found = {}
    try:
        from music21.tempo import MetronomeMark
        for mm in sf.recurse().getElementsByClass(MetronomeMark):
            bpm = mm.getQuarterBPM()
            if bpm and float(bpm) > 0:
                found[round(float(mm.getOffsetInHierarchy(sf)), 6)] = float(bpm)
    except Exception:
        pass
    if not found:
        return [(0.0, float(default))]
    tm = sorted(found.items())
    if tm[0][0] > 0:
        tm.insert(0, (0.0, tm[0][1]))
    return tm


def score_tempo(sf, default=72.0):
    """First tempo (quarter-note bpm) of a music21 score, or `default`."""
    return score_tempo_map(sf, default)[0][1]
