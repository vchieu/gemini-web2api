#-------------------------------------------------------------------------------
# Name:         PianoPlayer
# Purpose:      Real time piano sound engine (no external samples needed)
#-------------------------------------------------------------------------------
"""
Real-time piano sound engine for PianoPlayer.

The engine is a small *scheduler + mixer + synthesizer* that runs in a
background audio thread.  Notes are scheduled with `note_on()` / `note_off()`
on a common time line, so that:

 - the left and the right hand are heard **together** (no blocking call),
 - chords are truly simultaneous,
 - releasing a key actually damps the string (no endless sine beeps).

Each note is synthesized with an additive model that mimics the main
behaviour of a piano string:

 - inharmonic partials             f_n = f0 * n * sqrt(1 + B*n^2)
 - two slightly mistuned strings   on the low partials (natural beating)
 - strike position comb filter     (the hammer does not hit the string end)
 - partial dependent decay         (high partials die first)
 - hammer transient                (short filtered noise burst)
 - velocity dependent brightness, attack and level
 - damper on key release           (fast, partial dependent decay)

Backends, in order of preference: miniaudio, sounddevice, none.
"""
from __future__ import annotations

import collections
import heapq
import math
import os
import queue
import sys
import threading
import time
import wave

import numpy as np

try:
    import miniaudio
    HAS_MINIAUDIO = True
except Exception:  # pragma: no cover - optional dependency
    HAS_MINIAUDIO = False

try:
    import sounddevice as _sounddevice
    HAS_SOUNDDEVICE = True
except Exception:  # pragma: no cover - optional dependency
    HAS_SOUNDDEVICE = False


SAMPLE_RATE = 44100

_MAX_PARTIALS = 24     # hard cap of partials used by each note (C5: was 40;
                       # partials above ~9 kHz cost oscillators but no timbre)
_MIN_PARTIALS = 8      # ... but always keep at least this many
_BEAT_PARTIALS = 2     # lowest partials played by 2 mistuned strings
_DETUNE = 0.00016      # ~0.3 cent, gives the natural "chorus" of a piano
_STRIKE_POS = 0.115    # hammer position along the string (0..1)
_TAU_REL = 0.085       # nominal damper time [s]
_TAU_KILL = 0.005      # smooth-steal fade time [s]: a stolen voice dies in
                       # ~5 ms instead of being cut mid-period (C6)
_LN1000 = math.log(1000.0)
_FAST_KILL_RATE = _LN1000 / _TAU_KILL
_MAX_DYING_VOICES = 8  # stolen voices may exceed `polyphony` by this many (C6)
_PARTIAL_FLOOR = 1e-3  # a partial quieter than this leaves the mix (C5)


############################################################ note helpers
def midi_of(n) -> int:
    """Return the MIDI number of an `INote` (or of a music21 note/pitch)."""
    obj = getattr(n, "note21", n)
    if hasattr(obj, "pitch"):          # music21.note.Note
        obj = obj.pitch
    if hasattr(obj, "midi"):           # music21.pitch.Pitch
        return int(obj.midi)
    if hasattr(n, "chord21"):          # last resort: first pitch of the chord
        return int(n.chord21.pitches[0].midi)
    raise ValueError(f"cannot read pitch of {n!r}")


def midi_from_name(name: str, octave: int) -> int:
    """('C',4) -> 60, ('C#',4) -> 61."""
    semis = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
             "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
             "A#": 10, "Bb": 10, "B": 11}
    if name not in semis:
        raise ValueError(f"unknown note name {name}")
    return 12 * (octave + 1) + semis[name]


def frequency(midi: int) -> float:
    """A4 = 440 Hz."""
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


##################################################### note parameter cache
_NOTE_CACHE: dict[int, dict] = {}


def _note_params(midi: int) -> dict:
    """Velocity independent parameters of one MIDI note (cached)."""
    cached = _NOTE_CACHE.get(midi)
    if cached is not None:
        return cached

    f0 = frequency(midi)

    # string inharmonicity: higher (stiffer) strings are more inharmonic
    pos = (midi - 21) / 87.0
    B = 0.00010 + 0.00095 * pos ** 1.6

    # enough partials to reach ~9 kHz (never more than the cap, so that a
    # bass note does not cost 150 oscillators)
    nmax = int(np.clip(9000.0 / max(f0, 20.0), _MIN_PARTIALS, _MAX_PARTIALS))
    n = np.arange(1, nmax + 1, dtype=float)
    f = f0 * n * np.sqrt(1.0 + B * n * n)

    # the hammer does not strike the string at its end -> comb filter,
    # on top of the 1/n^1.5 spectral tilt of a struck string
    amp = np.abs(np.sin(np.pi * n * _STRIKE_POS)) / np.power(n, 1.5)

    # two slightly mistuned strings on the lowest partials -> natural beating
    b = _BEAT_PARTIALS
    freqs = np.concatenate([f[:b] * (1 - _DETUNE), f[:b] * (1 + _DETUNE), f[b:]])
    amps = np.concatenate([amp[:b] * 0.5, amp[:b] * 0.5, amp[b:]])
    partial_n = np.concatenate([n[:b], n[:b], n[b:]])

    # how long the note rings, and the faster death of the upper partials
    t60 = float(np.clip(8.0 * math.sqrt(440.0 / max(f0, 20.0)), 1.0, 14.0))
    t60n = np.maximum(t60 / np.power(partial_n, 0.85), 0.035)
    rates = _LN1000 / t60n

    # the damper stops the upper partials first
    rel_rate = _LN1000 / (_TAU_REL * np.power(partial_n, 0.45))

    peak = float(np.max(amps))
    if peak > 0:
        amps = amps / peak

    # float32 throughout: halves the memory traffic of the per-block sin/exp
    # matrices in _mix, at ~1e-7 relative precision (C5 step 1)
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
    params = dict(freqs=f32(freqs), amps=f32(amps), rates=f32(rates),
                  rel_rate=f32(rel_rate),
                  n_all=partial_n, f0=f0, t60=t60)
    _NOTE_CACHE[midi] = params
    return params


##################################################### one sounding note
class _Voice:
    """A single sounding note (one key press)."""
    __slots__ = ("midi", "params", "amps", "start_frame", "rel_frame",
                 "pan_l", "pan_r", "noise", "gain", "full_until",
                 "fast_kill")

    def __init__(self, midi, velocity, start_frame, pan, sr, gain=1.0):
        self.midi = midi
        self.params = p = _note_params(midi)
        v = float(min(max(velocity, 0.0), 1.0))

        # brighter when hit harder: one pole roll off above the cut frequency
        fc = 700.0 + 3500.0 * v ** 1.5
        amps = p["amps"] / np.sqrt(1.0 + (p["freqs"] / fc) ** 2)
        # normalise on the sum (not on the peak): the sum of the partials is
        # what the output level really is, so a note can never blow up
        self.amps = amps / (float(np.sum(amps)) or 1.0)

        # time until the first partial decays below _PARTIAL_FLOOR: before
        # this (and while no damper is on) the whole set is alive, so _mix
        # can skip the gate test and the index copies (C5 step 3)
        with np.errstate(divide="ignore"):
            t_drop = np.log(self.amps / _PARTIAL_FLOOR) / p["rates"]
        self.full_until = float(np.min(t_drop))

        self.start_frame = start_frame
        self.rel_frame = None
        self.fast_kill = False   # C6: dying after a smooth voice steal

        self.gain = (0.18 + 0.82 * v) ** 1.6 * gain
        c = 0.35 * pan
        self.pan_l = math.sqrt(max(0.0, 0.5 * (1.0 - c)))
        self.pan_r = math.sqrt(max(0.0, 0.5 * (1.0 + c)))

        # short, dark noise burst = hammer felt transient
        # (its envelope is baked in, so mixing it is just an addition)
        nn = max(64, int(0.030 * sr))
        rng = np.random.default_rng((midi * 977 + int(v * 500)) & 0xFFFFFFFF)
        nz = rng.standard_normal(nn)
        nz = np.convolve(nz, np.ones(10) / 10.0, mode="same")
        nz = np.convolve(nz, np.ones(10) / 10.0, mode="same")  # soft felt: ~2 kHz
        ramp = int(0.0015 * sr)
        if ramp > 1:
            nz[:ramp] *= np.linspace(0.0, 1.0, ramp)   # no click at onset
        nz *= np.exp(-np.arange(nn) / (0.010 * sr))
        thump = min(1.0 + 0.8 * (200.0 / max(p["f0"], 40.0)), 2.0)
        self.noise = np.ascontiguousarray(
            nz * (0.55 * (0.35 + 0.65 * v) * thump), dtype=np.float32)

    # ----------------------------------------------------------
    def energy_at(self, frame: int) -> float:
        """Rough instantaneous amplitude, used for voice stealing and cleanup."""
        t = max(frame - self.start_frame, 0) / 44100.0
        e = math.exp(-float(np.min(self.params["rates"])) * t)
        if self.rel_frame is not None:
            tr = (frame - self.rel_frame) / 44100.0
            if tr > 0:
                e *= math.exp(-60.0 * tr)
                if self.fast_kill:
                    e *= math.exp(-_FAST_KILL_RATE * tr)
        return e * float(np.max(self.amps)) * self.gain

    def alive_at(self, frame: int) -> bool:
        return self.energy_at(frame) > 2e-4


##################################################### synthesizer core
class SynthCore:
    """Sample accurate scheduler + additive piano synthesizer (numpy only)."""

    def __init__(self, sample_rate=SAMPLE_RATE, polyphony=32):
        self.sr = int(sample_rate)
        self.polyphony = int(polyphony)
        self.frame = 0                    # audio clock, in samples
        self.voices: list[_Voice] = []
        self._events: list[tuple] = []     # heap of (frame, seq, kind, ...)
        self._seq = 0
        self._pending: queue.SimpleQueue = queue.SimpleQueue()

    # ---------------------------------------------------------- scheduling
    def note_on(self, midi, velocity=0.6, when=None, pan=0.0):
        self._push(self._to_frame(when), 0, int(midi), float(velocity), float(pan))

    def note_off(self, midi, when=None):
        self._push(self._to_frame(when), 1, int(midi), 0.0, 0.0)

    def all_notes_off(self, when=None):
        frame = self._to_frame(when)
        for v in list(self.voices):
            # identify the voice by its start frame: unique and stable
            self._push(frame, 2, float(v.start_frame), 0.0, 0.0)

    def _to_frame(self, when):
        if when is None:
            return self.frame
        return max(int(round(when * self.sr)), self.frame)

    def _push(self, frame, kind, midi, vel, pan):
        self._seq += 1
        self._pending.put((frame, self._seq, kind, midi, vel, pan))

    def _drain(self):
        try:
            while True:
                heapq.heappush(self._events, self._pending.get_nowait())
        except queue.Empty:
            pass

    # ---------------------------------------------------------- rendering
    def render_block(self, nframes) -> np.ndarray:
        """Render `nframes` stereo samples, returns a float array of shape (n,2)."""
        nframes = int(nframes)
        out = np.zeros((nframes, 2), dtype=np.float64)
        self._drain()
        done = 0

        while done < nframes:
            # 1) apply every event that is due at the current clock position
            while self._events and self._events[0][0] <= self.frame:
                frame, _, kind, midi, vel, pan = heapq.heappop(self._events)
                if kind == 0:
                    self._spawn(midi, vel, pan, frame)
                elif kind == 1:
                    self._release(midi, frame)
                else:
                    self._release_by_start(int(midi), frame)

            # 2) render up to the next scheduled event (sample accurate)
            remaining = nframes - done
            nxt = self._events[0][0] if self._events else self.frame + remaining
            length = min(max(nxt - self.frame, 0), remaining)
            if length > 0:
                self._mix(out, done, length)
                self.frame += length
                done += length
            elif not self._events:
                break  # cannot happen, guards against a runaway loop

        return out

    def _spawn(self, midi, vel, pan, frame):
        live = [v for v in self.voices if not v.fast_kill]
        if len(live) >= self.polyphony:
            # smooth steal (C6): fade the weakest live voice over ~5 ms
            # instead of cutting it mid-period (heard as a click)
            weakest = min(live, key=lambda v: v.energy_at(self.frame))
            weakest.rel_frame = frame
            weakest.fast_kill = True
        dying = [v for v in self.voices if v.fast_kill]
        if len(dying) > _MAX_DYING_VOICES:
            dying.sort(key=lambda v: v.energy_at(self.frame))
            for v in dying[:len(dying) - _MAX_DYING_VOICES]:
                self.voices.remove(v)
        self.voices.append(_Voice(midi, vel, frame, pan, self.sr))

    def _release(self, midi, frame):
        target = None
        for v in self.voices:
            if v.midi == midi and v.rel_frame is None:
                if target is None or v.start_frame < target.start_frame:
                    target = v
        if target is not None:
            target.rel_frame = frame

    def _release_by_start(self, start_frame, frame):
        for v in self.voices:
            if v.start_frame == start_frame:
                if v.rel_frame is None or v.rel_frame > frame:
                    v.rel_frame = frame
                return

    def shift_pending(self, delta_sec):
        """Move every scheduled-but-unsounded event by `delta_sec` (F10).

        When the measured output latency grows mid-playback, every sample
        still inside the synth is heard later than planned; shifting the
        pending note on/off times by the same (negative) amount keeps them
        glued to the animation.  Returns the number of events moved.
        """
        dframes = int(round(float(delta_sec) * self.sr))
        if dframes == 0:
            return 0
        queued = []
        try:
            while True:
                queued.append(self._pending.get_nowait())
        except queue.Empty:
            pass
        for (fr, seq, kind, midi, vel, pan) in queued:
            self._pending.put((fr + dframes, seq, kind, midi, vel, pan))
        if self._events:
            self._events = [(fr + dframes, seq, kind, midi, vel, pan)
                            for (fr, seq, kind, midi, vel, pan) in self._events]
            heapq.heapify(self._events)
        return len(queued) + len(self._events)

    def _mix(self, out, offset, length):
        if length <= 0:
            return
        if not self.voices:
            out[offset:offset + length] = 0.0
            return

        sr = float(self.sr)
        start = self.frame
        # float32 time base and accumulators (C5 step 1)
        t_base = np.arange(length, dtype=np.float32) / np.float32(sr)
        left = np.zeros(length, dtype=np.float32)
        right = np.zeros(length, dtype=np.float32)
        alive = []
        two_pi = np.float32(2.0 * np.pi)

        for v in self.voices:
            p = v.params
            t0 = max((start - v.start_frame) / sr, 0.0)
            t = np.float32(t0) + t_base

            rel = -1.0
            if v.rel_frame is not None:
                rel = (start - v.rel_frame) / sr

            # --- active partials (C5 step 3): at the block start, drop every
            # --- partial whose current amplitude is below _PARTIAL_FLOOR;
            # --- it only decays further inside this block.  Young,
            # --- unreleased voices keep the whole set with no test at all.
            if rel < 0.0 and t0 < v.full_until:
                freqs = p["freqs"]
                amps = v.amps
                rates = p["rates"]
                rel_rate = p["rel_rate"]
                have = True
            else:
                gate = v.amps * np.exp(-p["rates"] * np.float32(t0))
                if rel >= 0.0:                   # damper is on
                    gate = gate * np.exp(-p["rel_rate"] * np.float32(rel))
                    if v.fast_kill:              # stolen: shed partials fast
                        gate = gate * np.exp(np.float32(-_FAST_KILL_RATE)
                                             * np.float32(rel))
                idx = np.nonzero(gate >= np.float32(_PARTIAL_FLOOR))[0]
                have = idx.size > 0
                if have:
                    freqs = p["freqs"][idx]
                    amps = v.amps[idx]
                    rates = p["rates"][idx]
                    rel_rate = p["rel_rate"][idx]

            if have:
                # --- envelope --------------------------------------------
                if t0 >= 0.004:
                    env = np.float32(1.0)        # attack long past: all ones
                else:                            # smooth 4 ms attack ...
                    env = np.minimum(t / np.float32(0.004), np.float32(1.0))
                    env = env * env * (np.float32(3.0) - np.float32(2.0) * env)
                env = env * np.exp(-rates[:, None] * t[None, :])

                if rel >= 0.0:                   # damper is on
                    env = env * np.exp(-rel_rate[:, None]
                                       * np.float32(rel))
                    if v.fast_kill:
                        # stolen: extra ~5 ms fade.  Per-sample, not per
                        # block: the fade is shorter than one render chunk,
                        # so a block-constant factor would just postpone the
                        # cut by a block instead of smoothing it.
                        env = env * np.exp(np.float32(-_FAST_KILL_RATE)
                                           * (np.float32(rel) + t_base)[None, :])

                sig = np.sin(freqs[:, None] * two_pi * t[None, :])
                sig = np.float32(v.gain) * (amps[:, None] * env * sig).sum(axis=0)
            else:
                sig = np.zeros(length, dtype=np.float32)

            # --- hammer transient ---------------------------------------
            i0 = int(t0 * sr)
            if i0 < len(v.noise):
                chunk = v.noise[i0:i0 + length]
                if len(chunk):
                    sig[:len(chunk)] += chunk

            left += sig * np.float32(v.pan_l)
            right += sig * np.float32(v.pan_r)

            if v.alive_at(start + length):
                alive.append(v)

        self.voices = alive
        out[offset:offset + length, 0] = left
        out[offset:offset + length, 1] = right

    # ---------------------------------------------------------- master
    @staticmethod
    def master(mix: np.ndarray, volume: float) -> np.ndarray:
        """
        Level control: perfectly transparent below the knee, soft (never
        clipped, never pumped) above it.
        """
        if volume != 1.0:
            mix = mix * float(volume)
        if not mix.size:
            return mix
        knee = 0.70
        a = np.abs(mix)
        over = a > knee
        if over.any():
            a[over] = knee + (1.0 - knee) * np.tanh(
                (a[over] - knee) / (1.0 - knee))
            mix = np.sign(mix) * a
        return np.clip(mix, -1.0, 1.0)


##################################################### audio backends
class _CushionBackend:
    """
    Pre-rendered audio ring shared by the real device backends.

    Two separate problems are solved here:

    1. *The device callback must answer within one device period* or the
       output underruns -- heard as stuttering.  So the callback never
       renders: it only *copies* bytes out of a ring of ready made chunks,
       which takes a fraction of a millisecond.

    2. *Filling that ring has to happen somewhere that can get the
       interpreter lock.*  Rendering is numpy work, and numpy drops the lock
       and then has to win it back; while the animation thread sits inside
       VTK -- which keeps the lock for a whole 3D render -- a background
       thread measured out at less than 1% of a core and the ring simply
       emptied.  The player therefore fills the ring itself through
       `pump()`, from the one thread that always owns the lock.  The feeder
       thread kept here only looks after engines nobody pumps.

    A sample rendered at engine time ``t`` is *heard* at ``t +
    output_latency``; the player subtracts that when it works out the audio
    epoch.  The figure is *measured*, not assumed: the device may ask for
    more frames than have been rendered while it primes its buffer before
    playback, the shortfall goes out as silence, and every later sample is
    then heard exactly that much later than the engine clock says.
    """

    CHUNK = 1024            # frames per pre-rendered chunk (~23 ms)
    CUSHION = 0.40          # ring level the pump aims for (seconds)
    BUFFER_MS = 60          # device period.  The device primes a few periods
                            # in one go before playback starts, so this also
                            # sets how much has to be pre-rendered: keep it
                            # well under CUSHION or the priming consumes the
                            # cushion and `_take` has to pad with silence,
                            # which shifts every later sample on the stream.
    # The OS hands the DAC a whole period before it can start, then mixes.
    ENGINE_LATENCY = BUFFER_MS / 1000.0 + 0.02
    # Debug hatch only -- not a CLI flag, not a GUI control.  Different sound
    # cards report different figures, so PIANOPLAYER_LATENCY_MS lets one
    # machine be recalibrated without touching the source:
    #   PIANOPLAYER_LATENCY_MS=120 python bin/pianoplayer scores/x.xml -v
    ENGINE_LATENCY_OVERRIDE = (
        float(os.environ["PIANOPLAYER_LATENCY_MS"]) / 1000.0
        if os.environ.get("PIANOPLAYER_LATENCY_MS") else None)
    IDLE_PUMP = 0.40        # feeder only takes over after this long unserved
    # Adaptive render budget (C5 step 4): a full ring only needs a top-up,
    # while a starving ring may spend a whole frame refilling.  The budget
    # is picked from the ring level, so a normal frame costs ~8 ms and only
    # a ring below 25% may use up to 60 ms.
    PUMP_BUDGET_FULL = 0.008    # ring above 50%: top-up only
    PUMP_BUDGET_EMPTY = 0.060   # ring below 25%: refill at any cost

    def __init__(self, synth: SynthCore, volume_getter):
        self.synth = synth
        self.volume_getter = volume_getter
        self.underruns = 0
        self._pulled = 0             # frames the device ever asked for
        self._chunks = collections.deque()
        self._partial = b""          # tail of the chunk being consumed
        self._lock = threading.Lock()
        self._render_lock = threading.Lock()   # one renderer at a time
        self._last_pump = time.perf_counter()
        self._stop = threading.Event()
        self._thread = None

    @property
    def output_latency(self) -> float:
        """
        Seconds between a sample being rendered and being heard.

        Two parts: the OS mixer + DAC, and whatever the stream got shifted
        by.  While the device primes its buffer it may pull more frames than
        the synth has rendered; `_take` pads that with silence, and from then
        on every sample reaches the speakers that much later than the engine
        clock claims.  Rather than assume a value, measure the gap.
        """
        with self._lock:
            ring = len(self._chunks) * self.CHUNK + len(self._partial) // 4
            shipped = self.synth.frame - ring    # frames actually handed over
            gap = self._pulled - shipped         # ...vs frames asked for
        return self._os_latency + max(gap, 0) / float(self.synth.sr)

    @property
    def _os_latency(self) -> float:
        """OS mixer + DAC, overridable for calibration (see the constant)."""
        if self.ENGINE_LATENCY_OVERRIDE is not None:
            return self.ENGINE_LATENCY_OVERRIDE
        return self.ENGINE_LATENCY

    # ------------------------------------------------------------- filling
    def _target(self):
        return int(self.CUSHION * self.synth.sr)

    def _render_chunk(self):
        mix = SynthCore.master(self.synth.render_block(self.CHUNK),
                               self.volume_getter())
        pcm = np.ascontiguousarray(mix * 32767.0, dtype=np.int16)
        return pcm.tobytes()

    def _buffered(self):
        with self._lock:
            return len(self._chunks) * self.CHUNK + len(self._partial) // 4

    def pump(self, budget=None) -> int:
        """
        Render audio ahead from the **calling** thread (the animation loop).

        Returns the number of chunks made.  Never blocks: if the fallback
        feeder happens to hold the render lock we simply try again on the
        next frame, which also makes a deadlock impossible.

        With `budget=None` (the default) the budget adapts to the ring
        level: 8 ms while the ring is over half full, up to 60 ms when it
        drops under a quarter, so one slow frame can still refill the whole
        cushion in a single call instead of turning a stall into a
        permanent deficit (heard as stuttering).
        """
        self._last_pump = time.perf_counter()
        target = self._target()
        buffered = self._buffered()
        if buffered >= target:
            return 0
        if budget is None:
            frac = buffered / max(target, 1)
            if frac >= 0.50:
                budget = self.PUMP_BUDGET_FULL
            elif frac <= 0.25:
                budget = self.PUMP_BUDGET_EMPTY
            else:                       # linear interpolation between the two
                budget = (self.PUMP_BUDGET_EMPTY + (self.PUMP_BUDGET_FULL
                          - self.PUMP_BUDGET_EMPTY) * (frac - 0.25) / 0.25)
        if not self._render_lock.acquire(blocking=False):
            return 0
        made = 0
        deadline = time.perf_counter() + budget
        try:
            while self._buffered() < target and time.perf_counter() < deadline:
                chunk = self._render_chunk()
                with self._lock:
                    self._chunks.append(chunk)
                made += 1
        finally:
            self._render_lock.release()
        return made

    def _feed(self):
        """Fallback renderer for an engine nobody pumps (see `pump()`)."""
        while not self._stop.is_set():
            if (time.perf_counter() - self._last_pump < self.IDLE_PUMP
                    or self._buffered() >= self._target()):
                time.sleep(0.05)
                continue
            if not self._render_lock.acquire(blocking=False):
                time.sleep(0.02)
                continue
            try:
                chunk = self._render_chunk()
                with self._lock:
                    self._chunks.append(chunk)
            finally:
                self._render_lock.release()
            time.sleep(0.01)

    # -------------------------------------------------------- device side
    def _take(self, frames):
        nbytes = frames * 4           # 16 bit, 2 channels
        with self._lock:
            self._pulled += frames
            while len(self._partial) < nbytes and self._chunks:
                self._partial += self._chunks.popleft()
            if len(self._partial) < nbytes:
                self.underruns += 1
                out = self._partial + b"\x00" * (nbytes - len(self._partial))
                self._partial = b""
                return out
            out = self._partial[:nbytes]
            self._partial = self._partial[nbytes:]
            return out

    # ------------------------------------------------------------- lifecycle
    def _pre_fill(self):
        # Fill the ring *before* the device opens: sample 0 then sits at the
        # front of the ring when playback begins.  The ring also has to be
        # deep enough to survive the device asking for several periods in one
        # go while it primes its buffer -- see `output_latency`.
        want = self._target()
        while self._buffered() < want:
            with self._lock:
                self._chunks.append(self._render_chunk())

    def _start_feeder(self):
        self._thread = threading.Thread(target=self._feed, daemon=True,
                                        name="pianoplayer-feeder")
        self._thread.start()

    def _shutdown(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


class _MiniAudioBackend(_CushionBackend):
    name = "miniaudio"

    def __init__(self, synth: SynthCore, volume_getter):
        super().__init__(synth, volume_getter)
        self.device = None

    def start(self):
        self._pre_fill()
        self.device = miniaudio.PlaybackDevice(
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=2,
            sample_rate=self.synth.sr,
            buffersize_msec=self.BUFFER_MS,
        )
        gen = self._generator()
        next(gen)                      # prime the generator
        self.device.start(gen)
        self._start_feeder()

    def _generator(self):
        required = yield b""
        while True:
            n = int(required) if required else self.CHUNK
            yield self._take(n)

    def stop(self):
        self._shutdown()
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
            self.device = None


class _SounddeviceBackend(_CushionBackend):
    name = "sounddevice"

    def __init__(self, synth: SynthCore, volume_getter):
        super().__init__(synth, volume_getter)
        self.stream = None

    def start(self):
        self._pre_fill()

        def cb(outdata, frames, time_info, status):
            raw = self._take(frames)
            outdata[:] = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)

        self.stream = _sounddevice.OutputStream(
            samplerate=self.synth.sr, channels=2, dtype="int16",
            blocksize=int(self.BUFFER_MS / 1000.0 * self.synth.sr),
            callback=cb,
        )
        self.stream.start()
        self._start_feeder()

    def stop(self):
        self._shutdown()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    @property
    def output_latency(self) -> float:
        """Measured gap + what PortAudio itself reports for the stream."""
        if self.ENGINE_LATENCY_OVERRIDE is not None:
            reported = self.ENGINE_LATENCY_OVERRIDE
        else:
            try:
                reported = float(self.stream.latency)
            except Exception:
                reported = self.ENGINE_LATENCY
        with self._lock:
            ring = len(self._chunks) * self.CHUNK + len(self._partial) // 4
            gap = self._pulled - (self.synth.frame - ring)
        return reported + max(gap, 0) / float(self.synth.sr)


class _NullBackend:
    """No audio device: keeps the engine clock running, renders silence."""
    name = "none"

    def __init__(self, synth, volume_getter):
        self.synth = synth
        self._stop = threading.Event()

    def start(self):
        def idle():
            step = int(0.05 * self.synth.sr)
            while not self._stop.is_set():
                self.synth.render_block(step)
                time.sleep(0.05)

        threading.Thread(target=idle, daemon=True).start()

    def stop(self):
        self._stop.set()


##################################################### public engine
class SoundEngine:
    """
    High level, thread safe piano sound engine.

    Typical use::

        eng = SoundEngine(volume=0.8)
        eng.start()
        eng.note_on(60, velocity=0.7, when=2.5)   # 2.5 s after the engine start
        eng.note_off(60, when=3.2)
        ...
        eng.stop()
    """

    def __init__(self, volume=0.8, enabled=True, backend="auto",
                 sample_rate=SAMPLE_RATE, polyphony=32):
        self.volume = float(volume)
        self.enabled = bool(enabled)
        self.sample_rate = int(sample_rate)
        self.backend_name = "none"
        self.synth = SynthCore(self.sample_rate, polyphony)
        self._backend = _NullBackend(self.synth, lambda: self.volume)
        self._backend_pref = backend
        self._started = False
        self._saved_switch = None

    # ----------------------------------------------------------- lifecycle
    def start(self):
        if self._started:
            return self
        self._started = True
        # Give the audio/feeder threads the interpreter lock more often: the
        # 3D thread can hold it for a whole VTK render otherwise.
        self._saved_switch = sys.getswitchinterval()
        if self._saved_switch > 0.001:
            sys.setswitchinterval(0.001)
        if self.enabled:
            backend = self._make_backend()
            try:
                backend.start()
                self._backend = backend
                self.backend_name = backend.name
            except Exception as exc:  # pragma: no cover - device problems
                print(f"Audio backend {backend.name!r} failed ({exc}); "
                      "continuing without sound.")
                self.backend_name = "none"
                self._backend = _NullBackend(self.synth, lambda: self.volume)
                self._backend.start()
        else:
            self._backend.start()
        return self

    def _make_backend(self):
        pref = self._backend_pref
        if pref in (None, "", "none"):
            return _NullBackend(self.synth, lambda: self.volume)
        if pref in ("auto", "miniaudio") and HAS_MINIAUDIO:
            return _MiniAudioBackend(self.synth, lambda: self.volume)
        if pref in ("auto", "sounddevice") and HAS_SOUNDDEVICE:
            return _SounddeviceBackend(self.synth, lambda: self.volume)
        if pref == "auto":
            print("No audio backend available (pip install miniaudio), "
                  "running silently.")
        return _NullBackend(self.synth, lambda: self.volume)

    def stop(self):
        if not self._started:
            return
        self.all_notes_off()
        try:
            self._backend.stop()
        except Exception:
            pass
        saved = getattr(self, "_saved_switch", None)
        if saved:
            sys.setswitchinterval(saved)
        self._started = False

    @property
    def underruns(self) -> int:
        """Audio callbacks that ran out of data (heard as a glitch)."""
        return int(getattr(self._backend, "underruns", 0))

    @property
    def output_latency(self) -> float:
        """Seconds between a sample being rendered and being heard."""
        return float(getattr(self._backend, "output_latency", 0.0))

    def pump(self, budget=None) -> int:
        """
        Render audio into the playback buffer ahead of time.

        Call this once per animation frame from the thread that drives the
        window: a background thread cannot get enough of the interpreter
        lock to keep up while a 3D render is running (see
        `_CushionBackend.pump`).  Does nothing for backends that render on
        demand or do not play at all.
        """
        fn = getattr(self._backend, "pump", None)
        if fn is None:
            return 0
        try:
            return int(fn(budget=budget))
        except TypeError:
            return int(fn())

    # ----------------------------------------------------------- transport
    @property
    def now(self) -> float:
        """Current engine time in seconds (since start())."""
        return self.synth.frame / float(self.sample_rate)

    def note_on(self, midi, velocity=0.6, when=None, pan=0.0):
        """
        Schedule a note.  `when` is an **absolute time on the engine clock**
        (seconds since start()), not a delay: use `eng.now + dt` to place a
        note `dt` seconds from now, and `eng.now` for "immediately".
        """
        if midi is None:
            return
        self.synth.note_on(midi, velocity, when, pan)

    def note_off(self, midi, when=None):
        """Release a note; `when` follows the same rule as in `note_on()`."""
        if midi is None:
            return
        self.synth.note_off(midi, when)

    def all_notes_off(self):
        self.synth.all_notes_off()

    def shift_pending(self, delta_sec):
        """Move pending note on/off events (see `SynthCore.shift_pending`)."""
        try:
            return int(self.synth.shift_pending(delta_sec))
        except Exception:
            return 0

    # ----------------------------------------------------------- offline
    def render(self, events, seconds) -> np.ndarray:
        """
        Render `events` offline.  `events` is an iterable of
        (time, 'on'|'off', midi, velocity, pan) tuples, returns (n, 2) floats.
        """
        sr = self.sample_rate
        total = max(int(seconds * sr), 1)
        local = SynthCore(sr, self.synth.polyphony)
        ordered = sorted(events, key=lambda e: (e[0], e[1] not in ("off", 1)))
        for (t, kind, midi, vel, pan) in ordered:
            when = int(round(t * sr)) / sr
            if kind in (0, "on", "onset"):
                local.note_on(midi, vel, when, pan)
            else:
                local.note_off(midi, when)
        chunks = []
        step = 4096
        while local.frame < total:
            chunks.append(local.render_block(min(step, total - local.frame)))
        mix = np.concatenate(chunks) if chunks else np.zeros((0, 2))
        return SynthCore.master(mix, self.volume)

    def export_wav(self, events, seconds, path) -> str:
        """Render `events` offline and write a 16 bit stereo WAV file."""
        mix = self.render(events, seconds)
        pcm = (mix * 32767.0).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm.tobytes())
        return path
