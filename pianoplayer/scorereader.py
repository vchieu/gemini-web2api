"""
Created on Thu Nov 26 19:22:20 2015

@author: marco musy
"""
from pianoplayer.utils import keypos


def _flat(stream):
    """music21>=7 removed Stream.flat in favour of Stream.flatten()."""
    if hasattr(stream, "flatten"):
        return stream.flatten()
    return stream.flat


#####################################################
class INote:
    def __init__(self):
        self.note21   = None
        self.name     = None
        self.isChord  = False
        self.isBlack  = False
        self.octave   = 0
        self.x        = 0.0
        self.time     = 0.0
        self.duration = 0.0
        self.time_real = 0.0     # true onset (quarter lengths), for playback
        self.dur_real  = 0.0     # true duration, for playback
        self.fingering= 0
        self.measure  = 0
        self.chordnr  = 0
        self.NinChord = 0
        self.chordID  = 0
        self.noteID   = 0


#####################################################
def reader(sf, beam=0):

    noteseq = []

    if hasattr(sf, 'parts'):
        if len(sf.parts) <= beam:
            return []
        strm = _flat(sf.parts[beam])
    elif hasattr(sf, 'elements'):
        if len(sf.elements)==1 and beam==1:
            strm = sf[0]
        else:
            if len(sf) <= beam:
                return []
            strm = sf[beam]
    else:
        strm = _flat(sf)

    print('Reading beam', beam, 'with', len(strm), 'objects in stream.')

    sfasam  = 0.05          # small stagger for the fingering algorithm only
    chordID = 0
    pending = {}            # midi -> INote that opened a tie still running
    last_off, dup, at_off = None, 0, set()   # elements sharing one onset

    def _absorb(midi, tie, qlen):
        """True when this note is the 2nd..last piece of a tie.  Its length is
        added to the note that started the tie (which is the only one played)."""
        if tie is None or tie.type not in ('continue', 'stop'):
            return False
        head = pending.get(midi)
        if head is not None:
            head.duration += qlen
            head.dur_real += qlen
            if tie.type == 'stop':
                del pending[midi]
        return True          # orphan continuation: skip, as before

    for n in strm:
        ql = n.duration.quarterLength
        if ql == 0 or not (n.isNote or n.isChord):
            continue

        # Several voices in one part: notes on the same onset are all kept
        # (they used to be dropped: no sound, no key, no finger).  The
        # algorithm gets a tiny time shift so they stay distinct events.
        off = float(n.offset)
        if last_off is not None and abs(off - last_off) < 1e-6:
            dup += 1
        else:
            last_off, dup, at_off = off, 0, set()
        shift = sfasam * dup

        if n.isNote:
            p = getattr(n, 'pitch', n)
            midi = int(p.midi)
            tie = getattr(n, 'tie', None)
            if _absorb(midi, tie, ql) or midi in at_off:
                continue                 # tie piece, or same key twice at once
            at_off.add(midi)

            an          = INote()
            an.noteID  += 1
            an.note21   = n
            an.isChord  = False
            an.name     = n.name
            an.octave   = n.octave
            an.measure  = n.measureNumber
            an.x        = keypos(n)
            an.time     = off + shift
            an.duration = ql
            an.time_real = off
            an.dur_real  = ql
            an.isBlack  = p.pitchClass in (1, 3, 6, 8, 10)
            if n.lyrics:
                # a lyric line is free text: only honour a real 1..5 hint,
                # anything else leaves the fingering open (0) for generate()
                try:
                    lv = int(str(n.lyric).strip())
                except (ValueError, TypeError):
                    lv = 0
                an.fingering = lv if 1 <= lv <= 5 else 0
            if tie is not None and tie.type == 'start':
                pending[midi] = an
            noteseq.append(an)

        else:   # chord: every pitch has its own tie (Chord.tie is only the first)
            cnotes = list(n.notes)
            kept = []
            for j, cn in enumerate(n.pitches):
                midi = int(cn.midi)
                tie = cnotes[j].tie if j < len(cnotes) else None
                if _absorb(midi, tie, ql) or midi in at_off:
                    continue
                at_off.add(midi)
                noteobj = cnotes[j] if j < len(cnotes) else None
                kept.append((cn, noteobj, midi, tie))
            if not kept:
                continue

            for j, (cn, noteobj, midi, tie) in enumerate(kept):
                an          = INote()
                an.chordID  = chordID
                an.noteID  += 1
                # a chord whose other pitches were all absorbed by ties is
                # down to one sounding pitch: not a chord anymore (F6), but
                # keep chord21/chordnr around. Its note21 must be the real
                # Note (a Pitch has no articulations/lyrics for generate()).
                an.isChord  = len(kept) > 1
                an.chord21  = n
                an.note21   = cn if len(kept) > 1 else (noteobj if noteobj is not None else cn)
                an.name     = cn.name
                an.chordnr  = j
                an.NinChord = len(kept)
                an.octave   = cn.octave
                an.measure  = n.measureNumber
                an.x        = keypos(cn)
                an.time     = off + shift - sfasam*j
                an.duration = ql + sfasam*(len(kept)-1)
                an.time_real = off
                an.dur_real  = ql
                an.isBlack  = cn.pitchClass in (1, 3, 6, 8, 10)
                if tie is not None and tie.type == 'start':
                    pending[midi] = an
                noteseq.append(an)

            chordID += 1

    if len(noteseq)<1:
        print("Beam is empty.")
        return []
    return noteseq


def PIG2Stream(fname, beam=0, time_unit=.5, fixtempo=0):
    """
    Convert a PIG text file to a music21 Stream object.

    time_unit must be multiple of 2.
    beam = 0, right hand
    beam = 1, left hand.
    """
    from music21 import stream, note, chord
    from music21.articulations import Fingering
    import numpy as np

    f = open(fname, "r")
    lines = f.readlines()
    f.close()

    #work out note type from distribution of durations
    # triplets are squashed to the closest figure
    durations = []
    firstonset = 0
    blines=[]
    for l in lines:
        if l.startswith('//'): continue
        _, onset, offset, name, _, _, channel, _ = l.split()
        onset, offset = float(onset), float(offset)
        if beam != int(channel): continue
        if not firstonset:
            firstonset = onset
        if offset-onset<0.0001: continue
        durations.append(offset-onset)
        blines.append(l)
    durations = np.array(durations)
    logdurs = -np.log2(durations)
    mindur = np.min(logdurs)
    expos = (logdurs-mindur).astype(int)
    if np.max(expos) > 3:
        mindur = mindur + 1
    #print(durations, '\nexpos=',expos, '\nmindur=', mindur)

    sf = stream.Part()
    sf.id = beam

    # first rest
    if not fixtempo and firstonset:
        r = note.Rest()
        logdur = -np.log2(firstonset)
        r.duration.quarterLength = 1.0/time_unit/pow(2, int(logdur-mindur))
        sf.append(r)

    n = len(blines)
    for i in range(n):
        if blines[i].startswith('//'): continue
        _, onset, offset, name, _, _, _, finger = blines[i].split()
        onset, offset = float(onset), float(offset)
        name = name.replace('b', '-')

        chordnotes = [name]
        for j in range(1, 5):
            if i+j<n:
                noteid1, onset1, offset1, name1, _, _, _, finger1 = blines[i+j].split()
                onset1 = float(onset1)
                if onset1 == onset:
                    name1 = name1.replace('b', '-')
                    chordnotes.append(name1)

        if len(chordnotes)>1:
            an = chord.Chord(chordnotes)
        else:
            an = note.Note(name)
            if '_' not in finger:
                x = Fingering(abs(int(finger)))
                x.style.absoluteY = 20
                an.articulations.append(x)

        if fixtempo:
            an.duration.quarterLength = fixtempo
        else:
            logdur = -np.log2(offset - onset)
            an.duration.quarterLength = 1.0/time_unit/pow(2, int(logdur-mindur))
        #print('note/chord:', an, an.duration.quarterLength, an.duration.type, 't=',onset)

        sf.append(an)

        # rest up to the next
        if i+1<n:
            _, onset1, _, _, _, _, _, _ = blines[i+1].split()
            onset1 = float(onset1)
            if onset1 - offset > 0:
                r = note.Rest()
                if fixtempo:
                    r.duration.quarterLength = fixtempo
                logdur = -np.log2(onset1 - offset)
                d = int(logdur-mindur)
                if d<4:
                    r.duration.quarterLength = 1.0/time_unit/pow(2, d)
                    sf.append(r)

    return sf








































