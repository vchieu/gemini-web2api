#-------------------------------------------------------------------------------
# Name:         PianoPlayer
# Purpose:      Find optimal fingering for piano scores
# Author:       Marco Musy
#-------------------------------------------------------------------------------
import copy

from music21.articulations import Fingering
import pianoplayer.utils as utils


# Phase B (Viterbi fingering) constants. All at the top so they are easy to tune.
HARD = 200.0            # cost added per violated skip/held rule (hard-soft: a
                        # violated rule never eliminates the path, fixing F4)
WEAK_FINGER_PENALTY = 3.0   # B3: light penalty, see pair_cost()
WEAK_FINGER_MAX_DT = 0.3    # B3: fires when the inter-onset gap is below this.
                            # noteseq times are in quarter notes; 0.3 quarters
                            # is ~0.15 s at the default 120 bpm.
WEAK_FINGER_MIN_JUMP = 3.0  # B3: ... and the finger itself travels more than
                            # this (cm).
HELD_EPS = 1e-3             # B2: overlap slack in quarter notes.


def _set_fingering(n21, fng, replace=True):
    """Attach a Fingering articulation.

    `replace=False` keeps the fingering already there: a chord gets
    one number per pitch, and only its first note (`chordnr == 0`)
    purges the previous run's numbers -- so re-running generate()
    neither drops nor doubles the annotation."""
    if replace:
        for a in [a for a in n21.articulations if isinstance(a, Fingering)]:
            n21.articulations.remove(a)
    n21.articulations.append(fng)


#####################################################
class Hand:
    def __init__(self, side="right", size='M'):

        self.LR        = side
        # fingers pos at rest first is dummy, (cm), asymmetry helps with scales
        self.frest     = [None,  -7.0,-2.8, 0.0, 2.8, 5.6]
        self.weights   = [None,   1.1, 1.0, 1.1, 0.9, 0.8] # finger relative strength
        self.bfactor   = [None,   0.3, 1.0, 1.1, 0.8, 0.7] # hit of black key bias
        self.noteseq   = []
        self.fingerseq = []
        self.fingerseq_index = None  # list (maybe empty) once generate() ran
        self.depth     = 9
        self.autodepth = True
        self.verbose   = True
        self.lyrics    = False  # show fingering numbers as lyrics in musescore
        self.size      = size
        # Phase B: False = Viterbi solve of the whole range (default);
        # True = legacy 9-note sliding-window search (optimize_seq).
        # The -d/--depth flag only has an effect in legacy mode.
        self.legacy_window = False

        self.hf = utils.handSizeFactor(size)
        for i in (1,2,3,4,5):
            if self.frest[i]: self.frest[i] *= self.hf
        print('Your hand span set to size-'+size, 'which is', 21*self.hf, 'cm')
        print('(max relaxed distance between thumb and pinkie)')
        self.cfps = list(self.frest) # hold current finger positions


    #####################################################
    def set_fingers_positions(self, fings, notes, i):

        fi = fings[i]
        ni = notes[i]
        ifx = self.frest[fi]
        if ifx is not None:
            for j in (1,2,3,4,5):
                jfx = self.frest[j]
                self.cfps[j] = (jfx-ifx) + ni.x


    #####################################################
    def ave_velocity(self, fingering, notes):
        ###calculate v for playing for notes in a given fingering combination

        self.set_fingers_positions(fingering, notes, 0)  #init fingers position

        vmean = 0.
        for i in range(1, self.depth):
            na = notes[i-1]
            nb = notes[i]
            fb = fingering[i]
            dx = abs(nb.x - self.cfps[fb])   # space travelled by finger fb
            dt = abs(nb.time - na.time) +0.1 # available time +smoothing term 0.1s
            v  = dx/dt                       # velocity
            if nb.isBlack:                   # penalty (by increasing speed)
                v /= self.weights[fb] * self.bfactor[fb]
            else:
                v /= self.weights[fb]
            vmean += v
            self.set_fingers_positions(fingering, notes, i) #update all fingers

        return vmean / (self.depth-1)


    #####################################################
    def _skip_penalty(self, fa, fb, na, nb):
        """Hard-soft version of the old optimize_seq.skip() rules.

        Returns HARD when the (fa on na -> fb on nb) transition breaks a
        rule, 0.0 otherwise. Unlike skip(), a violation never eliminates
        the combination, so a path always exists (fixes F4).

        The rules run on EVERY pair (fixes F2): two single notes keep the
        old rules; two notes of the same chord keep the old chord rules;
        any other mix (chord<->note, chord<->chord) gets the "same finger,
        different pitch" rule plus the single-note crossing rules on x.
        """
        xba = nb.x - na.x  # physical distance btw the second to first note, in cm

        if na.isChord and nb.isChord and na.chordID == nb.chordID:
            # na and nb are notes in the same chord
            if fa==fb: return HARD   # play different chord notes w/ same finger
            if fa<fb and self.LR=='left' : return HARD
            if fa>fb and self.LR=='right': return HARD
            axba = abs(xba)*self.hf /0.8
            # max normalized distance in cm btw 2 consecutive fingers
            if axba> 5 and (fa==3 and fb==4 or fa==4 and fb==3): return HARD
            if axba> 5 and (fa==4 and fb==5 or fa==5 and fb==4): return HARD
            if axba> 6 and (fa==2 and fb==3 or fa==3 and fb==2): return HARD
            if axba> 7 and (fa==2 and fb==4 or fa==4 and fb==2): return HARD
            if axba> 8 and (fa==3 and fb==5 or fa==5 and fb==3): return HARD
            if axba>11 and (fa==2 and fb==5 or fa==5 and fb==2): return HARD
            if axba>12 and (fa==1 and fb==2 or fa==2 and fb==1): return HARD
            if axba>14 and (fa==1 and fb==3 or fa==3 and fb==1): return HARD
            if axba>16 and (fa==1 and fb==4 or fa==4 and fb==1): return HARD
            return 0.0

        # single notes, and every mixed pair: the old single-note rules.
        if fa==fb and abs(xba)>1e-9 and na.duration<4:
            return HARD # play different notes w/ same finger
        if fa>1 : # if a is not thumb
            if fb>1 and (fb-fa)*xba<0: return HARD # non-thumb fingers are crossings
            if fb==1 and nb.isBlack and xba>0: return HARD # crossing thumb goes to black
        else: # a is played by thumb:
            # a is black  and  b is behind a  and  fb not thumb  and na.duration<2:
            if na.isBlack and xba<0 and fb>1 and na.duration<2: return HARD
        return 0.0


    #####################################################
    def pair_cost(self, na, fa, nb, fb):
        """Cost of playing nb with finger fb right after na with finger fa.

        The velocity term reproduces ave_velocity() exactly: reach_err is
        the distance finger fb itself travels, dt the inter-onset gap plus
        the 0.1 smoothing term, weights/bfactor the old finger strength and
        black-key penalties. On top come the hard-soft skip rules (B1), the
        held-finger constraint (B2) and the weak-finger penalty (B3).
        """
        if fa not in (1,2,3,4,5) or fb not in (1,2,3,4,5):
            return HARD  # e.g. a legacy finger 0: always losable, never fatal
        reach_err = abs(nb.x - (na.x + self.frest[fb] - self.frest[fa]))
        dt = abs(nb.time - na.time) +0.1 # available time +smoothing term 0.1s
        v  = reach_err/dt                # velocity
        if nb.isBlack:                   # penalty (by increasing speed)
            v /= self.weights[fb] * self.bfactor[fb]
        else:
            v /= self.weights[fb]
        cost = v + self._skip_penalty(fa, fb, na, nb)
        # B2: a finger still holding na when nb starts must not replay a
        # different key at the same time.
        if na.time_real + na.dur_real > nb.time_real + HELD_EPS:
            if fa==fb and abs(nb.x - na.x)>1e-9:
                cost += HARD
        # B3: weak fingers (4, 5) must not jump far on short notice.
        # (No extra thumb-on-black penalty: bfactor already covers it.)
        if fb in (4,5) and abs(nb.time - na.time) < WEAK_FINGER_MAX_DT \
                and reach_err > WEAK_FINGER_MIN_JUMP:
            cost += WEAK_FINGER_PENALTY
        return cost


    #####################################################
    def total_cost(self, notes, fingers):
        """Sum of pair_cost() over the consecutive pairs of a solution.

        Used to compare the Viterbi output against the legacy-window
        output with one and the same cost function (B5).
        """
        if len(notes) < 2 or len(fingers) < 2:
            return 0.0
        return float(sum(self.pair_cost(na, fa, nb, fb) for na, fa, nb, fb
                         in zip(notes, fingers, notes[1:], fingers[1:])))


    #####################################################
    def solve_viterbi(self, notes):
        """Viterbi-optimal finger sequence (1..5) for the whole note list.

        State is the finger pair (f_{i-1}, f_i) -- 25 states, 125
        transitions per step -- so that, besides the consecutive-pair
        cost, a finger held by note i-2 can also be honoured when note i
        starts (B2). Back-pointers recover the cheapest path; ties keep
        the first minimum, so repeated runs agree exactly.
        """
        n = len(notes)
        if n == 0:
            return []
        if n == 1:
            return [3]
        fingers = (1,2,3,4,5)
        # dp[(fa, fb)]: cheapest cost of a path ending with fa on
        # notes[i-1] and fb on notes[i].
        dp = {}
        for fa in fingers:
            for fb in fingers:
                dp[(fa,fb)] = self.pair_cost(notes[0], fa, notes[1], fb)
        back = {1: {k: k[0] for k in dp}}  # unused in traceback, kept for shape
        for i in range(2, n):
            na, nb, nc = notes[i-1], notes[i], notes[i-2]
            nc_held = nc.time_real + nc.dur_real > nb.time_real + HELD_EPS
            ndp, nback = {}, {}
            for fa in fingers:
                for fb in fingers:
                    pc = self.pair_cost(na, fa, nb, fb)
                    best, bestfp = None, None
                    for fp in fingers:
                        c = dp[(fp,fa)] + pc
                        if nc_held and fp==fb and abs(nb.x - nc.x)>1e-9:
                            c += HARD
                        if best is None or c < best:
                            best, bestfp = c, fp
                    ndp[(fa,fb)] = best
                    nback[(fa,fb)] = bestfp
            dp, back[i] = ndp, nback
        bfa, bfb = min(dp, key=lambda k: dp[k])
        out = [0]*n
        out[n-1], out[n-2] = bfb, bfa
        for i in range(n-1, 1, -1):
            out[i-2] = back[i][(out[i-1], out[i])]
        return out


    #####################################################
    def _repair_held(self, notes, fingers):
        """Greedy fix for same-finger overlaps deeper than the Viterbi
        order-2 memory (a note held across 3+ later notes, typical of
        multi-voice beams, B2/F3).

        Walks the notes in order; a note sharing its finger with any
        still-held earlier note on a different key takes the cheapest
        free finger (measured with pair_cost against both neighbours).
        At most a handful of notes are ever held at once, so a free
        finger always exists in practice. Returns the corrected list.
        """
        f = list(fingers)
        n = len(notes)
        fixed = 0
        for i in range(n):
            if f[i] not in (1,2,3,4,5):
                continue
            bi = notes[i]
            bi_end = bi.time_real + max(bi.dur_real, 0.0)
            blocked = set()
            for j in range(i):
                if f[j] not in (1,2,3,4,5):
                    continue
                aj = notes[j]
                if abs(aj.x - bi.x) < 1e-9:
                    continue  # same pitch: legal
                aj_end = aj.time_real + max(aj.dur_real, 0.0)
                if aj_end > bi.time_real + HELD_EPS \
                        and bi_end > aj.time_real + HELD_EPS:
                    blocked.add(f[j])
            if f[i] not in blocked:
                continue
            best, bestc = None, None
            for cand in (1,2,3,4,5):
                if cand in blocked:
                    continue
                c = 0.0
                if i > 0:
                    c += self.pair_cost(notes[i-1], f[i-1], bi, cand)
                if i+1 < n:
                    c += self.pair_cost(bi, cand, notes[i+1], f[i+1])
                if best is None or c < bestc:
                    best, bestc = cand, c
            if best is not None and best != f[i]:
                if self.verbose:
                    print('  held-finger repair: note %d (%s%s) finger %d -> %d'
                          % (i, bi.name, bi.octave, f[i], best))
                f[i] = best
                fixed += 1
            elif self.verbose:
                print('  held-finger repair: note %d (%s%s) keeps conflicting'
                      ' finger %d (no free finger)' % (i, bi.name, bi.octave, f[i]))
        if fixed and self.verbose:
            print('  held-finger repair: fixed %d note(s)' % fixed)
        return f


    #####################################################
    def optimize_seq(self, nseq, istart):
        '''Generate meaningful fingering for a note sequence of size depth'''

        #---------------------------------------------------------
        def skip(fa,fb, na,nb, level):
            ### two-consecutive-notes movement, skipping rules ###
            # fa is fingering for note na, level is passed only for debugging

            xba = nb.x - na.x  # physical distance btw the second to first note, in cm

            if not na.isChord and not nb.isChord: # neither of the 2 notes live in a chord
                if fa==fb and xba and na.duration<4:
                    return True # play different notes w/ same finger, skip
                if fa>1 : # if a is not thumb
                    if fb>1 and (fb-fa)*xba<0: return True # non-thumb fingers are crossings, skip
                    if fb==1 and nb.isBlack and xba>0: return True # crossing thumb goes to black, skip
                else: # a is played by thumb:
                    # skip if  a is black  and  b is behind a  and  fb not thumb  and na.duration<2:
                    if na.isBlack and xba<0 and fb>1 and na.duration<2: return True

            elif na.isChord and nb.isChord and na.chordID == nb.chordID:
                # na and nb are notes in the same chord
                if fa==fb: return True   # play different chord notes w/ same finger, skip
                if fa<fb and self.LR=='left' : return True
                if fa>fb and self.LR=='right': return True
                axba = abs(xba)*self.hf /0.8
                # max normalized distance in cm btw 2 consecutive fingers
                if axba> 5 and (fa==3 and fb==4 or fa==4 and fb==3): return True
                if axba> 5 and (fa==4 and fb==5 or fa==5 and fb==4): return True
                if axba> 6 and (fa==2 and fb==3 or fa==3 and fb==2): return True
                if axba> 7 and (fa==2 and fb==4 or fa==4 and fb==2): return True
                if axba> 8 and (fa==3 and fb==5 or fa==5 and fb==3): return True
                if axba>11 and (fa==2 and fb==5 or fa==5 and fb==2): return True
                if axba>12 and (fa==1 and fb==2 or fa==2 and fb==1): return True
                if axba>14 and (fa==1 and fb==3 or fa==3 and fb==1): return True
                if axba>16 and (fa==1 and fb==4 or fa==4 and fb==1): return True

            return False
        #---------------------------------------------------------------------------

        if self.autodepth:
            #choose depth based on time span of 3.5 seconds
            if nseq[0].isChord:
                self.depth = max(3, nseq[0].NinChord - nseq[0].chordnr + 1)
            else:
                tn0 = nseq[0].time
                for i in (4,5,6,7,8,9):
                    self.depth = i
                    if nseq[i-1].time - tn0 > 3.5:
                        break
        depth = self.depth

        fingers = (1,2,3,4,5)
        n1, n2, n3, n4, n5, n6, n7, n8, n9 = nseq
        u_start = [istart]
        if istart == 0:
            u_start = fingers

        #####################################
        out = ([0 for i in range(depth)], -1)
        minvel = 1.e+10
        for f1 in u_start:
            for f2 in fingers:
                if skip(f1,f2, n1,n2, 2): continue
                for f3 in fingers:
                    if f3 and skip(f2,f3, n2,n3, 3): continue
                    if depth<4: u=[False]
                    else:       u=fingers
                    for f4 in u:
                        if f4 and skip(f3,f4, n3,n4, 4): continue
                        if depth<5: u=[False]
                        else:       u=fingers
                        for f5 in u:
                            if f5 and skip(f4,f5, n4,n5, 5): continue
                            if depth<6: u=[False]
                            else:       u=fingers
                            for f6 in u:
                                if f6 and skip(f5,f6, n5,n6, 6): continue
                                if depth<7: u=[False]
                                else:       u=fingers
                                for f7 in u:
                                    if f7 and skip(f6,f7, n6,n7, 7): continue
                                    if depth<8: u=[False]
                                    else:       u=fingers
                                    for f8 in u:
                                        if f8 and skip(f7,f8, n7,n8, 8): continue
                                        if depth<9: u=[False]
                                        else:       u=fingers
                                        for f9 in u:
                                            if f9 and skip(f8,f9, n8,n9, 9): continue
                                            c = [f1,f2,f3,f4,f5,f6,f7,f8,f9]
                                            v = self.ave_velocity(c, nseq)
                                            if v < minvel:
                                                out = (c, v)
                                                minvel  = v
        # if out[1]==-1: exit() #no combination found
        return out


    ###########################################################################################
    def generate(self, start_measure=0, nmeasures=1000):
        """Find a finger (1..5) for every note in the [start_measure,
        start_measure + nmeasures) range (measures are 1-based on the
        command line; start_measure == 1 is mapped to 0 as before).

        Unless self.legacy_window is True, the whole in-range slice is
        solved at once with solve_viterbi() (B1): no dummy-note padding
        is needed for short scores, and a single note gets finger 3.
        The legacy 9-note sliding-window search lives in
        _generate_legacy(), where -d/--depth still applies.
        """

        if start_measure == 1:
            start_measure=0 # avoid confusion with python numbering

        if self.LR == "left":
            for anote in self.noteseq:
                if not getattr(anote, "_mirrored", False):
                    anote.x = -anote.x  # play left as a right on a mirrored keyboard
                    anote._mirrored = True   # ...only once: generate() may run again

        if self.legacy_window:
            return self._generate_legacy(start_measure, nmeasures)

        self.fingerseq = []        # fingertip fan of every processed note ..
        self.fingerseq_index = []  # .. and the noteseq index it belongs to

        kept = []
        for i, an in enumerate(self.noteseq):
            if an.measure:
                if an.measure < start_measure : continue
                if an.measure >= start_measure + nmeasures : break
            kept.append(i)
        if not kept:
            return

        if len(kept) == 1:
            fingers = [3]
        else:
            sub = [self.noteseq[i] for i in kept]
            fingers = self.solve_viterbi(sub)
            fingers = self._repair_held(sub, fingers)

        for k, i in enumerate(kept):##############

            an = self.noteseq[i]
            best_finger = fingers[k]

            an.fingering = best_finger
            for j in (1,2,3,4,5):
                self.cfps[j] = (self.frest[j]-self.frest[best_finger]) + an.x
            self.fingerseq.append(list(self.cfps))
            self.fingerseq_index.append(i)

            if best_finger>0:
                fng = Fingering(best_finger)
                if an.isChord:
                    if len(an.chord21.pitches) < 3:
                        # dont show fingering in the lyrics line for >3 note-chords
                        if self.lyrics:
                            nl = len(an.chord21.pitches) - an.chordnr
                            an.chord21.addLyric(best_finger, nl)
                        else:
                            # a chord appends one fingering per pitch;
                            # only its first note purges the old ones
                            _set_fingering(an.chord21, fng,
                                           replace=an.chordnr == 0)
                else:
                    if self.lyrics:
                        an.note21.addLyric(best_finger)
                    else:
                        _set_fingering(an.note21, fng)

            #---------------------------------------------------------------------------- print
            if self.verbose:
                if an.measure: print(f"meas.{an.measure: <3}", end=' ')
                print(f"finger_{best_finger}  plays  {an.name: >2}{an.octave}")
            else:
                if k and not k%100 and an.measure:
                    print('scanned', k, '/', len(kept),
                          'notes, measure', an.measure+1, ' for the', self.LR ,'hand...')


    ###########################################################################################
    def _generate_legacy(self, start_measure=0, nmeasures=1000):
        """Legacy 9-note sliding-window search (optimize_seq).

        Kept for comparison (B5) behind self.legacy_window; this is the
        only mode where self.depth / self.autodepth (-d/--depth) apply.
        start_measure is already converted (1 -> 0) by generate()."""

        start_finger, out, vel = 0, [0 for i in range(9)], 0
        entry_autodepth, entry_depth = self.autodepth, self.depth
        N = len(self.noteseq)
        if self.depth < 3: self.depth = 3
        if self.depth > 9: self.depth = 9

        # fewer than 9 notes: optimize_seq needs exactly 9, so pad the window
        # with dummy notes (same pitch, later in time) and solve it once.
        if 2 <= N < 9:
            pad = list(self.noteseq)
            last = pad[-1]
            for k in range(9 - N):
                q = copy.copy(last)
                q.isChord = False
                q.time = last.time + k + 1
                pad.append(q)
            self.autodepth = False
            self.depth = 9
            res, vel = self.optimize_seq(pad, 0)
            out = [0] + list(res)      # slot 0 is "consumed" by pop(1) below

        self.fingerseq = []        # fingertip fan of every processed note ..
        self.fingerseq_index = []  # .. and the noteseq index it belongs to

        for i in range(N):##############

            an = self.noteseq[i]
            if an.measure:
                if an.measure < start_measure : continue
                if an.measure >= start_measure + nmeasures : break

            if i > N-11:
                self.autodepth = False
                self.depth = 9

            best_finger = 0
            ninenotes = self.noteseq[i : i+9]
            if i > N-9:                 # window of 9 is still complete at i=N-9
                if len(out)>1: best_finger = out.pop(1)
            else:
                out, vel  = self.optimize_seq(ninenotes, start_finger)
                best_finger  = out[0]
                start_finger = out[1]

            an.fingering = best_finger
            outfan = list(out)
            if outfan: outfan[0] = best_finger
            self.set_fingers_positions(outfan, ninenotes, 0)
            self.fingerseq.append(list(self.cfps))
            self.fingerseq_index.append(i)

            if best_finger>0:
                fng = Fingering(best_finger)
                if an.isChord:
                    if len(an.chord21.pitches) < 3:
                        # dont show fingering in the lyrics line for >3 note-chords
                        if self.lyrics:
                            nl = len(an.chord21.pitches) - an.chordnr
                            an.chord21.addLyric(best_finger, nl)
                        else:
                            # a chord appends one fingering per pitch;
                            # only its first note purges the old ones
                            _set_fingering(an.chord21, fng,
                                           replace=an.chordnr == 0)
                else:
                    if self.lyrics:
                        an.note21.addLyric(best_finger)
                    else:
                        _set_fingering(an.note21, fng)

            #---------------------------------------------------------------------------- print
            if self.verbose:
                if not best_finger: best_finger = '?'
                if an.measure: print(f"meas.{an.measure: <3}", end=' ')
                print(f"finger_{best_finger}  plays  {an.name: >2}{an.octave}", end=' ')
                if i < N-10:
                    print(f"  v={round(vel,1)}", end='')
                    if self.autodepth:
                        print("\t "+str(out[0:self.depth]) + " d:" + str(self.depth))
                    else:
                        print("\t"+("   "*(i%self.depth))+str(out[0:self.depth]))
                else:
                    print()
            else:
                if i and not i%100 and an.measure:
                    print('scanned', i, '/', N,
                          'notes, measure', an.measure+1, ' for the', self.LR ,'hand...')

        # Restore the depth settings the caller had on entry: the scan
        # forces depth 9 on its last stretch, and leaving that behind
        # would make a second generate() solve other depths and find a
        # different fingering for the same notes.
        self.autodepth = entry_autodepth
        self.depth = entry_depth




