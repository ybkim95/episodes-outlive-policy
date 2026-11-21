"""MemoryDialogue v2: a synthetic long-horizon multi-turn environment with a sparse goal reward and
small per-turn interaction costs (goal progress minus interaction cost).

A hidden user goal g is observable only in reply to the agent's first T_INFO turns: the LOOKUP tool
returns it, ASK makes the user reveal it with prob p_ask, and a scripted teammate sometimes volunteers
it. With prob p_shift the user announces a new goal once, mid-episode ("NOW g'"). On the last turn the
user asks for the goal and whatever the agent replies is its answer: reward 1 iff the reply's argument is
the current goal (a forced reply: rewarding only the ANSWER verb let the policy collapse to always-PASS
under the speech cost and never find the goal reward). Speaking when not asked (ASK, or ANSWER before
the last turn) costs `cost`; PASS (stay silent) and tools (LOOKUP, WRITE) are free.
If the context window W is shorter than the episode, the goal scrolls out of context, so the agent must
WRITE it to a persistent memory slot, and rewrite it after a shift.

Vectorised over B episodes that advance in lockstep (fixed length H). Every message is 3 tokens:
[actor, word, arg]; the agent's own message is [<agent>, verb, arg].
"""
import numpy as np

SPECIALS = ["<pad>", "<mem>", "<user>", "<tool>", "<mate>", "<agent>"]
WORDS = ["START", "IS", "UNKNOWN", "CHAT", "RESULT", "ERROR", "SAYS", "QUESTION", "NOW"]
VERBS = ["ASK", "LOOKUP", "WRITE", "PASS", "ANSWER"]
N_GOALS = 8
ARGS = ["NONE"] + [f"G{i}" for i in range(N_GOALS)]
CHATS = [f"C{i}" for i in range(4)]
VOCAB = SPECIALS + WORDS + VERBS + ARGS + CHATS
TOK = {w: i for i, w in enumerate(VOCAB)}
VERB_IDS = np.array([TOK[v] for v in VERBS])
ARG_IDS = np.array([TOK[a] for a in ARGS])
GOAL_IDS = ARG_IDS[1:]
CHAT_IDS = np.array([TOK[c] for c in CHATS])
T_INFO = 3


class MemoryDialogue:
    def __init__(self, goals, H, rng, p_ask=0.5, p_mate=0.3, p_shift=0.5, cost=0.01, crn=1):
        assert H >= T_INFO + 2, "need room for the info phase and the final question"
        self.g0 = GOAL_IDS[np.asarray(goals)]
        self.B, self.H, self.rng = len(self.g0), H, rng
        self.p_ask, self.p_mate, self.p_shift, self.cost = p_ask, p_mate, p_shift, cost
        assert self.B % crn == 0
        self.crn = crn  # control: consecutive groups of crn episodes share one goal-shift schedule

    def _msg(self, actor, word, arg):
        return np.stack([np.broadcast_to(x, (self.B,)) for x in (actor, word, arg)], 1)

    def reset(self):
        self.t, self.g = 0, self.g0.copy()
        self.mem = np.full(self.B, TOK["NONE"])
        # always draw B schedules, so crn > 1 uses the same random numbers as crn = 1 (paired control)
        self.t_shift, self.has_shift, off = (np.repeat(x[:: self.crn], self.crn) for x in self._draw_shift(self.B))
        self.g_new = self._new_goal(self.g0, off)
        return self._msg(TOK["<user>"], TOK["START"], TOK["NONE"])

    def _draw_shift(self, n, rng=None):
        """Prior over the goal shift: when, whether, and the goal offset. It lands after the info phase
        and before the final question."""
        rng = self.rng if rng is None else rng
        lo, hi = max(T_INFO + 1, self.H // 4), min(self.H - 1, max(T_INFO + 2, 3 * self.H // 4))
        t_shift = rng.integers(lo, max(hi, lo + 1), n)
        has_shift = (rng.random(n) < self.p_shift) & (lo < hi)
        return t_shift, has_shift, rng.integers(1, N_GOALS, n)

    @staticmethod
    def _new_goal(g0, off):
        return GOAL_IDS[(np.searchsorted(GOAL_IDS, g0) + off) % N_GOALS]  # off in 1..N-1: always a different goal

    def fork(self, k, resample_rng=None):
        """Replicate every episode k times (rows stay grouped: episode i -> rows i*k .. i*k+k-1).
        Copies share the full state, including the pending goal shift; later randomness diverges.
        resample_rng (control): each copy redraws a shift it has not seen yet, from the prior given that
        none happened so far, so siblings share the history but not the user's future. The redraw uses
        its own generator, so the env's stream stays paired with a run that does not resample."""
        for name in ("g0", "g", "mem", "t_shift", "has_shift", "g_new"):
            setattr(self, name, np.repeat(getattr(self, name), k))
        self.B *= k
        resample = resample_rng is not None
        todo = ~(self.has_shift & (self.t_shift <= self.t)) if resample else np.zeros(self.B, bool)
        while todo.any():  # rejection sampling: drop draws whose shift would already have happened
            idx = np.nonzero(todo)[0]
            t_shift, has_shift, off = self._draw_shift(len(idx), resample_rng)
            ok = ~(has_shift & (t_shift <= self.t))
            idx = idx[ok]
            self.t_shift[idx], self.has_shift[idx] = t_shift[ok], has_shift[ok]
            self.g_new[idx] = self._new_goal(self.g0[idx], off[ok])
            todo[idx] = False

    def step(self, act):
        """act: (B, 2) [verb, arg] tokens for turn self.t. Returns (next_obs or None, reward, done)."""
        verb, arg = act[:, 0], act[:, 1]
        self.mem = np.where(verb == TOK["WRITE"], arg, self.mem)
        self.t += 1
        if self.t == self.H:  # the action just taken answered the final question
            return None, (arg == self.g).astype(np.float32), True
        reward = -self.cost * ((verb == TOK["ASK"]) | (verb == TOK["ANSWER"])).astype(np.float32)
        # draw every random number each step so rng consumption does not depend on the actions
        u_ask, u_mate = self.rng.random(self.B), self.rng.random(self.B)
        chat = self.rng.choice(CHAT_IDS, self.B)
        if self.t == self.H - 1:
            return self._msg(TOK["<user>"], TOK["QUESTION"], TOK["NONE"]), reward, False

        info = self.t - 1 < T_INFO  # replying to an action taken inside the info phase
        lookup, ask = verb == TOK["LOOKUP"], verb == TOK["ASK"]
        mate = info & ~lookup & ~ask & (u_mate < self.p_mate)
        told = info & ask & (u_ask < self.p_ask)
        shift = self.has_shift & (self.t == self.t_shift)
        self.g = np.where(shift, self.g_new, self.g)
        actor = np.select([shift, lookup, mate], [TOK["<user>"], TOK["<tool>"], TOK["<mate>"]], TOK["<user>"])
        word = np.select(
            [shift, lookup & info, lookup, mate, told, ask],
            [TOK["NOW"], TOK["RESULT"], TOK["ERROR"], TOK["SAYS"], TOK["IS"], TOK["UNKNOWN"]],
            TOK["CHAT"],
        )
        arg_out = np.select(
            [shift, lookup & info, lookup, mate | told, ask],
            [self.g, self.g, TOK["NONE"], self.g, TOK["NONE"]],
            chat,
        )
        return self._msg(actor, word, arg_out), reward, False
