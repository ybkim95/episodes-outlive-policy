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
    def __init__(self, goals, H, rng, p_ask=0.5, p_mate=0.3, p_shift=0.5, cost=0.01):
        assert H >= T_INFO + 2, "need room for the info phase and the final question"
        self.g0 = GOAL_IDS[np.asarray(goals)]
        self.B, self.H, self.rng = len(self.g0), H, rng
        self.p_ask, self.p_mate, self.p_shift, self.cost = p_ask, p_mate, p_shift, cost

    def _msg(self, actor, word, arg):
        return np.stack([np.broadcast_to(x, (self.B,)) for x in (actor, word, arg)], 1)

    def reset(self):
        self.t, self.g = 0, self.g0.copy()
        self.mem = np.full(self.B, TOK["NONE"])
        # the shift lands after the info phase and before the final question
        lo, hi = max(T_INFO + 1, self.H // 4), min(self.H - 1, max(T_INFO + 2, 3 * self.H // 4))
        self.t_shift = self.rng.integers(lo, max(hi, lo + 1), self.B)
        self.has_shift = (self.rng.random(self.B) < self.p_shift) & (lo < hi)
        gi = np.searchsorted(GOAL_IDS, self.g0)
        self.g_new = GOAL_IDS[(gi + self.rng.integers(1, N_GOALS, self.B)) % N_GOALS]  # always a different goal
        return self._msg(TOK["<user>"], TOK["START"], TOK["NONE"])

    def fork(self, k):
        """Replicate every episode k times (rows stay grouped: episode i -> rows i*k .. i*k+k-1).
        Copies share the full state, including the pending goal shift; later randomness diverges."""
        for name in ("g0", "g", "mem", "t_shift", "has_shift", "g_new"):
            setattr(self, name, np.repeat(getattr(self, name), k))
        self.B *= k

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
