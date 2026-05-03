"""
BaccaratGameEngine
==================
State machine that detects baccarat game events from per-frame card detections.

Baccarat rules implemented
--------------------------
  Card values : 1→1, 2-9→face, 10/J/Q/K→0.  Total = sum mod 10.
  Natural     : Either side reaches 8 or 9 from first two cards → no more draws.
  Player rule : total 0-5 → draw 3rd card;  6-7 → stand.
  Banker rule (when player drew):
      total 0-2 → always draw
      total 3   → draw unless player's 3rd card = 8
      total 4   → draw if player's 3rd card ∈ {2,3,4,5,6,7}
      total 5   → draw if player's 3rd card ∈ {4,5,6,7}
      total 6   → draw if player's 3rd card ∈ {6,7}
      total 7   → always stand
  Banker rule (when player stood):
      total 0-5 → draw;  6-7 → stand.
  Winner: closest to 9 wins; equal → TIE.

Events emitted (GameEvent.event strings)
-----------------------------------------
  GAME_START              first card appears on table
  CARD_COMMITTED          one card position stably classified
  INITIAL_TOTALS          both sides have 2 committed cards
  NATURAL                 8 or 9 after 2 cards → game ends immediately
  PLAYER_STANDS           player total 6-7, no third card
  PLAYER_DRAWS_PENDING    waiting to see player's third card
  PLAYER_DRAWS            player's third card committed
  BANKER_STANDS           banker stands
  BANKER_DRAWS_PENDING    waiting to see banker's third card
  BANKER_DRAWS            banker's third card committed
  GAME_END                winner determined (PLAYER / BANKER / TIE)
  GAME_RESET              table cleared, engine ready for next round

Usage
-----
    engine = BaccaratGameEngine(fps=15)

    # inside the per-frame loop:
    events = engine.update(player_labels, banker_labels, frame_number=fn)
    for ev in events:
        print(ev)          # or log / overlay on video

    # read current state at any time:
    engine.state           # e.g. "DEALING"
    engine.player_total()  # e.g. 7
    engine.banker_total()  # e.g. 5
    engine.hud_lines()     # list[str] ready to draw on the video frame
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Card value helpers
# ---------------------------------------------------------------------------

_RANK_VALUE = {
    '1': 1,                      # Ace
    '2': 2, '3': 3, '4': 4, '5': 5,
    '6': 6, '7': 7, '8': 8, '9': 9,
    '10': 0, 'j': 0, 'q': 0, 'k': 0,
}


def card_value(label: Optional[str]) -> int:
    """
    Return baccarat point value (0-9) for a card label like '7 heart',
    '10 clubs', 'J spade', '1 diamo'.
    Returns -1 for 'Back' (face-down) or None.
    """
    if not label or label.strip().lower() == 'back':
        return -1
    rank = label.strip().lower().split()[0]
    return _RANK_VALUE.get(rank, -1)


def hand_total(labels: List[Optional[str]]) -> int:
    """Baccarat hand total (mod 10), ignoring unknown cards."""
    return sum(v for v in (card_value(l) for l in labels if l) if v >= 0) % 10


# ---------------------------------------------------------------------------
# Game event
# ---------------------------------------------------------------------------

@dataclass
class GameEvent:
    event:     str
    frame:     int   = 0
    timestamp: float = field(default_factory=time.time)
    data:      dict  = field(default_factory=dict)

    def __str__(self):
        kv = "  ".join(f"{k}={v}" for k, v in self.data.items())
        return f"[{self.event}]  frame={self.frame}  {kv}"


# ---------------------------------------------------------------------------
# Card slot — stabilises per-frame noisy detections
# ---------------------------------------------------------------------------

class CardSlot:
    """
    Accumulates per-frame card classifications for one positional slot
    (e.g. "player card 1").  Commits once COMMIT_FRAMES consecutive frames
    agree on the same label.  Un-commits after CLEAR_FRAMES empty frames.
    """
    COMMIT_FRAMES = 5
    CLEAR_FRAMES  = 30

    def __init__(self):
        self._window:       List[str] = []
        self.committed:     Optional[str] = None
        self._empty:        int = 0
        self._just_committed = False

    def update(self, label: Optional[str]) -> bool:
        """
        Feed one frame's observation.
        Returns True whenever a label is committed or re-committed to a
        different card (corrects wrong early commits).
        """
        self._just_committed = False

        if label is None:
            self._empty += 1
            if self._empty >= self.CLEAR_FRAMES and self.committed:
                self.committed = None
                self._window.clear()
            return False

        self._empty = 0
        self._window.append(label)
        keep = self.COMMIT_FRAMES * 2
        if len(self._window) > keep:
            self._window = self._window[-keep:]

        # Commit (or re-commit) if the last COMMIT_FRAMES frames all agree
        # on the same label.  Re-commitment corrects slots that were locked
        # to a wrong card from an early noisy detection.
        recent = self._window[-self.COMMIT_FRAMES:]
        if len(recent) == self.COMMIT_FRAMES and len(set(recent)) == 1:
            candidate = recent[0]
            if candidate != self.committed:
                self.committed = candidate
                self._just_committed = True
                return True
        return False

    @property
    def just_committed(self) -> bool:
        return self._just_committed

    def reset(self):
        self._window.clear()
        self.committed     = None
        self._empty        = 0
        self._just_committed = False


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class BaccaratGameEngine:
    """
    Per-frame baccarat game state machine.

    Parameters
    ----------
    fps : int
        Video frame rate — used to scale RESET_FRAMES.
    commit_frames : int
        Consecutive identical classifications needed to commit a card (default 5).
    """

    # State names
    IDLE        = "IDLE"
    DEALING     = "DEALING"
    EVAL        = "EVALUATING"
    P_PENDING   = "PLAYER_DRAW_PENDING"
    B_PENDING   = "BANKER_DRAW_PENDING"
    GAME_OVER   = "GAME_OVER"

    def __init__(self, fps: int = 15, commit_frames: int = 5):
        self.fps           = fps
        CardSlot.COMMIT_FRAMES = commit_frames
        # empty frames before declaring table clear after GAME_OVER
        self._reset_threshold = fps * 3   # 3 seconds
        self._init_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        player_labels: List[Optional[str]],
        banker_labels: List[Optional[str]],
        frame_number: int = 0,
    ) -> List[GameEvent]:
        """
        Process one video frame.

        Parameters
        ----------
        player_labels : up to 3 card labels for Player side, sorted left→right.
                        Use None for a slot with no visible card.
        banker_labels : same for Banker side.
        frame_number  : current video frame index.

        Returns
        -------
        List[GameEvent]  — may be empty when nothing changes.
        """
        self._fn = frame_number
        events: List[GameEvent] = []

        # "Back" = face-down card in the shoe — not a dealt card, treat as absent
        def _strip_back(labels):
            return [None if (l and l.strip().lower() == 'back') else l
                    for l in labels]

        player_labels = _strip_back(player_labels)
        banker_labels = _strip_back(banker_labels)

        total_visible = sum(
            1 for l in player_labels + banker_labels if l is not None
        )

        # ── Table-clear / reset detection ────────────────────────────────
        if total_visible == 0:
            self._empty_streak += 1
            if (self._state == self.GAME_OVER
                    and self._empty_streak >= self._reset_threshold):
                events.append(self._ev("GAME_RESET"))
                self._init_state()
            return events
        self._empty_streak = 0

        # ── Update card slots (pad to 3 each) ────────────────────────────
        p = (list(player_labels) + [None, None, None])[:3]
        b = (list(banker_labels) + [None, None, None])[:3]

        for i, (slot, lbl) in enumerate(zip(self._p_slots, p)):
            if slot.update(lbl) and slot.committed:
                events.append(self._ev("CARD_COMMITTED", {
                    "side": "player", "index": i + 1,
                    "label": slot.committed,
                    "value": card_value(slot.committed),
                }))

        for i, (slot, lbl) in enumerate(zip(self._b_slots, b)):
            if slot.update(lbl) and slot.committed:
                events.append(self._ev("CARD_COMMITTED", {
                    "side": "banker", "index": i + 1,
                    "label": slot.committed,
                    "value": card_value(slot.committed),
                }))

        # ── IDLE → DEALING ───────────────────────────────────────────────
        if self._state == self.IDLE and total_visible > 0:
            self._state = self.DEALING
            events.append(self._ev("GAME_START"))

        # ── shorthand for current committed cards ────────────────────────
        pc = [s.committed for s in self._p_slots]  # [p1, p2, p3]
        bc = [s.committed for s in self._b_slots]  # [b1, b2, b3]
        p_n = sum(1 for c in pc if c)
        b_n = sum(1 for c in bc if c)

        # ── DEALING → EVALUATING (both sides have 2 cards) ───────────────
        if self._state == self.DEALING and p_n >= 2 and b_n >= 2:
            self._state = self.EVAL
            events += self._initial_eval(pc[:2], bc[:2])

        # ── PLAYER_DRAW_PENDING → saw 3rd player card ────────────────────
        elif self._state == self.P_PENDING and p_n >= 3 and pc[2]:
            p3v = card_value(pc[2])
            events.append(self._ev("PLAYER_DRAWS", {
                "card": pc[2], "value": p3v,
                "player_total": hand_total([c for c in pc if c]),
            }))
            self._state = self.B_PENDING
            events += self._banker_decision(bc[:2], p3v, player_stood=False, pc=pc)

        # ── BANKER_DRAW_PENDING → saw 3rd banker card ────────────────────
        elif self._state == self.B_PENDING and b_n >= 3 and bc[2]:
            events.append(self._ev("BANKER_DRAWS", {
                "card": bc[2], "value": card_value(bc[2]),
                "banker_total": hand_total([c for c in bc if c]),
            }))
            events += self._winner(
                [c for c in pc if c], [c for c in bc if c])
            self._state = self.GAME_OVER

        return events

    # ── Read-only state ──────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    def player_cards(self) -> List[Optional[str]]:
        return [s.committed for s in self._p_slots]

    def banker_cards(self) -> List[Optional[str]]:
        return [s.committed for s in self._b_slots]

    def player_total(self) -> int:
        return hand_total([c for c in self.player_cards() if c])

    def banker_total(self) -> int:
        return hand_total([c for c in self.banker_cards() if c])

    def hud_lines(self) -> List[str]:
        """Return ready-to-draw HUD text lines for the video overlay."""
        pc = [c for c in self.player_cards() if c]
        bc = [c for c in self.banker_cards() if c]
        lines = [
            f"State : {self._state}",
            f"Player: {', '.join(pc) if pc else '---'}  total={self.player_total()}",
            f"Banker: {', '.join(bc) if bc else '---'}  total={self.banker_total()}",
        ]
        if self._last_winner:
            lines.append(f">>> {self._last_winner} WINS <<<")
        return lines

    # ------------------------------------------------------------------
    # Baccarat rule engine
    # ------------------------------------------------------------------

    def _initial_eval(self, p2: list, b2: list) -> List[GameEvent]:
        events = []
        pt = hand_total([c for c in p2 if c])
        bt = hand_total([c for c in b2 if c])

        events.append(self._ev("INITIAL_TOTALS", {
            "player": f"{[c for c in p2 if c]}={pt}",
            "banker": f"{[c for c in b2 if c]}={bt}",
        }))

        # Natural
        if pt >= 8 or bt >= 8:
            events.append(self._ev("NATURAL", {
                "player_total": pt, "banker_total": bt,
            }))
            events += self._winner([c for c in p2 if c], [c for c in b2 if c])
            self._state = self.GAME_OVER
            return events

        # Player drawing rule
        if pt <= 5:
            events.append(self._ev("PLAYER_DRAWS_PENDING", {
                "player_total": pt,
                "reason": f"total {pt} ≤ 5 → must draw",
            }))
            self._state = self.P_PENDING
        else:
            events.append(self._ev("PLAYER_STANDS", {
                "player_total": pt,
                "reason": f"total {pt} ≥ 6 → stands",
            }))
            pc_full = [s.committed for s in self._p_slots]
            events += self._banker_decision(
                b2, player_third_value=None, player_stood=True, pc=pc_full)

        return events

    def _banker_decision(
        self,
        b2: list,
        player_third_value: Optional[int],
        player_stood: bool,
        pc: list,
    ) -> List[GameEvent]:
        events = []
        bt = hand_total([c for c in b2 if c])

        if player_stood:
            draw = bt <= 5
            reason = (f"Player stood; Banker {bt} "
                      + ("≤ 5 → draws" if draw else "≥ 6 → stands"))
        else:
            p3v = player_third_value
            if bt <= 2:
                draw, reason = True, f"Banker {bt} ≤ 2 → always draws"
            elif bt == 3:
                draw   = (p3v != 8)
                reason = f"Banker 3; player 3rd={p3v} → {'draws' if draw else 'stands (p3=8)'}"
            elif bt == 4:
                draw   = p3v in (2, 3, 4, 5, 6, 7)
                reason = f"Banker 4; player 3rd={p3v} → {'draws' if draw else 'stands'}"
            elif bt == 5:
                draw   = p3v in (4, 5, 6, 7)
                reason = f"Banker 5; player 3rd={p3v} → {'draws' if draw else 'stands'}"
            elif bt == 6:
                draw   = p3v in (6, 7)
                reason = f"Banker 6; player 3rd={p3v} → {'draws' if draw else 'stands'}"
            else:  # bt == 7
                draw, reason = False, "Banker 7 → always stands"

        if draw:
            events.append(self._ev("BANKER_DRAWS_PENDING", {
                "banker_total": bt, "reason": reason,
            }))
            self._state = self.B_PENDING
        else:
            events.append(self._ev("BANKER_STANDS", {
                "banker_total": bt, "reason": reason,
            }))
            events += self._winner([c for c in pc if c], [c for c in b2 if c])
            self._state = self.GAME_OVER

        return events

    def _winner(self, p_cards: list, b_cards: list) -> List[GameEvent]:
        pt = hand_total(p_cards)
        bt = hand_total(b_cards)
        if pt > bt:
            winner = "PLAYER"
        elif bt > pt:
            winner = "BANKER"
        else:
            winner = "TIE"
        self._last_winner = winner
        return [self._ev("GAME_END", {
            "winner":       winner,
            "player_total": pt,
            "banker_total": bt,
            "player_cards": p_cards,
            "banker_cards": b_cards,
        })]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ev(self, name: str, data: dict = None) -> GameEvent:
        return GameEvent(event=name, frame=self._fn,
                         timestamp=time.time(), data=data or {})

    def _init_state(self):
        self._state        = self.IDLE
        self._p_slots      = [CardSlot() for _ in range(3)]
        self._b_slots      = [CardSlot() for _ in range(3)]
        self._fn           = 0
        self._empty_streak = 0
        self._last_winner  = ""
