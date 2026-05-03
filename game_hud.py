"""
GameHUD
=======
Video-game style overlay for baccarat event display.

Design
------
  * ONE centred message at a time — appears for a fixed duration then disappears
  * Small persistent info panel — bottom-right corner (state + hand totals)

Centre messages
---------------
  GAME START           white   2 s   (every new game)
  NATURAL!             gold    3 s
  PLAYER STANDS        cyan    1.5 s
  PLAYER DRAWS 3rd     cyan    2 s
  BANKER STANDS        orange  1.5 s
  BANKER DRAWS 3rd     orange  2 s
  PLAYER WINS! X–Y     blue    4 s
  BANKER WINS! X–Y     red     4 s
  TIE!  X–X            gold    4 s
  GAME OVER            white   2 s   (lingers until table clears)

Usage
-----
    hud = GameHUD(width=960, height=608, fps=15)

    for ev in game_engine_events:
        hud.add_event(ev)
    hud.update_from_engine(engine)   # sync info panel each frame
    hud.draw(frame, frame_number)
"""

import cv2
import numpy as np
from typing import Optional, Tuple


# ── Colours (BGR) ────────────────────────────────────────────────────────────
WHITE  = (255, 255, 255)
BLACK  = (  0,   0,   0)
GREEN  = ( 80, 220,  80)
GOLD   = ( 30, 215, 255)
CYAN   = (220, 220,   0)
BLUE   = (220, 120,   0)
ORANGE = (  0, 150, 255)
RED    = (  0,  60, 220)

_WINNER_COL = {"PLAYER": BLUE, "BANKER": RED, "TIE": GOLD}

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blend_rect(frame, x1, y1, x2, y2, bgr=(20, 20, 20), alpha=0.60):
    """Draw a semi-transparent filled rectangle."""
    roi = frame[y1:y2, x1:x2]
    fill = np.full_like(roi, bgr)
    cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi


def _shadow_text(frame, text, x, y, scale, colour, thickness=1):
    """Text with a 1-px dark shadow for readability on any background."""
    cv2.putText(frame, text, (x + 1, y + 1), FONT, scale,
                BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale,
                colour, thickness, cv2.LINE_AA)


def _text_wh(text, scale, thickness=1):
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h


# ── Active message ─────────────────────────────────────────────────────────────

class _Msg:
    def __init__(self, line1, line2, colour, expire_frame, big):
        self.line1   = line1
        self.line2   = line2       # optional second line (e.g. score)
        self.colour  = colour
        self.expire  = expire_frame
        self.big     = big         # True → large scale, False → medium

    def alive(self, fn):
        return fn < self.expire


# ── Main HUD class ────────────────────────────────────────────────────────────

class GameHUD:
    """
    Parameters
    ----------
    width, height : video frame dimensions in pixels
    fps           : video frame rate (used to convert seconds to frames)
    """

    PANEL_W      = 300    # width of info panel
    PANEL_MARGIN = 10     # px from frame edge

    def __init__(self, width: int, height: int, fps: int = 15):
        self.W   = width
        self.H   = height
        self.fps = fps

        self._msg:         Optional[_Msg] = None
        self._game_over_msg: Optional[_Msg] = None   # winner stays until reset
        self._state        = "IDLE"
        self._player_info  = "---  =0"
        self._banker_info  = "---  =0"
        self._fn           = 0

    # ------------------------------------------------------------------
    # Feed events from the game engine
    # ------------------------------------------------------------------

    def add_event(self, ev):
        """
        Convert one GameEvent into a timed centre message.
        Call once per event, in order.
        """
        name = ev.event

        # ── game start ───────────────────────────────────────────────────
        if name == "GAME_START":
            self._game_over_msg = None
            self._player_info   = "---  =0"
            self._banker_info   = "---  =0"
            self._show("GAME START", "", WHITE, 2.0, big=True)

        # ── natural ──────────────────────────────────────────────────────
        elif name == "NATURAL":
            pt = ev.data.get("player_total", 0)
            bt = ev.data.get("banker_total", 0)
            self._show("NATURAL!", f"Player {pt}  Banker {bt}", GOLD, 3.0, big=True)

        # ── player decision ──────────────────────────────────────────────
        elif name == "PLAYER_DRAWS_PENDING":
            pt = ev.data.get("player_total", "?")
            self._show("PLAYER DRAWS 3rd CARD",
                       f"Player total: {pt}", CYAN, 2.0, big=False)

        elif name == "PLAYER_STANDS":
            pt = ev.data.get("player_total", "?")
            self._show("PLAYER STANDS",
                       f"Player total: {pt}", CYAN, 1.5, big=False)

        # ── banker decision ──────────────────────────────────────────────
        elif name == "BANKER_DRAWS_PENDING":
            bt = ev.data.get("banker_total", "?")
            self._show("BANKER DRAWS 3rd CARD",
                       f"Banker total: {bt}", ORANGE, 2.0, big=False)

        elif name == "BANKER_STANDS":
            bt = ev.data.get("banker_total", "?")
            self._show("BANKER STANDS",
                       f"Banker total: {bt}", ORANGE, 1.5, big=False)

        # ── game end ─────────────────────────────────────────────────────
        elif name == "GAME_END":
            winner = ev.data.get("winner", "?")
            pt     = ev.data.get("player_total", 0)
            bt     = ev.data.get("banker_total", 0)
            colour = _WINNER_COL.get(winner, WHITE)
            line1  = f"{winner} WINS!" if winner != "TIE" else "TIE!"
            line2  = f"Banker {bt}  vs  Player {pt}"
            msg    = self._show(line1, line2, colour, 4.0, big=True)
            # keep a permanent copy until reset
            self._game_over_msg = msg

        # ── game reset ───────────────────────────────────────────────────
        elif name == "GAME_RESET":
            self._game_over_msg = None
            self._msg           = None
            self._state         = "IDLE"
            self._player_info   = "---  =0"
            self._banker_info   = "---  =0"

    def update_from_engine(self, engine):
        """Sync the info panel from the live engine state (call every frame)."""
        self._state = engine.state
        # Only show card details once both sides have their initial 2 cards
        dealing_done = engine.state not in ("IDLE", "DEALING")
        if dealing_done:
            pc = [c for c in engine.player_cards() if c]
            bc = [c for c in engine.banker_cards() if c]
            if pc:
                self._player_info = f"{', '.join(pc)}  ={engine.player_total()}"
            if bc:
                self._banker_info = f"{', '.join(bc)}  ={engine.banker_total()}"

    # ------------------------------------------------------------------
    # Draw (call once per frame, after update_from_engine)
    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray, frame_number: int):
        """Render all HUD elements onto `frame` in-place."""
        self._fn = frame_number
        self._draw_info_panel(frame)
        self._draw_center(frame)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _show(self, line1: str, line2: str, colour, dur_sec: float,
              big: bool) -> _Msg:
        expire = self._fn + int(self.fps * dur_sec)
        msg = _Msg(line1, line2, colour, expire, big)
        self._msg = msg
        return msg

    # ── centre message ────────────────────────────────────────────────

    def _draw_center(self, frame):
        # Choose which message to show:
        #   - timed message while alive
        #   - winner message (permanent) once timed message expires
        msg = None
        if self._msg and self._msg.alive(self._fn):
            msg = self._msg
        elif self._game_over_msg:
            msg = self._game_over_msg

        if msg is None:
            return

        L1_SCALE_BIG  = 1.4
        L1_SCALE_MED  = 0.9
        L2_SCALE      = 0.65
        THICK_BIG     = 3
        THICK_MED     = 2
        PAD           = 18

        scale1 = L1_SCALE_BIG if msg.big else L1_SCALE_MED
        thick1 = THICK_BIG    if msg.big else THICK_MED

        w1, h1 = _text_wh(msg.line1, scale1, thick1)
        w2, h2 = (_text_wh(msg.line2, L2_SCALE, 1)
                  if msg.line2 else (0, 0))

        box_w = max(w1, w2) + PAD * 2
        box_h = h1 + (h2 + 10 if msg.line2 else 0) + PAD * 2

        cx = self.W // 2
        cy = self.H // 2

        bx1 = cx - box_w // 2
        by1 = cy - box_h // 2
        bx2 = bx1 + box_w
        by2 = by1 + box_h

        # dark semi-transparent background
        _blend_rect(frame, bx1, by1, bx2, by2, (10, 10, 10), alpha=0.70)
        # coloured border
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), msg.colour, 2)

        # line 1 — main text
        x1 = cx - w1 // 2
        y1 = by1 + PAD + h1
        _shadow_text(frame, msg.line1, x1, y1, scale1, msg.colour, thick1)

        # line 2 — sub-text
        if msg.line2:
            x2 = cx - w2 // 2
            y2 = y1 + h2 + 10
            _shadow_text(frame, msg.line2, x2, y2, L2_SCALE, WHITE, 1)

    # ── info panel (bottom-right) ─────────────────────────────────────

    def _draw_info_panel(self, frame):
        LINE_H = 20
        PAD    = 8
        lines  = [
            (f"STATE : {self._state}",   GREEN if self._state == "IDLE" else GOLD),
            (f"PLAYER: {self._player_info}", CYAN),
            (f"BANKER: {self._banker_info}", ORANGE),
        ]
        ph = len(lines) * LINE_H + PAD * 2
        pw = self.PANEL_W
        m  = self.PANEL_MARGIN

        x1 = self.W - pw - m
        y1 = self.H - ph - m
        x2 = self.W - m
        y2 = self.H - m

        _blend_rect(frame, x1, y1, x2, y2, (15, 15, 15), alpha=0.65)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)

        for i, (text, colour) in enumerate(lines):
            y = y1 + PAD + (i + 1) * LINE_H - 4
            _shadow_text(frame, text, x1 + PAD, y, 0.42, colour, 1)
