"""Shellback: a hermit crab carries a letter along the sea floor.

Urchins sit on the sand and must be jumped. Jellyfish drift at shell height and must be
ducked. Everything here is deterministic given the seed, and needs Pillow and no display.
The world advances only when `step` is called, so a slow player costs nothing.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from PIL import Image, ImageDraw

W, H = 480, 270
GROUND = 218                # the sand line, in pixels from the top
CRAB_X = 80
CRAB_W, CRAB_H, DUCK_H = 54, 52, 26
GRAVITY, JUMP_V = 0.9, 15.0
SPEED = 4.0                 # pixels per tick, constant: one frame shows everything
TICKS_PER_ACTION = 4        # the player acts once every this many ticks
TICKS_PER_SECOND = 30
MAX_TICKS = 3000            # a run that lasts this long ends, undefeated
ACTIONS = ("jump", "duck", "stay")


@dataclass
class Obstacle:
    kind: str               # "urchin" or "jelly"
    x: float
    wobble: float = 0.0

    @property
    def box(self) -> tuple[float, float, float, float]:
        if self.kind == "urchin":
            return (self.x + 9, GROUND - 34, self.x + 39, GROUND)
        return (self.x + 4, GROUND - 92, self.x + 52, GROUND - 32)   # the bell and its tentacles

    @property
    def right(self) -> float:
        return self.x + (48 if self.kind == "urchin" else 56)


@dataclass
class Shellback:
    seed: int
    tick: int = 0
    crab_y: float = 0.0     # height of the feet above the sand
    crab_v: float = 0.0
    ducking: bool = False
    alive: bool = True
    cleared: int = 0
    obstacles: list[Obstacle] = field(default_factory=list)

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        scene = random.Random(self.seed ^ 0x5eed)
        self.weeds = [(scene.randrange(0, W * 2), scene.randrange(26, 70), scene.random())
                      for _ in range(9)]
        self.pebbles = [(scene.randrange(0, W * 2), scene.randrange(GROUND + 8, H - 4),
                         scene.randrange(2, 5)) for _ in range(26)]
        self.bubbles = [(scene.randrange(0, W), scene.randrange(10, GROUND - 90),
                         scene.randrange(2, 5)) for _ in range(12)]
        self.next_spawn = self.rng.randrange(300, 900)

    @property
    def score(self) -> int:
        return self.tick // TICKS_PER_ACTION

    @property
    def over(self) -> bool:
        return not self.alive or self.tick >= MAX_TICKS

    def crab_box(self) -> tuple[float, float, float, float]:
        h = DUCK_H if self.ducking and self.crab_y == 0 else CRAB_H
        return (CRAB_X + 12, GROUND - self.crab_y - h, CRAB_X + CRAB_W - 8, GROUND - self.crab_y)

    def _spawn(self) -> None:
        kind = "urchin" if self.rng.random() < 0.6 else "jelly"
        self.obstacles.append(Obstacle(kind, W + 8.0, self.rng.random() * 6.28))
        # Far enough apart that a landing leaves time for the next move.
        self.next_spawn = self.rng.randrange(260, 480)

    def _tick(self, action: str) -> None:
        on_sand = self.crab_y == 0
        if action == "jump" and on_sand:
            self.crab_v = JUMP_V
            self.crab_y = 0.001
        self.ducking = action == "duck" and on_sand
        if self.crab_y > 0:
            self.crab_y = max(0.0, self.crab_y + self.crab_v)
            self.crab_v -= GRAVITY
            if self.crab_y == 0:
                self.crab_v = 0.0
        self.tick += 1
        self.next_spawn -= SPEED
        if self.next_spawn <= 0:
            self._spawn()
        for o in self.obstacles:
            o.x -= SPEED
        kept = []
        for o in self.obstacles:
            if o.right < CRAB_X:
                self.cleared += 1
            else:
                kept.append(o)
        self.obstacles = kept
        a = self.crab_box()
        for o in self.obstacles:
            b = o.box
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                self.alive = False

    def step(self, action: str) -> None:
        """One decision: the action holds for TICKS_PER_ACTION ticks."""
        if action not in ACTIONS:
            raise ValueError(action)
        for _ in range(TICKS_PER_ACTION):
            if not self.over:
                self._tick(action)

    # ---- drawing: everything the player gets --------------------------------------

    def render(self) -> Image.Image:
        im = Image.new("RGB", (W, H), (9, 58, 84))
        d = ImageDraw.Draw(im)
        for y in range(0, GROUND, 6):                       # light falling off with depth
            t = y / GROUND
            d.rectangle((0, y, W, y + 6), fill=(int(18 - 8 * t), int(112 - 58 * t), int(136 - 52 * t)))
        for x, y, r in self.bubbles:
            bx = (x - self.tick * 1) % W
            d.ellipse((bx - r, y - r, bx + r, y + r), outline=(120, 196, 208))
        offset = self.tick * SPEED
        for x, h, phase in self.weeds:
            wx = (x - offset * 0.5) % (W * 2) - 30
            top = GROUND - h
            sway = 6 * math.sin(self.tick / 14 + phase * 6)
            d.line((wx, GROUND, wx + sway, top), fill=(21, 92, 74), width=5)
        d.rectangle((0, GROUND, W, H), fill=(214, 186, 132))
        for x, y, r in self.pebbles:
            px = (x - offset) % (W * 2) - 20
            d.ellipse((px - r, y - r / 2, px + r, y + r / 2), fill=(176, 146, 100))
        for o in self.obstacles:
            (self._urchin if o.kind == "urchin" else self._jelly)(d, o)
        self._crab(d)
        return im

    def _urchin(self, d: ImageDraw.ImageDraw, o: Obstacle) -> None:
        cx, cy = o.x + 24, GROUND - 18
        for i in range(14):
            a = math.pi + i * math.pi / 13
            d.line((cx, cy, cx + 24 * math.cos(a), cy + 24 * math.sin(a)), fill=(38, 12, 54), width=3)
        d.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), fill=(92, 24, 110))
        d.ellipse((cx - 8, cy - 11, cx, cy - 4), fill=(164, 84, 180))

    def _jelly(self, d: ImageDraw.ImageDraw, o: Obstacle) -> None:
        x0, top = o.x + 4, GROUND - 92 + 3 * math.sin(self.tick / 7 + o.wobble)
        d.pieslice((x0, top, x0 + 48, top + 62), 180, 360, fill=(255, 122, 176))
        for i in range(5):
            tx = x0 + 6 + i * 9
            d.line((tx, top + 31, tx + 4 * math.sin(self.tick / 5 + i), top + 60),
                   fill=(255, 176, 208), width=3)
        d.ellipse((x0 + 11, top + 12, x0 + 19, top + 20), fill=(255, 255, 255))
        d.ellipse((x0 + 29, top + 12, x0 + 37, top + 20), fill=(255, 255, 255))

    def _crab(self, d: ImageDraw.ImageDraw) -> None:
        x, foot = CRAB_X, GROUND - self.crab_y
        tucked = self.ducking and self.crab_y == 0
        h = DUCK_H if tucked else CRAB_H
        top = foot - h
        for i in range(3):                                  # legs
            lx = x + 8 + i * 9
            d.line((lx, foot - 12, lx - 6, foot), fill=(196, 58, 30), width=3)
        d.pieslice((x + 2, top, x + 42, top + h * 1.5), 180, 360, fill=(244, 226, 196))   # the shell
        for i in range(3):
            d.arc((x + 8 + i * 5, top + 6 + i * 3, x + 36 - i * 5, top + h), 200, 340,
                  fill=(186, 128, 92), width=2)
        d.ellipse((x + 30, foot - 24, x + 52, foot - 6), fill=(228, 76, 40))              # the body
        d.rectangle((x + 8, foot - 24, x + 20, foot - 14), fill=(255, 244, 150))          # the letter
        if not tucked:
            d.line((x + 44, foot - 22, x + 46, foot - 34), fill=(196, 58, 30), width=2)
            d.ellipse((x + 42, foot - 40, x + 50, foot - 32), fill=(20, 20, 30))
