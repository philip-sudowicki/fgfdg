#!/usr/bin/env python3
"""
PAKO CHASE -- an endless low-poly police chase through a procedural city.

    python pako_chase.py                play
    python pako_chase.py --smoke-test   headless crash test (no window needed)

Controls: the mouse's horizontal distance from the screen centre sets the
steering angle (further out = sharper turn). Hold the LEFT mouse button to
handbrake / drift. Esc pauses, F3 toggles the FPS counter.

Only pygame and numpy are needed. The file is organised as:
    1. CONFIG ............ every tunable number / colour and the difficulty table
    2. MATH & RENDERER ... camera and a small software 3D pipeline: near-plane and
                           guard-band clipping, back-face culling, painter's sort,
                           flat shading from one sun, distance fog
    3. WORLD ............. chunked procedural city, colliders, line of sight
    4. ENTITIES .......... car meshes, car physics, particles, skid marks
    5. AI ................ police pursuit / navigation and the spawn director
    6. GAME STATES ....... menus, HUD, play / pause / game-over screens
    7. MAIN LOOP ......... Game object, main(), smoke test
"""

import json
import math
import os
import random
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame


# =============================================================================
# 1. CONFIG / CONSTANTS  -- tweak freely
# =============================================================================

# --- Display -----------------------------------------------------------------
SCREEN_W, SCREEN_H = 1280, 720
TARGET_FPS = 60
SHOW_FPS = True                  # F3 toggles it in game

# --- Camera ------------------------------------------------------------------
CAM_DISTANCE = 9.0               # metres behind the car
CAM_HEIGHT = 4.0                 # metres above the road
CAM_PITCH_DEG = -11.0            # negative = looking down
CAM_YAW_FOLLOW = 5.0             # how fast the camera swings in behind the car (1/s)
FOV_BASE_DEG = 70.0              # horizontal field of view when slow ...
FOV_FAST_DEG = 86.0              # ... widening to this at speed
FOV_FAST_SPEED = 70.0            # speed (m/s) at which the FOV is fully widened
NEAR_PLANE = 0.3                 # metres; polygons are clipped against this plane
VIEW_DIST = 270.0                # draw distance (m); fog reaches the sky colour here
FOG_START = 80.0                 # fog starts here (m)
GUARD_BAND = 2.0                 # off-screen polygons are clipped to this many screens

# --- Player car handling -----------------------------------------------------
STEER_DEADZONE = 0.03            # fraction of half the screen width that does nothing
STEER_SATURATE = 0.80            # mouse offset (fraction of half width) for full lock
STEER_CURVE = 1.4                # >1 gives finer control near the centre
STEER_RESPONSE = 10.0            # how fast the wheel follows the mouse (1/s)
MAX_YAW_RATE = 2.2               # turn rate at full lock when slow (rad/s)
GRIP_ACCEL = 30.0                # lateral grip (m/s^2): caps the turn rate at speed
GRIP_RATE = 7.0                  # how fast sideways sliding is killed (1/s)
SLIDE_KEEP = 0.93                # share of killed sideways speed returned as forward speed
HANDBRAKE_YAW_RATE = 2.7         # turn rate while drifting (rad/s), not capped by speed
HANDBRAKE_GRIP_RATE = 1.2        # sideways grip while drifting (low = long slides)
HANDBRAKE_SLIDE_KEEP = 0.72      # drifting scrubs more speed
HANDBRAKE_DRAG = 0.25            # fraction of speed lost per second on the handbrake
LOW_SPEED_BOOST = 9.0            # extra m/s^2 when well below the start speed (crash recovery)
MAX_SPEED_SANITY = 150.0         # physics safety net only (540 km/h)

# --- Collisions and health ---------------------------------------------------
CAR_RADIUS = 1.05                # cars collide as two circles ...
CAR_CIRCLE_OFFSET = 1.15         # ... this far ahead of / behind the car centre
WALL_RESTITUTION = 0.25
WALL_DAMAGE_PER_MS = 1.15        # health lost per m/s of impact speed into a wall ...
WALL_DAMAGE_THRESHOLD = 3.5      # ... above this impact speed
CAR_RESTITUTION = 0.35
CAR_HIT_DAMAGE_BASE = 3.0        # health lost per police ram ...
CAR_HIT_DAMAGE_PER_MS = 0.9      # ... plus this per m/s of closing speed above 2 m/s ...
CAR_HIT_DAMAGE_MAX = 25.0        # ... capped per hit
CAR_HIT_COOLDOWN = 0.7           # seconds between rams from the same police car
PIN_SPEED = 6.0                  # boxed in: touching police below this speed (m/s) ...
PIN_DAMAGE = 9.0                 # ... drains this much health per second
PLAYER_HEALTH = 100.0
POLICE_HEALTH = 70.0             # police wreck themselves on walls and each other

# --- Score -------------------------------------------------------------------
SCORE_PER_SECOND = 10.0          # score = time survived * this ...
SCORE_PER_METRE = 0.25           # ... + distance driven * this
SCORE_FILE = os.path.join(os.path.expanduser("~"), ".pako_chase_scores.json")

# --- World -------------------------------------------------------------------
WORLD_SEED = 20261019            # change for a different city
CELL = 60.0                      # chunk = one block plus the roads on its west/south side
ROAD_W = 12.0                    # two-lane road width (m)
AVENUE_W = 18.0                  # four-lane avenue width ...
AVENUE_EVERY = 4                 # ... on every 4th row / column
SIDEWALK_W = 3.0
CURB_H = 0.15
ZONE_SCALE = 6.0                 # size (in blocks) of downtown / suburb patches
LOAD_MARGIN = 1.2 * CELL         # chunks are generated this far beyond the draw distance
CHUNK_BUDGET = 3                 # max chunks generated per frame (spreads the cost)
LOD_MARKINGS = 110.0             # lane markings are skipped beyond this distance (m)
LOD_CROSSWALK = 60.0
LOD_DETAIL = 190.0               # roof details and trees
LOD_CAR_DETAIL = 90.0            # wheels and lights on cars

# --- Police ------------------------------------------------------------------
POLICE_SPAWN_MIN = 75.0          # police appear on roads in this ring around you (m) ...
POLICE_SPAWN_MAX = 135.0         # ... always out of view
POLICE_DESPAWN_DIST = 330.0      # police further away than this give up
POLICE_WRECK_TIME = 5.0          # wrecked police cars linger this long (s)
POLICE_BRAKE = 16.0              # police braking (m/s^2)
POLICE_LOS_INTERVAL = 0.25       # seconds between each car's line-of-sight checks

# --- Difficulty --------------------------------------------------------------
#   start_speed, accel ...... your speed at the start (m/s) and how fast it grows (m/s^2)
#   police_min_top .......... police top speed floor (m/s) ...
#   police_speed_ratio ...... ... or this multiple of your current speed, if higher
#   police_accel ............ police acceleration (m/s^2)
#   police_grip, police_yaw . police cornering grip (m/s^2) and max turn rate (rad/s)
#   police_lookahead ........ how many seconds ahead police predict where you'll be
#   police_ram .............. how much faster than you (m/s) police drive when ramming
#   max_police .............. most police chasing at once ...
#   start_police, ramp_every  ... starting at this many, +1 allowed every N seconds
#   first_spawn, spawn_interval  seconds before the first spawn / between spawns
#   damage_mult ............. scales all damage you take
#   regen, regen_delay ...... health regained per second after N seconds without a hit
DIFFICULTY_NAMES = ("Easy", "Normal", "Hard")
DIFFICULTY = {
    "Easy": dict(
        start_speed=15.0, accel=0.30,
        police_min_top=21.0, police_speed_ratio=0.88, police_accel=7.0,
        police_grip=16.0, police_yaw=1.35, police_lookahead=0.5, police_ram=3.0,
        max_police=2, start_police=1, ramp_every=35.0,
        first_spawn=7.0, spawn_interval=12.0,
        damage_mult=0.55, regen=3.0, regen_delay=5.0,
        blurb="Gentle acceleration, slow police, up to 2 chasers, health regenerates.",
    ),
    "Normal": dict(
        start_speed=17.0, accel=0.45,
        police_min_top=27.0, police_speed_ratio=1.0, police_accel=10.0,
        police_grip=22.0, police_yaw=1.8, police_lookahead=1.0, police_ram=5.0,
        max_police=4, start_police=1, ramp_every=18.0,
        first_spawn=4.0, spawn_interval=7.0,
        damage_mult=1.0, regen=1.0, regen_delay=8.0,
        blurb="Police keep pace with you. Up to 4 chasers.",
    ),
    "Hard": dict(
        start_speed=20.0, accel=0.65,
        police_min_top=29.0, police_speed_ratio=1.12, police_accel=14.0,
        police_grip=30.0, police_yaw=2.3, police_lookahead=1.5, police_ram=7.5,
        max_police=7, start_police=1, ramp_every=12.0,
        first_spawn=4.0, spawn_interval=4.5,
        damage_mult=1.2, regen=0.0, regen_delay=0.0,
        blurb="Fast, sharp-turning police that out-think you. Up to 7. No healing.",
    ),
}

# --- Colours -----------------------------------------------------------------
SKY_TOP = (86, 138, 214)
SKY_HORIZON = (228, 208, 190)    # also the fog colour
FOG_COLOR = SKY_HORIZON
LIGHT_DIR = (-0.62, 0.70, -0.30) # direction towards the sun
AMBIENT = 0.34
DIFFUSE = 0.72

COL_ROAD = (64, 66, 74)
COL_INTERSECTION = (70, 72, 80)
COL_SIDEWALK = (158, 154, 146)
COL_CURB = (180, 176, 168)
COL_LINE_YELLOW = (236, 188, 52)
COL_LINE_WHITE = (228, 228, 224)
COL_GRASS = (94, 152, 76)
COL_PLAZA = (188, 174, 152)
COL_WATER = (72, 146, 196)
COL_SHADOW = (36, 36, 42)
COL_SKID = (44, 44, 48)
COL_TRUNK = (112, 80, 54)
PAL_CITY = [(196, 178, 152), (170, 160, 150), (186, 120, 96), (150, 96, 80), (214, 200, 176),
            (128, 132, 140), (204, 168, 120), (162, 176, 168), (230, 222, 206), (120, 110, 104)]
PAL_GLASS = [(96, 140, 176), (70, 110, 150), (130, 170, 190), (80, 126, 132), (150, 160, 178),
             (110, 120, 150)]
PAL_HOUSE = [(236, 214, 170), (214, 150, 120), (176, 196, 210), (226, 226, 214), (196, 210, 160),
             (240, 190, 150), (200, 170, 200)]
PAL_ROOF = [(92, 92, 98), (110, 104, 100), (80, 84, 92), (126, 120, 112)]
PAL_HOUSE_ROOF = [(150, 70, 56), (110, 64, 50), (80, 84, 96), (160, 104, 66)]
PAL_TREE = [(70, 140, 70), (56, 120, 64), (96, 156, 70), (60, 132, 90)]

COL_PLAYER_BODY = (255, 152, 28)
COL_PLAYER_ROOF = (246, 128, 18)
COL_POLICE_BODY = (36, 40, 56)
COL_POLICE_ROOF = (238, 240, 244)
COL_GLASS = (52, 70, 94)
COL_TYRE = (28, 28, 32)
COL_TAIL = (255, 44, 44)
COL_HEAD = (255, 248, 210)
COL_SIREN_RED = (255, 36, 36)
COL_SIREN_BLUE = (40, 110, 255)

UI_ACCENT = (255, 172, 40)
UI_TEXT = (245, 245, 245)
UI_DIM = (172, 174, 186)
UI_PANEL = (16, 18, 26)
UI_DANGER = (236, 60, 60)
UI_OK = (92, 212, 112)


# =============================================================================
# 2. MATH & RENDERER
# =============================================================================
# Coordinates: x = east, y = up, z = north (metres). A heading of 0 faces +z and
# grows clockwise seen from above, so forward = (sin h, cos h), right = (cos h, -sin h).

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def lerp(a, b, t):
    return a + (b - a) * t


def wrap_angle(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % math.tau - math.pi


def ease(rate, dt):
    """Frame-rate independent smoothing: the fraction to move towards a target."""
    return 1.0 - math.exp(-rate * dt)


def hash_ints(*vals):
    """Deterministic 32-bit hash of integers (stable across runs, unlike hash())."""
    h = 0x811C9DC5
    for v in vals:
        h = ((h ^ (v & 0xFFFFFFFF)) * 0x01000193) & 0xFFFFFFFF
        h ^= h >> 15
        h = (h * 0x2C1B3C6D) & 0xFFFFFFFF
        h ^= h >> 12
    return h


_SUN = np.array(LIGHT_DIR, dtype=np.float64)
_SUN /= np.linalg.norm(_SUN)


def shade(colors, normals):
    """Flat shading from one directional light (wrapped Lambert, so no face goes black)."""
    lam = np.asarray(normals, dtype=np.float64) @ _SUN
    k = AMBIENT + DIFFUSE * (0.5 + 0.5 * lam)
    return np.minimum(np.asarray(colors, dtype=np.float64) * k[:, None], 255.0).astype(np.float32)


class MeshBuilder:
    """Collects quads (a triangle is a quad with its last vertex repeated) plus per-face data.

    Per-face attributes: lod = max draw distance, layer = ground draw order,
    intra = draw order inside one object, depth/roof0/roof1 = roof-detail ordering,
    fp = footprint (x0, x1, z0, z1) of the owning object (for the depth sort),
    tie = tiny unique sort offset per object, kind = emissive / siren flags on cars.
    """

    def __init__(self):
        self.q, self.n, self.c = [], [], []
        self.lod, self.layer, self.intra, self.kind = [], [], [], []
        self.depth, self.roof0, self.roof1 = [], [], []
        self.fp, self.tie = [], []
        self.group_fp = (0.0, 0.0, 0.0, 0.0)
        self.group_tie = 0.0

    def group(self, fp, tie=0.0):
        """Start a new depth-sorted object with footprint fp = (x0, x1, z0, z1)."""
        self.group_fp = fp
        self.group_tie = tie

    def add(self, verts, color, normal=None, center=None, lod=1e9, layer=0, intra=0,
            depth=0, roof0=0.0, roof1=0.0, kind=0):
        if len(verts) == 3:
            verts = (verts[0], verts[1], verts[2], verts[2])
        if normal is None:
            a, b, c = verts[0], verts[1], verts[2]
            ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            nx, ny, nz = nx / ln, ny / ln, nz / ln
            if center is not None:   # orient outwards, away from the part's centre
                mx = sum(v[0] for v in verts) * 0.25 - center[0]
                my = sum(v[1] for v in verts) * 0.25 - center[1]
                mz = sum(v[2] for v in verts) * 0.25 - center[2]
                if nx * mx + ny * my + nz * mz < 0.0:
                    nx, ny, nz = -nx, -ny, -nz
            normal = (nx, ny, nz)
        self.q.append(verts)
        self.n.append(normal)
        self.c.append(color)
        self.lod.append(lod)
        self.layer.append(layer)
        self.intra.append(intra)
        self.kind.append(kind)
        self.depth.append(depth)
        self.roof0.append(roof0)
        self.roof1.append(roof1)
        self.fp.append(self.group_fp)
        self.tie.append(self.group_tie)

    @staticmethod
    def _col(color, face):
        return color.get(face, color["*"]) if isinstance(color, dict) else color

    def box(self, x0, x1, y0, y1, z0, z1, color, faces="tnsew", **kw):
        """Axis-aligned box. color may be a dict {face letter: colour, '*': default}."""
        col = self._col
        if "t" in faces:
            self.add(((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)), col(color, "t"), (0.0, 1.0, 0.0), **kw)
        if "b" in faces:
            self.add(((x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0)), col(color, "b"), (0.0, -1.0, 0.0), **kw)
        if "n" in faces:
            self.add(((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)), col(color, "n"), (0.0, 0.0, 1.0), **kw)
        if "s" in faces:
            self.add(((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)), col(color, "s"), (0.0, 0.0, -1.0), **kw)
        if "e" in faces:
            self.add(((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)), col(color, "e"), (1.0, 0.0, 0.0), **kw)
        if "w" in faces:
            self.add(((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)), col(color, "w"), (-1.0, 0.0, 0.0), **kw)

    def frustum(self, bottom, top, y0, y1, color, faces="tnsew", **kw):
        """Tapered box: bottom/top rectangles (x0, x1, z0, z1) at heights y0/y1."""
        bx0, bx1, bz0, bz1 = bottom
        tx0, tx1, tz0, tz1 = top
        cen = ((bx0 + bx1 + tx0 + tx1) * 0.25, (y0 + y1) * 0.5, (bz0 + bz1 + tz0 + tz1) * 0.25)
        col = self._col
        if "t" in faces:
            self.add(((tx0, y1, tz0), (tx1, y1, tz0), (tx1, y1, tz1), (tx0, y1, tz1)), col(color, "t"), (0.0, 1.0, 0.0), **kw)
        if "n" in faces:
            self.add(((bx0, y0, bz1), (bx1, y0, bz1), (tx1, y1, tz1), (tx0, y1, tz1)), col(color, "n"), center=cen, **kw)
        if "s" in faces:
            self.add(((bx0, y0, bz0), (tx0, y1, tz0), (tx1, y1, tz0), (bx1, y0, bz0)), col(color, "s"), center=cen, **kw)
        if "e" in faces:
            self.add(((bx1, y0, bz0), (tx1, y1, tz0), (tx1, y1, tz1), (bx1, y0, bz1)), col(color, "e"), center=cen, **kw)
        if "w" in faces:
            self.add(((bx0, y0, bz0), (bx0, y0, bz1), (tx0, y1, tz1), (tx0, y1, tz0)), col(color, "w"), center=cen, **kw)

    def rect(self, x0, x1, z0, z1, y, color, **kw):
        """Flat, upward-facing ground quad."""
        self.add(((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)), color, (0.0, 1.0, 0.0), **kw)

    def arrays(self, lit=True):
        """Freeze into numpy arrays. Static geometry is pre-lit once here."""
        if not self.q:
            return None
        f32 = np.float32
        nrm = np.array(self.n, dtype=f32)
        col = np.array(self.c, dtype=f32)
        return dict(
            q=np.array(self.q, dtype=f32), n=nrm, c=shade(col, nrm) if lit else col,
            lod=np.array(self.lod, dtype=f32), layer=np.array(self.layer, dtype=np.int16),
            intra=np.array(self.intra, dtype=f32), kind=np.array(self.kind, dtype=np.int8),
            depth=np.array(self.depth, dtype=np.int8), roof0=np.array(self.roof0, dtype=f32),
            roof1=np.array(self.roof1, dtype=f32), fp=np.array(self.fp, dtype=f32),
            tie=np.array(self.tie, dtype=f32))


class Camera:
    """Perspective camera given by position, yaw and (fixed) pitch."""

    def __init__(self):
        self.x = self.y = self.z = 0.0
        self.yaw = 0.0
        self.pitch = math.radians(CAM_PITCH_DEG)
        self.hfov = math.radians(FOV_BASE_DEG)
        self.setup(SCREEN_W, SCREEN_H)

    def setup(self, w, h):
        cyw, syw = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        self.fwd = (syw * cp, sp, cyw * cp)
        self.right = (cyw, 0.0, -syw)
        self.up = (-sp * syw, cp, -sp * cyw)
        self.R = np.array((self.right, self.up, self.fwd), dtype=np.float32)
        self.pos = np.array((self.x, self.y, self.z), dtype=np.float32)
        self.tan_h = math.tan(self.hfov * 0.5)
        self.tan_v = self.tan_h * h / w
        self.cx, self.cy = w * 0.5, h * 0.5
        self.focal = self.cx / self.tan_h
        self.horizon_y = self.cy + self.focal * math.tan(self.pitch)

    def to_camera(self, x, y, z):
        dx, dy, dz = x - self.x, y - self.y, z - self.z
        r, u, f = self.right, self.up, self.fwd
        return (dx * r[0] + dy * r[1] + dz * r[2],
                dx * u[0] + dy * u[1] + dz * u[2],
                dx * f[0] + dy * f[1] + dz * f[2])

    def project(self, x, y, z):
        """World point -> (screen x, screen y, depth), or None if behind the near plane."""
        px, py, pz = self.to_camera(x, y, z)
        if pz < NEAR_PLANE:
            return None
        return self.cx + self.focal * px / pz, self.cy - self.focal * py / pz, pz


class Renderer:
    """Draws flat-shaded polygons with pygame.draw.polygon.

    Pipeline per batch (all vectorised with numpy except the final draw loop):
    back-face cull -> distance / LOD cull -> camera transform -> trivial frustum
    reject -> project -> fog -> sort -> draw. The few faces that cross the near
    plane or reach far off-screen are clipped in camera space (Sutherland-Hodgman)
    against the near plane and a guard band, which also keeps pygame's scanline
    filler from walking thousands of off-screen rows.
    """

    def __init__(self, surface):
        self.surface = surface
        self.w, self.h = surface.get_size()
        self.fog = np.array(FOG_COLOR, dtype=np.float32)
        self.sky = self._make_sky()
        self.cam = None
        self.planes = ()
        self.faces_drawn = 0

    def _make_sky(self):
        sky = pygame.Surface((self.w, self.h * 2))
        for y in range(self.h * 2):
            t = min(1.0, y / self.h) ** 1.8
            col = [int(lerp(SKY_TOP[i], SKY_HORIZON[i], t)) for i in range(3)]
            pygame.draw.line(sky, col, (0, y), (self.w, y))
        return sky

    def begin(self, cam):
        """Set up the camera for this frame and paint the sky (the horizon tracks the FOV)."""
        cam.setup(self.w, self.h)
        self.cam = cam
        self.surface.blit(self.sky, (0, int(cam.horizon_y) - self.h))
        g = GUARD_BAND * 0.98
        gh, gv = g * cam.tan_h, g * cam.tan_v
        # Clip planes as (a, b, c, d): inside when a*x + b*y + c*z + d >= 0 (camera space).
        self.planes = ((0.0, 0.0, 1.0, -NEAR_PLANE), (1.0, 0.0, gh, 0.0), (-1.0, 0.0, gh, 0.0),
                       (0.0, 1.0, gv, 0.0), (0.0, -1.0, gv, 0.0))
        self.faces_drawn = 0

    def draw_faces(self, q, nrm, col, lod, sort_keys=None):
        """Cull, project, fog, sort and draw a batch of world-space quads.

        q (n,4,3), nrm (n,3), col (n,3) lit colours, lod (n,) max distance.
        sort_keys: tuple of (n,) arrays for np.lexsort (last key is the primary one);
        faces are drawn in ascending lexsort order.
        """
        if q is None or len(q) == 0:
            return
        cam = self.cam
        cp = cam.pos
        # 1. back faces, distance and LOD (cheap, on all faces)
        front = np.einsum("ij,ij->i", nrm, q[:, 0, :] - cp) < 0.0
        cen = q.mean(axis=1)
        dx = cen[:, 0] - cp[0]
        dz = cen[:, 2] - cp[2]
        dist = np.sqrt(dx * dx + dz * dz)
        idx = np.nonzero(front & (dist < lod) & (dist < VIEW_DIST))[0]
        if not len(idx):
            return
        # 2. camera space + trivial reject against the view frustum
        v = ((q[idx].reshape(-1, 3) - cp) @ cam.R.T).reshape(-1, 4, 3)
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
        zh = z * cam.tan_h
        zv = z * cam.tan_v
        ok = ((z.max(axis=1) > NEAR_PLANE) & ~(x > zh).all(axis=1) & ~(x < -zh).all(axis=1)
              & ~(y > zv).all(axis=1) & ~(y < -zv).all(axis=1))
        sel = np.nonzero(ok)[0]
        if not len(sel):
            return
        idx, v, x, y, z = idx[sel], v[sel], x[sel], y[sel], z[sel]
        # 3. projection (faces needing clipping are flagged and redone below)
        zs = np.maximum(z, NEAR_PLANE)
        f = cam.focal
        sx = cam.cx + f * x / zs
        sy = cam.cy - f * y / zs
        needs_clip = ((z.min(axis=1) < NEAR_PLANE)
                      | (np.abs(sx - cam.cx).max(axis=1) > GUARD_BAND * cam.cx)
                      | (np.abs(sy - cam.cy).max(axis=1) > GUARD_BAND * cam.cy))
        # 4. fog towards the horizon colour
        t = np.clip((dist[idx] - FOG_START) / (VIEW_DIST - FOG_START), 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        c = col[idx]
        c = c + (self.fog - c) * t[:, None]
        # 5. painter's order
        if sort_keys:
            order = np.lexsort(tuple(k[idx] for k in sort_keys))
        else:
            order = np.arange(len(idx))
        pts = np.stack((sx, sy), axis=-1)[order].tolist()
        cols = c[order].astype(np.uint8).tolist()
        clip_at = np.nonzero(needs_clip[order])[0]
        if len(clip_at):
            clip = self._clip_project
            for k, verts in zip(clip_at.tolist(), v[order[clip_at]].tolist()):
                pts[k] = clip(verts)
        # 6. draw
        draw = pygame.draw.polygon
        surf = self.surface
        n = 0
        for colr, p in zip(cols, pts):
            if p:
                draw(surf, colr, p)
                n += 1
        self.faces_drawn += n

    def _clip_project(self, poly):
        """Clip a camera-space polygon to the near plane and guard band, then project."""
        for a, b, c, d in self.planes:
            out = []
            prev = poly[-1]
            pd = a * prev[0] + b * prev[1] + c * prev[2] + d
            for cur in poly:
                cd = a * cur[0] + b * cur[1] + c * cur[2] + d
                if cd >= 0.0:
                    if pd < 0.0:
                        s = pd / (pd - cd)
                        out.append((prev[0] + (cur[0] - prev[0]) * s, prev[1] + (cur[1] - prev[1]) * s,
                                    prev[2] + (cur[2] - prev[2]) * s))
                    out.append(cur)
                elif pd >= 0.0:
                    s = pd / (pd - cd)
                    out.append((prev[0] + (cur[0] - prev[0]) * s, prev[1] + (cur[1] - prev[1]) * s,
                                prev[2] + (cur[2] - prev[2]) * s))
                prev, pd = cur, cd
            if len(out) < 3:
                return None
            poly = out
        cam = self.cam
        f, cx, cy = cam.focal, cam.cx, cam.cy
        return [(cx + f * p[0] / p[2], cy - f * p[1] / p[2]) for p in poly]


# =============================================================================
# 3. WORLD GENERATION
# =============================================================================
# The city is an endless grid. Chunk (cx, cz) covers [cx*CELL, (cx+1)*CELL) on both
# axes and owns a road strip along its west edge, one along its south edge, the
# intersection in its south-west corner, and a raised block (sidewalk + lots) in the
# rest. Each chunk is generated from a seed derived from its coordinates, so an area
# always looks the same when you come back to it.

def road_w_col(i):
    """Width of the north-south road along the west edge of chunk column i."""
    return AVENUE_W if i % AVENUE_EVERY == 0 else ROAD_W


def road_w_row(j):
    """Width of the east-west road along the south edge of chunk row j."""
    return AVENUE_W if (j + 2) % AVENUE_EVERY == 0 else ROAD_W


def col_center(i):
    """x of the centre line of north-south road i."""
    return i * CELL + road_w_col(i) * 0.5


def row_center(j):
    """z of the centre line of east-west road j."""
    return j * CELL + road_w_row(j) * 0.5


def nearest_col(x):
    i = math.floor(x / CELL)
    return i if abs(x - col_center(i)) <= abs(x - col_center(i + 1)) else i + 1


def nearest_row(z):
    j = math.floor(z / CELL)
    return j if abs(z - row_center(j)) <= abs(z - row_center(j + 1)) else j + 1


def on_road(x, z):
    i = math.floor(x / CELL)
    if x - i * CELL < road_w_col(i):
        return True
    j = math.floor(z / CELL)
    return z - j * CELL < road_w_row(j)


def ground_height(x, z):
    return 0.0 if on_road(x, z) else CURB_H


def rect_dist(x, z, x0, x1, z0, z1):
    """Distance from a point to an axis-aligned rectangle (0 inside)."""
    dx = x0 - x if x < x0 else (x - x1 if x > x1 else 0.0)
    dz = z0 - z if z < z0 else (z - z1 if z > z1 else 0.0)
    return math.sqrt(dx * dx + dz * dz)


def _lattice(i, j):
    return (hash_ints(i, j, WORLD_SEED ^ 0x5F3759DF) & 0xFFFF) / 65535.0


def city_density(cx, cz):
    """Smooth value noise in [0, 1]: 0 = leafy suburb, 1 = downtown towers."""
    u, v = cx / ZONE_SCALE, cz / ZONE_SCALE
    i, j = math.floor(u), math.floor(v)
    fu, fv = u - i, v - j
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    a = lerp(_lattice(i, j), _lattice(i + 1, j), fu)
    b = lerp(_lattice(i, j + 1), _lattice(i + 1, j + 1), fu)
    return clamp((lerp(a, b, fv) - 0.2) / 0.62, 0.0, 1.0)


def _split(rng, a, b, n):
    """Cut [a, b] into n pieces of random-ish width."""
    w = [rng.uniform(0.7, 1.3) for _ in range(n)]
    s = sum(w)
    out, acc = [a], a
    for wi in w[:-1]:
        acc += (b - a) * wi / s
        out.append(acc)
    out.append(b)
    return out


def _tint(col, k):
    return tuple(min(255.0, c * k) for c in col)


def _markings(g, a0, a1, c0, c1, along_z):
    """Centre line, lane dividers and crosswalks on a road strip running along one axis.
    [a0, a1] is the extent along the road, [c0, c1] across it."""
    def quad(alo, ahi, clo, chi, color, lod):
        if along_z:
            g.rect(clo, chi, alo, ahi, 0.02, color, layer=2, lod=lod)
        else:
            g.rect(alo, ahi, clo, chi, 0.02, color, layer=2, lod=lod)

    w, mid = c1 - c0, (c0 + c1) * 0.5
    if w > ROAD_W + 0.1:          # avenue: double yellow + dashed white lanes
        s0, s1 = a0 + 4.5, a1 - 4.5
        for k in range(2):
            lo, hi = lerp(s0, s1, k / 2), lerp(s0, s1, (k + 1) / 2)
            quad(lo, hi, mid - 0.36, mid - 0.16, COL_LINE_YELLOW, LOD_MARKINGS)
            quad(lo, hi, mid + 0.16, mid + 0.36, COL_LINE_YELLOW, LOD_MARKINGS)
        for lane in (c0 + w * 0.25, c0 + w * 0.75):
            a = a0 + 5.0
            while a + 3.0 <= a1 - 5.0:
                quad(a, a + 3.0, lane - 0.12, lane + 0.12, COL_LINE_WHITE, LOD_MARKINGS)
                a += 7.0
    else:                          # two-lane road: dashed yellow centre line
        a = a0 + 5.0
        while a + 3.0 <= a1 - 5.0:
            quad(a, a + 3.0, mid - 0.15, mid + 0.15, COL_LINE_YELLOW, LOD_MARKINGS)
            a += 6.0
    for alo in (a0 + 0.8, a1 - 3.8):   # crosswalks at both ends
        c = c0 + 0.9
        while c + 0.8 <= c1 - 0.9:
            quad(alo, alo + 3.0, c, c + 0.8, COL_LINE_WHITE, LOD_CROSSWALK)
            c += 1.7


def _gable(o, x0, x1, z0, z1, y, rh, roof_col, wall_col):
    """Pitched roof (a triangular prism) on top of a house."""
    kw = dict(depth=1, roof0=y)
    yt = y + rh
    if x1 - x0 >= z1 - z0:     # ridge runs east-west
        zm = (z0 + z1) * 0.5
        cen = ((x0 + x1) * 0.5, y + rh * 0.3, zm)
        o.add(((x0, y, z0), (x0, yt, zm), (x1, yt, zm), (x1, y, z0)), roof_col, center=cen, **kw)
        o.add(((x0, y, z1), (x1, y, z1), (x1, yt, zm), (x0, yt, zm)), roof_col, center=cen, **kw)
        o.add(((x0, y, z0), (x0, y, z1), (x0, yt, zm)), wall_col, center=cen, **kw)
        o.add(((x1, y, z0), (x1, yt, zm), (x1, y, z1)), wall_col, center=cen, **kw)
    else:                      # ridge runs north-south
        xm = (x0 + x1) * 0.5
        cen = (xm, y + rh * 0.3, (z0 + z1) * 0.5)
        o.add(((x0, y, z0), (x0, y, z1), (xm, yt, z1), (xm, yt, z0)), roof_col, center=cen, **kw)
        o.add(((x1, y, z0), (xm, yt, z0), (xm, yt, z1), (x1, y, z1)), roof_col, center=cen, **kw)
        o.add(((x0, y, z0), (xm, yt, z0), (x1, y, z0)), wall_col, center=cen, **kw)
        o.add(((x0, y, z1), (x1, y, z1), (xm, yt, z1)), wall_col, center=cen, **kw)


def _building(o, rng, lot, dens, colliders, bfp):
    lx0, lx1, lz0, lz1 = lot
    m = 0.4 + 1.6 * (1.0 - dens)                 # suburbs get bigger gaps (alleys!)
    fx0, fx1 = lx0 + rng.uniform(0.3, m), lx1 - rng.uniform(0.3, m)
    fz0, fz1 = lz0 + rng.uniform(0.3, m), lz1 - rng.uniform(0.3, m)
    if fx1 - fx0 < 4.0 or fz1 - fz0 < 4.0:
        return 0.0
    tower = dens > 0.6 and rng.random() < 0.45
    if tower:
        h = rng.uniform(34.0, 78.0) * (0.7 + 0.3 * dens)
        pal = PAL_GLASS
    elif dens < 0.35:
        h = rng.uniform(4.5, 8.5)
        pal = PAL_HOUSE
    else:
        h = rng.uniform(6.5, 12.0) + dens * rng.uniform(4.0, 30.0)
        pal = PAL_CITY
    wall = _tint(rng.choice(pal), rng.uniform(0.88, 1.08))
    roof = rng.choice(PAL_ROOF)
    y0 = CURB_H
    top = y0 + h
    house = dens < 0.35 and h < 11.5 and rng.random() < 0.7
    o.group((fx0, fx1, fz0, fz1), rng.random() * 1e-3)
    o.box(fx0, fx1, y0, top, fz0, fz1, {"*": wall, "t": roof}, faces="nsew" if house else "tnsew")
    colliders.append((fx0, fx1, fz0, fz1, top))
    bfp.append((fx0, fx1, fz0, fz1))
    if house:
        rh = min(fx1 - fx0, fz1 - fz0) * rng.uniform(0.28, 0.42)
        _gable(o, fx0, fx1, fz0, fz1, top, rh, rng.choice(PAL_HOUSE_ROOF), wall)
        return top + rh
    if tower and rng.random() < 0.75:            # set-back crown, maybe an antenna
        ix = (fx1 - fx0) * rng.uniform(0.14, 0.26)
        iz = (fz1 - fz0) * rng.uniform(0.14, 0.26)
        ph = rng.uniform(4.0, 10.0)
        o.box(fx0 + ix, fx1 - ix, top, top + ph, fz0 + iz, fz1 - iz, {"*": _tint(wall, 0.9), "t": roof},
              lod=LOD_DETAIL, depth=1, roof0=top)
        if rng.random() < 0.6:
            ax, az = (fx0 + fx1) * 0.5, (fz0 + fz1) * 0.5
            ah = rng.uniform(5.0, 12.0)
            o.box(ax - 0.3, ax + 0.3, top + ph, top + ph + ah, az - 0.3, az + 0.3, (205, 205, 210),
                  lod=LOD_DETAIL, depth=2, roof0=top, roof1=top + ph)
            return top + ph + ah
        return top + ph
    if rng.random() < 0.6:                       # rooftop AC unit / water tank
        sx, sz, sh = rng.uniform(1.5, 3.5), rng.uniform(1.5, 3.5), rng.uniform(1.0, 2.6)
        if fx1 - fx0 > sx + 2.0 and fz1 - fz0 > sz + 2.0:
            px = rng.uniform(fx0 + 1.0, fx1 - 1.0 - sx)
            pz = rng.uniform(fz0 + 1.0, fz1 - 1.0 - sz)
            o.box(px, px + sx, top, top + sh, pz, pz + sz, _tint(roof, 1.35),
                  lod=LOD_DETAIL, depth=1, roof0=top)
            return top + sh
    return top


def _tree(o, rng, x, z, colliders):
    trunk = rng.uniform(1.6, 2.6)
    r = rng.uniform(1.4, 2.2)
    ch = rng.uniform(2.6, 4.2)
    y0 = CURB_H
    o.group((x - r, x + r, z - r, z + r), rng.random() * 1e-3)
    o.box(x - 0.22, x + 0.22, y0, y0 + trunk + 0.3, z - 0.22, z + 0.22, COL_TRUNK, faces="nsew",
          lod=LOD_DETAIL, intra=0)
    k = r * 0.45
    o.frustum((x - r, x + r, z - r, z + r), (x - k, x + k, z - k, z + k), y0 + trunk, y0 + trunk + ch,
              _tint(rng.choice(PAL_TREE), rng.uniform(0.9, 1.1)), lod=LOD_DETAIL, intra=1)
    colliders.append((x - 0.35, x + 0.35, z - 0.35, z + 0.35, y0 + trunk + ch))
    return y0 + trunk + ch


class Chunk:
    __slots__ = ("cx", "cz", "ground", "objects", "colliders", "rect", "bfp")

    def __init__(self, cx, cz, ground, objects, colliders, rect, bfp):
        self.cx, self.cz = cx, cz
        self.ground = ground          # dict of arrays (see MeshBuilder.arrays)
        self.objects = objects        # dict of arrays or None
        self.colliders = colliders    # list of (x0, x1, z0, z1, height)
        self.rect = rect              # (x0, x1, z0, z1, max height) for culling
        self.bfp = bfp                # (n, 4) building footprints (sorting cars vs buildings)


def build_chunk(cx, cz):
    rng = random.Random(hash_ints(cx, cz, WORLD_SEED))
    wc, wr = road_w_col(cx), road_w_row(cz)
    x0, z0 = cx * CELL, cz * CELL
    x1, z1 = x0 + CELL, z0 + CELL
    bx0, bz0 = x0 + wc, z0 + wr                 # the block is [bx0, x1] x [bz0, z1]
    g = MeshBuilder()

    # Roads (layer 0). Long strips are split so the per-face fog changes smoothly.
    for k in range(3):
        za, zb = lerp(bz0, z1, k / 3), lerp(bz0, z1, (k + 1) / 3)
        g.rect(x0, bx0, za, zb, 0.0, COL_ROAD, layer=0)
        xa, xb = lerp(bx0, x1, k / 3), lerp(bx0, x1, (k + 1) / 3)
        g.rect(xa, xb, z0, bz0, 0.0, COL_ROAD, layer=0)
    g.rect(x0, bx0, z0, bz0, 0.0, COL_INTERSECTION, layer=0)

    # Raised block: sidewalk-coloured top plus curbs (layer 1).
    mx, mz = (bx0 + x1) * 0.5, (bz0 + z1) * 0.5
    for xa, xb in ((bx0, mx), (mx, x1)):
        for za, zb in ((bz0, mz), (mz, z1)):
            g.rect(xa, xb, za, zb, CURB_H, COL_SIDEWALK, layer=1)
    h = CURB_H
    ck = dict(layer=1, lod=LOD_MARKINGS)
    g.add(((bx0, 0, bz0), (bx0, 0, z1), (bx0, h, z1), (bx0, h, bz0)), COL_CURB, (-1.0, 0.0, 0.0), **ck)
    g.add(((x1, 0, bz0), (x1, h, bz0), (x1, h, z1), (x1, 0, z1)), COL_CURB, (1.0, 0.0, 0.0), **ck)
    g.add(((bx0, 0, bz0), (bx0, h, bz0), (x1, h, bz0), (x1, 0, bz0)), COL_CURB, (0.0, 0.0, -1.0), **ck)
    g.add(((bx0, 0, z1), (x1, 0, z1), (x1, h, z1), (bx0, h, z1)), COL_CURB, (0.0, 0.0, 1.0), **ck)

    # Lane markings and crosswalks (layer 2).
    _markings(g, bz0, z1, x0, bx0, along_z=True)
    _markings(g, bx0, x1, z0, bz0, along_z=False)

    # Lots: buildings, parks and plazas.
    o = MeshBuilder()
    colliders, bfp = [], []
    max_h = 1.0
    dens = city_density(cx, cz)
    if dens > 0.62:
        grids, weights = ((1, 1), (1, 2), (2, 1), (2, 2)), (3, 2, 2, 3)
    elif dens > 0.3:
        grids, weights = ((2, 2), (1, 2), (2, 1), (3, 2), (2, 3)), (4, 2, 2, 1, 1)
    else:
        grids, weights = ((2, 2), (3, 2), (2, 3), (3, 1), (1, 3)), (3, 2, 2, 1, 1)
    nx, nz = rng.choices(grids, weights)[0]
    ix0, ix1 = bx0 + SIDEWALK_W, x1 - SIDEWALK_W
    iz0, iz1 = bz0 + SIDEWALK_W, z1 - SIDEWALK_W
    xs, zs = _split(rng, ix0, ix1, nx), _split(rng, iz0, iz1, nz)
    p_park = 0.05 + 0.16 * (1.0 - dens)
    p_plaza = p_park + 0.06
    for i in range(nx):
        for j in range(nz):
            lot = (xs[i], xs[i + 1], zs[j], zs[j + 1])
            lx0, lx1, lz0, lz1 = lot
            r = rng.random()
            if r < p_park:
                g.rect(lx0 + 0.4, lx1 - 0.4, lz0 + 0.4, lz1 - 0.4, CURB_H + 0.01, COL_GRASS, layer=2)
                spots = []
                for _ in range(rng.randint(2, 4)):
                    if lx1 - lx0 < 6.0 or lz1 - lz0 < 6.0:
                        break
                    tx, tz = rng.uniform(lx0 + 2.5, lx1 - 2.5), rng.uniform(lz0 + 2.5, lz1 - 2.5)
                    if all((tx - sx) ** 2 + (tz - sz) ** 2 > 20.0 for sx, sz in spots):
                        spots.append((tx, tz))
                        max_h = max(max_h, _tree(o, rng, tx, tz, colliders))
            elif r < p_plaza:
                g.rect(lx0 + 0.4, lx1 - 0.4, lz0 + 0.4, lz1 - 0.4, CURB_H + 0.01, COL_PLAZA, layer=2)
                if rng.random() < 0.6 and min(lx1 - lx0, lz1 - lz0) > 9.0:
                    fx, fz, fr = (lx0 + lx1) * 0.5, (lz0 + lz1) * 0.5, rng.uniform(1.8, 3.0)
                    o.group((fx - fr, fx + fr, fz - fr, fz + fr), rng.random() * 1e-3)
                    o.box(fx - fr, fx + fr, CURB_H, CURB_H + 0.7, fz - fr, fz + fr,
                          {"*": (176, 170, 164), "t": COL_WATER}, lod=LOD_DETAIL)
                    colliders.append((fx - fr, fx + fr, fz - fr, fz + fr, CURB_H + 0.7))
            else:
                max_h = max(max_h, _building(o, rng, lot, dens, colliders, bfp))

    bfp_arr = np.array(bfp, dtype=np.float32) if bfp else np.zeros((0, 4), dtype=np.float32)
    return Chunk(cx, cz, g.arrays(), o.arrays(), colliders, (x0, x1, z0, z1, max_h), bfp_arr)


class World:
    """Chunk cache around the player plus spatial queries used by physics and AI."""

    def __init__(self):
        self.chunks = {}
        self._dirty = True
        self._list = []
        self._rects = None
        self._corners = None
        self._settled = None

    def get(self, cx, cz):
        ch = self.chunks.get((cx, cz))
        if ch is None:
            ch = self.chunks[(cx, cz)] = build_chunk(cx, cz)
            self._dirty = True
        return ch

    def stream(self, x, z, budget=CHUNK_BUDGET):
        """Generate missing chunks around (x, z), nearest first, and drop far ones."""
        key = (math.floor(x / CELL), math.floor(z / CELL))
        if key == self._settled:
            return
        r = VIEW_DIST + LOAD_MARGIN
        n = int(r / CELL) + 1
        missing = []
        for i in range(key[0] - n, key[0] + n + 1):
            for j in range(key[1] - n, key[1] + n + 1):
                if (i, j) not in self.chunks:
                    d = rect_dist(x, z, i * CELL, (i + 1) * CELL, j * CELL, (j + 1) * CELL)
                    if d < r:
                        missing.append((d, i, j))
        missing.sort()
        for _, i, j in missing[:budget]:
            self.get(i, j)
        if len(missing) <= budget:
            self._settled = key
            far = r + CELL
            for k in [k for k in self.chunks
                      if rect_dist(x, z, k[0] * CELL, (k[0] + 1) * CELL, k[1] * CELL, (k[1] + 1) * CELL) > far]:
                del self.chunks[k]
                self._dirty = True

    def preload(self, x, z):
        self._settled = None
        self.stream(x, z, budget=10 ** 6)

    def visible(self, cam):
        """Chunks within the draw distance whose bounding box touches the view frustum."""
        if self._dirty:
            self._list = list(self.chunks.values())
            r = np.array([c.rect for c in self._list], dtype=np.float32).reshape(-1, 5)
            xs = r[:, [0, 1, 0, 1, 0, 1, 0, 1]]
            zs = r[:, [2, 2, 3, 3, 2, 2, 3, 3]]
            ys = np.zeros_like(xs)
            ys[:, 4:] = r[:, 4:5]
            self._rects, self._corners = r, np.stack((xs, ys, zs), axis=-1)
            self._dirty = False
        if not self._list:
            return []
        r, cp = self._rects, cam.pos
        dx = np.maximum(np.maximum(r[:, 0] - cp[0], cp[0] - r[:, 1]), 0.0)
        dz = np.maximum(np.maximum(r[:, 2] - cp[2], cp[2] - r[:, 3]), 0.0)
        near = dx * dx + dz * dz < VIEW_DIST * VIEW_DIST
        v = (self._corners - cp) @ cam.R.T
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
        zh, zv = z * cam.tan_h, z * cam.tan_v
        out = ((z < NEAR_PLANE).all(axis=1) | (x > zh).all(axis=1) | (x < -zh).all(axis=1)
               | (y > zv).all(axis=1) | (y < -zv).all(axis=1))
        return [self._list[i] for i in np.nonzero(near & ~out)[0]]

    def colliders_near(self, x, z, r):
        out = []
        for i in range(math.floor((x - r) / CELL), math.floor((x + r) / CELL) + 1):
            for j in range(math.floor((z - r) / CELL), math.floor((z + r) / CELL) + 1):
                out.extend(self.get(i, j).colliders)
        return out

    def point_blocked(self, x, z, margin=0.0):
        ch = self.chunks.get((math.floor(x / CELL), math.floor(z / CELL)))
        if ch is None:
            return False
        for x0, x1, z0, z1, _ in ch.colliders:
            if x0 - margin <= x <= x1 + margin and z0 - margin <= z <= z1 + margin:
                return True
        return False

    def raycast(self, ax, ay, az, bx, by, bz, first_hit=False):
        """Fraction (0..1) along the segment where it first enters a collider, else 1.0.
        Heights matter: the segment passes over anything lower than itself."""
        dx, dy, dz = bx - ax, by - ay, bz - az
        best = 1.0
        for i in range(math.floor(min(ax, bx) / CELL), math.floor(max(ax, bx) / CELL) + 1):
            for j in range(math.floor(min(az, bz) / CELL), math.floor(max(az, bz) / CELL) + 1):
                ch = self.chunks.get((i, j))
                if ch is None:
                    continue
                for x0, x1, z0, z1, h in ch.colliders:
                    t0, t1 = 0.0, best
                    if abs(dx) < 1e-9:
                        if ax < x0 or ax > x1:
                            continue
                    else:
                        u0, u1 = (x0 - ax) / dx, (x1 - ax) / dx
                        if u0 > u1:
                            u0, u1 = u1, u0
                        t0, t1 = max(t0, u0), min(t1, u1)
                        if t0 > t1:
                            continue
                    if abs(dz) < 1e-9:
                        if az < z0 or az > z1:
                            continue
                    else:
                        u0, u1 = (z0 - az) / dz, (z1 - az) / dz
                        if u0 > u1:
                            u0, u1 = u1, u0
                        t0, t1 = max(t0, u0), min(t1, u1)
                        if t0 > t1:
                            continue
                    if min(ay + dy * t0, ay + dy * t1) < h:
                        if first_hit:
                            return t0
                        best = t0
        return best

    def line_of_sight(self, ax, az, bx, bz, height=1.0):
        return self.raycast(ax, height, az, bx, height, bz, first_hit=True) >= 1.0


# =============================================================================
# 4. ENTITIES
# =============================================================================
# Car meshes are built once in local space (+z forward, +x right) and transformed
# every frame. Face "kind": 0 = lit, 1 = emissive light, 2/3 = red/blue siren.

def build_car_mesh(police):
    m = MeshBuilder()
    body = COL_POLICE_BODY if police else COL_PLAYER_BODY
    roof = COL_POLICE_ROOF if police else COL_PLAYER_ROOF
    # Draw order inside the car (intra): wheels, body, cabin, lights, siren bar.
    for wx0, wx1, faces in ((0.78, 1.06, "tnse"), (-1.06, -0.78, "tnsw")):
        for wz in (-1.35, 1.35):
            m.box(wx0, wx1, 0.0, 0.68, wz - 0.34, wz + 0.34, COL_TYRE, faces=faces,
                  lod=LOD_CAR_DETAIL, intra=0)
    m.box(-0.92, 0.92, 0.28, 0.95, -2.2, 2.2, body, intra=1)
    bumper = (58, 58, 64)
    m.add(((-0.92, 0.28, -2.205), (-0.92, 0.44, -2.205), (0.92, 0.44, -2.205), (0.92, 0.28, -2.205)),
          bumper, (0.0, 0.0, -1.0), intra=1, lod=LOD_CAR_DETAIL)
    m.add(((-0.92, 0.28, 2.205), (0.92, 0.28, 2.205), (0.92, 0.44, 2.205), (-0.92, 0.44, 2.205)),
          bumper, (0.0, 0.0, 1.0), intra=1, lod=LOD_CAR_DETAIL)
    stripe = COL_POLICE_ROOF if police else (40, 40, 46)
    if police:   # white doors
        m.add(((0.925, 0.42, -0.95), (0.925, 0.88, -0.95), (0.925, 0.88, 0.95), (0.925, 0.42, 0.95)),
              stripe, (1.0, 0.0, 0.0), intra=1, lod=LOD_CAR_DETAIL)
        m.add(((-0.925, 0.42, -0.95), (-0.925, 0.42, 0.95), (-0.925, 0.88, 0.95), (-0.925, 0.88, -0.95)),
              stripe, (-1.0, 0.0, 0.0), intra=1, lod=LOD_CAR_DETAIL)
    else:        # racing stripes on bonnet and boot
        m.rect(-0.22, 0.22, 0.95, 2.2, 0.952, stripe, intra=1, lod=LOD_CAR_DETAIL)
        m.rect(-0.22, 0.22, -2.2, -1.3, 0.952, stripe, intra=1, lod=LOD_CAR_DETAIL)
    m.frustum((-0.86, 0.86, -1.25, 0.85), (-0.74, 0.74, -0.85, 0.30), 0.95, 1.55,
              {"*": COL_GLASS, "t": roof}, intra=2)
    lk = dict(intra=3, kind=1, lod=LOD_CAR_DETAIL)
    for xa, xb in ((-0.86, -0.5), (0.5, 0.86)):
        m.add(((xa, 0.62, -2.21), (xa, 0.82, -2.21), (xb, 0.82, -2.21), (xb, 0.62, -2.21)),
              COL_TAIL, (0.0, 0.0, -1.0), **lk)
        m.add(((xa, 0.62, 2.21), (xb, 0.62, 2.21), (xb, 0.82, 2.21), (xa, 0.82, 2.21)),
              COL_HEAD, (0.0, 0.0, 1.0), **lk)
    if police:
        m.box(-0.64, -0.05, 1.55, 1.76, -0.42, 0.06, COL_SIREN_RED, intra=4, kind=2)
        m.box(0.05, 0.64, 1.55, 1.76, -0.42, 0.06, COL_SIREN_BLUE, intra=4, kind=3)
    return m.arrays(lit=False)


_CAR_MESHES = {}


def car_mesh(police):
    if police not in _CAR_MESHES:
        _CAR_MESHES[police] = build_car_mesh(police)
    return _CAR_MESHES[police]


class Car:
    """Arcade car: the heading turns with the wheel and the velocity is dragged
    towards the heading by tyre grip (weak grip on the handbrake = drift)."""

    def __init__(self, x, z, heading, police=False, speed=0.0):
        self.x, self.z, self.y = x, z, 0.0
        self.heading = heading
        self.vx, self.vz = math.sin(heading) * speed, math.cos(heading) * speed
        self.police = police
        self.mass = 1.0 if police else 1.4      # rams hurt the police car more than you
        self.mesh = car_mesh(police)
        self.health = POLICE_HEALTH if police else PLAYER_HEALTH
        self.steer = 0.0
        self.wrecked = False
        self.drifting = False
        self.slip = 0.0
        self.flash = 0.0                  # white hit flash
        self.siren_phase = random.random() * 2.0

    @property
    def speed(self):
        return math.hypot(self.vx, self.vz)

    def forward_speed(self):
        return self.vx * math.sin(self.heading) + self.vz * math.cos(self.heading)

    def drive(self, dt, steer, handbrake, accel, grip_accel, max_yaw):
        """Integrate one step. steer in [-1, 1]; accel in m/s^2 along the heading
        (negative brakes, then reverses)."""
        sp = math.hypot(self.vx, self.vz)
        if handbrake:
            yaw = steer * HANDBRAKE_YAW_RATE * clamp(sp / 6.0, 0.3, 1.0)
        else:   # grip caps how fast you can turn at speed
            yaw = steer * min(max_yaw, grip_accel / max(sp, 1.0)) * clamp(sp / 5.0, 0.35, 1.0)
        if self.forward_speed() < -0.5:
            yaw = -yaw                     # reversing swings the nose the other way
        self.heading = wrap_angle(self.heading + yaw * dt)
        sh, ch = math.sin(self.heading), math.cos(self.heading)
        vf = self.vx * sh + self.vz * ch   # along the heading
        vr = self.vx * ch - self.vz * sh   # sideways (right = (cos h, -sin h))
        rate, keep = ((HANDBRAKE_GRIP_RATE, HANDBRAKE_SLIDE_KEEP) if handbrake
                      else (GRIP_RATE, SLIDE_KEEP))
        vr_new = vr * math.exp(-rate * dt)
        if vf >= 0.0:   # tyres turn part of the killed slide back into forward speed
            vf = math.sqrt(vf * vf + (vr * vr - vr_new * vr_new) * keep)
        vr = vr_new
        vf += accel * dt
        if handbrake:
            vf -= vf * HANDBRAKE_DRAG * dt
        vf = clamp(vf, -12.0, MAX_SPEED_SANITY)
        self.vx, self.vz = sh * vf + ch * vr, ch * vf - sh * vr
        self.x += self.vx * dt
        self.z += self.vz * dt
        self.slip = vr
        self.drifting = abs(vr) > 2.5

    def circles(self):
        sh, ch = math.sin(self.heading), math.cos(self.heading)
        o = CAR_CIRCLE_OFFSET
        return ((self.x + sh * o, self.z + ch * o), (self.x - sh * o, self.z - ch * o))

    def collide_world(self, world):
        """Push the car out of buildings/trees; returns the hardest impact speed (m/s)."""
        R = CAR_RADIUS
        cols = world.colliders_near(self.x, self.z, CAR_CIRCLE_OFFSET + R + 0.5)
        impact = 0.0
        for off in (CAR_CIRCLE_OFFSET, -CAR_CIRCLE_OFFSET):
            sh, ch = math.sin(self.heading), math.cos(self.heading)
            px, pz = self.x + sh * off, self.z + ch * off
            for x0, x1, z0, z1, _h in cols:
                if px < x0 - R or px > x1 + R or pz < z0 - R or pz > z1 + R:
                    continue
                qx = x0 if px < x0 else (x1 if px > x1 else px)
                qz = z0 if pz < z0 else (z1 if pz > z1 else pz)
                dx, dz = px - qx, pz - qz
                d2 = dx * dx + dz * dz
                if d2 >= R * R:
                    continue
                if d2 > 1e-12:
                    d = math.sqrt(d2)
                    nx, nz, pen = dx / d, dz / d, R - d
                else:   # centre got inside: leave through the nearest side
                    dmin, nx, nz = min((px - x0, -1.0, 0.0), (x1 - px, 1.0, 0.0),
                                       (pz - z0, 0.0, -1.0), (z1 - pz, 0.0, 1.0))
                    pen = dmin + R
                self.x += nx * pen
                self.z += nz * pen
                px += nx * pen
                pz += nz * pen
                vn = self.vx * nx + self.vz * nz
                if vn < 0.0:
                    impact = max(impact, -vn)
                    self.vx -= (1.0 + WALL_RESTITUTION) * vn * nx
                    self.vz -= (1.0 + WALL_RESTITUTION) * vn * nz
                    scrape = 1.0 - min(0.3, 0.02 * -vn)
                    self.vx *= scrape
                    self.vz *= scrape
                    if sh * nx + ch * nz < -0.15:   # nose points into the wall: swing it along
                        tx, tz = -nz, nx
                        if tx * sh + tz * ch < 0.0:
                            tx, tz = -tx, -tz
                        self.heading = wrap_angle(self.heading + wrap_angle(math.atan2(tx, tz) - self.heading) * 0.3)
                        sh, ch = math.sin(self.heading), math.cos(self.heading)
        return impact

    def world_faces(self, t):
        """This car's faces transformed to world space, with lighting and sirens."""
        m = self.mesh
        c, s = math.cos(self.heading), math.sin(self.heading)
        rot = np.array(((c, 0.0, -s), (0.0, 1.0, 0.0), (s, 0.0, c)), dtype=np.float32)
        q = m["q"] @ rot + np.array((self.x, self.y, self.z), dtype=np.float32)
        n = m["n"] @ rot
        col = shade(m["c"], n)
        kind = m["kind"]
        emissive = kind == 1
        col[emissive] = m["c"][emissive]
        if self.police:
            on = (not self.wrecked) and ((t * 6.0 + self.siren_phase) % 2.0) < 1.0
            lit = not self.wrecked
            col[kind == 2] = COL_SIREN_RED if (on and lit) else (96, 22, 22)
            col[kind == 3] = COL_SIREN_BLUE if (not on and lit) else (22, 36, 96)
        if self.wrecked:
            col *= 0.42
        if self.flash > 0.0:
            col += (255.0 - col) * min(1.0, self.flash * 4.0)
        return q, n, col

    def aabb(self):
        sh, ch = abs(math.sin(self.heading)), abs(math.cos(self.heading))
        hx = 1.06 * ch + 2.2 * sh
        hz = 1.06 * sh + 2.2 * ch
        return self.x - hx, self.x + hx, self.z - hz, self.z + hz


def collide_cars(a, b):
    """Circle-pair collision with an equal-mass impulse. Returns the closing speed
    (a tiny positive value for a resting touch, 0 when apart)."""
    if abs(a.x - b.x) > 6.0 or abs(a.z - b.z) > 6.0:
        return 0.0
    r2 = 2.0 * CAR_RADIUS
    best = 0.0
    for ax, az in a.circles():
        for bx, bz in b.circles():
            dx, dz = ax - bx, az - bz
            d2 = dx * dx + dz * dz
            if d2 >= r2 * r2 or d2 < 1e-12:
                continue
            best = max(best, 1e-3)        # touching, even if not closing
            d = math.sqrt(d2)
            nx, nz = dx / d, dz / d
            wa = b.mass / (a.mass + b.mass)          # the lighter car gets pushed further
            pen = r2 - d
            a.x += nx * pen * wa
            a.z += nz * pen * wa
            b.x -= nx * pen * (1.0 - wa)
            b.z -= nz * pen * (1.0 - wa)
            rel = (a.vx - b.vx) * nx + (a.vz - b.vz) * nz
            if rel < 0.0:
                j = -(1.0 + CAR_RESTITUTION) * rel / (1.0 / a.mass + 1.0 / b.mass)
                cx, cz = (ax + bx) * 0.5, (az + bz) * 0.5
                for car, s in ((a, j / a.mass), (b, -j / b.mass)):
                    car.vx += s * nx
                    car.vz += s * nz
                    # Off-centre hits twist the car (a PIT manoeuvre on a rear corner).
                    sh, ch = math.sin(car.heading), math.cos(car.heading)
                    rf = (cx - car.x) * sh + (cz - car.z) * ch       # lever arm, forward
                    rr = (cx - car.x) * ch - (cz - car.z) * sh       # lever arm, right
                    ff = s * (nx * sh + nz * ch)                     # impulse, forward
                    fr = s * (nx * ch - nz * sh)                     # impulse, right
                    car.heading = wrap_angle(car.heading + clamp((rf * fr - rr * ff) * 0.05, -0.25, 0.25))
                best = max(best, -rel)
    return best


def shadow_quads(cars):
    """Dark quads under each car (drawn with the ground), nudged away from the sun."""
    ox, oz = -_SUN[0] / _SUN[1] * 0.45, -_SUN[2] / _SUN[1] * 0.45
    local = ((-1.1, -2.35), (1.1, -2.35), (1.1, 2.35), (-1.1, 2.35))
    out = []
    for car in cars:
        c, s = math.cos(car.heading), math.sin(car.heading)
        y = car.y + 0.03
        out.append([(car.x + lx * c + lz * s + ox, y, car.z - lx * s + lz * c + oz) for lx, lz in local])
    return np.array(out, dtype=np.float32).reshape(-1, 4, 3)


class Particles:
    """Sparks and smoke, drawn as screen-space dots after the 3D pass."""

    MAX = 240

    def __init__(self):
        self.items = []

    def sparks(self, x, y, z, n, vx=0.0, vz=0.0):
        for _ in range(n):
            if len(self.items) >= self.MAX:
                return
            a, s = random.uniform(0.0, math.tau), random.uniform(2.0, 8.0)
            life = random.uniform(0.25, 0.6)
            self.items.append([x, y, z, vx * 0.35 + math.cos(a) * s, random.uniform(1.5, 5.0),
                               vz * 0.35 + math.sin(a) * s, life, life, 0])

    def smoke(self, x, y, z, dark=False):
        if len(self.items) >= self.MAX:
            return
        life = random.uniform(1.0, 1.8)
        self.items.append([x + random.uniform(-0.4, 0.4), y, z + random.uniform(-0.4, 0.4),
                           random.uniform(-0.5, 0.5), random.uniform(1.0, 2.2), random.uniform(-0.5, 0.5),
                           life, life, 2 if dark else 1])

    def update(self, dt):
        keep = []
        for p in self.items:
            p[6] -= dt
            if p[6] <= 0.0:
                continue
            p[0] += p[3] * dt
            p[1] += p[4] * dt
            p[2] += p[5] * dt
            if p[8] == 0:
                p[4] -= 14.0 * dt
                if p[1] < 0.05:
                    p[1], p[4] = 0.05, -p[4] * 0.3
            else:
                p[3] *= 0.98
                p[5] *= 0.98
            keep.append(p)
        self.items = keep

    def draw(self, surf, cam):
        rect = pygame.draw.rect
        circle = pygame.draw.circle
        f = cam.focal
        for x, y, z, _vx, _vy, _vz, life, life0, kind in self.items:
            pr = cam.project(x, y, z)
            if pr is None:
                continue
            sx, sy, d = pr
            k = life / life0
            if kind == 0:
                r = max(1.0, 0.09 * f / d)
                rect(surf, (255, int(140 + 110 * k), int(60 * k)), (sx - r, sy - r, 2 * r, 2 * r))
            elif d > 2.5:   # smoke right in front of the lens would cover the screen
                age = 1.0 - k
                r = clamp((0.3 + 0.8 * age) * f / d, 1.0, 26.0)
                g = 70 if kind == 2 else 150
                v = int(g + (FOG_COLOR[0] - g) * age * 0.6)
                circle(surf, (v, v, v + 4), (sx, sy), r)


class SkidMarks:
    """Ring buffer of dark ground quads laid behind drifting tyres."""

    MAX = 140

    def __init__(self):
        self.q = np.zeros((self.MAX, 4, 3), dtype=np.float32)
        self.n = np.tile(np.array((0.0, 1.0, 0.0), dtype=np.float32), (self.MAX, 1))
        self.c = shade(np.tile(np.array(COL_SKID, dtype=np.float32), (self.MAX, 1)), self.n)
        self.lod = np.full(self.MAX, LOD_MARKINGS, dtype=np.float32)
        self.count = 0
        self.head = 0
        self.last = {}

    def track(self, key, x, z, y, active):
        if not active:
            self.last.pop(key, None)
            return
        prev = self.last.get(key)
        if prev is None:
            self.last[key] = (x, z)
            return
        px, pz = prev
        dx, dz = x - px, z - pz
        d = math.hypot(dx, dz)
        if d < 0.9:
            return
        self.last[key] = (x, z)
        if d > 6.0:
            return
        wx, wz = -dz / d * 0.17, dx / d * 0.17
        self.q[self.head] = ((px - wx, y, pz - wz), (px + wx, y, pz + wz), (x + wx, y, z + wz), (x - wx, y, z - wz))
        self.head = (self.head + 1) % self.MAX
        self.count = min(self.count + 1, self.MAX)


# =============================================================================
# 5. AI
# =============================================================================

class Police(Car):
    """A pursuing police car.

    Steering is pure pursuit on a *predicted* target: where the player will be after
    the time it takes to get there (capped by the difficulty's lookahead). When a
    building blocks the line of sight the car follows the road grid instead,
    hopping between intersections greedily towards the target. Up close it rams at
    a controlled speed and drops back after each hit; facing you head-on it slows
    down to block. A short feeler ray steers it around walls, and it backs out if
    it gets stuck.
    """

    def __init__(self, x, z, heading, speed=0.0):
        super().__init__(x, z, heading, police=True, speed=speed)
        self.waypoint = None
        self.nav_timer = 0.0
        self.los_timer = random.uniform(0.0, POLICE_LOS_INTERVAL)
        self.sees_target = self.sees_player = False
        self.stuck_time = 0.0
        self.reverse_time = 0.0
        self.reverse_steer = 1.0
        self.wreck_time = 0.0
        self.hit_cooldown = 0.0
        self.backoff = 0.0            # after a ram, drop back for a moment

    def think(self, dt, player, world, diff):
        """Returns (steer, accel) for this frame."""
        if self.reverse_time > 0.0:
            self.reverse_time -= dt
            return self.reverse_steer, -9.0

        dx, dz = player.x - self.x, player.z - self.z
        dist = math.hypot(dx, dz)
        sp = self.speed
        look = clamp(dist / max(sp, 12.0), 0.0, diff["police_lookahead"])
        tx, tz = player.x + player.vx * look, player.z + player.vz * look

        self.los_timer -= dt
        self.nav_timer -= dt
        if self.los_timer <= 0.0:
            self.los_timer = POLICE_LOS_INTERVAL
            self.sees_target = world.line_of_sight(self.x, self.z, tx, tz)
            self.sees_player = self.sees_target or world.line_of_sight(self.x, self.z, player.x, player.z)
        if self.sees_target or dist < 18.0:
            gx, gz = tx, tz
            self.waypoint = None
        elif self.sees_player:
            gx, gz = player.x, player.z
            self.waypoint = None
        else:
            gx, gz = self.road_target(tx, tz)

        err = wrap_angle(math.atan2(gx - self.x, gz - self.z) - self.heading)
        steer = clamp(err * 2.2, -1.0, 1.0)
        top = max(diff["police_min_top"], diff["police_speed_ratio"] * player.speed)
        target_speed = top * clamp(1.15 - abs(err) / 1.4, 0.32, 1.0)   # brake for corners
        if dist < 35.0:
            # Close in at a controlled speed: from behind they ram at your speed plus
            # police_ram; coming head-on they slow right down and try to block you.
            if math.cos(player.heading - self.heading) < -0.3:
                target_speed = min(target_speed, 10.0)
            elif self.backoff > 0.0:
                target_speed = min(target_speed, player.speed * 0.85)
            else:
                target_speed = min(target_speed, player.speed + diff["police_ram"])
        self.backoff -= dt

        # Feeler: a wall straight ahead? steer to the open side and slow down.
        reach = 4.0 + sp * 0.35
        h = self.heading
        if world.point_blocked(self.x + math.sin(h) * reach, self.z + math.cos(h) * reach, 0.8):
            left = world.point_blocked(self.x + math.sin(h - 0.7) * reach, self.z + math.cos(h - 0.7) * reach, 0.8)
            right = world.point_blocked(self.x + math.sin(h + 0.7) * reach, self.z + math.cos(h + 0.7) * reach, 0.8)
            if left != right:
                steer = 1.0 if left else -1.0
            else:
                steer = 1.0 if err >= 0.0 else -1.0
            target_speed *= 0.55

        # Stuck against something? Back out with the wheel turned the other way.
        if sp < 2.0:
            self.stuck_time += dt
            if self.stuck_time > 1.3:
                self.stuck_time = 0.0
                self.reverse_time = 1.1
                self.reverse_steer = -1.0 if err >= 0.0 else 1.0
        else:
            self.stuck_time = max(0.0, self.stuck_time - dt)

        fs = self.forward_speed()
        if fs < target_speed:
            accel = diff["police_accel"]
        else:
            accel = -POLICE_BRAKE * clamp((fs - target_speed) / 5.0, 0.2, 1.0)
        return steer, accel

    def road_target(self, tx, tz):
        """Next intersection to drive to when the target is hidden behind buildings."""
        if self.waypoint is not None and self.nav_timer > 0.0:
            wx, wz = col_center(self.waypoint[0]), row_center(self.waypoint[1])
            if math.hypot(wx - self.x, wz - self.z) > 7.0:
                return wx, wz
        i, j = nearest_col(self.x), nearest_row(self.z)
        on_v = abs(self.x - col_center(i)) < road_w_col(i) * 0.5 + 1.5
        on_h = abs(self.z - row_center(j)) < road_w_row(j) * 0.5 + 1.5
        if on_v and on_h:          # inside an intersection: any of the four neighbours
            opts = ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))
        elif on_v:                 # on a north-south road: the intersections either side
            j0 = math.floor(self.z / CELL)
            if row_center(j0) > self.z:
                j0 -= 1
            opts = ((i, j0), (i, j0 + 1))
        elif on_h:
            i0 = math.floor(self.x / CELL)
            if col_center(i0) > self.x:
                i0 -= 1
            opts = ((i0, j), (i0 + 1, j))
        else:                      # off-road (plaza, alley): nearest intersection
            opts = ((i, j),)
        fx, fz = math.sin(self.heading), math.cos(self.heading)
        best = None
        for oi, oj in opts:
            wx, wz = col_center(oi), row_center(oj)
            ddx, ddz = wx - self.x, wz - self.z
            cost = math.hypot(tx - wx, tz - wz) + 0.3 * math.hypot(ddx, ddz)
            if ddx * fx + ddz * fz < 0.0:
                cost += 35.0       # behind us: turning round is slow
            if best is None or cost < best[0]:
                best = (cost, oi, oj)
        self.waypoint = (best[1], best[2])
        self.nav_timer = 1.5
        return col_center(best[1]), row_center(best[2])


class PoliceDirector:
    """Decides when and where police join the chase (always out of view)."""

    def __init__(self, diff):
        self.diff = diff
        self.timer = diff["first_spawn"]

    def allowed(self, t):
        d = self.diff
        return min(d["max_police"], d["start_police"] + int(t / d["ramp_every"]))

    def update(self, dt, play):
        active = sum(1 for p in play.police if not p.wrecked)
        self.timer -= dt
        if self.timer <= 0.0 and active < self.allowed(play.time):
            if self.spawn(play):
                self.timer = self.diff["spawn_interval"] * random.uniform(0.8, 1.2)
            else:
                self.timer = 0.5
        player = play.player
        play.police = [p for p in play.police
                       if not (p.wrecked and p.wreck_time > POLICE_WRECK_TIME)
                       and math.hypot(p.x - player.x, p.z - player.z) < POLICE_DESPAWN_DIST]

    def spawn(self, play):
        player, cam, world = play.player, play.scene.cam, play.world
        d = self.diff
        for _ in range(16):
            a = random.uniform(0.0, math.tau)
            r = random.uniform(POLICE_SPAWN_MIN, POLICE_SPAWN_MAX)
            px, pz = player.x + math.sin(a) * r, player.z + math.cos(a) * r
            i, j = nearest_col(px), nearest_row(pz)
            if abs(px - col_center(i)) < abs(pz - row_center(j)):   # snap onto a road centre line
                px = col_center(i)
                heading = 0.0 if player.z > pz else math.pi
            else:
                pz = row_center(j)
                heading = math.pi * 0.5 if player.x > px else -math.pi * 0.5
            if math.hypot(px - player.x, pz - player.z) < POLICE_SPAWN_MIN * 0.8:
                continue
            ang = abs(wrap_angle(math.atan2(px - cam.x, pz - cam.z) - cam.yaw))
            hidden = ang > cam.hfov * 0.5 + 0.15 or not world.line_of_sight(cam.x, cam.z, px, pz, height=2.0)
            if not hidden or world.point_blocked(px, pz, 2.0):
                continue
            if any(math.hypot(o.x - px, o.z - pz) < 8.0 for o in play.police):
                continue
            speed = 0.7 * max(d["police_min_top"], d["police_speed_ratio"] * player.speed)
            play.police.append(Police(px, pz, heading, speed))
            return True
        return False


# =============================================================================
# 6. GAME STATES & UI
# =============================================================================

def _footprint_dist(fp, x, z):
    """Distance from (x, z) to each footprint rectangle (n, 4) = x0, x1, z0, z1."""
    dx = np.maximum(np.maximum(fp[:, 0] - x, x - fp[:, 1]), 0.0)
    dz = np.maximum(np.maximum(fp[:, 2] - z, z - fp[:, 3]), 0.0)
    return np.sqrt(dx * dx + dz * dz)


def _car_sort_key(car, cam, bfp, bkeys):
    """Painter's key for a car, made consistent with nearby buildings.

    Base key = distance to the car's footprint (same metric as buildings). Then, for
    each nearby building, a separating axis between the two footprints tells which
    of them can hide the other from the camera; the key is clamped so the car is
    drawn after buildings it is in front of and before buildings that can hide it.
    """
    x0, x1, z0, z1 = car.aabb()
    key = rect_dist(cam.x, cam.z, x0, x1, z0, z1)
    if not len(bfp):
        return key
    near = (bfp[:, 0] < x1 + 30.0) & (bfp[:, 1] > x0 - 30.0) & (bfp[:, 2] < z1 + 30.0) & (bfp[:, 3] > z0 - 30.0)
    if not near.any():
        return key
    b, k = bfp[near], bkeys[near]
    bx0, bx1, bz0, bz1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    cx, cz = cam.x, cam.z
    bld_front = (((bx1 <= x0) & (cx < bx1)) | ((bx0 >= x1) & (cx > bx0))
                 | ((bz1 <= z0) & (cz < bz1)) | ((bz0 >= z1) & (cz > bz0)))
    car_front = (((x1 <= bx0) & (cx < x1)) | ((x0 >= bx1) & (cx > x0))
                 | ((z1 <= bz0) & (cz < z1)) | ((z0 >= bz1) & (cz > z0))
                 | ((bx0 < x1) & (bx1 > x0) & (bz0 < z1) & (bz1 > z0)))   # touching: keep car visible
    lo = bld_front & ~car_front
    hi = car_front & ~bld_front
    if lo.any():
        key = max(key, float(k[lo].max()) + 0.01)
    if hi.any():
        key = min(key, float(k[hi].min()) - 0.01)
    return key


_UP = np.array((0.0, 1.0, 0.0), dtype=np.float32)


class Scene:
    """The world, cars and effects seen through a chase camera.
    Used both by the game and by the menu's attract mode."""

    def __init__(self, game):
        self.game = game
        self.world = game.world
        self.renderer = game.renderer
        self.cam = Camera()
        self.particles = Particles()
        self.skids = SkidMarks()
        self.cam_yaw = 0.0
        self.fov = math.radians(FOV_BASE_DEG)
        self.shake = 0.0
        self._glow_cache = {}
        self._glow_base = {c: self._make_glow(c) for c in (COL_SIREN_RED, COL_SIREN_BLUE)}

    @staticmethod
    def _make_glow(col):
        size = 96
        s = pygame.Surface((size, size))
        for r in range(size // 2, 0, -2):
            k = (1.0 - r / (size / 2)) ** 2
            pygame.draw.circle(s, (int(col[0] * k * 0.8), int(col[1] * k * 0.8), int(col[2] * k * 0.8)),
                               (size // 2, size // 2), r)
        return s

    def follow(self, car, dt, snap=False, yaw_offset=0.0, side=0.0):
        """Chase camera: swings in behind the car; side > 0 shifts it to the right."""
        target = car.heading
        if car.speed > 4.0:          # lean towards the direction of travel so drifts read well
            target += wrap_angle(math.atan2(car.vx, car.vz) - car.heading) * 0.35
        if snap:
            self.cam_yaw = target
        else:
            self.cam_yaw = wrap_angle(self.cam_yaw + wrap_angle(target - self.cam_yaw) * ease(CAM_YAW_FOLLOW, dt))
        yaw = self.cam_yaw + yaw_offset
        fx, fz = math.sin(yaw), math.cos(yaw)
        cx, cy, cz = car.x - fx * CAM_DISTANCE, car.y + CAM_HEIGHT, car.z - fz * CAM_DISTANCE
        cx += fz * side
        cz -= fx * side
        ay = car.y + 1.5               # keep the camera out of buildings
        t = self.world.raycast(car.x, ay, car.z, cx, cy, cz)
        if t < 1.0:
            t = max(0.15, t - 0.08)
            cx, cy, cz = car.x + (cx - car.x) * t, ay + (cy - ay) * t, car.z + (cz - car.z) * t
        if self.shake > 0.0:
            s = self.shake * 0.25
            cx += random.uniform(-s, s)
            cy += random.uniform(-s, s)
            cz += random.uniform(-s, s)
            self.shake = max(0.0, self.shake - dt * 2.5)
        cam = self.cam
        cam.x, cam.y, cam.z, cam.yaw = cx, cy, cz, yaw
        want = math.radians(lerp(FOV_BASE_DEG, FOV_FAST_DEG, clamp(car.speed / FOV_FAST_SPEED, 0.0, 1.0)))
        self.fov += (want - self.fov) * (1.0 if snap else ease(3.0, dt))
        cam.hfov = self.fov

    def render(self, cars, t):
        r, cam, world = self.renderer, self.cam, self.world
        r.begin(cam)
        chunks = world.visible(cam)

        # Ground pass: flat stuff in layer order (roads, block tops, markings, shadows).
        gs = [c.ground for c in chunks]
        q = [g["q"] for g in gs]
        n = [g["n"] for g in gs]
        col = [g["c"] for g in gs]
        lod = [g["lod"] for g in gs]
        layer = [g["layer"] for g in gs]
        extra = []
        if cars:
            extra.append((shadow_quads(cars), COL_SHADOW, 140.0))
        sk = self.skids
        if sk.count:
            extra.append((sk.q[:sk.count], COL_SKID, LOD_MARKINGS))
        for eq, ecol, elod in extra:
            m = len(eq)
            en = np.tile(_UP, (m, 1))
            q.append(eq)
            n.append(en)
            col.append(shade(np.tile(np.array(ecol, dtype=np.float32), (m, 1)), en))
            lod.append(np.full(m, elod, dtype=np.float32))
            layer.append(np.full(m, 3, dtype=np.int16))
        if q:
            r.draw_faces(np.concatenate(q), np.concatenate(n), np.concatenate(col),
                         np.concatenate(lod), (np.concatenate(layer),))

        # Object pass: buildings, details, trees and cars, sorted per object.
        objs = [c.objects for c in chunks if c.objects is not None]
        q, n, col, lod, intra, key = [], [], [], [], [], []
        if objs:
            sq = np.concatenate([o["q"] for o in objs])
            fp = np.concatenate([o["fp"] for o in objs])
            depth = np.concatenate([o["depth"] for o in objs])
            roof0 = np.concatenate([o["roof0"] for o in objs])
            roof1 = np.concatenate([o["roof1"] for o in objs])
            base_intra = np.concatenate([o["intra"] for o in objs])
            # Roof details are drawn after their building when the camera is above the
            # roof they stand on, before it otherwise (two levels of stacking).
            s0 = np.where(cam.y > roof0, 1.0, -1.0)
            s1 = np.where(cam.y > roof1, 0.5, -0.5)
            q.append(sq)
            n.append(np.concatenate([o["n"] for o in objs]))
            col.append(np.concatenate([o["c"] for o in objs]))
            lod.append(np.concatenate([o["lod"] for o in objs]))
            intra.append(np.where(depth == 0, base_intra, np.where(depth == 1, s0, s0 + s1)).astype(np.float32))
            key.append(_footprint_dist(fp, cam.x, cam.z) + np.concatenate([o["tie"] for o in objs]))
        if cars:
            bfp = [c.bfp for c in chunks if len(c.bfp)]
            bfp = np.concatenate(bfp) if bfp else np.zeros((0, 4), dtype=np.float32)
            bkeys = _footprint_dist(bfp, cam.x, cam.z)
            for car in cars:
                cq, cn, cc = car.world_faces(t)
                m = car.mesh
                q.append(cq)
                n.append(cn)
                col.append(cc)
                lod.append(m["lod"])
                intra.append(m["intra"])
                key.append(np.full(len(cq), _car_sort_key(car, cam, bfp, bkeys), dtype=np.float32))
        if q:
            r.draw_faces(np.concatenate(q), np.concatenate(n), np.concatenate(col), np.concatenate(lod),
                         (np.concatenate(intra), -np.concatenate(key)))

        self.particles.draw(r.surface, cam)
        self._siren_glows(cars, t)

    def _siren_glows(self, cars, t):
        """Additive red/blue flares on visible police light bars."""
        cam, surf, world = self.cam, self.renderer.surface, self.world
        for car in cars:
            if not car.police or car.wrecked:
                continue
            c, s = math.cos(car.heading), math.sin(car.heading)
            on = ((t * 6.0 + car.siren_phase) % 2.0) < 1.0
            lx = 0.35 if not on else -0.35
            wx, wy, wz = car.x + lx * c - 0.18 * s, car.y + 1.68, car.z - lx * s - 0.18 * c
            pr = cam.project(wx, wy, wz)
            if pr is None or not (-80 < pr[0] < self.renderer.w + 80 and -80 < pr[1] < self.renderer.h + 80):
                continue
            if pr[2] > 160.0 or world.raycast(cam.x, cam.y, cam.z, wx, wy, wz, first_hit=True) < 0.98:
                continue
            size = int(clamp(4.5 * cam.focal / pr[2], 10, 200)) // 12 * 12 + 12   # few sizes to cache
            colr = COL_SIREN_RED if on else COL_SIREN_BLUE
            g = self._glow_cache.get((colr, size))
            if g is None:
                g = self._glow_cache[(colr, size)] = pygame.transform.smoothscale(self._glow_base[colr], (size, size))
            surf.blit(g, (pr[0] - size / 2, pr[1] - size / 2), special_flags=pygame.BLEND_ADD)


class UI:
    """Cached text rendering and simple widgets."""

    def __init__(self):
        self.fonts = {}
        self.cache = {}

    def font(self, size):
        f = self.fonts.get(size)
        if f is None:
            f = self.fonts[size] = pygame.font.Font(None, size)
        return f

    def text(self, text, size, color):
        k = (text, size, color)
        s = self.cache.get(k)
        if s is None:
            if len(self.cache) > 600:
                self.cache.clear()
            s = self.cache[k] = self.font(size).render(text, True, color)
        return s

    def draw(self, surf, text, size, color, pos, anchor="center", shadow=True):
        s = self.text(text, size, color)
        rect = s.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
        if shadow:
            surf.blit(self.text(text, size, (0, 0, 0)), rect.move(2, 2))
        surf.blit(s, rect)
        return rect

    def panel(self, surf, rect, alpha=165, radius=12, color=UI_PANEL):
        k = ("panel", rect.size, alpha, radius, color)
        s = self.cache.get(k)
        if s is None:
            s = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*color, alpha), s.get_rect(), border_radius=radius)
            self.cache[k] = s
        surf.blit(s, rect.topleft)


class Button:
    def __init__(self, label, center, action, w=300, h=54, size=36):
        self.label = label
        self.rect = pygame.Rect(0, 0, w, h)
        self.rect.center = (int(center[0]), int(center[1]))
        self.action = action
        self.size = size
        self.selected = False

    def draw(self, surf, ui, mouse):
        hover = self.rect.collidepoint(mouse)
        if self.selected:
            fill, border, txt = UI_ACCENT, (255, 220, 150), (24, 20, 16)
        elif hover:
            fill, border, txt = (74, 80, 100), UI_ACCENT, UI_TEXT
        else:
            fill, border, txt = (36, 40, 54), (90, 96, 116), UI_TEXT
        pygame.draw.rect(surf, fill, self.rect, border_radius=10)
        pygame.draw.rect(surf, border, self.rect, width=2, border_radius=10)
        ui.draw(surf, self.label, self.size, txt, self.rect.center, shadow=False)

    def handle(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.rect.collidepoint(e.pos):
            self.action()
            return True
        return False


def steer_from_mouse(mx, width):
    """Mouse x -> steering in [-1, 1]: dead zone, then a curve up to full lock."""
    off = (mx - width * 0.5) / (width * 0.5)
    a = abs(off)
    if a <= STEER_DEADZONE:
        return 0.0
    a = clamp((a - STEER_DEADZONE) / (STEER_SATURATE - STEER_DEADZONE), 0.0, 1.0) ** STEER_CURVE
    return math.copysign(a, off)


def fmt_time(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def wrap_text(font, text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if font.size(test)[0] <= width or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


class State:
    cursor_visible = True

    def __init__(self, game):
        self.game = game

    def handle(self, e):
        pass

    def update(self, dt):
        pass

    def draw(self, surf):
        pass


class MenuState(State):
    """Main menu over an attract-mode chase."""

    def __init__(self, game):
        super().__init__(game)
        self.scene = Scene(game)
        self.t = 0.0
        self.hero = Car(col_center(0) + 2.5, 40.0, 0.0, speed=19.0)
        self.cop = Police(col_center(0) - 1.0, 26.0, 0.0, speed=19.0)
        game.world.preload(self.hero.x, self.hero.z)
        self._place_demo(0.0)
        self.scene.follow(self.hero, 0.0, snap=True)
        H = SCREEN_H
        cx = self.COL_X
        self.buttons = [
            Button("PLAY", (cx, H * 0.335), game.start_run, w=330, h=62, size=46),
            Button("CONTROLS", (cx, H * 0.735), lambda: game.change(ControlsState(game, self))),
            Button("QUIT", (cx, H * 0.83), game.quit),
        ]
        self.diff_buttons = [
            Button(name, (cx + (k - 1) * 142, H * 0.475), lambda n=name: game.set_difficulty(n), w=134, h=46, size=32)
            for k, name in enumerate(DIFFICULTY_NAMES)]

    COL_X = 290          # the menu sits on the left so the chase behind it stays visible

    def _place_demo(self, dt):
        self.t += dt
        t, base = self.t, col_center(0)
        for car, lane, lag, ph in ((self.hero, 2.5, 0.0, 0.0), (self.cop, 1.0, 13.0 + 2.0 * math.sin(t * 0.3), 0.9)):
            car.x = base + lane + math.sin(t * 0.5 - ph) * 1.6
            car.z = 40.0 + 19.0 * t - lag
            car.vx, car.vz = 0.8 * math.cos(t * 0.5 - ph), 19.0
            car.heading = math.atan2(car.vx, car.vz)

    def update_demo(self, dt):
        self._place_demo(dt)
        self.game.world.stream(self.hero.x, self.hero.z)
        self.scene.follow(self.hero, dt, yaw_offset=math.sin(self.t * 0.23) * 0.45, side=-3.5)

    def handle(self, e):
        for b in self.buttons + self.diff_buttons:
            if b.handle(e):
                return
        if e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_RETURN, pygame.K_SPACE):
                self.game.start_run()
            elif e.key in (pygame.K_LEFT, pygame.K_RIGHT):
                k = DIFFICULTY_NAMES.index(self.game.difficulty) + (1 if e.key == pygame.K_RIGHT else -1)
                self.game.set_difficulty(DIFFICULTY_NAMES[k % len(DIFFICULTY_NAMES)])

    def update(self, dt):
        self.update_demo(dt)

    def draw_background(self, surf):
        self.scene.render([self.cop, self.hero], self.t)

    def draw(self, surf):
        self.draw_background(surf)
        g, ui = self.game, self.game.ui
        H = surf.get_height()
        cx = self.COL_X
        mouse = g.read_mouse()[:2]
        ui.panel(surf, pygame.Rect(cx - 250, 24, 500, H - 48), alpha=150, radius=18)
        ui.draw(surf, "PAKO", 120, UI_ACCENT, (cx, H * 0.105))
        ui.draw(surf, "CHASE", 64, UI_TEXT, (cx, H * 0.19))
        ui.draw(surf, "Outrun the police. It only gets faster.", 25, UI_DIM, (cx, H * 0.245), shadow=False)
        ui.draw(surf, "DIFFICULTY", 24, UI_DIM, (cx, H * 0.415), shadow=False)
        for b in self.diff_buttons:
            b.selected = b.label == g.difficulty
        for b in self.buttons + self.diff_buttons:
            b.draw(surf, ui, mouse)
        y = H * 0.53
        for line in wrap_text(ui.font(23), DIFFICULTY[g.difficulty]["blurb"], 430):
            ui.draw(surf, line, 23, UI_TEXT, (cx, y), shadow=False)
            y += 22
        ui.draw(surf, f"Best on {g.difficulty}: {g.best.get(g.difficulty, 0)}", 28, UI_ACCENT, (cx, H * 0.645))
        ui.draw(surf, "Enter = play    Left/Right = difficulty", 20, UI_DIM, (cx, H * 0.91), shadow=False)


class ControlsState(State):
    LINES = (
        ("STEER", "Move the mouse left or right of the screen centre. The further from the centre, "
                  "the sharper the turn. Keep it near the middle to drive straight."),
        ("HANDBRAKE / DRIFT", "Hold the LEFT mouse button. The car slides and swings round fast for "
                              "tight corners, at the cost of some speed."),
        ("SPEED", "Your car accelerates by itself and never stops getting faster. Staying in "
                  "control is the challenge."),
        ("DAMAGE", "Hitting buildings or police costs health. At zero health you are BUSTED."),
        ("PAUSE", "Esc pauses the game. F3 toggles the FPS counter."),
        ("TIPS", "Start drifts before the intersection, lure police into walls and watch the "
                 "radar for cars coming from side streets."),
    )

    def __init__(self, game, menu):
        super().__init__(game)
        self.menu = menu
        self.back = Button("BACK", (SCREEN_W * 0.5, SCREEN_H * 0.88), lambda: game.change(menu))

    def handle(self, e):
        if self.back.handle(e):
            return
        if e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_BACKSPACE):
            self.game.change(self.menu)

    def update(self, dt):
        self.menu.update_demo(dt)

    def draw(self, surf):
        self.menu.draw_background(surf)
        ui = self.game.ui
        W, H = surf.get_size()
        ui.panel(surf, pygame.Rect(W // 2 - 420, int(H * 0.05), 840, int(H * 0.9)), alpha=190, radius=18)
        ui.draw(surf, "CONTROLS", 72, UI_ACCENT, (W * 0.5, H * 0.12))
        y = H * 0.2
        font = ui.font(25)
        for head, body in self.LINES:
            ui.draw(surf, head, 28, UI_ACCENT, (W * 0.5 - 380, y), anchor="topleft", shadow=False)
            y += 26
            for line in wrap_text(font, body, 740):
                ui.draw(surf, line, 25, UI_TEXT, (W * 0.5 - 380, y), anchor="topleft", shadow=False)
                y += 23
            y += 12
        self.back.draw(surf, ui, self.game.read_mouse()[:2])


class PlayState(State):
    cursor_visible = False

    def __init__(self, game, difficulty):
        super().__init__(game)
        self.diff_name = difficulty
        self.diff = DIFFICULTY[difficulty]
        self.world = game.world
        self.scene = Scene(game)
        sx, sz = col_center(0), 22.0
        self.world.preload(sx, sz)
        self.player = Car(sx, sz, 0.0, speed=self.diff["start_speed"])
        self.police = []
        self.director = PoliceDirector(self.diff)
        self.time = self.distance = 0.0
        self.top_speed = 0.0
        self.wrecked_cops = 0
        self.since_hit = 99.0
        self.hurt = 0.0
        self.busted = False
        self.busted_time = 0.0
        self.boxed_in = 0.0
        self.handbrake = False
        self.smoke_timer = 0.0
        self.scene.follow(self.player, 0.0, snap=True)

    # --- simulation -----------------------------------------------------------
    def score(self):
        return int(self.time * SCORE_PER_SECOND + self.distance * SCORE_PER_METRE)

    def chasers(self):
        return sum(1 for p in self.police if not p.wrecked)

    def damage(self, amount):
        amount *= self.diff["damage_mult"]
        if amount <= 0.0 or self.busted:
            return
        p = self.player
        p.health = max(0.0, p.health - amount)
        p.flash = 0.12
        self.since_hit = 0.0
        self.hurt = min(1.0, self.hurt + 0.25 + amount / 30.0)
        self.scene.shake = min(1.6, self.scene.shake + amount / 14.0)
        if p.health <= 0.0:
            self.busted = True

    def handle(self, e):
        focus_lost = e.type == getattr(pygame, "WINDOWFOCUSLOST", -1)   # alt-tab pauses too
        if (focus_lost or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE)) and not self.busted:
            self.game.change(PauseState(self.game, self))

    def update(self, dt):
        p, diff = self.player, self.diff
        mx, _my, held = self.game.read_mouse()
        if self.busted:
            target, self.handbrake = 0.0, False
            self.busted_time += dt
            if self.busted_time > 2.2:
                self.game.change(GameOverState(self.game, self))
                return
        else:
            target, self.handbrake = steer_from_mouse(mx, SCREEN_W), bool(held)
            self.time += dt
            self.distance += p.speed * dt
            self.top_speed = max(self.top_speed, p.speed)
        p.steer += (target - p.steer) * ease(STEER_RESPONSE, dt)

        # Police decide once per frame; physics runs in sub-steps so fast cars can't tunnel.
        orders = [cop.think(dt, p, self.world, diff) if not cop.wrecked else (0.0, 0.0) for cop in self.police]
        for cop, (steer, _a) in zip(self.police, orders):
            cop.steer += (steer - cop.steer) * ease(8.0, dt)
        fastest = max([p.speed] + [c.speed for c in self.police])
        steps = max(1, min(8, int(math.ceil(fastest * dt / 0.6))))
        h = dt / steps
        for _ in range(steps):
            self.step(h, orders)

        self.since_hit += dt
        if not self.busted and diff["regen"] > 0.0 and self.since_hit > diff["regen_delay"]:
            p.health = min(PLAYER_HEALTH, p.health + diff["regen"] * dt)
        for car in [p] + self.police:
            car.flash = max(0.0, car.flash - dt)
            car.y = ground_height(car.x, car.z)
            if car.police:
                car.hit_cooldown = max(0.0, car.hit_cooldown - dt)
                if car.wrecked:
                    car.wreck_time += dt
        self.hurt = max(0.0, self.hurt - dt * 1.8)
        self.boxed_in = max(0.0, self.boxed_in - dt)
        self._effects(dt)
        self.world.stream(p.x, p.z)
        self.director.update(dt, self)
        self.scene.particles.update(dt)
        self.scene.follow(p, dt)

    def step(self, dt, orders):
        p, world, diff = self.player, self.world, self.diff
        fs = p.forward_speed()
        if self.busted:
            accel = -clamp(fs * 3.0, -14.0, 14.0)
        else:
            accel = diff["accel"] + LOW_SPEED_BOOST * clamp(1.0 - fs / diff["start_speed"], 0.0, 1.5)
        p.drive(dt, p.steer, self.handbrake, accel, GRIP_ACCEL, MAX_YAW_RATE)
        impact = p.collide_world(world)
        if impact > WALL_DAMAGE_THRESHOLD:
            self.damage((impact - WALL_DAMAGE_THRESHOLD) * WALL_DAMAGE_PER_MS)
            sh, ch = math.sin(p.heading), math.cos(p.heading)
            self.scene.particles.sparks(p.x + sh * 2.0, 0.6, p.z + ch * 2.0, int(4 + impact * 0.6), p.vx, p.vz)

        for cop, (steer, accel) in zip(self.police, orders):
            if cop.wrecked:
                cfs = cop.forward_speed()
                cop.drive(dt, 0.0, False, -clamp(cfs * 2.0, -10.0, 10.0), diff["police_grip"], diff["police_yaw"])
            else:
                cop.drive(dt, cop.steer, False, accel, diff["police_grip"], diff["police_yaw"])
            hit = cop.collide_world(world)
            if hit > 6.0:
                self._hurt_cop(cop, (hit - 6.0) * 1.6)
                self.scene.particles.sparks(cop.x, 0.6, cop.z, 6, cop.vx, cop.vz)

        cars = [p] + self.police
        boxed_in = False
        for i in range(len(cars)):
            a = cars[i]
            for j in range(i + 1, len(cars)):
                b = cars[j]
                closing = collide_cars(a, b)
                if closing <= 0.0:
                    continue
                if closing > 4.0:
                    self.scene.particles.sparks((a.x + b.x) * 0.5, 0.7, (a.z + b.z) * 0.5, int(3 + closing * 0.4))
                if a is p:
                    boxed_in |= not b.wrecked
                    if closing > 2.0 and b.hit_cooldown <= 0.0 and not b.wrecked:
                        b.hit_cooldown = CAR_HIT_COOLDOWN
                        b.backoff = random.uniform(1.0, 2.0)
                        self.damage(min(CAR_HIT_DAMAGE_MAX,
                                        CAR_HIT_DAMAGE_BASE + CAR_HIT_DAMAGE_PER_MS * (closing - 2.0)))
                    if closing > 2.0:
                        self._hurt_cop(b, closing * 0.8)
                else:
                    self._hurt_cop(a, max(0.0, closing - 4.0) * 1.2)
                    self._hurt_cop(b, max(0.0, closing - 4.0) * 1.2)
        # Pinned by police at low speed: health drains away until you break free.
        if boxed_in and p.speed < PIN_SPEED and not self.busted:
            self.boxed_in = 0.25
            self.since_hit = 0.0
            p.health = max(0.0, p.health - PIN_DAMAGE * diff["damage_mult"] * dt)
            if p.health <= 0.0:
                self.busted = True

    def _hurt_cop(self, cop, amount):
        if cop.wrecked or amount <= 0.0:
            return
        cop.health -= amount
        cop.flash = 0.1
        if cop.health <= 0.0:
            cop.wrecked = True
            cop.wreck_time = 0.0
            self.wrecked_cops += 1
            self.scene.particles.sparks(cop.x, 1.0, cop.z, 18, cop.vx, cop.vz)

    def _effects(self, dt):
        p, sc = self.player, self.scene
        sh, ch = math.sin(p.heading), math.cos(p.heading)
        skid = (self.handbrake and p.speed > 6.0) or abs(p.slip) > 3.0
        for k, side in enumerate((-0.86, 0.86)):
            wx = p.x + side * ch - 1.35 * sh
            wz = p.z - side * sh - 1.35 * ch
            sc.skids.track(k, wx, wz, p.y + 0.03, skid)
        self.smoke_timer -= dt
        if self.smoke_timer <= 0.0:
            self.smoke_timer = 0.12
            if p.health < 35.0:
                sc.particles.smoke(p.x + sh * 1.8, p.y + 1.0, p.z + ch * 1.8, dark=p.health < 15.0)
            for cop in self.police:
                if cop.wrecked and random.random() < 0.35:
                    sc.particles.smoke(cop.x, cop.y + 1.2, cop.z, dark=True)

    # --- drawing --------------------------------------------------------------
    def draw(self, surf, hud=True):
        self.scene.render([self.player] + self.police, self.time)
        if self.hurt > 0.0:
            self._hurt_vignette(surf)
        if hud:
            self.draw_hud(surf)
            if self.boxed_in > 0.0 and not self.busted and (self.time * 4.0) % 1.0 < 0.6:
                self.game.ui.draw(surf, "BOXED IN!  GET MOVING", 44, UI_DANGER, (SCREEN_W * 0.5, SCREEN_H * 0.3))
        if self.busted and hud:
            ui = self.game.ui
            k = min(1.0, self.busted_time * 2.0)
            ui.draw(surf, "BUSTED!", int(90 + 60 * k), UI_DANGER, (SCREEN_W * 0.5, SCREEN_H * 0.35))

    def _hurt_vignette(self, surf):
        """Red frame around the screen after a hit (plain fills: blended ones are slow)."""
        W, H = surf.get_size()
        k = min(1.0, self.hurt)
        for i, (shade_k, width) in enumerate(((0.55, 22), (0.8, 12), (1.0, 5))):
            t = int(width * k)
            if t <= 0:
                continue
            col = (int(150 + 90 * shade_k * k), int(30 * (1 - k)), int(30 * (1 - k)))
            surf.fill(col, (0, 0, W, t))
            surf.fill(col, (0, H - t, W, t))
            surf.fill(col, (0, 0, t, H))
            surf.fill(col, (W - t, 0, t, H))

    def draw_hud(self, surf):
        ui, p = self.game.ui, self.player
        W, H = surf.get_size()
        # top left: time and distance
        ui.panel(surf, pygame.Rect(16, 16, 290, 84))
        ui.draw(surf, "TIME", 20, UI_DIM, (32, 26), "topleft", shadow=False)
        ui.draw(surf, fmt_time(self.time), 40, UI_TEXT, (32, 46), "topleft", shadow=False)
        ui.draw(surf, "DISTANCE", 20, UI_DIM, (170, 26), "topleft", shadow=False)
        ui.draw(surf, f"{self.distance / 1000.0:.2f} km", 34, UI_TEXT, (170, 50), "topleft", shadow=False)
        # top right: score and chasers
        ui.panel(surf, pygame.Rect(W - 306, 16, 290, 84))
        ui.draw(surf, "SCORE", 20, UI_DIM, (W - 290, 26), "topleft", shadow=False)
        ui.draw(surf, str(self.score()), 40, UI_ACCENT, (W - 290, 46), "topleft", shadow=False)
        n = self.chasers()
        ui.draw(surf, "POLICE", 20, UI_DIM, (W - 120, 26), "topleft", shadow=False)
        blink = COL_SIREN_RED if (self.time * 3.0) % 2.0 < 1.0 else COL_SIREN_BLUE
        if n:
            pygame.draw.circle(surf, blink, (W - 108, 63), 7)
        ui.draw(surf, str(n), 40, UI_TEXT if n else UI_DIM, (W - 92, 46), "topleft", shadow=False)
        # bottom left: speed and health
        ui.panel(surf, pygame.Rect(16, H - 112, 320, 96))
        ui.draw(surf, f"{p.speed * 3.6:.0f}", 64, UI_TEXT, (32, H - 106), "topleft", shadow=False)
        ui.draw(surf, "km/h", 24, UI_DIM, (140, H - 82), "topleft", shadow=False)
        if self.handbrake:
            ui.draw(surf, "DRIFT", 26, UI_ACCENT, (230, H - 84), "topleft", shadow=False)
        hp = p.health / PLAYER_HEALTH
        bar = pygame.Rect(32, H - 44, 288, 16)
        pygame.draw.rect(surf, (50, 20, 24), bar, border_radius=6)
        hc = UI_OK if hp > 0.5 else (UI_ACCENT if hp > 0.25 else UI_DANGER)
        if hp > 0.0:
            pygame.draw.rect(surf, hc, (bar.x, bar.y, max(6, int(bar.w * hp)), bar.h), border_radius=6)
        pygame.draw.rect(surf, (230, 230, 230), bar, width=2, border_radius=6)
        # bottom centre: steering gauge;  bottom right: radar
        self._draw_steer(surf, W * 0.5, H - 30)
        self._draw_radar(surf, W - 92, H - 92, 74)
        # steering cursor
        if not self.busted:
            mx, my, _ = self.game.read_mouse()
            pygame.draw.circle(surf, (255, 255, 255), (int(mx), int(my)), 9, 2)
            pygame.draw.circle(surf, UI_ACCENT, (int(mx), int(my)), 3)
            pygame.draw.line(surf, (255, 255, 255), (W // 2, H // 2 - 10), (W // 2, H // 2 + 10), 1)

    def _draw_steer(self, surf, cx, cy):
        half = 170
        pygame.draw.line(surf, (230, 230, 230), (cx - half, cy), (cx + half, cy), 2)
        dz = half * STEER_DEADZONE / STEER_SATURATE
        pygame.draw.line(surf, UI_DIM, (cx - dz, cy - 6), (cx - dz, cy + 6), 2)
        pygame.draw.line(surf, UI_DIM, (cx + dz, cy - 6), (cx + dz, cy + 6), 2)
        x = cx + self.player.steer * half
        pygame.draw.circle(surf, UI_ACCENT if not self.handbrake else UI_DANGER, (int(x), int(cy)), 8)

    def _draw_radar(self, surf, cx, cy, rad):
        ui, p = self.game.ui, self.player
        ui.panel(surf, pygame.Rect(cx - rad - 6, cy - rad - 6, 2 * rad + 12, 2 * rad + 12), alpha=150, radius=rad + 6)
        pygame.draw.circle(surf, (90, 96, 116), (cx, cy), rad, 1)
        pygame.draw.circle(surf, (70, 74, 90), (cx, cy), rad // 2, 1)
        yaw = self.scene.cam.yaw
        c, s = math.cos(yaw), math.sin(yaw)
        scale = rad / 160.0
        for cop in self.police:
            dx, dz = cop.x - p.x, cop.z - p.z
            rx, ry = (dx * c - dz * s) * scale, (dx * s + dz * c) * scale
            d = math.hypot(rx, ry)
            if d > rad - 4:
                rx, ry = rx * (rad - 4) / d, ry * (rad - 4) / d
            if cop.wrecked:
                colr = (110, 110, 110)
            else:
                colr = COL_SIREN_RED if ((self.time * 4.0 + cop.siren_phase) % 2.0) < 1.0 else COL_SIREN_BLUE
            pygame.draw.circle(surf, colr, (int(cx + rx), int(cy - ry)), 5)
        rel = wrap_angle(p.heading - yaw)
        pts = [(cx + math.sin(rel + a) * r, cy - math.cos(rel + a) * r) for a, r in ((0.0, 9), (2.5, 7), (-2.5, 7))]
        pygame.draw.polygon(surf, UI_ACCENT, pts)


class PauseState(State):
    def __init__(self, game, play):
        super().__init__(game)
        self.play = play
        self.snapshot = game.screen.copy()
        self.snapshot.fill((105, 105, 115), special_flags=pygame.BLEND_MULT)
        cx, cy = SCREEN_W * 0.5, SCREEN_H * 0.5
        self.buttons = [Button("RESUME", (cx, cy + 10), self.resume),
                        Button("QUIT TO MENU", (cx, cy + 80), game.to_menu)]

    def resume(self):
        self.game.change(self.play)

    def handle(self, e):
        for b in self.buttons:
            if b.handle(e):
                return
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            self.resume()

    def draw(self, surf):
        surf.blit(self.snapshot, (0, 0))
        ui = self.game.ui
        ui.draw(surf, "PAUSED", 96, UI_ACCENT, (SCREEN_W * 0.5, SCREEN_H * 0.3))
        mouse = self.game.read_mouse()[:2]
        for b in self.buttons:
            b.draw(surf, ui, mouse)


class GameOverState(State):
    def __init__(self, game, play):
        super().__init__(game)
        self.play = play
        self.score = play.score()
        self.new_best = game.record_score(play.diff_name, self.score)
        self.t = 0.0
        cx = SCREEN_W * 0.5
        self.buttons = [Button("RESTART", (cx - 165, SCREEN_H * 0.86), game.start_run),
                        Button("MENU", (cx + 165, SCREEN_H * 0.86), game.to_menu)]

    def handle(self, e):
        if self.t < 0.6:     # ignore the click you were holding when you crashed
            return
        for b in self.buttons:
            if b.handle(e):
                return
        if e.type == pygame.KEYDOWN and e.key in (pygame.K_RETURN, pygame.K_SPACE):
            self.game.start_run()

    def update(self, dt):
        self.t += dt
        sc = self.play.scene
        sc.particles.update(dt)
        sc.follow(self.play.player, dt, yaw_offset=self.t * 0.35)

    def draw(self, surf):
        play, ui, g = self.play, self.game.ui, self.game
        play.draw(surf, hud=False)
        surf.fill((120, 115, 120), special_flags=pygame.BLEND_MULT)
        W, H = surf.get_size()
        ui.panel(surf, pygame.Rect(W // 2 - 340, int(H * 0.06), 680, int(H * 0.72)), alpha=180, radius=18)
        ui.draw(surf, "BUSTED!", 120, UI_DANGER, (W * 0.5, H * 0.16))
        ui.draw(surf, f"{play.diff_name.upper()}", 30, UI_DIM, (W * 0.5, H * 0.25))
        rows = (("Time survived", fmt_time(play.time)),
                ("Distance", f"{play.distance / 1000.0:.2f} km"),
                ("Top speed", f"{play.top_speed * 3.6:.0f} km/h"),
                ("Police wrecked", str(play.wrecked_cops)))
        y = H * 0.31
        for label, val in rows:
            ui.draw(surf, label, 30, UI_TEXT, (W * 0.5 - 220, y), "topleft", shadow=False)
            ui.draw(surf, val, 30, UI_TEXT, (W * 0.5 + 220, y), "topright", shadow=False)
            y += 34
        ui.draw(surf, f"SCORE  {self.score}", 64, UI_ACCENT, (W * 0.5, H * 0.56))
        if self.new_best:
            ui.draw(surf, "NEW BEST!", 34, UI_OK, (W * 0.5, H * 0.625))
        x = W * 0.5 - 230
        for name in DIFFICULTY_NAMES:
            colr = UI_ACCENT if name == play.diff_name else UI_DIM
            ui.draw(surf, f"Best {name}: {g.best.get(name, 0)}", 26, colr, (x, H * 0.70), shadow=False)
            x += 230
        mouse = g.read_mouse()[:2]
        for b in self.buttons:
            b.draw(surf, ui, mouse)


# =============================================================================
# 7. MAIN LOOP
# =============================================================================

class Game:
    def __init__(self, save_scores=True):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("PAKO CHASE")
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)
        self.ui = UI()
        self.world = World()
        self.save_scores = save_scores
        self.best = self.load_scores()
        self.difficulty = "Normal"
        self.show_fps = SHOW_FPS
        self.mouse_override = None     # (x, y, held): scripted input for the smoke test
        self.running = True
        self.state = None
        self.change(MenuState(self))

    def read_mouse(self):
        if self.mouse_override is not None:
            return self.mouse_override
        x, y = pygame.mouse.get_pos()
        return x, y, pygame.mouse.get_pressed()[0]

    def change(self, state):
        self.state = state
        pygame.mouse.set_visible(state.cursor_visible)
        pygame.event.set_grab(not state.cursor_visible)   # keep the mouse in the window while driving

    def set_difficulty(self, name):
        self.difficulty = name

    def start_run(self):
        self.change(PlayState(self, self.difficulty))

    def to_menu(self):
        self.change(MenuState(self))

    def quit(self):
        self.running = False

    def load_scores(self):
        best = {n: 0 for n in DIFFICULTY_NAMES}
        if self.save_scores:
            try:
                with open(SCORE_FILE, encoding="utf-8") as f:
                    data = json.load(f)
                for n in best:
                    best[n] = int(data.get(n, 0))
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        return best

    def record_score(self, difficulty, score):
        """Store a finished run; returns True for a new best."""
        if score <= self.best.get(difficulty, 0):
            return False
        self.best[difficulty] = score
        if self.save_scores:
            try:
                with open(SCORE_FILE, "w", encoding="utf-8") as f:
                    json.dump(self.best, f)
            except OSError:
                pass
        return True

    def frame(self, dt, events):
        for e in events:
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_F3:
                self.show_fps = not self.show_fps
            else:
                self.state.handle(e)
        self.state.update(dt)
        self.state.draw(self.screen)
        if self.show_fps:
            self.ui.draw(self.screen, f"{self.clock.get_fps():.0f} FPS   {self.renderer.faces_drawn} polys",
                         20, UI_TEXT, (SCREEN_W * 0.5, 8), "midtop")

    def run(self):
        while self.running:
            dt = min(self.clock.tick(TARGET_FPS) / 1000.0, 0.05)
            self.frame(dt, pygame.event.get())
            pygame.display.flip()


def _autopilot(play, route):
    """Smoke-test driver: follows a staircase of intersections, drifting into turns."""
    p = play.player
    while True:
        wx, wz = col_center(route[0][0]), row_center(route[0][1])
        if math.hypot(wx - p.x, wz - p.z) > 14.0 or len(route) == 1:
            break
        route.pop(0)
    err = wrap_angle(math.atan2(wx - p.x, wz - p.z) - p.heading)
    s = clamp(err * 2.0, -1.0, 1.0)
    off = STEER_DEADZONE + abs(s) ** (1.0 / STEER_CURVE) * (STEER_SATURATE - STEER_DEADZONE) if s else 0.0
    mx = SCREEN_W * 0.5 * (1.0 + math.copysign(off, s))
    return mx, SCREEN_H * 0.5, abs(err) > 0.55 and p.speed > 14.0


def smoke_test(frames=600):
    """Headless crash test: menu and controls screens, then a scripted run on every
    difficulty (pause/resume, forced BUSTED, restart, back to the menu)."""
    import traceback
    game = Game(save_scores=False)
    dt = 1.0 / 60.0
    Event = pygame.event.Event

    def click(button):
        game.frame(dt, [Event(pygame.MOUSEBUTTONDOWN, button=1, pos=button.rect.center)])

    def run(n):
        for _ in range(n):
            game.frame(dt, [])

    def finite(car):
        return all(math.isfinite(v) for v in (car.x, car.z, car.vx, car.vz, car.heading, car.health))

    try:
        run(30)
        assert isinstance(game.state, MenuState)
        click(game.state.buttons[1])
        assert isinstance(game.state, ControlsState), "Controls button"
        run(10)
        click(game.state.back)
        assert isinstance(game.state, MenuState), "Back button"
        print("menu + controls screens: ok")

        for name in DIFFICULTY_NAMES:
            menu = game.state
            assert isinstance(menu, MenuState)
            click(next(b for b in menu.diff_buttons if b.label == name))
            assert game.difficulty == name
            click(menu.buttons[0])
            play = game.state
            assert isinstance(play, PlayState) and play.diff_name == name, "Play button"
            route = [(0, 2), (1, 2), (1, 4), (2, 4), (2, 6), (3, 6), (3, 8), (4, 8), (4, 10), (5, 10),
                     (5, 12), (6, 12), (6, 14), (7, 14), (7, 16), (8, 16), (8, 18), (9, 18)]
            times, peak, paused = [], 0, False
            for f in range(frames):
                if game.state is play:
                    game.mouse_override = _autopilot(play, route)
                events = []
                if f == frames // 2 and game.state is play:
                    events = [Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode="\x1b", scancode=41)]
                t0 = time.perf_counter()
                game.frame(dt, events)
                times.append(time.perf_counter() - t0)
                if events:
                    assert isinstance(game.state, PauseState), "Esc should pause"
                    run(5)
                    click(game.state.buttons[0])
                    assert game.state is play, "Resume button"
                    paused = True
                if f == 120:    # make sure several police are around to exercise the AI
                    for _ in range(8):
                        if play.chasers() < play.diff["max_police"]:
                            play.director.spawn(play)
                assert finite(play.player), "player state went non-finite"
                for cop in play.police:
                    assert finite(cop), "police state went non-finite"
                assert play.chasers() <= play.diff["max_police"], "too many police"
                peak = max(peak, play.chasers())
            times.sort()
            p = play.player
            print(f"{name:6s}: {frames} frames, avg {1000 * sum(times) / len(times):5.1f} ms, "
                  f"p95 {1000 * times[int(len(times) * 0.95)]:5.1f} ms, police peak {peak}, "
                  f"speed {p.speed * 3.6:4.0f} km/h, health {p.health:5.1f}, score {play.score()}, "
                  f"polys {game.renderer.faces_drawn}, paused {'ok' if paused else 'skipped (busted)'}")

            if game.state is play:          # force a bust and walk through game over
                play.damage(1e6)
            for _ in range(240):
                if isinstance(game.state, GameOverState):
                    break
                game.frame(dt, [])
            over = game.state
            assert isinstance(over, GameOverState), "BUSTED should lead to game over"
            run(45)
            if name != DIFFICULTY_NAMES[-1]:
                click(over.buttons[0])                      # Restart
                assert isinstance(game.state, PlayState) and game.state is not play, "Restart"
                run(30)
                game.frame(dt, [Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode="\x1b", scancode=41)])
                click(game.state.buttons[1])                # Quit to menu
            else:
                click(over.buttons[1])                      # Menu
            assert isinstance(game.state, MenuState), "back to menu"
            game.mouse_override = None
        run(20)
        print("SMOKE TEST PASSED")
        return 0
    except Exception:
        traceback.print_exc()
        print("SMOKE TEST FAILED")
        return 1
    finally:
        pygame.quit()


def main():
    if "--smoke-test" in sys.argv[1:]:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        sys.exit(smoke_test())
    game = Game()
    try:
        game.run()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
