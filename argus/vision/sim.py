"""Physically-motivated simulator of a robot doing operator rounds.

The simulator is the "world" the agent acts in.  Actions really change the
evidence: specular glare moves when the camera moves (mirror geometry), motion
blur scales with shutter time, sensor noise follows shot/read noise with gain,
occluders in front of a dial show parallax, autofocus sometimes fails.

Ground truth (true value, true homography) is kept on the scene and is never
exposed to the agent's tools; it is used only for evaluation and tests.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..registry import CATALOGUE, GaugeSpec, Registry, make_spec
from . import geometry as g
from . import textures as tx

LIN = ((np.arange(256) / 255.0) ** 2.2).astype(np.float32)
GAMMA_LUT = ((np.linspace(0, 1, 4096) ** (1 / 2.2)) * 255.0 + 0.5).astype(np.uint8)

K_EXPO = 0.65  # radiance 1.0 at 10 ms, 0 dB -> 0.65 of full well
FULL_WELL = 2000.0
READ_E = 4.0
CAM_Y = -1150.0  # camera mast height (Y is negative upwards from the floor)
CORRIDOR_Z = -1300.0  # robot walkway, 1.3 m from the wall
VIEW_OFFSETS = {"front": 0.0, "left": -650.0, "right": 650.0}
FPS = 15.0


@dataclass
class SimGauge:
    spec: GaugeSpec
    center: np.ndarray
    true_value: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    fault: Optional[str] = None  # stuck_low | overrange
    flutter_amp: float = 0.0  # half amplitude, value units
    flutter_hz: float = 0.0
    flutter_phase: float = 0.0
    vibration_mm: float = 0.0
    vib_dir: float = 90.0
    haze: float = 0.0
    occluder: Optional[str] = None
    occluder_seed: int = 0
    glare: Optional[dict] = None
    style_seed: int = 0
    history: list = field(default_factory=list)
    conditions: list = field(default_factory=list)
    _asm: list = field(default=None, repr=False)
    _occ: tuple = field(default=None, repr=False)

    # ---- geometry
    def basis(self):
        Rg = g.rot_y(self.yaw) @ g.rot_x(self.pitch)
        cr, sr = math.cos(math.radians(self.roll)), math.sin(math.radians(self.roll))
        u = Rg @ np.array([cr, sr, 0.0])
        v = Rg @ np.array([-sr, cr, 0.0])
        n = Rg @ np.array([0.0, 0.0, -1.0])  # faces the room
        return u, v, n

    def needle_theta(self, t: float) -> float:
        s = self.spec
        if self.fault == "stuck_low":
            return s.theta_min - 4.5
        if self.fault == "overrange":
            return s.theta_max + 3.5
        val = self.true_value + self.flutter_amp * math.sin(2 * math.pi * self.flutter_hz * t + self.flutter_phase)
        return g.value_to_theta(val, s.min_value, s.max_value, s.theta_min, s.theta_max)

    def true_status(self) -> str:
        if self.fault == "stuck_low":
            return "instrument_fault"
        if self.fault == "overrange":
            return "alarm"
        return self.spec.status_of(self.true_value)

    def assembly_mips(self):
        if self._asm is None:
            base = tx.render_assembly(self.spec, self.style_seed)
            mips = [base]
            for _ in range(4):
                mips.append(cv2.pyrDown(mips[-1]))
            self._asm = mips
        return self._asm

    def occluder_tex(self):
        if self.occluder and self._occ is None:
            self._occ = tx.render_occluder(self.occluder, self.occluder_seed)
        return self._occ


@dataclass
class Scene:
    seed: int
    gauges: list
    ambient: float
    night: bool
    af_fail: float
    wall: dict
    profile: str = "random"

    @property
    def registry(self) -> Registry:
        r = Registry()
        for sg in self.gauges:
            r.add(sg.spec)
        return r

    def gauge(self, gid: str) -> SimGauge:
        for sg in self.gauges:
            if sg.spec.id == gid:
                return sg
        raise KeyError(gid)


# ---------------------------------------------------------------------------
# Rendering helpers


def _pick_mip(H_tex0: np.ndarray, center_px, n_levels: int) -> int:
    s = g.local_scale(H_tex0, center_px)
    lvl = 0
    while lvl < n_levels - 1 and s * (2 ** (lvl + 1)) <= 1.4:
        lvl += 1
    return lvl


def _tex_to_plane(ppm: float, a0: float, b0: float, level: int) -> np.ndarray:
    f = (2 ** level) / ppm
    return np.array([[f, 0, a0], [0, f, b0], [0, 0, 1]], np.float64)


def _roi_warp(src: np.ndarray, H_src_to_img: np.ndarray, W: int, Hh: int, flags=cv2.INTER_LINEAR):
    h, w = src.shape[:2]
    corners = g.apply_h(H_src_to_img, [(0, 0), (w, 0), (w, h), (0, h)])
    x0 = int(max(0, math.floor(corners[:, 0].min()) - 1))
    y0 = int(max(0, math.floor(corners[:, 1].min()) - 1))
    x1 = int(min(W, math.ceil(corners[:, 0].max()) + 1))
    y1 = int(min(Hh, math.ceil(corners[:, 1].max()) + 1))
    if x1 <= x0 or y1 <= y0:
        return None
    T = g.translate(-x0, -y0) @ H_src_to_img
    out = cv2.warpPerspective(src, T, (x1 - x0, y1 - y0), flags=flags, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)
    return x0, y0, x1, y1, out


def _composite(canvas: np.ndarray, roi) -> Optional[tuple]:
    if roi is None:
        return None
    x0, y0, x1, y1, w = roi
    a = (w[..., 3].astype(np.float32) / 255.0)[..., None]
    col = LIN[w[..., :3]]
    sub = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = sub * (1 - a) + col * a
    return x0, y0, x1, y1, a[..., 0]


def _line_kernel(length: float, angle_deg: float) -> np.ndarray:
    L = max(1, int(round(length)))
    if L <= 1:
        return np.ones((1, 1), np.float32)
    size = L if L % 2 == 1 else L + 1
    k = np.zeros((size, size), np.float32)
    c = size // 2
    t = math.radians(angle_deg)
    for i in range(L):
        s = i - (L - 1) / 2.0
        x = int(round(c + s * math.cos(t)))
        y = int(round(c + s * math.sin(t)))
        k[y, x] = 1.0
    return k / k.sum()


def sensor(radiance: np.ndarray, exposure_ms: float, gain_db: float, rng: np.random.Generator) -> np.ndarray:
    """Linear radiance -> 8-bit sRGB through a shot/read-noise sensor model.

    electrons e ~ N(mu, mu + read^2) (Gaussian approximation of Poisson shot noise plus read noise),
    clipped at full well, amplified by analogue gain, quantised, gamma-encoded.
    """
    e = cv2.multiply(np.minimum(radiance, 8.0), K_EXPO * exposure_ms / 10.0 * FULL_WELL)
    noise = np.empty_like(e)
    cv2.setRNGSeed(int(rng.integers(0, 2 ** 31 - 1)))
    cv2.randn(noise, 0.0, 1.0)
    std = cv2.sqrt(e + READ_E ** 2)
    e = np.minimum(e + noise * std, FULL_WELL)
    v = np.clip(e * (10 ** (gain_db / 20.0) / FULL_WELL * 4095.0), 0, 4095)
    return GAMMA_LUT[v.astype(np.int32)]


# ---------------------------------------------------------------------------


class RobotCamera:
    """A legged/wheeled inspection robot with a PTZ camera and a spotlight."""

    def __init__(self, scene: Scene, seed: int = 0, width: int = 1280, height: int = 720,
                 jpeg_quality: int = 90):
        self.scene = scene
        self.rng = np.random.default_rng(seed)
        self.W, self.H = width, height
        self.jpeg_quality = jpeg_quality
        self.f_wide = (width / 2) / math.tan(math.radians(35))
        self.pos = np.array([scene.gauges[0].center[0], CAM_Y, CORRIDOR_Z])
        self.at = (None, None)
        self.time_s = 0.0
        self.sim_t = 0.0  # process time (drives needle flutter)
        self.light_on = False
        self.captures = 0

    # ---- motion
    def viewpoint(self, sg: SimGauge, view: str) -> np.ndarray:
        return np.array([sg.center[0] + VIEW_OFFSETS[view], CAM_Y, CORRIDOR_Z])

    def move_to(self, gid: str, view: str = "front") -> float:
        sg = self.scene.gauge(gid)
        target = self.viewpoint(sg, view)
        if self.at == (gid, view):
            return 0.0
        dx = float(np.linalg.norm(target - self.pos))
        dt = (1.5 if self.at[0] == gid else 2.0) + dx / 1000.0 * 2.2
        self.pos = target
        self.at = (gid, view)
        self._advance(dt)
        return dt

    def _advance(self, dt: float):
        self.time_s += dt
        self.sim_t += dt

    def set_light(self, on: bool) -> float:
        if on != self.light_on:
            self.light_on = on
            self._advance(0.3)
            return 0.3
        return 0.0

    # ---- optics
    def pose_for(self, sg: SimGauge, zoom) -> tuple[np.ndarray, np.ndarray]:
        """Aim at the middle of the assembly (dial + fiducial tag) and set the focal length.

        ``zoom='auto'`` frames the whole assembly (dial radius ~26% of frame height);
        a number is a multiple of the auto framing (e.g. 1.4 = tighter on the dial).
        """
        _, v, _ = sg.basis()
        target = sg.center + 28.0 * v
        R = g.look_at(self.pos, target)
        dist = float(np.linalg.norm(target - self.pos))
        f_auto = 0.26 * self.H * dist / sg.spec.dial_radius_mm
        if zoom in (None, "auto"):
            f = f_auto
        else:
            f = f_auto * float(zoom)
        f = float(np.clip(f, self.f_wide, self.f_wide * 40))
        return R, g.intrinsics(f, self.W, self.H)

    def face_homography(self, sg: SimGauge, R, K, C=None) -> np.ndarray:
        u, v, _ = sg.basis()
        return g.plane_homography(K, R, self.pos if C is None else C, sg.center, u, v)

    def predicted_face_homography(self, gid: str, R, K) -> np.ndarray:
        """What the robot *believes* from its map + localisation (with realistic error).

        Uses the registry pose (no knowledge of an installer's roll/yaw) plus
        position and pan/tilt encoder noise.
        """
        sg = self.scene.gauge(gid)
        rng = np.random.default_rng(zlib.crc32(f"{gid}:{self.captures}:{self.scene.seed}".encode()))
        C = self.pos + rng.normal(0, 12.0, 3)
        dR = g.rot_y(rng.normal(0, 0.35)) @ g.rot_x(rng.normal(0, 0.35))
        Rn = dR @ R
        u, v = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        return g.plane_homography(K, Rn, C, sg.center, u, v)

    # ---- capture
    def auto_exposure(self, radiance: np.ndarray, ev_bias: float, fast_shutter: bool) -> tuple[float, float]:
        h, w = radiance.shape[:2]
        c = radiance[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)]
        m = float(np.mean(np.minimum(c, 2.0))) + 1e-4
        g_total = 0.40 / (m * K_EXPO) * (2.0 ** ev_bias)
        t_max = 4.0 if fast_shutter else 25.0
        t = float(np.clip(10.0 * g_total, 0.3, t_max))
        gain = max(1.0, 10.0 * g_total / t)
        gain_db = float(min(24.0, 20 * math.log10(gain)))
        return t, gain_db

    def render_radiance(self, R, K, t_frame: float, exposure_ms: float, gauges=None):
        """Scene radiance (linear float32) for one exposure, plus per-gauge truth."""
        sc = self.scene
        W, H = self.W, self.H
        canvas = np.zeros((H, W, 3), np.float32)
        wall = sc.wall
        H_wall = g.plane_homography(K, R, self.pos, np.array([wall["x0"], wall["y0"], 0.0]),
                                    np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
        lvl = _pick_mip(H_wall @ _tex_to_plane(wall["ppm"], 0, 0, 0),
                        (wall["mips"][0].shape[1] / 2, wall["mips"][0].shape[0] / 2), len(wall["mips"]))
        Hw = H_wall @ _tex_to_plane(wall["ppm"], 0, 0, lvl)
        bg = cv2.warpPerspective(wall["mips"][lvl], Hw, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        canvas[:] = LIN[bg]
        truth = {}
        glare_layers = []
        haze_layers = []
        for sg in (gauges or sc.gauges):
            Hf = self.face_homography(sg, R, K)
            cpx = g.apply_h(Hf, [(0, 0)])[0]
            u, v, n = sg.basis()
            if np.dot(self.pos - sg.center, n) <= 0:
                continue
            if not (-0.5 * W < cpx[0] < 1.5 * W and -0.5 * H < cpx[1] < 1.5 * H):
                continue
            mips = sg.assembly_mips()
            A0 = _tex_to_plane(tx.PPM, tx.ASM_A0, tx.ASM_B0, 0)
            lvl = _pick_mip(Hf @ A0, ((0 - tx.ASM_A0) * tx.PPM, (0 - tx.ASM_B0) * tx.PPM), len(mips))
            asm_roi = _composite(canvas, _roi_warp(mips[lvl], Hf @ _tex_to_plane(tx.PPM, tx.ASM_A0, tx.ASM_B0, lvl),
                                                   W, H))
            # needle (sub-sampled across the exposure if it is moving)
            ang_amp = sg.flutter_amp / sg.spec.span * (sg.spec.theta_max - sg.spec.theta_min)
            sweep_in_exposure = ang_amp * 2 * math.pi * sg.flutter_hz * exposure_ms / 1000.0
            n_sub = 1 if sweep_in_exposure < 0.4 else 5
            thetas = [sg.needle_theta(t_frame + (i / max(1, n_sub - 1)) * exposure_ms / 1000.0)
                      for i in range(n_sub)]
            ndl = tx.render_needle(sg.spec, thetas if n_sub > 1 else thetas[0])
            nm = [ndl]
            nl = _pick_mip(Hf @ _tex_to_plane(tx.PPM, tx.NDL_A0, tx.NDL_B0, 0),
                           (-tx.NDL_A0 * tx.PPM, -tx.NDL_B0 * tx.PPM), 5)
            for _ in range(nl):
                nm.append(cv2.pyrDown(nm[-1]))
            _composite(canvas, _roi_warp(nm[nl], Hf @ _tex_to_plane(tx.PPM, tx.NDL_A0, tx.NDL_B0, nl), W, H))
            # occluder in a plane 70 mm in front of the dial (parallax!)
            if sg.occluder:
                occ, oa0, ob0, oppm = sg.occluder_tex()
                Ho = g.plane_homography(K, R, self.pos, sg.center + n * 70.0, u, v)
                _composite(canvas, _roi_warp(occ, Ho @ _tex_to_plane(oppm, oa0, ob0, 0), W, H))
            # body vibration -> directional motion blur of the assembly
            if sg.vibration_mm > 0 and asm_roi is not None:
                ppm_img = g.local_scale(Hf, (0, 0))
                L = 2 * sg.vibration_mm * ppm_img * exposure_ms / 10.0
                if L > 1.2:
                    x0, y0, x1, y1, a = asm_roi
                    kern = _line_kernel(L, sg.vib_dir)
                    sub = canvas[y0:y1, x0:x1]
                    blurred = cv2.filter2D(sub, -1, kern, borderType=cv2.BORDER_REPLICATE)
                    m = cv2.filter2D(a, -1, kern)[..., None]
                    canvas[y0:y1, x0:x1] = sub * (1 - m) + blurred * m
            # glare & haze layers in face coordinates
            glare_layers.append((sg, Hf))
            if sg.haze > 0:
                haze_layers.append((sg, Hf))
            truth[sg.spec.id] = dict(H_face_to_img=Hf, center_px=cpx.tolist(),
                                     theta=float(np.mean(thetas)),
                                     radius_px=g.local_scale(Hf, (0, 0)) * sg.spec.dial_radius_mm)
        # illumination: ambient with a ceiling-light gradient, vignetting, optional spotlight
        base_ill, spot_ill = self._illum_maps()
        illum = sc.ambient * base_ill
        if self.light_on:
            illum = illum + 0.6 * spot_ill
        radiance = canvas * illum[..., None]
        for sg, Hf in haze_layers:
            hz = self._glass_texture(sg, lambda A, B, r: sg.haze * (0.75 + 0.25 * np.sin(A / 9.0 + B / 13.0)))
            roi = _roi_warp(hz, Hf @ _tex_to_plane(2.0, -60, -60, 0), W, H)
            if roi:
                x0, y0, x1, y1, hm = roi
                hm = hm[..., None]
                sub = radiance[y0:y1, x0:x1]
                veil = 0.72 * illum[y0:y1, x0:x1, None]
                mixed = sub * (1 - hm) + veil * hm
                radiance[y0:y1, x0:x1] = mixed * (1 - hm) + cv2.GaussianBlur(mixed, (0, 0), 1.6) * hm
        for sg, Hf in glare_layers:
            for light, inten, sig, elong, ang in self._glare_sources(sg):
                layer = self._glare_texture(sg, light, sig, elong, ang)
                if layer is None:
                    continue
                roi = _roi_warp(layer, Hf @ _tex_to_plane(2.0, -60, -60, 0), W, H)
                if roi:
                    x0, y0, x1, y1, gl = roi
                    radiance[y0:y1, x0:x1] += (gl * inten)[..., None]
        return radiance, truth

    def _illum_maps(self):
        if getattr(self, "_ill", None) is None:
            H, W = self.H, self.W
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            rn = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (0.75 * W) ** 2
            self._ill = ((1.05 - 0.15 * yy / H) * (1 - 0.25 * rn), np.exp(-rn * 1.6))
        return self._ill

    def _glass_texture(self, sg: SimGauge, fn) -> np.ndarray:
        n = 240
        b, a = np.mgrid[0:n, 0:n].astype(np.float32)
        A = a / 2.0 - 60
        B = b / 2.0 - 60
        r = np.sqrt(A ** 2 + B ** 2)
        R = sg.spec.dial_radius_mm * 1.06
        edge = np.clip((R - r) / 1.5, 0, 1)
        return (fn(A, B, r) * edge).astype(np.float32)

    def _glare_sources(self, sg: SimGauge):
        out = []
        if sg.glare:
            gl = sg.glare
            out.append((np.array(gl["light"]), gl["intensity"], gl["sigma_mm"], gl["elong"], gl["angle"]))
        if self.light_on:
            L = self.pos + np.array([0.0, -120.0, 0.0])
            d = float(np.linalg.norm(sg.center - self.pos))
            out.append((L, 2.2 * (1300.0 / d) ** 2, 9.0, 1.0, 0.0))
        return out

    def _glare_texture(self, sg: SimGauge, light, sig, elong, ang):
        u, v, n = sg.basis()
        G = sg.center
        C = self.pos
        if np.dot(light - G, n) <= 0:
            return None
        Lm = light - 2 * np.dot(light - G, n) * n
        den = np.dot(Lm - C, n)
        if abs(den) < 1e-9:
            return None
        t = np.dot(G - C, n) / den
        Q = C + t * (Lm - C)
        qa, qb = float(np.dot(Q - G, u)), float(np.dot(Q - G, v))
        if math.hypot(qa, qb) > sg.spec.dial_radius_mm * 1.06 + 3 * sig * elong:
            return None
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))

        def fn(A, B, r):
            da, db = A - qa, B - qb
            p = da * ca + db * sa
            q = -da * sa + db * ca
            return np.exp(-(p ** 2) / (2 * (sig * elong) ** 2) - (q ** 2) / (2 * sig ** 2))
        return self._glass_texture(sg, fn)

    def _finish(self, radiance, exposure_ms, gain_db, defocus_sigma):
        if defocus_sigma > 0:
            radiance = cv2.GaussianBlur(radiance, (0, 0), defocus_sigma)
        img = sensor(radiance, exposure_ms, gain_db, self.rng)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def capture(self, gid: str, view: str = "front", zoom="auto", ev_bias: float = 0.0, burst: int = 1,
                hdr: bool = False, light: Optional[bool] = None, fast_shutter: bool = False) -> dict:
        """Point at gauge ``gid`` from ``view`` and capture frame(s).  Returns frames + settings + truth."""
        sg = self.scene.gauge(gid)
        travel = self.move_to(gid, view)
        if light is not None:
            travel += self.set_light(bool(light))
        R, K = self.pose_for(sg, zoom)
        self._advance(1.2 + 0.4)  # PTZ slew + autofocus
        self.captures += 1
        af_p = self.scene.af_fail * (1.8 if self.scene.night and not self.light_on else 1.0)
        defocus = float(self.rng.uniform(2.2, 4.5)) if self.rng.random() < af_p else 0.0
        # meter on a preview (radiance does not depend on exposure for a static gauge)
        rad_prev, truth_prev = self.render_radiance(R, K, self.sim_t, 10.0, gauges=[sg])
        t_ms, gain_db = self.auto_exposure(rad_prev, ev_bias, fast_shutter)
        static = sg.flutter_amp == 0 and sg.vibration_mm == 0
        evs = [-2.5, -1.0, 0.5] if hdr else [0.0]
        frames, truths, settings = [], [], []
        n_frames = 3 if hdr else int(np.clip(burst, 1, 9))
        for i in range(n_frames):
            ev = evs[i] if hdr else 0.0
            te = float(np.clip(t_ms * (2 ** ev), 0.1, 66.0))
            if static:
                rad, truth = rad_prev, truth_prev
            else:
                rad, truth = self.render_radiance(R, K, self.sim_t, te, gauges=[sg])
            frames.append(self._finish(rad, te, gain_db, defocus))
            truths.append(truth.get(gid))
            settings.append({"exposure_ms": round(te, 2), "gain_db": round(gain_db, 1)})
            self._advance(max(1.0 / FPS, te / 1000.0))
        return dict(frames=frames, truth=truths, settings=settings, K=K, R=R, travel_s=travel, hdr=hdr,
                    defocus=defocus, light=self.light_on, view=view, gauge_id=gid,
                    predicted_H=self.predicted_face_homography(gid, R, K), f_px=float(K[0, 0]))

    def overview(self) -> dict:
        xs = [sg.center[0] for sg in self.scene.gauges]
        ys = [sg.center[1] for sg in self.scene.gauges]
        half = (max(xs) - min(xs)) / 2 + 450.0
        dist = half / math.tan(math.radians(33))
        self.pos = np.array([float(np.mean(xs)), CAM_Y, -dist])
        self.at = (None, "overview")
        self._advance(3.0)
        target = np.array([float(np.mean(xs)), float(np.mean(ys)), 0.0])
        R = g.look_at(self.pos, target)
        K = g.intrinsics(self.f_wide, self.W, self.H)
        rad_prev, _ = self.render_radiance(R, K, self.sim_t, 10.0)
        t_ms, gain_db = self.auto_exposure(rad_prev, 0.0, False)
        rad, truth = self.render_radiance(R, K, self.sim_t, t_ms)
        img = self._finish(rad, t_ms, gain_db, 0.0)
        self.captures += 1
        return dict(frames=[img], truth=[truth], settings=[{"exposure_ms": t_ms, "gain_db": gain_db}], K=K, R=R)


# ---------------------------------------------------------------------------
# Scenario generation

PROFILES = {
    "random": "Mixed conditions sampled per gauge (the evaluation distribution).",
    "glare": "Day shift: sunlight and overhead lights reflecting on the gauge glass.",
    "night": "Night shift: low ambient light, the robot must manage exposure, noise and its own spotlight.",
    "vibration": "Pump skid running: needle flutter and body vibration.",
    "clean": "Benign conditions: well lit, frontal, static.",
    "hard": "Everything at once: glare, vibration, haze, occlusion, oblique mounting, night.",
}


def _sample_value(spec: GaugeSpec, rng: np.random.Generator, cls: str) -> float:
    lo, hi = spec.min_value, spec.max_value
    span = spec.span
    la, lw, hw, ha = spec.low_alarm, spec.low_warn, spec.high_warn, spec.high_alarm
    n_lo = lw if lw is not None else (la if la is not None else lo)
    n_hi = hw if hw is not None else (ha if ha is not None else hi)
    m = 0.03 * span
    if cls == "normal":
        return float(rng.uniform(n_lo + m, n_hi - m))
    if cls == "near":
        thr = [t for t in (la, lw, hw, ha) if t is not None]
        t = float(rng.choice(thr))
        return float(np.clip(t + rng.choice([-1, 1]) * rng.uniform(0.006, 0.025) * span, lo + 0.01 * span,
                             hi - 0.01 * span))
    if cls == "warning":
        opts = []
        if hw is not None:
            opts.append((hw + 0.01 * span, (ha if ha is not None else hi) - 0.01 * span))
        if lw is not None:
            opts.append(((la if la is not None else lo) + 0.01 * span, lw - 0.01 * span))
        if not opts:
            return _sample_value(spec, rng, "normal")
        a, b = opts[rng.integers(0, len(opts))]
        return float(rng.uniform(a, b)) if b > a else float(a)
    if cls == "alarm":
        opts = []
        if ha is not None:
            opts.append((ha + 0.01 * span, hi - 0.02 * span))
        if la is not None:
            opts.append((lo + 0.02 * span, la - 0.01 * span))
        if not opts:
            return _sample_value(spec, rng, "warning")
        a, b = opts[rng.integers(0, len(opts))]
        return float(rng.uniform(a, b)) if b > a else float(a)
    return float(rng.uniform(lo, hi))


def _place_glare(sg: SimGauge, rng, on_needle: bool, intensity: float):
    """Put a light where its mirror image lands on the dial as seen from the FRONT view."""
    u, v, n = sg.basis()
    C = np.array([sg.center[0], CAM_Y, CORRIDOR_Z])
    R = sg.spec.dial_radius_mm
    if on_needle:
        th = sg.needle_theta(0.0) + rng.uniform(-18, 18)
    else:
        th = rng.uniform(-180, 180)
    a, b = g.face_point(th, rng.uniform(0.2, 0.7) * R)
    Q = sg.center + a * u + b * v
    d = Q - C
    # point beyond the plane along the ray, at distance ~1.6 m behind it, then mirror back
    tt = 1.0 + rng.uniform(0.9, 1.6)
    Lm = C + tt * d
    L = Lm - 2 * np.dot(Lm - sg.center, n) * n
    sg.glare = dict(light=L.tolist(), intensity=float(intensity), sigma_mm=float(rng.uniform(7, 14)),
                    elong=float(rng.uniform(1.0, 2.6)), angle=float(rng.uniform(0, 180)))


def make_scene(seed: int = 0, profile: str = "random", n_gauges: int = 5, catalogue_offset: Optional[int] = None,
               condition_rates: Optional[dict] = None) -> Scene:
    rng = np.random.default_rng(seed)
    rates = dict(glare=0.35, vibration=0.25, haze=0.12, occluder=0.12, oblique=0.18, roll=0.3, pitch=0.2,
                 night=0.3, fault=0.07)
    if profile == "clean":
        rates = {k: 0.0 for k in rates}
    elif profile == "glare":
        rates.update(glare=0.85, vibration=0.0, night=0.0, haze=0.1, occluder=0.0, oblique=0.1)
    elif profile == "night":
        rates.update(night=1.0, glare=0.0, vibration=0.3, occluder=0.0)
    elif profile == "vibration":
        rates.update(vibration=0.9, glare=0.15, night=0.0)
    elif profile == "hard":
        rates.update(glare=0.6, vibration=0.5, haze=0.3, occluder=0.3, oblique=0.35, night=0.5)
    if condition_rates:
        rates.update(condition_rates)
    night = bool(rng.random() < rates["night"])
    ambient = float(rng.uniform(0.045, 0.09)) if night else float(rng.uniform(0.75, 1.1))
    off = int(rng.integers(0, len(CATALOGUE))) if catalogue_offset is None else catalogue_offset
    gauges = []
    x = 0.0
    for i in range(n_gauges):
        spec = make_spec((off + i) % len(CATALOGUE), marker_id=i + 1)
        spec.location = f"Bay {1 + i // 3}, position {i % 3 + 1}"
        y = -float(rng.uniform(1080, 1420))
        sg = SimGauge(spec=spec, center=np.array([x, y, 0.0]), true_value=0.0,
                      style_seed=int(rng.integers(0, 1 << 30)))
        cls = rng.choice(["normal", "warning", "alarm", "near"], p=[0.64, 0.14, 0.13, 0.09])
        sg.true_value = _sample_value(spec, rng, str(cls))
        if rng.random() < rates["fault"]:
            sg.fault = str(rng.choice(["stuck_low", "overrange"]))
            sg.conditions.append("fault")
        if rng.random() < rates["oblique"]:
            sg.yaw = float(rng.choice([-1, 1]) * rng.uniform(32, 52))
            sg.conditions.append("oblique")
        if rng.random() < rates["roll"]:
            sg.roll = float(rng.uniform(-8, 8))
        if rng.random() < rates["pitch"]:
            sg.pitch = float(rng.uniform(-14, 14))
        if rng.random() < rates["vibration"]:
            sg.flutter_amp = float(rng.uniform(0.008, 0.03)) * spec.span
            sg.flutter_hz = float(rng.uniform(4, 12))
            sg.flutter_phase = float(rng.uniform(0, 2 * math.pi))
            sg.vibration_mm = float(rng.uniform(0.3, 1.1))
            sg.vib_dir = float(rng.uniform(0, 180))
            sg.conditions.append("vibration")
        if rng.random() < rates["haze"]:
            sg.haze = float(rng.uniform(0.25, 0.5))
            sg.conditions.append("haze")
        if rng.random() < rates["occluder"]:
            sg.occluder = str(rng.choice(["cable", "tag"]))
            sg.occluder_seed = int(rng.integers(0, 1 << 30))
            sg.conditions.append("occlusion")
        if rng.random() < rates["glare"]:
            _place_glare(sg, rng, on_needle=bool(rng.random() < 0.55), intensity=float(rng.uniform(1.6, 4.5)))
            sg.conditions.append("glare")
        if night:
            sg.conditions.append("night")
        if not sg.conditions:
            sg.conditions.append("clean")
        base = float(rng.uniform(0.35, 0.6)) * spec.span + spec.min_value
        sg.history = [round(float(base + rng.normal(0, 0.02 * spec.span)), 3) for _ in range(6)]
        gauges.append(sg)
        x += float(rng.uniform(900, 1300))
    xs = [sg.center[0] for sg in gauges]
    x0, x1 = min(xs) - 2500.0, max(xs) + 2500.0
    wall_ppm = 0.6
    labels = [(sg.center[0], sg.center[1] - 165.0, sg.spec.id) for sg in gauges]
    tex = tx.render_wall(x1 - x0, 2700.0, wall_ppm, xs, labels, seed=seed, x0_mm=x0, y0_mm=-2700.0)
    mips = [tex]
    for _ in range(3):
        mips.append(cv2.pyrDown(mips[-1]))
    wall = dict(x0=x0, y0=-2700.0, ppm=wall_ppm, mips=mips)
    return Scene(seed=seed, gauges=gauges, ambient=ambient, night=night, af_fail=0.06, wall=wall, profile=profile)
