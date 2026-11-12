"""Synthetic localizer stimuli + their design descriptions.

Every experiment writes:
  <name>.mp4 or <name>.wav   the stimulus TRIBE sees
  <name>_design.npz          what was shown when, sampled at 10 Hz (dt = 0.1 s)

Visual coordinates: degrees from the frame centre, +x right, +y UP.
Polar angle: 0 = upper vertical meridian, +90 = right horizontal meridian,
+/-180 = lower vertical meridian, -90 = left horizontal meridian.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile

DT = 0.1            # design sampling (s)
GRID = 64           # design aperture grid (GRID x GRID covering the frame)


# --------------------------------------------------------------------- geometry
def deg_grid(n: int, max_ecc: float):
    """x, y (deg), eccentricity and polar angle for an n x n pixel grid."""
    c = (np.arange(n) + 0.5) / n * 2 - 1           # -1..1
    x = c[None, :] * max_ecc
    y = -c[:, None] * max_ecc
    x, y = np.broadcast_arrays(x, y)
    r = np.hypot(x, y)
    theta = np.degrees(np.arctan2(x, y))            # 0 = up, 90 = right
    return x, y, r, theta


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def checkerboard(r, theta, phase: int, n_rings: float = 6.0, n_wedges: int = 24):
    ring = np.floor(n_rings * np.log(r + 0.3)).astype(int)
    wedge = np.floor((theta + 180.0) / (360.0 / n_wedges)).astype(int)
    return ((ring + wedge + phase) % 2).astype(np.uint8)


def _ring_band(centre: float, ring_min_ecc: float, max_ecc: float, area_match: bool):
    """Inner/outer radius of the ring aperture at this centre eccentricity.

    Default: a fixed log-width band (centre/1.5 .. centre*1.5), so stimulated area
    grows with eccentricity. area_match=True instead holds pi*(outer^2-inner^2)
    constant across the whole sweep (same geometric-mean radius `centre`), to test
    whether an eccentricity effect is driven by position or by stimulated area."""
    if not area_match:
        return centre / 1.5, centre * 1.5
    # K = area constant (as r_outer^2 - r_inner^2), chosen to match the default
    # band's area at the sweep's geometric-mean centre so the two controls are
    # comparable in typical stimulated area.
    ref_centre = np.sqrt(ring_min_ecc * max_ecc)
    K = ref_centre ** 2 * (1.5 ** 2 - (1.0 / 1.5) ** 2)
    c2 = centre ** 2
    u = (-K + np.sqrt(K ** 2 + 4 * c2 ** 2)) / 2.0           # r_inner^2, from
    inner = np.sqrt(max(u, 1e-9))                            # r_inner*r_outer=c2 and
    outer = c2 / inner                                        # r_outer^2-r_inner^2=K
    return inner, outer


# --------------------------------------------------------------------- writers
def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


class VideoWriter:
    """Pipes RGB frames into ffmpeg (H.264) and adds a SILENT audio track,
    because TRIBE extracts audio from every video it is given."""

    def __init__(self, path: Path, size: int, fps: int):
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}", "-r", str(fps), "-i", "-",
               "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
               "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
               "-c:a", "aac", str(path)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_gray_or_rgb: np.ndarray) -> None:
        f = frame_gray_or_rgb
        if f.ndim == 2:
            f = np.repeat(f[:, :, None], 3, axis=2)
        self.p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def add_fixation(frame: np.ndarray, size_px: int) -> np.ndarray:
    n = frame.shape[0]
    c = n // 2
    out = frame if frame.ndim == 3 else np.repeat(frame[:, :, None], 3, axis=2)
    out[c - size_px:c + size_px, c - size_px:c + size_px] = (220, 40, 40)
    return out


# --------------------------------------------------------------------- retinotopy
def _aperture(kind: str, t_in: float, cfg_r: dict, max_ecc: float, r, theta, x):
    """Binary aperture at time t_in (s since content onset) for one run kind."""
    if kind == "wedge":
        P = cfg_r["wedge_period"]
        n = cfg_r["wedge_cycles_each_direction"]
        direction = 1.0 if t_in < n * P else -1.0           # clockwise, then counter-clockwise
        phi = direction * 360.0 * ((t_in % P) / P)
        return (np.abs(wrap180(theta - phi)) <= cfg_r["wedge_width_deg"] / 2) & (r <= max_ecc), phi
    if kind == "ring":
        P = cfg_r["ring_period"]
        n = cfg_r["ring_cycles_each_direction"]
        ph = (t_in % P) / P
        if t_in >= n * P:                                   # contracting half
            ph = 1.0 - ph
        lo, hi = np.log(cfg_r["ring_min_ecc_deg"]), np.log(max_ecc)
        centre = np.exp(lo + ph * (hi - lo))
        inner, outer = _ring_band(centre, cfg_r["ring_min_ecc_deg"], max_ecc,
                                   cfg_r.get("ring_area_match", False))
        return (r >= inner) & (r <= outer) & (r <= max_ecc), centre
    if kind == "left":
        return (x < 0) & (r <= max_ecc), np.nan
    if kind == "right":
        return (x > 0) & (r <= max_ecc), np.nan
    raise ValueError(kind)


def make_retinotopy(out_dir: Path, cfg: dict, kind: str) -> dict:
    """kind = 'wedge' | 'ring'"""
    S, fps = cfg["screen"]["size_px"], cfg["screen"]["fps"]
    E, bg = cfg["screen"]["max_ecc_deg"], cfg["screen"]["background"]
    R = cfg["retinotopy"]
    pad = R["blank_pad"]
    P = R[f"{kind}_period"]
    content = 2 * R[f"{kind}_cycles_each_direction"] * P
    dur = pad + content + pad
    _, _, r, th = deg_grid(S, E)
    xg, _, _, _ = deg_grid(S, E)
    gx, _, gr, gth = deg_grid(GRID, E)
    name = f"retino_{kind}"
    n_rings, n_wedges = R.get("check_n_rings", 6.0), R.get("check_n_wedges", 24)
    vw = VideoWriter(out_dir / f"{name}.mp4", S, fps)
    flip_every = max(1, int(round(fps / (2 * R["flicker_hz"]))))
    for f in range(int(round(dur * fps))):
        t = f / fps
        frame = np.full((S, S), bg, np.uint8)
        if pad <= t < pad + content:
            ap, _ = _aperture(kind, t - pad, R, E, r, th, xg)
            cb = checkerboard(r, th, (f // flip_every) % 2, n_rings, n_wedges)
            frame[ap] = np.where(cb[ap] == 1, 255, 0)
        vw.write(add_fixation(frame, cfg["screen"]["fixation_px"]))
    vw.close()
    # design at 10 Hz
    T = int(round(dur / DT))
    apert = np.zeros((T, GRID, GRID), np.uint8)
    param = np.full(T, np.nan)
    for i in range(T):
        t = (i + 0.5) * DT
        if pad <= t < pad + content:
            ap, p = _aperture(kind, t - pad, R, E, gr, gth, gx)
            apert[i] = ap
            param[i] = p
    np.savez_compressed(out_dir / f"{name}_design.npz", aperture=apert, param=param, dt=DT,
                        max_ecc=E, duration=dur, kind=kind)
    return {"name": name, "file": f"{name}.mp4", "modality": "video", "duration": dur, "kind": "retinotopy"}


def make_hemifield(out_dir: Path, cfg: dict) -> dict:
    S, fps = cfg["screen"]["size_px"], cfg["screen"]["fps"]
    E, bg = cfg["screen"]["max_ecc_deg"], cfg["screen"]["background"]
    H = cfg["hemifield"]
    pad = cfg["retinotopy"]["blank_pad"]
    blocks = ["left", "right", "blank"] * H["repeats"]
    dur = pad + len(blocks) * H["block"] + pad
    _, _, r, th = deg_grid(S, E)
    x, _, _, _ = deg_grid(S, E)
    vw = VideoWriter(out_dir / "hemifield.mp4", S, fps)
    flip_every = max(1, int(round(fps / (2 * cfg["retinotopy"]["flicker_hz"]))))

    def side_at(t):
        if t < pad or t >= pad + len(blocks) * H["block"]:
            return "pad"
        return blocks[int((t - pad) // H["block"])]

    for f in range(int(round(dur * fps))):
        t = f / fps
        frame = np.full((S, S), bg, np.uint8)
        s = side_at(t)
        if s in ("left", "right"):
            ap, _ = _aperture(s, 0, cfg["retinotopy"], E, r, th, x)
            cb = checkerboard(r, th, (f // flip_every) % 2)
            frame[ap] = np.where(cb[ap] == 1, 255, 0)
        vw.write(add_fixation(frame, cfg["screen"]["fixation_px"]))
    vw.close()
    T = int(round(dur / DT))
    code = {"pad": -1, "blank": 2, "left": 0, "right": 1}
    side = np.array([code[side_at((i + 0.5) * DT)] for i in range(T)])
    np.savez_compressed(out_dir / "hemifield_design.npz", side=side, dt=DT, duration=dur)
    return {"name": "hemifield", "file": "hemifield.mp4", "modality": "video", "duration": dur, "kind": "hemifield"}


# --------------------------------------------------------------------- motion
def make_motion(out_dir: Path, cfg: dict, seed: int = 0) -> dict:
    """Moving (alternately expanding / contracting) vs static dots, MT+ localizer."""
    S, fps = cfg["screen"]["size_px"], cfg["screen"]["fps"]
    E, bg = cfg["screen"]["max_ecc_deg"], cfg["screen"]["background"]
    M = cfg["motion"]
    pad = cfg["retinotopy"]["blank_pad"]
    rng = np.random.default_rng(seed)
    content = 2 * M["cycles"] * M["block"]
    dur = pad + content + pad
    px_per_deg = (S / 2) / E
    n = M["n_dots"]

    def respawn(k):
        rr = np.sqrt(rng.uniform(0.3 ** 2, E ** 2, k))
        aa = rng.uniform(0, 2 * np.pi, k)
        return np.c_[rr * np.sin(aa), rr * np.cos(aa)]

    pos = respawn(n)
    pol = rng.choice([0, 255], n)
    last_refresh = -1.0
    vw = VideoWriter(out_dir / "motion.mp4", S, fps)
    d = M["dot_px"]

    def state(t):
        if t < pad or t >= pad + content:
            return "pad"
        return "moving" if int((t - pad) // M["block"]) % 2 == 0 else "static"

    for f in range(int(round(dur * fps))):
        t = f / fps
        st = state(t)
        frame = np.full((S, S), bg, np.uint8)
        if st == "moving":
            sign = 1.0 if int((t - pad) // M["direction_switch_s"]) % 2 == 0 else -1.0
            r = np.hypot(pos[:, 0], pos[:, 1])
            step = sign * M["speed_deg_s"] / fps
            pos = pos * ((r + step) / np.maximum(r, 1e-6))[:, None]
            r = np.hypot(pos[:, 0], pos[:, 1])
            bad = (r > E) | (r < 0.3)
            pos[bad] = respawn(bad.sum())
        elif st == "static" and (t - last_refresh) >= M["direction_switch_s"]:
            pos = respawn(n)                       # same refresh rate as direction switches
            last_refresh = t
        if st != "pad":
            cols = (pos[:, 0] * px_per_deg + S / 2).astype(int)
            rows = (-pos[:, 1] * px_per_deg + S / 2).astype(int)
            for rr_, cc_, pv in zip(rows, cols, pol):
                frame[max(rr_ - d, 0):rr_ + d, max(cc_ - d, 0):cc_ + d] = pv
        vw.write(add_fixation(frame, cfg["screen"]["fixation_px"]))
    vw.close()
    T = int(round(dur / DT))
    code = {"pad": -1, "static": 0, "moving": 1}
    moving = np.array([code[state((i + 0.5) * DT)] for i in range(T)])
    np.savez_compressed(out_dir / "motion_design.npz", moving=moving, dt=DT, duration=dur)
    return {"name": "motion", "file": "motion.mp4", "modality": "video", "duration": dur, "kind": "motion"}


# --------------------------------------------------------------------- tonotopy
def band_edges(cfg_t: dict) -> np.ndarray:
    k = np.arange(cfg_t["n_bands"] + 1)
    return cfg_t["f_low_hz"] * 2 ** (k * cfg_t["band_semitones"] / 12.0)


# ISO 226:2003 equal-loudness-level contours: standard table (frequency, af, Lu, Tf),
# log-frequency interpolated. Used only when tonotopy.equalize_loudness is set, to build
# a tone set that is approximately equally LOUD across bands rather than equal in
# amplitude -- ruling out "best frequency = loudest frequency" as an alternative account
# of a flat/uniform best-frequency map (see paper Section 4, loudness control).
_ISO226_F = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
                      630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                      10000, 12500], float)
_ISO226_AF = np.array([0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330,
                       0.315, 0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244,
                       0.243, 0.243, 0.243, 0.242, 0.242, 0.245, 0.254, 0.271, 0.301], float)
_ISO226_LU = np.array([-31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5,
                       -3.1, -2.0, -1.1, -0.4, 0.0, 0.3, 0.5, 0.0, -2.7, -4.1, -1.0, 1.7,
                       2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1], float)
_ISO226_TF = np.array([78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4,
                       8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6.0, -5.4, -1.5,
                       6.0, 12.6, 13.9, 12.3], float)


def iso226_spl(freq_hz, phon: float = 60.0) -> float:
    """SPL (dB) of a pure tone at freq_hz perceived as loud as `phon` phon
    (= phon dB SPL at 1 kHz by definition), per ISO 226:2003."""
    f = np.clip(freq_hz, _ISO226_F[0], _ISO226_F[-1])
    logf = np.log(_ISO226_F)
    af = np.interp(np.log(f), logf, _ISO226_AF)
    Lu = np.interp(np.log(f), logf, _ISO226_LU)
    Tf = np.interp(np.log(f), logf, _ISO226_TF)
    Af = 4.47e-3 * (10 ** (0.025 * phon) - 1.15) + (0.4 * 10 ** ((Tf + Lu) / 10 - 9)) ** af
    return (10.0 / af) * np.log10(Af) - Lu + 94.0


def equal_loudness_gain_db(freq_hz, ref_hz: float = 1000.0, phon: float = 60.0) -> float:
    """dB gain (relative to a tone at ref_hz) that makes freq_hz perceptually as
    loud as ref_hz, per the ISO 226:2003 equal-loudness contour at `phon`."""
    return iso226_spl(ref_hz, phon) - iso226_spl(freq_hz, phon)


def make_tonotopy(out_dir: Path, cfg: dict, direction: str, seed: int = 0) -> dict:
    """Tone-pip 'motifs' (4 tones) drawn from one frequency band at a time,
    stepping through the bands (as in the NNDb-3T+ tonotopy task)."""
    T_ = cfg["tonotopy"]
    equalize = T_.get("equalize_loudness", False)
    rng = np.random.default_rng(seed + (0 if direction == "up" else 1))
    sr = T_["sample_rate"]
    edges = band_edges(T_)
    nb = T_["n_bands"]
    order = list(range(nb)) if direction == "up" else list(range(nb - 1, -1, -1))
    seq = order * T_["sweeps_each_direction"]
    pad = T_["blank_pad"]
    dur = pad + len(seq) * T_["band_duration"] + pad
    audio = np.zeros(int(round(dur * sr)), np.float32)
    amp = 10 ** (T_["level_dbfs"] / 20)
    tone_n = int(sr * T_["tone_ms"] / 1000)
    gap_n = int(sr * T_["gap_ms"] / 1000)
    ramp = int(0.01 * sr)
    env = np.ones(tone_n)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    tt = np.arange(tone_n) / sr
    for i, b in enumerate(seq):
        start = int((pad + i * T_["band_duration"]) * sr)
        end = int((pad + (i + 1) * T_["band_duration"]) * sr)
        p = start
        while p + tone_n <= end:
            f = np.exp(rng.uniform(np.log(edges[b]), np.log(edges[b + 1])))
            a = amp * 10 ** (equal_loudness_gain_db(f) / 20) if equalize else amp
            audio[p:p + tone_n] += a * env * np.sin(2 * np.pi * f * tt)
            p += tone_n + gap_n
    name = f"tono_{direction}"
    wavfile.write(out_dir / f"{name}.wav", sr, (audio * 32767).astype(np.int16))
    T = int(round(dur / DT))
    band = np.full(T, -1)
    for i in range(T):
        t = (i + 0.5) * DT
        if pad <= t < pad + len(seq) * T_["band_duration"]:
            band[i] = seq[int((t - pad) // T_["band_duration"])]
    centres = np.sqrt(edges[:-1] * edges[1:])
    np.savez_compressed(out_dir / f"{name}_design.npz", band=band, band_centres_hz=centres,
                        dt=DT, duration=dur)
    return {"name": name, "file": f"{name}.wav", "modality": "audio", "duration": dur, "kind": "tonotopy"}
