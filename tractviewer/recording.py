"""Writing videos / GIFs from RGB frames, through ffmpeg.

Two entry points:
  - write_gif(frames, path, fps): infinitely looping GIF (optimized palette).
  - write_video(frames, path, ...): video, with codec selection/fallback.

Both try ffmpeg (system binary or imageio-ffmpeg) and raise an explicit
RuntimeError when no backend is available.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union
import contextlib
import shutil
import subprocess
import tempfile
import warnings
import numpy as np

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None


def write_gif(frames: List[np.ndarray], output_path: Union[str, Path], fps: int = 10) -> str:
    """Write an infinitely looping GIF. Tries ffmpeg (palette), falls back to imageio."""
    try:
        return _gif_with_ffmpeg(frames, output_path, fps=fps)
    except Exception as e:
        warnings.warn(f"ffmpeg GIF creation failed: {e}. Falling back to imageio.")
        if imageio is None:
            raise RuntimeError("imageio unavailable for the GIF fallback.") from e
        imageio.mimsave(str(output_path), frames, format="GIF", duration=1000 / fps, loop=0)
        return str(output_path)


def write_video(
    frames: List[np.ndarray],
    output_path: Union[str, Path],
    fps: int = 30,
    quality: int = 5,
    codec: str = "mpeg4",
    bitrate: Optional[str] = None,
    crf: Optional[int] = None,
    auto_codec_fallback: bool = True,
    prefer_software: bool = True,
) -> str:
    """Write a video. Tries imageio-ffmpeg (direct stream), falls back to the ffmpeg binary."""
    if imageio is None:
        raise RuntimeError(
            "imageio unavailable. Install: pip install imageio imageio-ffmpeg, or a system ffmpeg."
        )

    if _has_imageio_ffmpeg():
        return _video_with_imageio(frames, output_path, fps, quality, codec, bitrate, crf)

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return _video_with_ffmpeg_binary(
            frames, output_path, ffmpeg_bin, fps, codec, bitrate, crf,
            auto_codec_fallback, prefer_software,
        )

    raise RuntimeError(
        "No working video backend.\n"
        "- Install imageio-ffmpeg: pip install imageio-ffmpeg\n"
        "or\n"
        "- Install ffmpeg (e.g. apt-get install ffmpeg)\n"
        "Otherwise write a .gif file."
    )


# ------------------------------
# GIF
# ------------------------------
def _gif_with_ffmpeg(frames, output_path, fps, scale_width=None) -> str:
    """GIF through ffmpeg with palette generation (two passes) for quality."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found. Install it: apt-get install ffmpeg")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if scale_width is None:
        scale_width = frames[0].shape[1]

    with _temp_frames(frames) as temp_dir:
        pattern = str(temp_dir / "frame%05d.png")
        palette = temp_dir / "palette.png"
        _run_ffmpeg(
            [ffmpeg_bin, "-y", "-framerate", str(fps), "-i", pattern,
             "-vf", f"fps={fps},scale={scale_width}:-1:flags=lanczos,"
                    "palettegen=max_colors=256:stats_mode=diff",
             str(palette)],
            error="Palette generation failed",
        )
        _run_ffmpeg(
            [ffmpeg_bin, "-y", "-framerate", str(fps), "-i", pattern, "-i", str(palette),
             "-filter_complex",
             f"fps={fps},scale={scale_width}:-1:flags=lanczos[x];"
             "[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
             "-loop", "0", str(output_path)],
            error="GIF creation failed",
        )
    return str(output_path)


# ------------------------------
# Video
# ------------------------------
def _has_imageio_ffmpeg() -> bool:
    try:
        import imageio_ffmpeg  # noqa: F401
        return True
    except Exception:
        return False


def _video_with_imageio(frames, output_path, fps, quality, codec, bitrate, crf) -> str:
    output_path = Path(output_path)
    writer_kwargs = dict(fps=fps, quality=quality, codec=codec)
    if bitrate:
        writer_kwargs["bitrate"] = bitrate
    elif crf is not None:
        writer_kwargs["output_params"] = ["-crf", str(crf)]

    writer = None
    try:
        try:
            writer = imageio.get_writer(str(output_path), **writer_kwargs)
        except TypeError:  # older versions: fewer kwargs
            writer = imageio.get_writer(str(output_path), fps=fps, quality=quality)
        for frame in frames:
            writer.append_data(frame)
    finally:
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.close()
    return str(output_path)


def _video_with_ffmpeg_binary(
    frames, output_path, ffmpeg_bin, fps, codec, bitrate, crf,
    auto_codec_fallback, prefer_software,
) -> str:
    output_path = Path(output_path)
    candidates = (
        _codec_candidates(ffmpeg_bin, codec, prefer_software) if auto_codec_fallback else [codec]
    )
    with _temp_frames(frames) as temp_dir:
        pattern = str(temp_dir / "frame%05d.png")
        last_err = None
        for idx, cdc in enumerate(candidates):
            if cdc and cdc != codec:
                warnings.warn(
                    f"Codec '{codec}' unavailable -> fallback '{cdc}'." if idx == 0
                    else f"Trying fallback codec '{cdc}'."
                )
            cmd = _video_command(ffmpeg_bin, pattern, fps, cdc, bitrate, crf, output_path)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                return str(output_path)
            last_err = f"Codec '{cdc or 'auto'}' failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    raise RuntimeError(last_err or "Video encoding failed.")


def _video_command(ffmpeg_bin, pattern, fps, cdc, bitrate, crf, output_path) -> List[str]:
    cmd = [ffmpeg_bin, "-y", "-framerate", str(fps), "-i", pattern]
    if cdc:
        cmd += ["-c:v", cdc]
    if crf is not None and not bitrate and (
        cdc.startswith(("libx264", "libx265")) or cdc in ("libvpx_vp9", "")
    ):
        cmd += ["-crf", str(crf)]
    if bitrate:
        cmd += ["-b:v", bitrate]
    if cdc.startswith(("libx264", "libx265")):
        cmd += ["-preset", "slow"]
    cmd += ["-pix_fmt", "yuv420p", str(output_path)]
    return cmd


def _codec_candidates(ffmpeg_bin: str, requested: str, prefer_software: bool = True) -> List[str]:
    """Ordered list of available codecs to try (requested first).

    Always ends with "" (let ffmpeg choose) as a last resort.
    """
    software = ["mpeg4", "libx264", "libopenh264", "libxvid",
                "libvpx_vp9", "libvpx_vp8", "ffv1", "rawvideo"]
    ordered = [requested] if requested else []
    ordered += [c for c in software if c not in ordered]

    available = [c for c in dict.fromkeys(ordered)
                 if c and _encoder_available(ffmpeg_bin, c)]
    available.append("")  # "" => ffmpeg default codec
    return available


def _encoder_available(ffmpeg_bin: str, encoder: str) -> bool:
    try:
        out = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return False
        return any(
            line.strip().endswith(encoder) or f" {encoder} " in line
            for line in out.stdout.splitlines()
        )
    except Exception:
        return False


# ------------------------------
# Helpers
# ------------------------------
@contextlib.contextmanager
def _temp_frames(frames: List[np.ndarray]):
    """Write the frames as PNGs into a temporary folder, removed on exit."""
    temp_dir = Path(tempfile.mkdtemp(prefix="tractviewer_frames_"))
    try:
        for i, frame in enumerate(frames):
            path = temp_dir / f"frame{i:05d}.png"
            if imageio is not None:
                imageio.imwrite(path, frame)
            else:
                from PIL import Image
                Image.fromarray(frame).save(path)
        yield temp_dir
    finally:
        with contextlib.suppress(Exception):
            shutil.rmtree(temp_dir)


def _run_ffmpeg(cmd: List[str], error: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{error}:\n{proc.stderr}")
