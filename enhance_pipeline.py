#!/usr/bin/env python3
"""Prepare CCTV recordings for inspection and YOLO dataset annotation.

The original videos are never modified. Real source frames are sampled at a
time interval and saved in both original and gently enhanced form. Optional
motion interpolation creates a smoother viewing copy, never training frames.
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".ts", ".m4v"}


def find_ffmpeg():
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def make_smooth_video(source, destination, target_fps):
    """Create a motion-interpolated playback copy. Do not use it for training."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("  FFmpeg unavailable; skipping optional smooth video.")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-i", str(source),
        "-vf", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        lines = (exc.stderr or str(exc)).strip().splitlines()
        print(f"  Smooth-video creation failed: {lines[-1] if lines else exc}")
        return False


def enhance_frame(frame, denoise_strength=3, use_clahe=True):
    """Apply conservative denoising and local contrast to a sampled frame."""
    if denoise_strength > 0:
        frame = cv2.fastNlMeansDenoisingColored(
            frame, None, denoise_strength, denoise_strength, 7, 21
        )
    if use_clahe:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        light, a, b = cv2.split(lab)
        light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
        frame = cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)
    return frame


def fit_inside(frame, width, height):
    """Resize proportionally and pad; never stretch the camera view."""
    if not width or not height:
        return frame
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(
        frame, (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def blur_score(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def process_video(source, relative_path, frames_root, rows, args, enhanced_video_path=None):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"  Cannot open, skipping: {source}")
        rows.append([str(relative_path), "", "ERROR", "", "", "Could not open video"])
        return 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps) or fps <= 0:
        print(f"  Invalid FPS metadata, skipping: {source}")
        rows.append([str(relative_path), "", "ERROR", "", "", "Invalid source FPS metadata"])
        cap.release()
        return 0, 0

    rel_stem = relative_path.with_suffix("")
    original_dir = frames_root / "original" / rel_stem
    enhanced_dir = frames_root / "enhanced" / rel_stem
    original_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    video_writer = None
    if enhanced_video_path is not None:
        output_width = args.width or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = args.height or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        enhanced_video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(
            str(enhanced_video_path), cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (output_width, output_height),
        )
        if not video_writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create enhanced video: {enhanced_video_path}")
    frame_index = sample_index = kept = filtered = sampled = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        target_index = round(sample_index * args.every_n_seconds * fps)
        selected = frame_index >= target_index
        enhanced = None
        if video_writer is not None or selected:
            enhanced = enhance_frame(frame, args.denoise_strength, not args.no_clahe)
            enhanced = fit_inside(enhanced, args.width, args.height)
        if video_writer is not None:
            video_writer.write(enhanced)
        if selected:
            sample_index += 1
            sampled += 1
            seconds = frame_index / fps
            original = fit_inside(frame, args.width, args.height)
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            sharpness = blur_score(gray)
            brightness = float(np.mean(gray))
            notes = []
            if sharpness < args.blur_threshold:
                notes.append(f"possible blur (score={sharpness:.1f})")
            if brightness < args.dark_threshold:
                notes.append(f"dark (brightness={brightness:.1f})")
            if brightness > args.bright_threshold:
                notes.append(f"very bright (brightness={brightness:.1f})")
            is_filtered = bool(notes) and args.filter_quality
            filename = f"{source.stem}_t{seconds:010.3f}s_f{frame_index:09d}.jpg"
            if is_filtered:
                filtered += 1
                status = "FILTERED"
            else:
                ok_original = cv2.imwrite(str(original_dir / filename), original)
                ok_enhanced = cv2.imwrite(str(enhanced_dir / filename), enhanced)
                if not (ok_original and ok_enhanced):
                    rows.append([str(relative_path), f"{seconds:.3f}", "ERROR", "", "", f"Could not write {filename}"])
                    frame_index += 1
                    continue
                kept += 1
                status = "REVIEW" if notes else "KEPT"
            rows.append([
                str(relative_path), f"{seconds:.3f}", status,
                str(Path("frames") / "original" / rel_stem / filename),
                str(Path("frames") / "enhanced" / rel_stem / filename),
                "; ".join(notes),
            ])
        frame_index += 1
    cap.release()
    if video_writer is not None:
        video_writer.release()
    print(f"  {sampled} sampled; {kept} original/enhanced pairs saved; {filtered} filtered")
    return kept, filtered


def main():
    parser = argparse.ArgumentParser(description="Prepare CCTV frames for YOLO dataset review and labeling")
    parser.add_argument("--input", required=True, help="One video file or a folder containing videos")
    parser.add_argument("--output", required=True, help="Destination for images and logs")
    parser.add_argument("--every-n-seconds", type=float, default=1.0, help="Sample interval; default one frame per second")
    parser.add_argument("--smooth-video-fps", type=float, default=0.0, help="Optional interpolated playback copy; 0 disables it")
    parser.add_argument("--create-enhanced-video", action="store_true", help="Write a full-length enhanced MP4 at the source FPS")
    parser.add_argument("--denoise-strength", type=int, default=0, help="Optional, slow non-local-means denoising strength; 0 disables it")
    parser.add_argument("--no-clahe", action="store_true", help="Skip local contrast enhancement")
    parser.add_argument("--width", type=int, default=0, help="Optional output canvas width; 0 keeps source size")
    parser.add_argument("--height", type=int, default=0, help="Optional output canvas height; 0 keeps source size")
    parser.add_argument("--filter-quality", action="store_true", help="Drop frames flagged by simple quality checks")
    parser.add_argument("--blur-threshold", type=float, default=60.0)
    parser.add_argument("--dark-threshold", type=float, default=25.0)
    parser.add_argument("--bright-threshold", type=float, default=235.0)
    args = parser.parse_args()

    if args.every_n_seconds <= 0:
        parser.error("--every-n-seconds must be greater than zero")
    if args.smooth_video_fps < 0 or args.denoise_strength < 0:
        parser.error("FPS and denoising strength cannot be negative")
    if (args.width == 0) != (args.height == 0) or args.width < 0 or args.height < 0:
        parser.error("Set both --width and --height to positive values, or leave both at 0")
    input_path, output_dir = Path(args.input).expanduser().resolve(), Path(args.output).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")
    if not input_path.is_dir() and (not input_path.is_file() or input_path.suffix.lower() not in VIDEO_EXTS):
        parser.error(f"Input must be a supported video file or a folder: {input_path}")
    if input_path.is_dir() and (input_path == output_dir or input_path in output_dir.parents):
        parser.error("Output folder must be outside the input folder")
    if input_path.is_file():
        input_dir = input_path.parent
        videos = [input_path]
    else:
        input_dir = input_path
        videos = sorted(
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS and output_dir not in p.parents
        )
    if not videos:
        print(f"No video files found under {input_dir}")
        return 1

    frames_root = output_dir / "frames"
    videos_root = output_dir / "smooth_videos"
    enhanced_videos_root = output_dir / "enhanced_videos"
    frames_root.mkdir(parents=True, exist_ok=True)
    rows, total_kept, total_filtered = [], 0, 0
    print(f"Found {len(videos)} video(s); sampling real frames.\n")
    for source in videos:
        relative = source.relative_to(input_dir)
        print(f"Processing: {relative}")
        if args.smooth_video_fps > 0:
            dest = videos_root / relative.with_suffix(".mp4")
            print(f"  Creating optional {args.smooth_video_fps:g} FPS interpolated viewing copy...")
            if not make_smooth_video(source, dest, args.smooth_video_fps):
                rows.append([str(relative), "", "ERROR", "", str(dest), "Smooth video creation failed"])
        enhanced_video_path = None
        if args.create_enhanced_video:
            enhanced_video_path = enhanced_videos_root / relative.with_name(
                f"{relative.stem}_enhanced.mp4"
            )
        kept, filtered = process_video(
            source, relative, frames_root, rows, args, enhanced_video_path
        )
        total_kept += kept
        total_filtered += filtered

    report = output_dir / "frame_log.csv"
    with report.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_video", "time_seconds", "status", "original_frame", "enhanced_frame", "review_notes"])
        writer.writerows(rows)
    print(f"\nDone: {total_kept} frame pairs saved, {total_filtered} filtered.")
    print(f"Original and enhanced frame folders: {frames_root}")
    print(f"Review log: {report}")
    if args.smooth_video_fps == 0:
        print("Interpolated video creation is disabled; training images use camera-recorded frames only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
