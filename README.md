# CCTV frame preparation for warehouse YOLO projects

The script keeps camera recordings unchanged and prepares sampled images for inspection and labeling. For each sampled real frame, it saves both an original and a gently enhanced version. Enhancement applies light denoising and local contrast adjustment. Compare both versions on your footage; enhancement can help in poor lighting, but can also remove fine detail if too strong.

## Run

Install the packages:

```bash
python -m pip install -r requirements.txt
```

### Upload one recording in a browser

Start the upload page from this project folder:

```bash
python -m streamlit run upload_app.py
```

Your browser opens a local page. Choose a recording, set how often to sample frames, and click **Process recording**. While it runs, the page shows elapsed time. When processing finishes, it shows the total time and offers a ZIP download containing the original and enhanced frames plus `frame_log.csv`.

On Windows with a virtual environment, run these commands from the project folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run upload_app.py
```

Run the install and launch commands separately. To reload a changed upload page, stop Streamlit with `Ctrl+C` and launch it again.

### Run from the command line

You can also pass one video file directly:

```bash
python enhance_pipeline.py --input "C:/videos/camera1.mp4" --output "C:/prepared/camera1"
```

Place videos in camera folders:

```text
raw_videos/
  camera_01/shift_day.mp4
  camera_01/shift_night.mp4
  camera_02/shift_day.mp4
```

Run:

```bash
python enhance_pipeline.py --input ./raw_videos --output ./prepared
```

By default the script samples one real frame per second, keeps source resolution, and saves paired images:

```text
prepared/
  frames/
    original/camera_01/shift_day/
    enhanced/camera_01/shift_day/
  frame_log.csv
```

Originals are in `frames/original/`; enhanced candidates are in `frames/enhanced/`. The CSV records source video, timestamp, both image paths, and review hints for possible blur or unusual brightness. Hints are not automatic proof that an image is unusable; nothing is filtered by default.

## Useful options

Sample fewer frames:

```bash
python enhance_pipeline.py --input ./raw_videos --output ./prepared --every-n-seconds 2
```

Reduce denoising if fine details look soft, or disable it:

```bash
python enhance_pipeline.py --input ./raw_videos --output ./prepared --denoise-strength 1
python enhance_pipeline.py --input ./raw_videos --output ./prepared --denoise-strength 0
```

Skip contrast adjustment:

```bash
python enhance_pipeline.py --input ./raw_videos --output ./prepared --no-clahe
```

Optional: make an interpolated higher-FPS video for smoother viewing. These guessed frames are **not** used in the training image folders:

```bash
python enhance_pipeline.py --input ./raw_videos --output ./prepared --smooth-video-fps 50
```

Keep original size for training unless a fixed input size is needed. Optional width/height fit the image proportionally and pad it rather than stretching it.

## Preparing a useful YOLO dataset

1. Check that targets are visible and large enough in the images. A file advertised as 2MP should normally be around 1920×1080; this script does not create missing detail by enlarging a smaller export.
2. Review the original and enhanced pairs. Choose the version that makes your target easiest to label; do not place both near-identical versions in different dataset splits.
3. Label the objects YOLO should detect (for example, people, forklifts, or pallets).
4. Keep all frames from the same video recording in one split. Use separate recordings or cameras for validation so the score reflects performance on new footage.
5. Check model results per camera and collect examples where it misses objects.

Image cleanup cannot guarantee high accuracy. Camera view, focus, lighting, target size in pixels, clear labels, and a varied dataset matter. This script prepares candidate images; it does not annotate them or train YOLO.
