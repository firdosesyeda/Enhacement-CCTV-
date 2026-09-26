# CCTV frame preparation for warehouse YOLO projects

The upload app keeps the source recording unchanged, creates a full enhanced video, runs a pretrained YOLO person detector on that enhanced video, and prepares sampled original/enhanced frames for inspection and labeling. The final video shows person boxes and confidence values. Local contrast adjustment is enabled by default. Denoising is optional and off by default because it can take a long time on every frame; compare original and enhanced frames because cleanup can soften small details.

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

Your browser opens a local page. Choose a recording, set how often to sample frames, choose denoising/contrast options and the minimum detection confidence, then click **Process recording**. The app enhances every video frame first, then runs YOLO on the enhanced video. It shows progress and elapsed time. When processing finishes, download a ZIP containing the enhanced video, person-detection video, sampled frames, and `frame_log.csv`.

The app uses `yolo26n.pt`, a small pretrained Ultralytics model. On first use, Ultralytics downloads its model weights, so an internet connection is needed. YOLO is restricted to its `person` class (COCO class 0). The confidence slider defaults to 0.25; lowering it may find more people but can add false detections. This is pretrained detection, not warehouse-specific fine-tuning.

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
  enhanced_videos/camera_01/shift_day_enhanced.mp4
  person_detection/shift_day_person_detected.mp4
  frames/
    original/camera_01/shift_day/
    enhanced/camera_01/shift_day/
  frame_log.csv
```

The enhanced video keeps the source FPS and image size. The detection video also keeps that FPS and size, with person boxes drawn on each frame; audio is not copied to the output video. Originals are in `frames/original/`; enhanced candidates are in `frames/enhanced/`. The CSV records source video, timestamp, both image paths, and review hints for possible blur or unusual brightness. Hints are not automatic proof that an image is unusable; nothing is filtered by default.

## Useful options

Sample fewer frames:

```bash
python enhance_pipeline.py --input ./raw_videos --output ./prepared --every-n-seconds 2
```

Optional denoising is slow on long videos. Keep it at 0 for faster processing, or enable a low level when the recording has visible grain:

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

Image cleanup cannot guarantee high detection accuracy or find every person. Small, distant, blurred, or hidden people may be missed. Camera view, focus, lighting, target size in pixels, representative examples, and accurate labels all affect results. The app uses pretrained YOLO for video detection but does not fine-tune YOLO; use labeled warehouse frames for later fine-tuning.
