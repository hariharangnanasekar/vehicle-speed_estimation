# vehicle-speed_estimation

This project is a Python-based system to detect and track vehicles from video footage and calculate their speeds using computer vision techniques. It leverages the **YOLOv8 object detection model**, OpenCV for video processing, and ByteTrack for object tracking.

---

## **Features**

- Detects vehicles such as cars, trucks, and buses in a video.
- Tracks each vehicle using unique IDs.
- Calculates vehicle speeds in **km/h** based on movement across frames.
- Annotates the output video with:
  - Bounding boxes
  - Tracking IDs
  - Vehicle speeds
- Saves the output as a playable annotated video.

---

## **Installation**

### **Prerequisites**

- Python 3.8 or higher
- Install dependencies using `pip`:

```bash
pip install opencv-python-headless numpy supervision ultralytics
```

### **Clone Repository**

```bash
git clone https://github.com/hariharangnanasekar/vehicle speed_estimation.git
cd vehicle speed_estimation
```

---

## **Usage**

1. **Prepare Input Video**:

   - Ensure you have a video file of traffic or vehicle footage. Save it in the project directory.

2. **Run the Script**:

   ```bash
   python main.py --source_video_path <input_video_path.mp4> --target_video_path <output_video_path.mp4>
   ```

   Replace `<input_video_path.mp4>` with your input video file path and `<output_video_path.mp4>` with the desired output file name.

3. **Output**:

   - The processed video with annotations will be saved in the project directory as `<output_video_path.mp4>`.

---

## **Command-Line Arguments**

- `--source_video_path`: Path to the input video file (required).
- `--target_video_path`: Path for the output video file (required).
- `--model_id`: YOLOv8 model version (`yolov8n`, `yolov8s`, `yolov8x`, etc.) [default: `yolov8x`].
- `--confidence_threshold`: Confidence threshold for YOLO detections (optional, default: `0.5`).
- `--iou_threshold`: IOU threshold for non-max suppression (optional, default: `0.4`).

---

## **Output Example**

The output video includes:

- **Bounding boxes** around detected vehicles.
- **Tracking IDs** for each vehicle.
- **Speed values** (e.g., "Car #1: 60 km/h").

---

## **File Structure**

```
project/
├── main.py                # Main script for the project
├── requirements.txt       # Dependency list
├── README.md              # Project description 
├── yolov8x.pt             # YOLOv8 pre-trained model weights 
├── input_video.mp4        # Example input video 
└── output_video.mp4       # Processed video 
```

---

## **Dependencies**

- **OpenCV**: For video processing and visualization.
- **Numpy**: For numerical computations.
- **Supervision**: For annotations and labeling.
- **Ultralytics YOLOv8**: For object detection.
- **ByteTrack**: For multi-object tracking.

---

## **Future Improvements**

- Add **real-time processing** for live video streams.
- Implement **multi-lane detection** for better tracking.
- Include **speed limit violation alerts**.
- Optimize performance for low-spec hardware.

---

## **License**

This project is open-source and available under the **MIT License**.

---



