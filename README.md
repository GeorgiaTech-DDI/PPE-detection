# PPE Safety Goggles Detection

Real-time PPE detection system using a Raspberry Pi 5, Hailo-10H AI accelerator, YOLOv8s, Raspberry Pi camera, and an addressable LED display.

The system detects:

- `Person`
- `Glasses`

It provides immediate visual feedback:

- Person + Glasses detected → GREEN
- Person detected without Glasses → RED
- No person detected → LED OFF

The model runs locally on the Raspberry Pi using the Hailo accelerator.

---

## Hardware

This project is designed to run on:

- Raspberry Pi 5
- Hailo-10H AI accelerator / AI HAT
- Raspberry Pi camera
- 8x32 WS2812-compatible addressable LED matrix/strip
- GPIO connection for LED data
- Appropriate power supply

Current LED configuration:

```python
LED_WIDTH = 32
LED_HEIGHT = 8
LED_PIN = 18
LED_BRIGHTNESS = 80
LED_PIXEL_ORDER = "GRB"
LED_SIMULATE = False
```

If your LED data wire uses a GPIO other than GPIO 18, update `LED_PIN` in `led_config.py`.

---

## Software

The project uses:

- Python
- HailoRT
- Hailo Apps
- Hailo-10H runtime
- OpenCV
- NumPy
- `rpi_ws281x`
- YOLOv8s
- Hailo Executable Format (`.hef`)

The trained YOLOv8s model has already been compiled into:

```text
yolov8s.hef
```

You do **not** need to retrain or recompile the model just to run the existing system.

---

## Project Structure

The runtime directory should contain:

```text
PPE/
├── goggle_detection.py
├── goggle_post_process.py
├── goggle_labels.txt
├── config.json
├── led_config.py
├── led_display.py
└── yolov8s.hef
```

Additional log files may also be present:

```text
hailort.log
hailort.1.log
```

---

## Label Configuration

`goggle_labels.txt` contains:

```text
Glasses
Person
```

The order of these labels matters because the model's class IDs correspond to these entries.

---

## Detection Configuration

`config.json` controls detection and tracking thresholds.

Current configuration:

```json
{
  "visualization_params": {
    "score_thres": 0.15,
    "max_boxes_to_draw": 500,
    "tracker": {
      "track_thresh": 0.1,
      "track_buffer": 30,
      "match_thresh": 0.9,
      "aspect_ratio_thresh": 2.0,
      "min_box_area": 500,
      "mot20": false
    }
  }
}
```

The current object detection confidence threshold is:

```text
0.15
```

Detections below this threshold are ignored.

---

# Raspberry Pi Setup

This application should be run on the Raspberry Pi, not directly on a normal Mac or PC.

The Hailo hardware and Hailo Apps environment must already be installed on the Raspberry Pi.

The code currently expects the Hailo Apps repository at:

```text
/home/pi/hailo-apps
```

The Python files add this path to `sys.path` so they can import the Hailo Apps libraries.

---

## Start the Hailo Environment

On the Raspberry Pi:

```bash
cd ~/hailo-apps
source setup_env.sh
```

Then enter the PPE project directory:

```bash
cd PPE
```

Your shell should now be inside something similar to:

```text
/home/pi/hailo-apps/PPE
```

---

# Verify the Files

Run:

```bash
ls
```

You should see at least:

```text
goggle_detection.py
goggle_post_process.py
goggle_labels.txt
config.json
led_config.py
led_display.py
yolov8s.hef
```

`config.json` should be in the current working directory because `goggle_detection.py` loads it using:

```python
load_json_file("config.json")
```

---

# Test Command-Line Arguments

Before running the detector, check the available options:

```bash
python goggle_detection.py --help
```

This is useful because many command-line arguments are provided by the Hailo Apps standalone parser.

---

# Run the Detector

For the Raspberry Pi camera:

```bash
python goggle_detection.py \
    --input rpi \
    --hef-path yolov8s.hef \
    --labels goggle_labels.txt
```

The application will:

1. Initialize the Hailo accelerator.
2. Load `yolov8s.hef`.
3. Load the custom labels.
4. Open the Raspberry Pi camera.
5. Preprocess camera frames.
6. Send frames to the Hailo accelerator.
7. Perform YOLO inference.
8. Post-process detections.
9. Draw detection boxes.
10. Determine PPE status.
11. Update the LED display.
12. Continue processing frames in real time.

---

# Important: `--labels` vs `--labels-json`

The original report gives a command similar to:

```bash
python goggle_detection.py \
    --input rpi \
    --hef-path yolov8s.hef \
    --labels-json goggle_labels.txt
```

However, the current `goggle_detection.py` defines:

```text
--labels
```

and:

```text
-l
```

Therefore, use:

```bash
--labels goggle_labels.txt
```

instead of:

```bash
--labels-json goggle_labels.txt
```

Correct command:

```bash
python goggle_detection.py \
    --input rpi \
    --hef-path yolov8s.hef \
    --labels goggle_labels.txt
```

---

# Important: Custom Post-Processing Import

The custom PPE and LED logic is located in:

```text
goggle_post_process.py
```

The detector should therefore use:

```python
from goggle_post_process import inference_result_handler
```

There is currently a potential issue in `goggle_detection.py`.

The normal Hailo import block contains:

```python
from hailo_apps.python.standalone_apps.object_detection.object_detection_post_process import inference_result_handler
```

while the fallback import block uses:

```python
from goggle_post_process import inference_result_handler
```

This means that when Hailo Apps imports successfully, the program may use Hailo's default object-detection post-processing instead of the custom PPE logic.

To make sure the PPE/LED system is used, change the import to:

```python
from goggle_post_process import inference_result_handler
```

The detector should always ultimately use the custom `inference_result_handler`.

---

# LED Behavior

The LED system is initialized by `goggle_post_process.py`.

The logic is:

```python
if person_found and not glasses_found:
    _led.set_state("unsafe")
elif person_found and glasses_found:
    _led.set_state("safe")
else:
    _led.set_state("off")
```

States correspond to:

```text
unsafe = red
safe   = green
off    = black/off
```

Other colors supported by `LEDDisplay` include:

```text
idle    = dim blue
warning = orange
```

---

# Testing Without the LED Hardware

If you want to test the detection pipeline before connecting the LED display, open:

```text
led_config.py
```

and change:

```python
LED_SIMULATE = False
```

to:

```python
LED_SIMULATE = True
```

Simulation mode prevents the program from trying to access the physical WS2812 LEDs.

Instead, LED actions will be printed to the terminal.

For example:

```text
[LED] Simulation mode enabled
[LED] fill -> (255, 0, 0)
```

This is recommended when initially debugging the camera/model pipeline.

---

# LED Dependencies

The physical LED implementation uses:

```python
from rpi_ws281x import PixelStrip, Color
```

If LED simulation is disabled and `rpi_ws281x` is unavailable, the program will raise an import error.

Install the library in the Raspberry Pi environment if necessary.

The LED panel is currently configured as:

```text
Width:      32
Height:     8
Pixels:     256
GPIO:       18
Brightness: 80
Order:      GRB
```

The display code assumes a serpentine/winding layout.

If the physical panel displays pixels in the wrong locations, the `xy_to_index()` mapping in `led_display.py` may need to be changed.

---

# Detection Pipeline

The main application uses three primary processing stages running in separate threads:

```text
Camera/Input
     |
     v
Preprocessing
     |
     v
Hailo Inference
     |
     v
Post-processing
     |
     v
Detection Visualization
     |
     v
PPE Logic
     |
     v
LED Feedback
```

Queues are used between preprocessing, inference, and visualization.

Hailo inference runs asynchronously so multiple inference operations can be managed efficiently.

---

# Tracking

Object tracking can optionally be enabled using:

```bash
--track
```

Example:

```bash
python goggle_detection.py \
    --input rpi \
    --hef-path yolov8s.hef \
    --labels goggle_labels.txt \
    --track
```

Tracking uses ByteTrack.

Tracking configuration comes from `config.json`.

The current tracker settings include:

```text
track_thresh:        0.1
track_buffer:        30
match_thresh:        0.9
aspect_ratio_thresh: 2.0
min_box_area:        500
```

---

# Detection Trails

When tracking is enabled, motion trails can also be displayed.

Use:

```bash
--draw-trail
```

Example:

```bash
python goggle_detection.py \
    --input rpi \
    --hef-path yolov8s.hef \
    --labels goggle_labels.txt \
    --track \
    --draw-trail
```

The current trail history stores approximately the previous 30 tracked positions.

---

# FPS Monitoring

The Hailo Apps parser also supports FPS measurement.

Depending on the installed Hailo Apps version, this can typically be enabled using:

```bash
--show-fps
```

Check:

```bash
python goggle_detection.py --help
```

for the exact supported options.

---

# Recommended First Boot

For the first test, disable the real LEDs:

```python
LED_SIMULATE = True
```

Then run:

```bash
cd ~/hailo-apps
source setup_env.sh
cd PPE
```

Check available arguments:

```bash
python goggle_detection.py --help
```

Then start detection:

```bash
python goggle_detection.py \
    --input rpi \
    --hef-path yolov8s.hef \
    --labels goggle_labels.txt
```

Once camera inference works correctly, switch:

```python
LED_SIMULATE = False
```

and test the physical LED panel.

---

# Model

The detector is based on a custom fine-tuned YOLOv8s model.

The dataset was simplified to two classes:

```text
Glasses
Person
```

This decision was made because distinguishing prescription glasses from actual safety goggles proved difficult in real-world conditions.

The model was trained using a relabeled PPE dataset exported in YOLOv8 format.

The original training configuration used approximately:

```text
epochs = 100
batch  = 16
imgsz  = 640
```

A representative Ultralytics training command is:

```bash
yolo task=detect \
    mode=train \
    model=yolov8s.pt \
    data=/path/to/data.yaml \
    epochs=100 \
    batch=16 \
    imgsz=640 \
    plots=True
```

---

# Model Export

After training, the best model weights are typically:

```text
best.pt
```

The PyTorch model is exported to ONNX:

```bash
yolo export \
    model=/path/to/best.pt \
    format=onnx \
    imgsz=640
```

This produces an ONNX model for Hailo compilation.

---

# Hailo Compilation

The ONNX model must be compiled to Hailo Executable Format:

```text
.onnx
  |
  v
.hef
```

A representative compilation command from the project is:

```bash
hailomz compile yolov8s \
    --ckpt=/path/to/best.onnx \
    --hw-arch hailo10h \
    --calib-path /path/to/train/images \
    --classes 2 \
    --performance
```

The calibration image folder is used by the Hailo compiler when optimizing/quantizing the model.

The final result is:

```text
yolov8s.hef
```

This `.hef` file is what runs on the Hailo-10H.

---

# Important Model Compatibility Note

One of the major issues encountered during development was ONNX compatibility.

Different versions of Ultralytics may produce slightly different ONNX model structures.

Some versions of exported YOLOv8 ONNX models may not be compatible with the Hailo Model Zoo / Dataflow Compiler pipeline.

If retraining or recompiling the model, keep the Ultralytics version consistent with the version used during the working compilation process.

The project report used:

```text
Ultralytics 8.3.252
```

for training.

---

# Existing HEF

If `yolov8s.hef` is already available, model compilation is not necessary.

The normal runtime workflow is simply:

```text
Camera
  |
  v
yolov8s.hef
  |
  v
Hailo-10H
  |
  v
Person / Glasses detections
  |
  v
PPE decision
  |
  v
LED output
```

---

# Hailo Runtime Verification

Existing Hailo logs from the project show that the Raspberry Pi successfully created a Hailo virtual device and recognized a network group named:

```text
yolov8s
```

This indicates that the HEF has previously been loaded by the Hailo runtime on the Raspberry Pi.

If the application stops working, check whether the Hailo accelerator is still detected before debugging the Python code.

---

# Troubleshooting

## `ModuleNotFoundError: hailo_apps`

Make sure the Hailo environment is initialized:

```bash
cd ~/hailo-apps
source setup_env.sh
```

Then run the project from the correct directory.

The Python code currently also expects:

```text
/home/pi/hailo-apps
```

to exist.

---

## `config.json` not found

Run the application from the directory containing:

```text
config.json
```

For example:

```bash
cd ~/hailo-apps/PPE
```

Then run the Python program.

---

## `rpi_ws281x is not installed`

Either install the LED dependency or temporarily enable simulation:

```python
LED_SIMULATE = True
```

---

## LEDs do not change color

First confirm that `goggle_detection.py` imports:

```python
from goggle_post_process import inference_result_handler
```

and not only Hailo's default object detection post-processor.

Then verify:

```python
LED_SIMULATE = False
```

and make sure the LED data line is connected to the configured GPIO.

Current GPIO:

```text
GPIO 18
```

---

## Camera does not open

Verify that:

- The Raspberry Pi camera is physically connected.
- The camera is recognized by Raspberry Pi OS.
- `--input rpi` is supported by the installed Hailo Apps version.

Check:

```bash
python goggle_detection.py --help
```

for the available input options.

---

## HEF/model does not load

Confirm that:

```text
yolov8s.hef
```

exists in the current directory.

Run:

```bash
ls -lh yolov8s.hef
```

Also verify that the Hailo accelerator is recognized and that the HEF was compiled for:

```text
hailo10h
```

---

## Labels look incorrect

Make sure `goggle_labels.txt` contains exactly:

```text
Glasses
Person
```

The class ordering must match the model.

---

## Detection is too sensitive

Increase:

```json
"score_thres": 0.15
```

inside `config.json`.

For example:

```json
"score_thres": 0.30
```

A higher value requires higher-confidence detections.

---

## Detection misses objects

Lower the score threshold in `config.json`.

Be aware that lowering it too much can create additional false detections.

---

# Current Limitations

The model detects the general class:

```text
Glasses
```

rather than reliably distinguishing:

```text
Safety Goggles
```

from:

```text
Prescription Glasses
```

This was an intentional simplification because distinguishing the two categories reliably in real-world images was difficult.

Therefore, the current system should be treated as a PPE detection prototype rather than a guaranteed safety-compliance system.

Other project challenges included:

- ONNX export compatibility
- Hailo toolchain integration
- Limited documentation for custom YOLO models on Hailo-10H
- Raspberry Pi deployment configuration
- Dataset limitations
- Hardware integration
- LED integration

---

# Quick Start

```bash
cd ~/hailo-apps
source setup_env.sh
cd PPE

python goggle_detection.py \
    --input rpi \
    --hef-path yolov8s.hef \
    --labels goggle_labels.txt
```

For initial testing, it is recommended to first set:

```python
LED_SIMULATE = True
```

in:

```text
led_config.py
```

Once detection works correctly, enable the real LED hardware again:

```python
LED_SIMULATE = False
```

---

# System Summary

The full system is:

```text
Raspberry Pi Camera
        |
        v
Frame Preprocessing
        |
        v
YOLOv8s HEF
        |
        v
Hailo-10H Accelerator
        |
        v
Person / Glasses Detection
        |
        v
Custom PPE Decision Logic
        |
        +----------------------+
        |                      |
 Person + Glasses       Person, No Glasses
        |                      |
        v                      v
      GREEN                   RED

No Person
   |
   v
 LED OFF
```

The goal is to provide a low-latency, fully local PPE compliance reminder using edge AI without requiring cloud inference.