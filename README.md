# PPE Detection

## Overview

This project implements a **Personal Protective Equipment (PPE) detection system** designed to identify whether a person is wearing safety goggles. It combines face detection, goggles detection, and edge AI inference to support safety monitoring in environments where eye protection is required such as the Invention Studio @ GT. The system includes model inference components, image post-processing, configuration files, and LED display logic. It is designed to support deployment on compatible edge AI hardware using Hailo-compiled inference models.

## Features

- **Face Detection:** Uses the SCRFD face detection model to locate faces in camera images.
- **Safety Goggles Detection:** Uses a YOLOv8-based model to detect goggles and related PPE classes.
- **Edge AI Inference:** Includes Hailo-compatible `.hef` model files for accelerated inference on supported hardware.
- **Detection Post-Processing:** Processes raw model outputs and prepares detection results for further use.
- **LED Status Display:** Includes configuration and display modules for communicating detection or safety status through an LED interface.
- **Configurable Settings:** Uses JSON configuration files and Python modules to manage model and system settings.
- **Modular Architecture:** Separates face detection, goggles detection, common utilities, and display functionality.

## Key Words

- **Python** — Core application logic and inference pipeline.
- **SCRFD** — Face detection.
- **YOLOv8** — Object detection for safety goggles.
- **Hailo AI Accelerator** — Hardware-accelerated inference on compatible devices.
- **OpenCV** — Potentially used for image or video processing, depending on the installed dependencies.
- **JSON** — Model and application configuration.
- **LED Interface** — Visual indication of detection or safety status.

## Major Requirements

The project requires a compatible Python environment and the libraries specified in `PPE/requirements.txt`.

Depending on the deployment hardware and software configuration, the system may also require:

- A compatible Hailo AI accelerator.
- HailoRT and the corresponding hardware drivers.
- A supported camera or video input source.
- Compatible `.hef` model files.
- Any hardware-specific libraries needed to control the LED display.

**Important:** Hailo runtime and device drivers are platform-dependent. Installing Python packages alone may not be sufficient to run inference.

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd PPE-DETECTION-MAIN
```

Replace `<repository-url>` with the actual URL of the Git repository.

### 2. Set Up a Python Environment

Create and activate a virtual environment using a Python version compatible with the project's dependencies.

```bash
python3 -m venv .venv
```

On Linux or Raspberry Pi OS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

Install the Python dependencies listed in the project requirements file:

```bash
pip install -r PPE/requirements.txt
```

Additional system libraries or hardware-specific dependencies may be required depending on the target platform.

### 4. Verify the Model Files

Confirm that the required model files are present:

- `PPE/face-detection/scrfd_10g.hef`
- `PPE/goggle-detection/yolov8s.hef`

These files are compiled inference models intended for compatible Hailo hardware and runtime software.

## Running the Project

The repository includes `PPE/scrfd_test.py` for testing the SCRFD face detection component.

From the repository root, start by reviewing the script's supported arguments:

```bash
python PPE/scrfd_test.py --help
```

If the script does not support a `--help` argument, inspect its source code for the expected inputs and execution instructions.

The goggles detection and LED display components are located in `PPE/goggle-detection/`, with additional modules in `PPE/`. Review the corresponding scripts and configuration files to identify the intended application entry point and required hardware settings.

**Note:** The exact launch command depends on the implementation of the entry-point scripts, available hardware, and configuration settings.

## Configuration

The project contains JSON configuration files and Python configuration modules that may be used to control model parameters and application behavior.

Relevant files include:

- `PPE/config.json`
- `PPE/goggle-detection/config.json`
- `PPE/goggle-detection/led_config.py`
- `PPE/goggle-detection/goggle_labels.txt`

Before running the application, verify that model paths, detection labels, camera settings, inference settings, and LED hardware parameters match the deployment environment.

## Detection Pipeline

1. **Image Input:** Obtain an image from a supported camera or other input source.
2. **Face Detection:** Run SCRFD inference to locate faces.
3. **Goggles Detection:** Run YOLOv8 inference to identify safety goggles.
4. **Post-Processing:** Process model outputs and associate detections as required by the application.
5. **Safety Status:** Determine the appropriate detection or safety status according to the implemented logic.
6. **LED Output:** Use the LED display modules to communicate status when the supported hardware is available.

## Troubleshooting

### Hailo Device or Runtime Errors

Verify that the accelerator is connected, the appropriate drivers are installed, and the Hailo runtime is compatible with the compiled model files.

### Missing Python Packages

Activate the intended virtual environment and install the dependencies from `PPE/requirements.txt`. Check for additional system-level dependencies if an installation fails.

### Model File Not Found

Confirm that the `.hef` files exist at the expected paths. Relative paths may depend on the directory from which the application is launched.

### Incorrect Detection Results

Check the model configuration, class labels, confidence thresholds, image preprocessing, and post-processing implementation. Results should be evaluated under representative lighting conditions and camera positions.

### LED Display Not Working

Verify the LED configuration, physical connections, hardware permissions, and any required display-control libraries. The application may need to run on the target hardware rather than a standard desktop environment.

## Safety Considerations

This project is intended to support PPE detection and safety monitoring. Detection results can be affected by lighting, camera placement, occlusion, model accuracy, and environmental conditions.

The system should be tested under representative operating conditions before deployment. A positive detection should not be treated as a guarantee that a person is compliant with all applicable safety requirements, and uncertain or missing detections should be handled conservatively according to the intended safety workflow.

## Project Documentation

Additional project documentation from previous semesters are available in the repository:

- `PPE Using Edge AI Acceleration final report.pdf`
- `PPE_final_presentation.pdf`

These documents provide supplementary information about the project design and implementation.

## Contributors

@jvogt23
@Daniel1464
@kaikai101k
@austinchan12
@vthokchom25
