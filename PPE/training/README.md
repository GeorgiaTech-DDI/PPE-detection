# Introduction
This folder contains the model trainer and model compiler, which have these functions:
- model_trainer.ipynb: A Jupyter notebook intended to be uploaded to Google Colab. Runs training on a dataset from RoboFlow to train into a YOLOv11 or YOLO26 model.
- model_compiler.py: A script to compile a YOLOv11 or YOLO26 model into an HEF so that it can run on a Hailo processor.

# Software Requirements
## For the Model Trainer
The model trainer must be run on Google Colab with a GPU-enabled runtime selected. All other software requirements for this program will be installed when running the notebook.

**IMPORTANT**: The model trainer must be run one cell at a time, as some of the cells must take in input from the user before they execute.

## For the Model Compiler
The model compiler relies on the Hailo Dataflow Compiler, which is a Python package. This package is not published on PyPI, and is instead found exclusively on the Hailo Developer Zone website. It also only supports Python versions 3.10, 3.11, and 3.12.

The software dependencies, therefore, are:
- Python 3.10-3.12
- The Ultralytics library
- Hailo Dataflow Compiler

Begin by creating a virtual environment. For example, in Conda, do the following:

```
conda create --name <env_name_here> python=3.12
conda activate <env_name_here>
```

Then, download the Hailo Dataflow Compiler from [here](https://hailo.ai/developer-zone/software-downloads). You can install this using Pip.
**IMPORTANT**: Install both Ultralytics and the Hailo Dataflow Compiler in one command.
They are __extremely__ picky about dependencies and will fight if installed in separate commands.

```
pip install path/to/hailo/dataflow/compiler.whl ultralytics
```

# Running the Script
## Arguments
hailo_compiler.py takes in the following arguments:
- trained_weights: Required positional argument. The .pt file we are compiling.
- calibration_data: Required positional argument. The data that will be used in model calibration
- hailo-target: Optional. The target architecture. Defaults to hailo10h.
- imgsz: Optional. The size of the image. Defaults to 640.

## How to run
- Set up a virtual environment through conda or venv and install Ultralytics and the Hailo Dataflow compiler
- Run with python3 model_compiler.py path/to/weights path/to/dataset.