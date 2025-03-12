# RoboSet Dataset Preparation Tool
## Overview
[RoboSet](https://robopen.github.io/roboset/) is an open dataset for robotic learning that contains rich robot manipulation data and visual information. The dataset provides multi-view robot interaction videos alongside planning and control information that can be used to train and evaluate robot learning algorithms.

This project provides a processing tool to extract specified frame count video clips and corresponding planning objects from the RoboSet dataset for further research and applications.

## Features
- Automatically downloads tar.gz archives from the RoboSet dataset
- Extracts and processes .h5 files containing robot manipulation data
- Generates video sequences from multiple viewpoints (left, right, top, wrist)
- Uniformly samples frames to create videos of fixed length
- Creates corresponding task description JSON files for each video
- Supports three different types of datasets: Autonomous, Kinesthetic, and Teleoperation

## Usage
python prepare_Autonomous.py [--data TYPE] [--frame COUNT]

### Parameters
- **--data or -d**: Specify the dataset type  
    - **a**: Autonomous dataset (default)
    - **k**: Kinesthetic dataset
    - **t**: Teleoperation dataset
- **--frame or -f**: Specify the number of frames to extract from each operation (default: 25)
- **--parallel or -p**: Enable parallel processing for faster extraction (default: False)

### Example
```bash
# Process Autonomous dataset with default 25 frames
python download.py

# Process Kinesthetic dataset
python download.py --data k

# Process Teleoperation dataset with 30 frames
python download.py -d t -f 30

# Process Autonomous dataset with 25 frames using parallel processing
python download.py -p
```

### Output Structure
Processed data will be saved in the following directory structure:
```
./[DatasetType]/
  └── [Task_Name]/
      ├── left/
      │   ├── [filename]-[Trial]-left.mp4
      │   └── [filename]-[Trial]-left.json
      ├── right/
      │   ├── [filename]-[Trial]-right.mp4
      │   └── [filename]-[Trial]-right.json
      ├── top/
      │   ├── [filename]-[Trial]-top.mp4
      │   └── [filename]-[Trial]-top.json
      └── wrist/
          ├── [filename]-[Trial]-wrist.mp4
          └── [filename]-[Trial]-wrist.json
```