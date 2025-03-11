import os
import csv
import requests
import tarfile
import h5py
import numpy as np
import json
import cv2
import glob
import shutil
from tqdm import tqdm
import time
import re
import argparse

def sanitize_filename(name):
    """Replace special characters in filename with underscores"""
    return re.sub(r'[^\w\-\.]', '_', name)

def ensure_dir(directory):
    """Make sure the directory exists, create it if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_file(url, destination):
    """Download file to specified path with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def extract_tarfile(tar_path, extract_to='.'):
    """Extract tar.gz file to specified directory and return the top level directory name"""
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    
    # Get the top level directory name
    with tarfile.open(tar_path, 'r:gz') as tar:
        top_level_dirs = {member.name.split('/')[0] for member in tar.getmembers() if member.name != '.'}
        return list(top_level_dirs)[0] if top_level_dirs else None

def find_h5_files(directory):
    """Find all .h5 files in the directory and its subdirectories"""
    return glob.glob(os.path.join(directory, '**', '*.h5'), recursive=True)

def process_h5_file(h5_path, task_name, output_folder, frame_count=25):
    """Process a single h5 file, extract frames and generate videos"""
    print(f"Processing {h5_path}")
    
    base_filename = os.path.basename(h5_path).replace('.h5', '')
    
    try:
        with h5py.File(h5_path, 'r') as file:
            # Check for Trial keys
            trials = sorted([k for k in file.keys() if 'Trial' in k], 
                           key=lambda x: int(x.replace('Trial', '')))
            
            for trial in trials:
                if 'data' not in file[trial]:
                    continue
                
                data = file[trial]['data']
                
                # Check for RGB views
                rgb_views = ['rgb_left', 'rgb_right', 'rgb_top', 'rgb_wrist']
                for view in rgb_views:
                    if view not in data:
                        continue
                    
                    # Check frame count and channels
                    frames = data[view][:]
                    if len(frames.shape) != 4 or frames.shape[0] < frame_count or frames.shape[3] != 3:
                        continue
                    
                    # Evenly sample frames
                    total_frames = frames.shape[0]
                    indices = np.linspace(0, total_frames-1, frame_count, dtype=int)
                    selected_frames = frames[indices]
                    
                    # Generate video filename
                    view_name = view.split('_')[1]  # left, right, top, wrist
                    video_name = f"{base_filename}-{trial}-{view_name}.mp4"
                    
                    # Save to the corresponding view subfolder
                    view_folder = os.path.join(output_folder, view_name)
                    video_path = os.path.join(view_folder, video_name)
                    
                    # Make sure width and height are even (required by some codecs)
                    height, width, channels = selected_frames[0].shape
                    if width % 2 == 1:
                        width -= 1
                    if height % 2 == 1:
                        height -= 1
                    
                    # Use more compatible encoder and parameters
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
                    
                    for frame in selected_frames:
                        # Resize if necessary
                        if frame.shape[0] != height or frame.shape[1] != width:
                            frame = cv2.resize(frame, (width, height))
                        
                        # Convert RGB to BGR for OpenCV
                        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        video.write(bgr_frame)
                    
                    video.release()
                    print(f"Saved video: {video_path}")
                    
                    # Save corresponding JSON file to the same view subfolder
                    json_name = f"{base_filename}-{trial}-{view_name}.json"
                    json_path = os.path.join(view_folder, json_name)
                    
                    with open(json_path, 'w') as json_file:
                        json.dump({"0": task_name}, json_file)
                    
                    print(f"Saved JSON: {json_path}")
                    
    except Exception as e:
        print(f"Error processing {h5_path}: {e}")

def process_dataset(csv_file, base_dir, frame_count=25):
    """Process the dataset based on the CSV file"""
    # Make sure the base directory exists
    ensure_dir(base_dir)
    
    # Read CSV file
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            task_name = row['task_name'].strip('"')
            url = row['url'].strip('"')
            
            print("-----------------------------")
            print(f"Downloading {task_name} with {url}")
            
            # Create task directory (handling special characters)
            sanitized_task_name = sanitize_filename(task_name)
            task_dir = os.path.join(base_dir, sanitized_task_name)
            ensure_dir(task_dir)
            
            # Create four view subfolders
            view_dirs = ['left', 'right', 'top', 'wrist']
            for view_dir in view_dirs:
                ensure_dir(os.path.join(task_dir, view_dir))
            
            # Download tar.gz file
            filename = os.path.basename(url)
            tar_path = os.path.join(".", filename)
            
            try:
                # Download the file
                download_file(url, tar_path)
                
                # Extract the file
                print(f"Extracting {tar_path}")
                extracted_dir = extract_tarfile(tar_path)
                
                if extracted_dir:
                    extracted_path = os.path.join(".", extracted_dir)
                    
                    # Remove original tar file
                    os.remove(tar_path)
                    print(f"Removed {tar_path}")
                    
                    # Find .h5 files
                    h5_files = find_h5_files(extracted_path)
                    
                    if h5_files:
                        print(f"Found {len(h5_files)} h5 files")
                        
                        # Process each h5 file
                        for h5_file in h5_files:
                            process_h5_file(h5_file, task_name, task_dir, frame_count)
                    else:
                        print("No h5 files found")
                    
                    # Remove extracted folder
                    shutil.rmtree(extracted_path)
                    print(f"Removed {extracted_path}")
                else:
                    print("Failed to extract directory")
            
            except Exception as e:
                print(f"Error processing {task_name}: {e}")
                # Clean up temporary files if needed
                if os.path.exists(tar_path):
                    os.remove(tar_path)
                
                continue
            
            # Pause briefly between tasks to avoid overloading network
            time.sleep(1)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process robot dataset files')
    parser.add_argument('--data', '-d', type=str, default='a', 
                        choices=['a', 'k', 't'], 
                        help='Dataset type: a=Autonomous (default), k=Kinesthetic, t=Teleoperation')
    parser.add_argument('--frame', '-f', type=int, default=25, 
                        help='Number of frames to extract (default: 25)')
    
    args = parser.parse_args()
    
    # Determine which dataset to process
    if args.data == 'k':
        csv_file = "Kinesthetic.csv"
        base_dir = "./Kinesthetic"
    elif args.data == 't':
        csv_file = "Teleoperation.csv"
        base_dir = "./Teleoperation"
    else:  # default is 'a'
        csv_file = "Autonomous.csv"
        base_dir = "./Autonomous"
    
    # Process the dataset
    process_dataset(csv_file, base_dir, args.frame)

if __name__ == "__main__":
    main()