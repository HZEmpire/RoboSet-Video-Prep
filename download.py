import os
import csv
import re
import requests
import time
import argparse
from tqdm import tqdm
import concurrent.futures
import functools
from processing import process_tarfile, ensure_dir

def sanitize_filename(name):
    """Replace special characters in filename with underscores"""
    return re.sub(r'[^\w\-\.]', '_', name)

def download_file(url, destination, position=0):
    """Download file to specified path with progress bar; display progress at given position"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            position=position,
            leave=True,
        ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def download_and_process_item(item, base_dir, frame_count=25):
    """
    Download and process a single data item.
    :param item: A tuple (row, pos), where pos is the unique progress bar position.
    """
    row, pos = item
    task_name = row['task_name'].strip('"')
    url = row['url'].strip('"')
    
    print(f"Processing task: {task_name} (position {pos})")
    
    # Create task directory (handling special characters)
    sanitized_task_name = sanitize_filename(task_name)
    task_dir = os.path.join(base_dir, sanitized_task_name)
    ensure_dir(task_dir)
    
    # Download tar.gz file
    filename = os.path.basename(url)
    tar_path = os.path.join(".", filename)
    
    try:
        # Download the file with assigned progress bar position
        download_file(url, tar_path, position=pos)
        
        # Process the downloaded tar file
        process_tarfile(tar_path, task_name, task_dir, frame_count)
        
    except Exception as e:
        print(f"Error processing {task_name}: {e}")
        # Clean up temporary files if needed
        if os.path.exists(tar_path):
            os.remove(tar_path)
    
    return f"Completed: {task_name}"

def process_dataset(csv_file, base_dir, frame_count=25, parallel=False):
    """Process the dataset based on the CSV file"""
    # Make sure the base directory exists
    ensure_dir(base_dir)
    
    # Read CSV file and create a list of (row, position) tuples for progress bar positioning
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    
    # 为每个任务分配唯一的 position
    items = [(row, idx) for idx, row in enumerate(rows)]
    
    if parallel:
        print("Enabling parallel processing mode, processing multiple tasks concurrently...")
        # Use ProcessPoolExecutor for parallel processing.
        max_workers = max(os.cpu_count() - 1, 1)
        print(f"Using {max_workers} worker processes")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Note: functools.partial cannot fix the 'item' parameter here since it varies.
            results = list(executor.map(
                functools.partial(download_and_process_item, base_dir=base_dir, frame_count=frame_count),
                items
            ))
            
            # Print results
            for result in results:
                print(result)
    else:
        # Sequential processing: iterate with assigned progress bar positions.
        for item in items:
            result = download_and_process_item(item, base_dir, frame_count)
            print(result)
            time.sleep(1)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Download robot dataset files')
    parser.add_argument('--data', '-d', type=str, default='a', 
                        choices=['a', 'k', 't'], 
                        help='Dataset type: a=Autonomous (default), k=Kinesthetic, t=Teleoperation')
    parser.add_argument('--frame', '-f', type=int, default=25, 
                        help='Number of frames to extract (default: 25)')
    parser.add_argument('--parallel', '-p', action='store_true',
                        help='Enable parallel processing for faster execution')
    
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
    
    # Process the dataset with parallel option
    process_dataset(csv_file, base_dir, args.frame, args.parallel)

if __name__ == "__main__":
    main()