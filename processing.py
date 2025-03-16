import os
import tarfile
import h5py
import numpy as np
import json
import cv2
import glob
import shutil
from tqdm import tqdm
import imageio

def ensure_dir(directory):
    """Make sure the directory exists, create it if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_tarfile(tar_path, extract_to='.'):
    """Extract tar.gz file to specified directory and return the top level directory name"""
    with tarfile.open(tar_path, 'r:*') as tar:
        tar.extractall(path=extract_to)
    
    # Get the top level directory name
    with tarfile.open(tar_path, 'r:*') as tar:
        top_level_dirs = {member.name.split('/')[0] for member in tar.getmembers() if member.name != '.'}
        return list(top_level_dirs)[0] if top_level_dirs else None

def find_h5_files(directory):
    """Find all .h5 files in the directory and its subdirectories"""
    return glob.glob(os.path.join(directory, '**', '*.h5'), recursive=True)

def adjust_dimensions_to_multiple_of_16(frames):
    """Adjust video frame dimensions to be multiples of 16 to ensure compatibility"""
    height, width, channels = frames[0].shape
    new_width = ((width + 15) // 16) * 16
    new_height = ((height + 15) // 16) * 16
    
    # If dimensions are already multiples of 16, no adjustment is needed
    if width == new_width and height == new_height:
        return frames
    
    # Adjust the size of each frame
    adjusted_frames = []
    for frame in frames:
        adjusted_frame = cv2.resize(frame, (new_width, new_height))
        adjusted_frames.append(adjusted_frame)
    
    return np.array(adjusted_frames)

def save_video_with_imageio(frames, output_path, fps=10):
    """Save video using imageio which has better compatibility"""
    try:
        # Convert frames to uint8 if they're not already
        if frames.dtype != np.uint8:
            frames = (frames * 255).astype(np.uint8)
        
        # Adjust dimensions to multiples of 16 to avoid warnings and ensure compatibility
        frames = adjust_dimensions_to_multiple_of_16(frames)
        
        #writer = imageio.get_writer(output_path, fps=fps, quality=8)
        writer = imageio.get_writer(output_path, quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        return True
    except Exception as e:
        print(f"Error saving video with imageio: {e}")
        return False

def process_h5_file(h5_path, task_name, output_folder, frame_count=25, max_pairs=500):
    """
Process a single h5 file, extract frames and generate videos.
    Returns True if processed completely, or False if early termination occurred.
    """    
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
                    # Make sure the view folder exists
                    ensure_dir(view_folder)
                    
                    # Check if the view folder has reached the limit
                    existing_files = os.listdir(view_folder)
                    if len(existing_files) >= 2 * max_pairs:
                        print(f"Skipping {view_name}: reached limit ({len(existing_files)//2} pairs)")
                        continue
                    
                    video_path = os.path.join(view_folder, video_name)
                    
                    # Try to save with imageio first (better compatibility)
                    if save_video_with_imageio(selected_frames, video_path):
                        # print(f"Saved video with imageio: {video_path}")
                        pass
                    else:
                        # Fallback to OpenCV if imageio fails
                        # Make sure dimensions are multiples of 16 for better compatibility
                        height, width = selected_frames[0].shape[:2]
                        new_width = ((width + 15) // 16) * 16
                        new_height = ((height + 15) // 16) * 16
                        
                        # Try H.264 codec if available
                        try:
                            # On macOS, 'avc1' is usually available for H.264
                            fourcc = cv2.VideoWriter_fourcc(*'avc1')
                            video = cv2.VideoWriter(video_path, fourcc, 10, (new_width, new_height))
                            
                            if not video.isOpened():
                                # Fall back to mp4v if avc1 is not available
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                video = cv2.VideoWriter(video_path, fourcc, 10, (new_width, new_height))
                        except:
                            # If H.264 is not available, use mp4v
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video = cv2.VideoWriter(video_path, fourcc, 10, (new_width, new_height))
                        
                        for frame in selected_frames:
                            # Adjust dimensions to multiples of 16
                            frame = cv2.resize(frame, (new_width, new_height))
                            
                            # Convert RGB to BGR for OpenCV
                            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            video.write(bgr_frame)
                        
                        video.release()
                        #print(f"Saved video with OpenCV: {video_path}")
                    
                    # Save corresponding JSON file to the same view subfolder
                    json_name = f"{base_filename}-{trial}-{view_name}.json"
                    json_path = os.path.join(view_folder, json_name)
                    
                    with open(json_path, 'w') as json_file:
                        json.dump({"0": task_name}, json_file)
                    
                    # Check if the view folder has reached the limit
                all_done = True
                for view in rgb_views:
                    view_name = view.split('_')[1]
                    folder = os.path.join(output_folder, view_name)
                    if len(os.listdir(folder)) < 2 * max_pairs:
                        all_done = False
                        break
                if all_done:
                    print("All views reached limit; skipping remaining h5 files.")
                    return False
            return True                    
    except Exception as e:
        print(f"Error processing {h5_path}: {e}")
        import traceback
        # Print detailed error traceback
        traceback.print_exc()
        return True

def process_tarfile(tar_path, task_name, output_folder, frame_count=25, max_pairs=500):
    """Process the dataset based on the tar file"""
    # Make sure the output folder exists
    ensure_dir(output_folder)
    
    # Create subfolders for each view
    view_dirs = ['left', 'right', 'top', 'wrist']
    for view_dir in view_dirs:
        ensure_dir(os.path.join(output_folder, view_dir))
    
    # Extract the tar file
    print(f"Extracting {tar_path}")
    extracted_dir = extract_tarfile(tar_path)
    
    if extracted_dir:
        extracted_path = os.path.join(".", extracted_dir)
        
        # Find .h5 files
        h5_files = find_h5_files(extracted_path)
        
        if h5_files:
            print(f"Found {len(h5_files)} h5 files")
            
            # Process each h5 file
            for h5_file in h5_files:
                cont = process_h5_file(h5_file, task_name, output_folder, frame_count, max_pairs)
                if not cont:
                    break
        else:
            print("No h5 files found")
        
        # Remove extracted folder
        shutil.rmtree(extracted_path)
        print(f"Removed {extracted_path}")
        
        # Remove tar file
        if os.path.exists(tar_path):
            os.remove(tar_path)
            print(f"Removed {tar_path}")
    else:
        print("Failed to extract directory")

if __name__ == "__main__":
    # Process the dataset
    process_tarfile("./Autonomous_RoboSet_Set_9_Pick_Wooden_Block_542_9.tar.gz", "example_task", "output", frame_count=25)