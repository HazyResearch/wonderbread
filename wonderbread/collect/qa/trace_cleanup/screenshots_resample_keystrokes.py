import datetime
import json
import os
import pandas as pd
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from PIL import Image
from config import path_to_dir, path_to_df_valid

df_valid: str = pd.read_csv(path_to_df_valid)

for path_to_task_dir in tqdm(os.listdir(path_to_dir)):
    # check if path_to_task_dir is a directory
    if not os.path.isdir(os.path.join(path_to_dir, path_to_task_dir)):
        continue
    # check if path_to_task_dir is in df_valid (i.e. is a valid task)
    if path_to_task_dir not in df_valid['folder_name'].values:
        continue
    # Loop thru all files in directory
    for path_to_file in os.listdir(os.path.join(path_to_dir, path_to_task_dir)):
        try:
            # Find file that ends in '.json' and does not start with '['
            if path_to_file.endswith('.json') and not path_to_file.startswith('['):
                path_to_json_file: str = os.path.join(path_to_dir, path_to_task_dir, path_to_file)
                # Load JSON
                with open(path_to_json_file, 'r') as f:
                    json_data = json.load(f)
                trace = json_data['trace']
                
                # Get .mp4 path
                demo_name: str = path_to_file.split('.')[0]
                path_to_screen_recording: str = os.path.join(path_to_dir, path_to_task_dir, demo_name + '.mp4')
                
                # Get screenshots/ dir path
                path_to_screenshots_dir: str = os.path.join(path_to_dir, path_to_task_dir, 'screenshots/')
                
                # Get start state's timestamps (for calculating secs_from_start later)
                start_state_timestamp: str = trace[0]['data']['timestamp']
                start_state_secs_from_start: str = trace[0]['data']['secs_from_start']

                last_state = None
                for event in trace:
                    if event['type'] == 'state':
                        last_state = event
                    elif event['type'] == 'action':
                        if event['data']['type'] == 'keystroke':
                            # Found keystroke action
                            path_to_screenshot: str = last_state['data']['path_to_screenshot']

                            # Take new screenshot from .mp4
                            print(f"Replacing screenshot in `{demo_name}` for `{event['data']['type']}` @ `{event['data']['start_timestamp']}` for `{os.path.basename(path_to_screenshot)}`")
                            with VideoFileClip(path_to_screen_recording) as video:
                                # We should take the corresponding screenshot at `secs_from_start`` relative to `start_timestamp` and not `timestamp`
                                # (so that we don't take the screenshot for the previous state after the following keystroke action has started)
                                # NOTE: This is unique for `keystroke` events given they last so long
                                timestamp: float = (
                                    datetime.datetime.fromisoformat(last_action['data']['timestamp'] if last_action else event['data']['start_timestamp']) - datetime.datetime.fromisoformat(start_state_timestamp)
                                ).total_seconds() + start_state_secs_from_start
                                try:
                                    frame = video.get_frame(timestamp)
                                    img: Image = Image.fromarray(frame)
                                    # Replace screenshot with updated one
                                    img.save(os.path.join(path_to_screenshots_dir, os.path.basename(path_to_screenshot)))
                                except Exception as e:
                                    print(f"Error replacing screenshot in `{demo_name}` for `{event['data']['type']}` @ `{event['data']['start_timestamp']}` for `{os.path.basename(path_to_screenshot)}`")
                                    print(str(e))
                        last_action = event
        except Exception as e:
            print(f"Error processing `{path_to_file}`")
            print(str(e))
            with open('./error_log.txt', 'a') as f:
                f.write(f"Error processing `{path_to_file}`\n")
                f.write(str(e) + '\n')
                f.write('\n=====================\n\n')
            continue