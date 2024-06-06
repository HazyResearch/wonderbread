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
                trace_json = json_data['trace']
                
                # Get .mp4 path
                demo_name: str = path_to_file.split('.')[0]
                path_to_screen_recording: str = os.path.join(path_to_dir, path_to_task_dir, demo_name + '.mp4')
                
                # Get screenshots/ dir path
                path_to_screenshots_dir: str = os.path.join(path_to_dir, path_to_task_dir, 'screenshots/')
                
                # Get start state's timestamps (for calculating secs_from_start later)
                start_state_timestamp: str = trace_json[0]['data']['timestamp']
                start_state_secs_from_start: str = trace_json[0]['data']['secs_from_start']
                
                # Remove all screenshots in screenshots/ dir
                for file in os.listdir(path_to_screenshots_dir):
                    os.remove(os.path.join(path_to_screenshots_dir, file))

                # Extract screenshots from video
                with VideoFileClip(path_to_screen_recording) as video:
                    img_idx: int = 0
                    video_start_timestamp: datetime.datetime = datetime.datetime.fromtimestamp(
                        os.stat(path_to_screen_recording).st_birthtime
                    )
                    for event_idx, event in tqdm(
                        enumerate(trace_json), desc="Extracting Video => Screenshots", total=len([ x for x in trace_json if x['type'] == 'state' ])
                    ):
                        if event["type"] == "state":
                            # Our trace is: (S, A, S', A', ...)
                            # For S', we want to take the screenshot immediately before A' (for page loading / animations / etc.)
                            # So we actually should ignore the time of S', and instead use slightly before the time of A' for our extracted frame
                            # (Except for the last S, which use its own time for since there is no A after it)
                            timestamp: float = (
                                datetime.datetime.fromisoformat(trace_json[event_idx + 1 if len(trace_json) > event_idx + 1 else event_idx]["data"]["timestamp"])
                                - video_start_timestamp
                            ).total_seconds()
                            try:
                                frame = video.get_frame(timestamp)
                                img: Image = Image.fromarray(frame)
                                path_to_screenshot: str = os.path.join(path_to_screenshots_dir, f"{img_idx}.png")
                                img.save(path_to_screenshot)
                                trace_json[event_idx]["data"]["path_to_screenshot"] = path_to_screenshot
                                # trace_json[event_idx]["data"]["screenshot_base64"] = encode_image(path_to_screenshot)
                                img_idx += 1
                            except Exception as e:
                                print(
                                    f"====> FAILED to extract screenshot: event_idx={event_idx} | timestamp={timestamp}..."
                                )
        except Exception as e:
            print(f"Error processing `{path_to_file}`")
            print(str(e))
            with open('./error_log.txt', 'a') as f:
                f.write(f"Error processing `{path_to_file}`\n")
                f.write(str(e) + '\n')
                f.write('\n=====================\n\n')
            continue