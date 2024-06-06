import json
import datetime
from functools import partial
import os
from pynput import mouse, keyboard
import argparse
import multiprocessing
from moviepy.editor import VideoFileClip
from PIL import Image
from tqdm import tqdm
from wonderbread.helpers import get_rel_path
from wonderbread.collect.record_utils import (
    merge_consecutive_keystrokes,
    merge_consecutive_scrolls,
    merge_consecutive_states,
    remove_action_type,
    remove_esc_key,
    State,
    UserAction,
    Trace,
    ScreenRecorder,
    Observer,
    Environment,
)
from typing import Dict, List, Any, Optional

TRACE_END_KEY = keyboard.Key.esc  # Press this key to terminate the trace

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_output_dir",
        default="./outputs/",
        type=str,
        required=False,
        help="Path to output directory",
    )
    parser.add_argument(
        "--is_webarena",
        action="store_true",
        help="Set to TRUE if task is webarena task",
    )
    parser.add_argument(
        "--name", type=str, required=False, help="Name of task being demonstrated"
    )
    return parser.parse_args()


def print_(*args):
    """Hacky fix needed to get printed statements to left-align in terminal (prob caused by calling `screencapture` command)"""
    print(*args, '\r')
    

def is_string_in_integer_range(s: str, min_value: int, max_value: int):
    """Assert that the string 's' represents an integer within the specified range [min_value, max_value]."""
    try:
        value = int(s)
        return min_value <= value <= max_value
    except ValueError:
        return False


def execute_scripts(env: Environment):
    """Helper function that re-executes all JavaScript scripts whenever webpage is changed."""
    path: str = "./event_listeners.js"
    with open(path, "r") as fd:
        # Map clicks/keystrokes to specific elements on the webpage
        js_script: str = fd.read()
    print_("EXECUTE SCRIPTS")
    env.execute_script(js_script)


def get_last_element_attributes(env: Environment, key: str) -> Optional[Dict[str, str]]:
    """Return the last element attributes for the given key as parsed JSON dict."""
    elem: Optional[str] = env.execute_script(f"return window.{key} ? JSON.stringify(window.{key}) : localStorage.getItem('{key}');")
    return json.loads(elem) if elem else None

####################################################
####################################################
#
# Mouse/Keyboard listeners
#
####################################################
####################################################


def on_action_worker(
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    path_to_screenshots_dir: str,
):
    env: Environment = Environment('selenium')
    env.start()
    print_("\r>>>>>>>> GOOD TO START RECORDING WORKFLOW <<<<<<<<<<")
    observer: Observer = Observer(
        env=env,
        path_to_screenshots_dir=path_to_screenshots_dir,
        is_take_screenshots=False,
        is_delete_xpath_from_json_state=False,
    )
    while True:
        data: Dict[str, Any] = task_queue.get()  # This will block if the queue is empty
        if data is None:
            result_queue.put(None)
            break

        # Ignore mousedown / keyrelease
        if data["type"] == "mousedown":
            continue
        elif data["type"] == "keyrelease":
            continue

        # Get state
        state: State = observer.run()
        state.timestamp = data["timestamp"]
        result_queue.put(state)

        # Get action
        is_application_browser: bool = state.active_application_name in ["Google Chrome"]
        action: Dict[str, Any] = {
            key: val
            for key, val in data.items()
            if not (
                key == "element_attributes" and not is_application_browser
            )  # Don't save element_attributes if not using browser
        }
        result_queue.put(UserAction(**action))

        print_({key: val for key, val in action.items() if key != "element_attributes"})

        # Re-execute scripts if webpage has changed
        if is_application_browser and not env.execute_script('return window.isEventListenerLoaded'):
            execute_scripts(env)
    env.stop()
    exit()


def on_scroll(
    task_queue: multiprocessing.Queue,
    env: Environment,
    x: int,
    y: int,
    dx: int,
    dy: int,
):
    timestamp: datetime.datetime = datetime.datetime.now()
    task_queue.put(
        {
            "type": "scroll",
            "timestamp": timestamp,
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
            "element_attributes": get_last_element_attributes(env, 'lastScrolled'),
        }
    )


def on_click(
    task_queue: multiprocessing.Queue,
    env: Environment,
    x: int,
    y: int,
    button: mouse.Button,
    pressed: bool,
):
    timestamp: datetime.datetime = datetime.datetime.now()
    task_queue.put(
        {
            "type": "mousedown" if pressed else "mouseup",
            "timestamp": timestamp,
            "x": x,
            "y": y,
            "is_right_click": button == mouse.Button.right,
            "pressed": pressed,
            "element_attributes": get_last_element_attributes(env, 'lastMouseDown' if pressed else 'lastMouseUp'),
        }
    )


def on_key_press(
    task_queue: multiprocessing.Queue, env: Environment, key: keyboard.Key
):
    timestamp: datetime.datetime = datetime.datetime.now()
    task_queue.put(
        {
            "type": "keypress",
            "timestamp": timestamp,
            "key": str(key),
            "element_attributes": get_last_element_attributes(env, 'lastKeyDown'),
        }
    )
    # Quit if ESC is pressed
    if key == TRACE_END_KEY:
        mouse_listener.stop()
        keyboard_listener.stop()


def on_key_release(
    task_queue: multiprocessing.Queue, env: Environment, key: keyboard.Key
):
    timestamp: datetime.datetime = datetime.datetime.now()
    task_queue.put(
        {
            "type": "keyrelease",
            "timestamp": timestamp,
            "key": str(key),
            "element_attributes": get_last_element_attributes(env,  'lastKeyUp'),
        }
    )


if __name__ == "__main__":
    args = parse_args()
    args.name = args.name if args.name is not None else "trace"
    trace_name: str = (
        f"{args.name} @ {datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    path_to_output_dir: str = os.path.join(args.path_to_output_dir, trace_name)
    path_to_screenshots_dir: str = os.path.join(path_to_output_dir, f"screenshots/")
    path_to_webarena_tasks: str = get_rel_path(__file__, "../benchmark/webarena/")
    path_to_screen_recording: str = os.path.join(
        path_to_output_dir, f"{trace_name}.mp4"
    )

    # make dirs
    os.makedirs(path_to_output_dir, exist_ok=True)
    os.makedirs(path_to_screenshots_dir, exist_ok=True)

    # Setup webarena tasks
    webarena_task: Optional[str] = None
    webarena_start_url: Optional[str] = None
    if args.is_webarena:
        assert is_string_in_integer_range(
            args.name, 0, 812
        ), f"Invalid task name for WebArena: `{args.name}`"
        for filename in os.listdir(path_to_webarena_tasks):
            if not filename.endswith(".json") or filename.startswith("test"):
                # Skip non-JSON files and test files
                continue
            task_id: int = int(filename.split(".")[0])
            if int(args.name) == task_id:
                task = json.load(
                    open(os.path.join(path_to_webarena_tasks, filename), "r")
                )
                webarena_task = task["intent"]
                webarena_start_url: str = task["start_url"]

    # make dirs
    os.makedirs(path_to_output_dir, exist_ok=True)
    os.makedirs(path_to_screenshots_dir, exist_ok=True)

    # Attach to Chrome session running on port 9222
    env: Environment = Environment('selenium')
    env.start()
    if args.is_webarena:
        env.get(webarena_start_url)

    # Start Javascript scripts
    execute_scripts(env)

    # Start screen recorder
    print_("Starting screen recorder...")
    screen_recorder = ScreenRecorder(path_to_screen_recording)
    screen_recorder.start()

    # Queues for multiprocessing
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Log initial state
    observer: Observer = Observer(
        env=env,
        path_to_screenshots_dir=path_to_screenshots_dir,
        is_take_screenshots=False,
        is_delete_xpath_from_json_state=False,
    )
    initial_state: State = observer.run()
    initial_state.timestamp = datetime.datetime.now()
    result_queue.put(initial_state)

    # Start listeners for mouse/keyboard interactions
    action_logger_process = multiprocessing.Process(
        target=on_action_worker,
        args=(task_queue, result_queue, path_to_screenshots_dir),
    )
    action_logger_process.start()
    with mouse.Listener(
        on_click=partial(on_click, task_queue, env),
        on_scroll=partial(on_scroll, task_queue, env),
    ) as mouse_listener:
        with keyboard.Listener(
            on_press=partial(on_key_press, task_queue, env),
            on_release=partial(on_key_release, task_queue, env),
        ) as keyboard_listener:
            keyboard_listener.join()
            mouse_listener.join()

    # Save trace
    task_queue.put(None)
    trace: Trace = Trace()
    while True:
        result = result_queue.get()
        if result is None:
            break
        if isinstance(result, UserAction):
            trace.log_action(result)
        elif isinstance(result, State):
            trace.log_state(result)
        else:
            raise ValueError(f"Unknown result type: {type(result)}")

    print_("Done with tracing. Savings results...")
    print_("# of events:", len(trace.log))

    # Stop screen recording and save to disk
    print_("Stopping screen recorder and saving to disk...")
    screen_recorder.stop()

    # Close processes
    action_logger_process.join()
    action_logger_process.close()
    
    # Get trace
    trace_json: List[Dict[str, Any]] = trace.to_json()
    # Compress `json_state` list into string for readability when printed out to trace.json
    trace_json = [ { **x, 'data': { **x['data'], 'json_state' : json.dumps(x.get('data', {}).get('json_state', None)) if x.get('data', {}).get('json_state', None) is not None else None } } if x['type'] == 'state' else x for x in trace_json]

    # Save raw trace
    json.dump(
        {"trace": trace_json},
        open(os.path.join(path_to_output_dir, f"[raw] {trace_name}.json"), "w"),
        indent=2,
    )

    # Post processing
    # Merge consecutive scroll events
    trace_json = merge_consecutive_scrolls(trace_json)
    # Remove ESC keypresses
    trace_json = remove_esc_key(trace_json)
    # Remove keyrelease and mousedown events
    trace_json = remove_action_type(trace_json, "keyrelease")
    trace_json = remove_action_type(trace_json, "mousedown")
    # Merge consecutive keystrokes in same input field
    trace_json = merge_consecutive_keystrokes(trace_json)
    # Merge consecutive states without intermediate actions (only keep first + last)
    trace_json = merge_consecutive_states(trace_json)
    # Reset step indices
    trace_json = [ { **x, 'data' : { **x['data'], 'step': (i+1) // 2, 'id' : i } } for i, x in enumerate(trace_json) ]

    # Save trace to disk
    json.dump(
        {"trace": trace_json},
        open(os.path.join(path_to_output_dir, f"{trace_name}.json"), "w"),
        indent=2,
    )


    # Create dummy SOP.txt
    with open(os.path.join(path_to_output_dir, f"SOP - {trace_name}.txt"), "w") as fd:
        fd.write("")

    # Pull out screenshots from screen recording
    print_("Creating screenshots from screen recording...")
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
                    print_(
                        f"====> FAILED to extract screenshot: event_idx={event_idx} | timestamp={timestamp}..."
                    )
        print_('')

    # Save updated trace with screenshot filenames
    n_screenshots: int = len([ x for x in os.listdir(path_to_screenshots_dir) if x.endswith('.png') ])
    assert n_screenshots == len([ x for x in trace_json if x['type'] == 'state' ]), f"Number of screenshots ({n_screenshots}) does not match number of states ({len([ x for x in trace_json if x['type'] == 'state' ])})"
    json.dump(
        {"trace": trace_json},
        open(os.path.join(path_to_output_dir, f"{trace_name}.json"), "w"),
        indent=2,
    )

    print_(f"DONE. Saved to: `{path_to_output_dir}`")
    os.system("stty sane") # fix terminal line spacing
