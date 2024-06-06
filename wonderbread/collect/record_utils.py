import signal
import subprocess
import os
import datetime
import json
import platform
from selenium.webdriver.common.by import By
from selenium import webdriver
from playwright.sync_api import sync_playwright, Browser
from typing import Dict, List, Tuple, Union, Any, Optional, Callable

LIST_OF_BROWSER_APPLICATIONS = ["Google Chrome", "Firefox", "Safari"]

class LoggedItemBaseClass:
    def __init__(
        self,
        id: Optional[int] = None,
        step: Optional[int] = None,
        timestamp: Optional[datetime.datetime] = None,
        secs_from_start: Optional[float] = None,
    ):
        self.id: Optional[int] = id
        self.step: Optional[int] = step
        self.timestamp: Optional[datetime.datetime] = timestamp
        self.secs_from_start: Optional[float] = secs_from_start

    def __repr__(self) -> Dict[str, Any]:
        return self.__dict__

    def __str__(self) -> str:
        return str(self.__dict__)

    def to_json(self) -> Dict[str, Any]:
        return self.__dict__ | {
            "id": self.id,
            "step": self.step,
            "timestamp": self.timestamp.isoformat(),
            "secs_from_start": self.secs_from_start,
        }


class State(LoggedItemBaseClass):
    def __init__(
        self,
        url: Optional[str],
        tab: Optional[str],
        json_state: Optional[List[Dict[str, str]]],
        html: Optional[str],
        screenshot_base64: Optional[str],
        path_to_screenshot: Optional[str],
        window_position: Dict[str, int], # position of active application's focused window
        window_size: Dict[str, int], # size of active application's focused window
        active_application_name: str, # name of active application (e.g. "Google Chrome")
        screen_size: Dict[str, str], # size of entire laptop screen
        is_headless: bool = False, # if TRUE, then state was taken from browser is running in headless mode
    ):
        super().__init__()
        self.url: Optional[str] = url
        self.tab: Optional[str] = tab
        self.json_state: Optional[List[Dict[str, str]]] = json_state
        self.html: Optional[str] = html
        self.screenshot_base64: Optional[str] = screenshot_base64
        self.path_to_screenshot: Optional[str] = path_to_screenshot
        self.window_position: Dict[str, int] = window_position  # x, y
        self.window_size: Dict[str, int] = window_size  # width, height
        self.active_application_name: str = active_application_name
        self.screen_size: Dict[str, str] = screen_size  # width, height of entire laptop screen
        self.is_headless: bool = is_headless

    def to_json(self) -> Dict[str, Any]:
        return super().to_json() | {
            "screenshot_base64": None,  # don't save screenshot to json b/c messy
        }

class UserAction:
    """Store an action taken by the user.

    Possible args:
        self.timestamp: Optional[str] = None # seconds since start
        self.type: Optional[str] = None
        self.x: Optional[int] = None
        self.y: Optional[int] = None
        self.button: Optional[str] = None
        self.pressed: Optional[bool] = None
        self.key: Optional[str] = None
        self.element_attributes: Optional[str] = None
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Trace:
    """Store a trace of user actions on their computer."""

    def __init__(self):
        self.log: List[Tuple[str, Union[State, UserAction]]] = []
        self.last_state: Optional[State] = None
        self.last_action: Optional[UserAction] = None

    def log_state(self, state: State):
        # Only save webpage stuff if using a browser
        self.log.append(
            {
                "type": "state",
                "data": state,
            }
        )
        self.last_state = state

    def log_action(self, action: UserAction):
        # Only save element_attributes if using browser
        if self.last_state is not None and (
            self.last_state.active_application_name not in LIST_OF_BROWSER_APPLICATIONS
        ):
            if hasattr(action, "element_attributes"):
                del action.element_attributes

        self.log.append(
            {
                "type": "action",
                "data": action,
            }
        )
        self.last_action = action

    def to_json(self) -> List[Dict[str, str]]:
        jsonified: List[Dict[str, str]] = []
        start_timestamp: Optional[datetime.datetime] = None
        for obj in self.log:
            data: Dict = obj.get("data").__dict__

            # Validation
            assert "timestamp" in data, f"Timestamp not found in data: {data}"

            # Add fields
            if start_timestamp is None:
                start_timestamp = data["timestamp"]
            data["secs_from_start"] = (
                (data["timestamp"] - start_timestamp).total_seconds()
                if start_timestamp is not None
                else 0
            )

            # JSONify
            for key, value in data.items():
                if isinstance(value, datetime.datetime):
                    data[key] = value.isoformat()

            jsonified.append(
                {
                    "type": obj.get("type"),
                    "data": data,
                }
            )
        return jsonified


class ScreenRecorder:
    """
    Helper class for screen recording.

    Usage:
    ```
        recorder = ScreenRecorder("test.mp4")
        recorder.start() # starts recording
        recorder.stop() # stops recording, saves recording to file
    """

    def __init__(self, path_to_screen_recording: str) -> None:
        self.path_to_screen_recording: str = path_to_screen_recording

    def start(self) -> None:
        self.proc = subprocess.Popen(
            ["screencapture", "-v", self.path_to_screen_recording],
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        os.kill(self.proc.pid, signal.SIGINT)
        # Poll until screen recording is done writing to file
        while self.proc.poll() is None:
            pass

class Environment:
    """
        Wrapper around selenium + playwright.
        Tries to conform to Selenium API as closely as possible.
    """

    ALLOWED_ENVS: List[str] = ["selenium", "playwright"]

    def __init__(self, 
                 env_type: str = "selenium"):
        self.env_type: str = env_type
        
        assert env_type in self.ALLOWED_ENVS, f"Invalid env_type: {env_type}"
    
    def start(self, *args, is_headless: bool = False, record_video_dir: Optional[str] = None, **kwargs):
        """Creates a new browser instance."""
        self.is_headless: bool = is_headless
        if self.env_type == "selenium":
            self.selenium_driver: webdriver.Chrome = setup_chrome_driver(*args, is_headless=is_headless, **kwargs)
        elif self.env_type == "playwright":
            self.playwright, self.playwright_browser = setup_playwright_driver(*args, is_headless=is_headless, **kwargs)
            self.playwright_context = self.playwright_browser.new_context(record_video_dir=record_video_dir)
            self.playwright_page = self.playwright_context.new_page()

    def stop(self):
        if self.env_type == "selenium":
            self.selenium_driver.quit()
        elif self.env_type == "playwright":
            self.playwright_context.close()
            self.playwright_browser.close()
            self.playwright.stop()

    def reset(self):
        if self.env_type == "selenium":
            self.selenium_driver.delete_all_cookies()
        elif self.env_type == "playwright":
            self.playwright_context.close()
            self.playwright_context = self.playwright_browser.new_context()
            self.playwright_page = self.playwright_context.new_page()

    def get(self, url: str):
        """Navigates to the given URL."""
        if self.env_type == "selenium":
            self.selenium_driver.get(url)
        elif self.env_type == "playwright":
            self.playwright_page.goto(url)
    
    def find_elements(self, css_selector: str):
        if self.env_type == "selenium":
            return self.selenium_driver.find_elements(By.CSS_SELECTOR, css_selector)
        elif self.env_type == "playwright":
            return self.playwright_page.query_selector_all(css_selector)
    
    def find_element(self, css_selector: str):
        if self.env_type == "selenium":
            return self.selenium_driver.find_element(By.CSS_SELECTOR, css_selector)
        elif self.env_type == "playwright":
            return self.playwright_page.query_selector(css_selector)
    
    def type_in_element(self, css_selector: str, text: str):
        """Types `text` in the element specified by `css_selector`."""
        if self.env_type == "selenium":
            self.find_element(css_selector).send_keys(text)
        elif self.env_type == "playwright":
            self.find_element(css_selector).fill(text)

    def click_element(self, css_selector: str):
        """Clicks the element specified by `css_selector`."""
        if self.env_type == "selenium":
            self.find_element(css_selector).click()
        elif self.env_type == "playwright":
            self.find_element(css_selector).click()

    @property
    def current_url(self) -> str:
        """Get current URL."""
        if self.env_type == "selenium":
            return self.selenium_driver.current_url
        elif self.env_type == "playwright":
            return self.playwright_page.url
    
    @property
    def title(self) -> str:
        """Get current tab name."""
        if self.env_type == "selenium":
            return self.selenium_driver.title
        elif self.env_type == "playwright":
            return self.playwright_page.title()
    
    def get_window_rect(self) -> Dict[str, int]:
        """Get coordinates of browser on screen."""
        if self.env_type == "selenium":
            return self.selenium_driver.get_window_rect()
        elif self.env_type == "playwright":
            return {
                'x' : 0,
                'y' : 0,
                'width' : self.playwright_page.viewport_size['width'],
                'height' : self.playwright_page.viewport_size['height'],
            }
    
    def content(self) -> str:
        """Gets the full HTML contents of the page, including the doctype"""
        if self.env_type == "selenium":
            return self.selenium_driver.page_source
        elif self.env_type == "playwright":
            return self.playwright_page.content()

    def execute_script(self, script: str, is_playwright_use_wrapper: bool = True) -> str:
        """Executes JS script on the current webpage"""
        if self.env_type == "selenium":
            return self.selenium_driver.execute_script(script)
        elif self.env_type == "playwright":
            # Note: For playwright, we need to inject () => {}
            return self.playwright_page.evaluate(f"() => {{ {script} }}" if is_playwright_use_wrapper else script)
    
    def save_screenshot(self, path_to_output: str, is_async: bool = False):
        """Saves screenshot to `path_to_output`"""
        if self.is_headless:
            if self.env_type == "selenium":
                self.selenium_driver.save_screenshot(path_to_output)
            elif self.env_type == "playwright":
                self.playwright_page.screenshot(path=path_to_output)
        else:
            save_screenshot(path_to_output, is_async=is_async)

class BaseClass:
    def __init__(self, 
                 model_kwargs: Optional[Dict[str, str ]] = None, 
                 env: Optional[Environment] = None):
        self.logger: Callable = lambda x : x
        self.model_kwargs: Dict[str, str] = model_kwargs or {}
        self.env: Optional[Environment] = env

    def set_logger(self, logger: Callable):
        """Define the function that we'll call to log things."""
        self.logger = logger

class Observer(BaseClass):
    """
    Purpose: Generate state representations (JSON, image, etc.) of the current webpage
    """

    def __init__(
        self,
        env: Environment,
        path_to_screenshots_dir: str,
        is_take_screenshots: bool = True,
        is_delete_xpath_from_json_state: bool = True,
    ):
        super().__init__(env=env)
        self.path_to_screenshots_dir: Optional[str] = path_to_screenshots_dir
        self.is_take_screenshots: bool = is_take_screenshots
        self.is_delete_xpath_from_json_state: bool = is_delete_xpath_from_json_state

    def run(self, is_take_screenshots: Optional[bool] = None) -> State:
        """Return state of application."""
        import pyautogui
        # Get screenshot
        path_to_screenshot: Optional[str] = None
        if self.is_take_screenshots and (is_take_screenshots is None or is_take_screenshots):
            path_to_screenshot = os.path.join(
                self.path_to_screenshots_dir, f"{int(datetime.datetime.now().timestamp() * 1000)}.png"
            )
            try:
                # NOTE: happens asynchronously in background process, so encode_image may fail
                # Must be async otherwise it will cause all pynput recordings to lag
                self.env.save_screenshot(path_to_screenshot, is_async=True) 
            except Exception as e:
                print(str(e))
                print(f"Error taking screenshot for path: `{path_to_screenshot}`")

        # Get active application window specs
        active_application_state: Dict[str, Any] = get_active_application_state(self.env)
        is_application_browser: bool = (
            active_application_state["name"] in LIST_OF_BROWSER_APPLICATIONS
        )

        # Get webpage state as JSON list of elements
        url: Optional[str] = None
        tab: Optional[str] = None
        json_state: Optional[List[Dict[str, str]]] = None
        if is_application_browser:
            json_state = self.convert_webpage_to_json_elements(self.env)
            url = self.env.current_url
            tab = self.env.title

        return State(
            url=url,
            tab=tab,
            json_state=json_state,
            html=self.env.content() if is_application_browser else None,
            screenshot_base64=None, # must be set later
            path_to_screenshot=path_to_screenshot,
            active_application_name=active_application_state["name"],
            window_size={
                k: v
                for k, v in active_application_state.items()
                if k in ["width", "height"]
            },
            window_position={
                k: v for k, v in active_application_state.items() if k in ["x", "y"]
            },
            screen_size={
                'width': pyautogui.size().width,
                'height': pyautogui.size().height,
            },
            is_headless=self.env.is_headless,
        )

    def convert_webpage_to_json_elements(
        self,
        env: Environment,
    ) -> List[Dict[str, str]]:
        """Converts the current webpage into a JSON list of dicts, where each dict is a visible/relevant HTML element and their attributes."""
        # Get current state as JSON blob
        with open("./get_webpage_state.js", "r") as fd:
            js_script: str = fd.read()
        json_state: Dict[str, str] = json.loads(env.execute_script(js_script))

        # Adjust (x,y) coordinates to account for browser window position
        browser_width: int = env.execute_script(
            "return window.outerWidth;"
        )  # width of browser window
        browser_viewport_width: int = env.execute_script(
            "return window.innerWidth;"
        )  # width of webpage itself
        browser_height: int = env.execute_script(
            "return window.outerHeight;"
        )  # height of browser window
        browser_viewport_height: int = env.execute_script(
            "return window.innerHeight;"
        )  # height of webpage itself
        browser_chrome_width: int = browser_width - browser_viewport_width
        browser_chrome_height: int = browser_height - browser_viewport_height
        browser_coords: Dict[
            str, int
        ] = env.get_window_rect() # coords of browser on screen
        browser_x, browser_y = browser_coords["x"], browser_coords["y"]
        for element in json_state:
            # Account for positioning of browser window on screen
            if env.env_type == "playwright":
                # Ignore chrome since playwright ignores chrome (i.e. positions (0,0) as top-left of viewport)
                element['x'] += browser_x
                element['y'] += browser_y
            elif env.env_type == 'selenium':
                element["x"] += browser_x + browser_chrome_width
                element["y"] += browser_y + browser_chrome_height
            # Center (x,y) within element
            element["x"] += element["width"] / 2
            element["y"] += element["height"] / 2

        # Drop xpath
        for element in json_state:
            if self.is_delete_xpath_from_json_state:
                del element["xpath"]
            for key in ["role", "text", "type", "label"]:
                if key in element and element[key] is None:
                    del element[key]

        # Add chrome omnibox to state (for navigation)
        chrome_omnibox: Dict[str, Any] = {
            "x": browser_coords["x"] + 250,
            "y": browser_coords["y"] + browser_chrome_height / 2 + 20,
            "height": browser_chrome_height,
            "width": browser_width,
            "tag": "chrome_omnibox",
            "role": "Address bar / Search box",
            "text": "",
            "type": "input",
            "label": "Chrome Omnibox - This is an address bar that can be used to search a query on Google or navigate to a URL",
        }
        json_state.append(chrome_omnibox)

        return json_state


def get_active_application_state(env: Environment) -> Dict[str, Any]:
    """Get the name of the currently active desktop application."""
    import Quartz
    from AppKit import NSWorkspace
    if env.env_type == 'playwright':
        return {
            'name': 'Google Chrome',
            'x': 0,
            'y': 0,
            'width': env.playwright_page.viewport_size['width'],
            'height': env.playwright_page.viewport_size['height'],
        }
    active_app_name: str = NSWorkspace.sharedWorkspace().activeApplication()[
        "NSApplicationName"
    ]
    active_app_pid: int = NSWorkspace.sharedWorkspace().activeApplication()[
        "NSApplicationProcessIdentifier"
    ]
    window_info_list: List = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly
        | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    x, y, width, height = None, None, None, None
    for window_info in window_info_list:
        if window_info["kCGWindowOwnerPID"] == active_app_pid:
            x: int = int(window_info["kCGWindowBounds"]["X"])
            y: int = int(window_info["kCGWindowBounds"]["Y"])
            width = int(window_info["kCGWindowBounds"]["Width"])
            height = int(window_info["kCGWindowBounds"]["Height"])
            break

    return {
        "name": active_app_name,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }

def save_screenshot(path_to_screenshot: str, is_async: bool = False):
    """
    Takes a screenshot and saves it to `path_to_screenshot`
    Source: https://github.com/OthersideAI/self-operating-computer/blob/main/operate/main.py
    """
    user_platform: str = platform.system()
    if user_platform == "Darwin":  # (Mac OS)
        # Use the screencapture utility to capture the screen with the cursor
        proc = subprocess.Popen(f'screencapture -C "{path_to_screenshot}"', shell=True)
        if not is_async:
            # Poll until screenshot is saved to file
            while proc.poll() is None:
                pass
    else:
        raise ValueError(
            f"The platform you're using ({user_platform}) is not currently supported"
        )
        
def setup_chrome_driver(is_headless: bool = False) -> webdriver.Chrome:
    """Attach Selenium driver to Chrome session running on port 9222"""
    print(f"Selenium is starting in {'headless' if is_headless else 'UI'} mode...")
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutoReload")
    options.debugger_address = "127.0.0.1:9222"
    options.headless = is_headless
    driver = webdriver.Chrome(options=options)
    print("Selenium is running...")
    return driver


def setup_playwright_driver(is_headless: bool = False) -> Browser:
    """Sping up Playwright instance"""
    print(f"Playwright is starting in {'headless' if is_headless else 'UI'} mode...")
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(channel="chrome", headless=is_headless)
    print("Playwright is running...")
    return playwright, browser

def merge_consecutive_scrolls(trace_json: List[Dict[str, Any]], pixel_margin_of_error: float = 0.0) -> List[Dict[str, Any]]:
    '''Merge consecutive scroll events into a single scroll action.
    This is lenient by +/- N pixels in x/y coords
    '''
    last_action: Dict[str, Any] = None
    idxs_to_remove: List[int] = []
    for idx, event in enumerate(trace_json):
        if event["type"] == "action":
            if (
                last_action is not None
                and last_action['data']['type'] == 'scroll' 
                and event['data']['type'] == 'scroll'
                and abs(last_action['data']['x'] - event['data']['x']) <= pixel_margin_of_error
                and abs(last_action['data']['y'] - event['data']['y']) <= pixel_margin_of_error
            ):
                # We found two consecutive scroll events at the same(-ish) location
                last_action['data']['dx'] += event['data']['dx']
                last_action['data']['dy'] += event['data']['dy']
                idxs_to_remove.append(idx)
            else:
                last_action = event
    return [event for idx, event in enumerate(trace_json) if idx not in idxs_to_remove]

def merge_consecutive_states(trace_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    '''Merge consecutive states into one state (the last one).'''
    idxs_to_remove: List[int] = []
    consecutive_event_idxs: List[int] = []
    for idx, event in enumerate(trace_json):
        if event["type"] == "state":
            consecutive_event_idxs.append(idx)
        if event['type'] != 'state' or idx == len(trace_json) - 1:
            # Found a non-state or at end of trace, so clear out our consecutive state tracker
            if len(consecutive_event_idxs) > 1:
                # keep the last state
                idxs_to_remove += consecutive_event_idxs[:-1]
            consecutive_event_idxs = []
    return [event for idx, event in enumerate(trace_json) if idx not in idxs_to_remove]

def remove_action_type(trace_json: List[Dict[str, Any]], action_type: str) -> List[Dict[str, Any]]:
    '''Remove all actions with type == `action_type`'''
    return [
        event 
        for event in trace_json 
        if (
            event['type'] != 'action' 
            or event['data']['type'] != action_type
        )
    ]

def remove_esc_key(trace_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    '''Remove all keypresses with key == 'esc' '''
    return [
        event 
        for event in trace_json 
        if (
            event['type'] != 'action' 
            or event['data']['type'] not in ['keypress', 'keyrelease']
            or event['data']['key'] != 'Key.esc'
        )
    ]

def merge_consecutive_keystrokes(trace_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    '''Merge consecutive keypresses/keyreleases into the same field into one atomic entry.'''
    last_action: Dict[str, Any] = None
    idxs_to_remove: List[int] = []
    prior_state: Dict[str, Any] = None # State immediately before current action
    prior_prior_state: Dict[str, Any] = None # State before last action (i.e. before the state immediately before to this action)
    for idx, event in enumerate(trace_json):
        if event["type"] == "state":
            prior_prior_state = prior_state
            prior_state = event['data']
        elif event["type"] == "action":
            if (
                last_action is not None # There is a previous action
                and last_action['data']['type'] in ['keypress', 'keyrelease', 'keystroke', ] # Previous action was key event
                and event['data']['type'] in ['keypress', 'keyrelease', 'keystroke',] # Current action is key event
                and prior_state['active_application_name'] == prior_prior_state['active_application_name'] # In same application
                and (
                    ( # If in web browser, then we need to be in the same HTML input field
                        prior_state['active_application_name'] in LIST_OF_BROWSER_APPLICATIONS # In web browser
                        and 'element_attributes' in last_action['data']
                        and 'element_attributes' in event['data']
                        and last_action['data']['element_attributes'] is not None
                        and event['data']['element_attributes'] is not None
                        and last_action['data']['element_attributes'].get('xpath', None) == event['data']['element_attributes'].get('xpath', None)
                    )
                    or ( # If not in web browser, then don't check HTML input field
                        prior_state['active_application_name'] not in LIST_OF_BROWSER_APPLICATIONS # Not in web browser
                    )
                )
                and (not event['data']['key'].startswith('Key.') or event['data']['key'] in ['Key.space', 'Key.shift', 'Key.shift_r', 'Key.caps_lock', 'Key.backspace']) # Ignore non-space/Shift special keys
            ):
                # We found two consecutive non-special-key keystroke events in the same HTML field (i.e. identical xpath)
                if event['data']['type'] == 'keypress':
                    # only record keypresses (i.e. ignore keyrelease events so that we don't double count keypresses)
                    last_action['data']['key'] += ' ' + event['data']['key']
                last_action['data']['type'] = 'keystroke' # merge into one atomic keystroke
                last_action['data']['end_timestamp'] = event['data']['timestamp']
                last_action['data']['timestamp'] = event['data']['timestamp'] # use end_timestamp as timestamp for this action, so that we know its finished by the time we record it as having "happened"; needed for long keystroke events
                last_action['data']['secs_from_start'] = event['data']['secs_from_start']
                idxs_to_remove.append(idx)
            else:
                last_action = event
                last_action['data']['start_timestamp'] = last_action['data']['timestamp']
    return [event for idx, event in enumerate(trace_json) if idx not in idxs_to_remove]
    