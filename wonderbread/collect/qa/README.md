# QA Checks

This folder contains scripts for running QA checks on the dataset.

## Installation

```bash
conda activate demo_env
pip3 install -r requirements.txt
```

Then, follow the [instructions here to setup Google Drive authentication.](https://docs.iterative.ai/PyDrive2/quickstart/#authentication).

Important notes:
1. Create the API under your **personal gmail**, as your **@stanford.edu** is not authorized to create APIs
2. Add your **@stanford.edu** email address as a **Test User** when you create your API, as the Google Drive folder is shared with that email address.

## How to Run

```bash
python3 main.py
```

This will run all the QA checks on the Google Drive folder, and output a CSV where each row is a `(task, person)` pair, and each column is a QA check. The value of each QA check cell is either `True` or `False`, where `True` means it passed the QA check, and `False` means it failed the QA check.

## Adding a New QA Check

To add a new QA check, create a new function in the file with your name in the `qa` folder. 

Your function should accept a single argument of type `List[Task]`, which is simply a list of `Task` objects (as defined in `qa > utils.py`). The function should return a list of `QACheck` objects (one for each `Task`), with the appropriate field set as defined in `qa > utils.py`.

These functions will be successively called by `main.py` and the results will be outputted to a CSV file.

NOTE: To access the SOP or trace.json of a task, you must use the `get_sop()` and `get_trace()` methods of the `Task` object. These methods will automatically download the SOP or trace.json from Google Drive if it is not already downloaded.

## Other notes

The `Process Mining Task Demonstrations.xlsx` file was downloaded from the version marked `FINAL` in Google Drive.


### Trace Cleanup

Run all of these in order:

```bash
cd trace_cleanup/
python trace_merge_scrolls.py # DONE
python trace_merge_states.py # DONE
python trace_fix_coords.py # DONE -- note: not idempotent 
python screenshots_resample_keystrokes.py # DONE
python trace_add_webarena_task_json.py # DONE
```