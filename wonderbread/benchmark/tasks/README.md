# Experiments

This folder contains our WONDERBREAD benchmark tasks. The tasks are divided into three folders: documentation, improvement, knowledge-transfer.

- Each task has a:
  - A `main.py` file that runs the experiment.
  - A `run_experiments.py` file that runs all experiments across all ablations.

## Installation

```bash
conda activate demo_env
pip3 install -r requirements.txt
```

Next, you need to download [the Google Drive folder here](https://drive.google.com/file/d/1_OYan6_wPfqLcFc5qpAMipkiFRLp-y-Y/view?usp=drive_link). It contains a subset of 30 separate tasks for testing plus 4 tasks with 5 demos each (so a total of 50 demonstrations).

Unzip this file, then move it into the `data/` directory in this repo. You should now have many demonstration folders (potentially multiple per task) in `data/Test] WebArena Tasks/`.

Use the `Process Mining Task Demonstrations.xlsx` file to determine which demos have corresponding Gold SOPs (located in `data/Test WebArena Tasks/Gold SOPs`) and should be included in Gold SOP experiments.

## How to Run

You should be able to go into an tasks subfolder and simply run `bash run_experiments.py` to generate all results.

```bash
cd documentation/sop_generation/
python3 run_experiments.py
```
