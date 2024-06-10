# Experiments

This folder contains our WONDERBREAD benchmark tasks. The tasks are divided into three folders: documentation, improvement, knowledge-transfer.

- Each task has a:
  - A `main.py` file that runs the experiment.
  - A `run_experiments.py` file that runs all experiments across all settings.

## Installation

Follow the instructions in the main `README.md` to (1) install the conda env; (2) download the dataset.

Move the dataset into the `data/` directory in this repo. You should now have many demonstration folders (potentially multiple per task).

## How to Run

You should be able to go into an tasks subfolder and simply run `python3 run_experiments.py` to generate all results.

```bash
cd documentation/sop_generation/
python3 run_experiments.py --model <one of: GPT4 | Gemini | Claude3>
```
