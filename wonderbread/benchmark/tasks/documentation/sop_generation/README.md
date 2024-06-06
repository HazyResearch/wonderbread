# Evaluations

This directory contains scripts to run auto-evaluations of model generated standard operating procedures (SOPs). It contains the following:

- [`eval.py`](./eval.py) - The main script used to run auto-evaluations of the model's generated SOPs against the "gold standard" SOPs.

- [`example_sops/`](./example_sops/) - A directory containing a few SOPs that can be used to debug the evaluation script. The SOPs in this directory are included in the repository for debugging purposes only and are not used in the actual evaluation. The directory is structured as follows:
    ```
    example_sops/
    ├── task_0/
    │   ├── rank_1.txt # The gold standard SOP for task 0
    │   ├── rank_2.txt # The rank 2 SOP for task 0
    |   └── ... 
    ├── task_1/
    │   ├── rank_1.txt # The gold standard SOP for task 1
    │   ├── rank_2.txt # The rank 2 SOP for task 1
    |   └── ...
    └── ...
    ```

- [`metrics.py`](./metrics.py) - Contains functions to calculate various metrics for evaluating the model's generated SOPs.

- [`eval_completion.py`](./eval_completion.py) - Contains functions related to fetching and caching evaluation completions from OpenAI's GPT-4 API. 

- [`eval_prompts.py`](./eval_prompts.py) - Contains a class abstraction for generating evaluation prompts for the model's generated SOPs. This class is used by the `eval.py` script to generate prompts for the model's SOPs.