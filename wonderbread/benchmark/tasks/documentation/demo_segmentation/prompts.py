prompt__intro: str = lambda n_tasks, task_descriptions: f"""# Task
You are a process mining automation tool. Your are given a recording of a worker doing multiple workflows (potentially overlapping). 
Your job is to segment this recording into discrete workflows -- i.e. identify which actions correspond to which workflow.
Workflow segmentation is important for process mining because it allows us to analyze the performance of each workflow separately.

# Workflow

The {n_tasks} workflows being executed in the recording are as follows:
{task_descriptions}

# Workflow Demonstration

You are given the following recording of the worker completing these {n_tasks} workflows over the course of this recording.

The recording is presented in chronological order.
The workflows are executed in sequence, but may be present in any order. You can assume that the worker always finishes a workflow before starting the next one.
The recording may include both screenshots and the actions taken to transition between screenshots. 

Each screenshot and action is labeled with a unique identifier ("UUID"). We will use these UUIDs to refer to specific screenshots and actions in the recording when segmenting the recording into the {n_tasks} workflows.

Here is the overall recording:"""

prompt__close_uuid: str = lambda n_tasks, task_descriptions, sops : f"""
# Instructions

Given what you observe in the previous recording, please classify each UUID as belonging to one of the {n_tasks} workflows.

As a reminder, the workflows are as follows. Each workflow is assigned a classification letter:
{task_descriptions}

The workflows may be present in the recording in any order. You can assume that the worker always finishes a workflow before starting the next one, so there are no overlapping workflows.

{sops if sops else ""}

Provide your answer as a JSON dictionary with the following format:
{{
    "UUID_1": <workflow classification>,
    "UUID_2": <workflow classification>,
    ...
}}

Please write your JSON below:
"""

prompt__close_start_end: str = lambda n_tasks, task_descriptions, sops : f"""
# Instructions

Given what you observe in the previous recording, please tell me the start and end UUIDs for each of the {n_tasks} workflows.

As a reminder, the workflows are as follows. Each workflow is assigned a classification letter:
{task_descriptions}

The workflows may be present in the recording in any order. You can assume that the worker always finishes a workflow before starting the next one, so there are no overlapping workflows.

{sops if sops else ""}

Provide your answer as a JSON dictionary with the following format:
{{
    "A": {{
        "start": <start UUID for workflow A>,
        "end": <end UUID for workflow B>
    }},
    "B": {{
        "start": <start UUID for workflow A>,
        "end": <end UUID for workflow B>
    }},
    ...
}}

You must respond with valid JSON. Please write your JSON below:
"""