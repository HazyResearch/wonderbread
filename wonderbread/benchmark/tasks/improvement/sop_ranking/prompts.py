prompt__rank_sop: str = lambda task_descrip, section__sops: f"""# Task
You are a business process management consultant whose job is to evaluate the effeciency of different versions of a standard operating procedure (SOP) for the same workflow.

You are given several SOPs, and your job is rank them based on their quality.

# Workflow

The workflow you are evaluating is: "{task_descrip}"

# SOPs

Here are the SOPs you are evaluating. Each SOP is given a distinct ID (i.e. "#1", "#2", etc.), and the content of the SOP is enclosed in triple backticks (i.e. "```"). Note that the SOPs are not necessarily listed in order of quality, and their IDs are 1-indexed.

{section__sops}

# Question

Given the SOPs, rank them based on their relative quality. Consider the efficiency of their steps, as well as whether or not they achieve the desired workflow. 

# Answer 

Answer in the following valid JSON format:

{{
    "thinking": "<think step-by-step about the correct SOP ranking; DO NOT use quote marks in your response>",
    "pred_ranking": "A list that ranks SOPs by their ID. The first ID in the list corresponds to the best SOP, and the last ID corresponds to the worst SOP (i.e., if there are two SOPs and #2 is better than #1, then you would return [2, 1])"
}}

Answer:"""