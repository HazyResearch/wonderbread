prompt__rewrite_sop__intro: str = (
    lambda task_descrip: f"""# Task
Your job is to improve upon the Standard Operating Procedure (SOP) for the workflow that is demonstrated in the following sequence of screenshots and actions.

# SOP Rubric
- Element Specification: Each element referenced in the SOP has a descriptive name and location (i.e., "Accounting tab under the Finances Section")
- Action Type: The only actions referenced in the SOP should be one of the following: Press, Delete, Click, Type, Scroll.
- Edge Case Coverage: the SOP describes any edge cases that the user might encounter, and how to solve them  (i.e., "if you don't see button, scroll down")
- Discrete Action: The SOP only contains one discrete action per step (i.e., the action "click on the text bar and type "hello"" should be converted to two separate steps: (1) click on the text bar and (2) type "hello")
- Action Relevance: Each action should be true to the task  (i.e., if the task is to find the "grey t-shirt" clothing item, then an action which instructs the user to type text in the search bar should type the text "grey t-shirt")
- Generality: The steps of the SOP should reflect how to do this task in general and not overfit to the specific window size or screen of the demonstration (i.e., "Scroll until you find the row with your order" rather than "Scroll 130 pixels down")

# Workflow

The workflow is: "{task_descrip if task_descrip else 'unspecified'}"

# User Interface

The workflow was executed within the web application shown in the screenshots.

# Workflow Demonstration

You are given the following sequence of screenshots which were sourced from a demonstration of the workflow. 
The screenshots are presented in chronological order.

Between each screenshot, you are also provided the action that was taken to transition between screenshots. 

Here are the screenshots and actions of the workflow:"""
)

prompt__rewrite_sop__intro_kf: str = (
    lambda task_descrip: f"""# Task
Your job is to improve upon the Standard Operating Procedure (SOP) for the workflow that is demonstrated in the following sequence of screenshots and actions.

# SOP Rubric
- Element Specification: Each element referenced in the SOP has a descriptive name and location (i.e., "Accounting tab under the Finances Section")
- Action Type: The only actions referenced in the SOP should be one of the following: Press, Delete, Click, Type, Scroll.
- Edge Case Coverage: the SOP describes any edge cases that the user might encounter, and how to solve them  (i.e., "if you don't see button, scroll down")
- Discrete Action: The SOP only contains one discrete action per step (i.e., the action "click on the text bar and type "hello"" should be converted to two separate steps: (1) click on the text bar and (2) type "hello")
- Action Relevance: Each action should be true to the task  (i.e., if the task is to find the "grey t-shirt" clothing item, then an action which instructs the user to type text in the search bar should type the text "grey t-shirt")
- Generality: The steps of the SOP should reflect how to do this task in general and not overfit to the specific window size or screen of the demonstration (i.e., "Scroll until you find the row with your order" rather than "Scroll 130 pixels down")

# Workflow

The workflow is: "{task_descrip if task_descrip else 'unspecified'}"

# User Interface

The workflow was executed within the web application shown in the screenshots.

# Workflow Demonstration

You are given the following sequence of screenshots which were sourced from a demonstration of the workflow. 
The screenshots are presented in chronological order.

Here are the screenshots of the workflow:"""
)

prompt__rewrite_sop__intro_act: str = (
    lambda task_descrip: f"""# Task
Your job is to improve upon the Standard Operating Procedure (SOP) for the workflow that is demonstrated in the following sequence of screenshots and actions.

# SOP Rubric
- Element Specification: Each element referenced in the SOP has a descriptive name and location (i.e., "Accounting tab under the Finances Section")
- Action Type: The only actions referenced in the SOP should be one of the following: Press, Delete, Click, Type, Scroll.
- Edge Case Coverage: the SOP describes any edge cases that the user might encounter, and how to solve them  (i.e., "if you don't see button, scroll down")
- Discrete Action: The SOP only contains one discrete action per step (i.e., the action "click on the text bar and type "hello"" should be converted to two separate steps: (1) click on the text bar and (2) type "hello")
- Action Relevance: Each action should be true to the task  (i.e., if the task is to find the "grey t-shirt" clothing item, then an action which instructs the user to type text in the search bar should type the text "grey t-shirt")
- Generality: The steps of the SOP should reflect how to do this task in general and not overfit to the specific window size or screen of the demonstration (i.e., "Scroll until you find the row with your order" rather than "Scroll 130 pixels down")

# Workflow

The workflow is: "{task_descrip if task_descrip else 'unspecified'}"

# Workflow Demonstration

You are given the following sequence of actions which were sourced from a demonstration of the workflow. 
The actions are presented in chronological order.

Here are the actions of the workflow:"""
)

prompt__rewrite_sop__close: str = (
    lambda sop: f"""

# Standard Operating Procedure

Here are the sequence of steps that you should have followed to complete this workflow:

{sop}

NOTE: The screenshots may not map 1-to-1 to the steps in the Standard Operating Procedure. i.e. screenshot #3 may correspond to step #2 (or multiple steps) in the Standard Operating Procedure.
However, as long as the general flow of the workflow is the same, then the workflow is considered to have accurately followed the Standard Operating Procedure.
Also note that elements may be interchangeably referred to as buttons or links (the distinction is not important).

# Instructions

Given what you observed in the previous sequence of screenshots and actions, rewrite the Standard Operating Procedure to increase clarity and accuracy. If any of the steps are missing, or if any of the steps were performed out of order, then the Standard Operating Procedure should be updated to correct these mistakes.

Provide your answer as a numbered list with the following format:
1. The first action to be taken goes here
2. The second action to be taken goes here
3. The third action goes here ...

Please write the new updated Standard Operating Procedure below using the guidelines from the SOP Rubric above:
"""
)

prompt__rewrite_sop__close_kf: str = (
    lambda sop: f"""

# Standard Operating Procedure

Here are the sequence of steps that you should have followed to complete this workflow:

{sop}

NOTE: The screenshots may not map 1-to-1 to the steps in the Standard Operating Procedure. i.e. screenshot #3 may correspond to step #2 (or multiple steps) in the Standard Operating Procedure.
However, as long as the general flow of the workflow is the same, then the workflow is considered to have accurately followed the Standard Operating Procedure.
Also note that elements may be interchangeably referred to as buttons or links (the distinction is not important).

# Instructions

Given what you observed in the previous sequence of screenshots, rewrite the Standard Operating Procedure to increase clarity and accuracy. If any of the steps are missing, or if any of the steps were performed out of order, then the Standard Operating Procedure should be updated to correct these mistakes.

Provide your answer as a numbered list with the following format:
1. The first action to be taken goes here
2. The second action to be taken goes here
3. The third action goes here ...

Please write the new updated Standard Operating Procedure below using the guidelines from the SOP Rubric above:
"""
)

prompt__rewrite_sop__close_act: str = (
    lambda sop: f"""

# Standard Operating Procedure

Here are the sequence of steps that you should have followed to complete this workflow:

{sop}

NOTE: The actions may not map 1-to-1 to the steps in the Standard Operating Procedure. i.e. action #3 may correspond to step #2 (or multiple steps) in the Standard Operating Procedure.
However, as long as the general flow of the workflow is the same, then the workflow is considered to have accurately followed the Standard Operating Procedure.
Also note that elements may be interchangeably referred to as buttons or links (the distinction is not important).

# Instructions

Given what you observed in the previous sequence of actions, rewrite the Standard Operating Procedure to increase clarity and accuracy. If any of the steps are missing, or if any of the steps were performed out of order, then the Standard Operating Procedure should be updated to correct these mistakes.

Provide your answer as a numbered list with the following format:
1. The first action to be taken goes here
2. The second action to be taken goes here
3. The third action goes here ...

Please write the new updated Standard Operating Procedure below using the guidelines from the SOP Rubric above:
"""
)