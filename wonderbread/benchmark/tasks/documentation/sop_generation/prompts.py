
############################################
#
# Full
#
############################################

prompt__start: str = lambda task_descrip, ui_name : f"""# Task
Your job is to write a standard operating procedure (SOP) for a workflow.

# Workflow

The workflow is: "{task_descrip if task_descrip else 'Some unspecified digital task'}"

# User Interface

The workflow will be executed within a web application. The web application is called: "{ui_name}"
"""

prompt__end: str = lambda : f"""Here is a sample format for what your SOP should look like:
```
1. Type the name of the repository in the search bar at the top left of the screen. The placeholder text in the search bar is "Find a repository...", and it is located directly to the right of the site logo.
2. A list of repositories will appear on the next page. Scroll down until you see a repository with the desired name. The name of the repository will be on the lefthand side of the row in bold font. Stop when you find the name of the repository.
3. Click on the relevant repository to go to the repository's main page.
```

Note, the above SOP is just an example. Use the same format, but the actions will be different for your workflow.

Be as detailed as possible. Each step should be a discrete action that reflects what you see in the corresponding step. Don't skip steps.

Please write your SOP below:"""

prompt__td_intro: str = lambda task_descrip, ui_name: f"""{prompt__start(task_descrip, ui_name)}"""

prompt__td_close: str = lambda : f"""
# Instructions

Write an SOP for completing this workflow on this website. The SOP should simply contain an enumerated list of actions taken by the user to complete the given workflow.
In your SOP, list all of the actions taken (i.e., buttons clicked, fields entered, mouse scrolls etc.). Be descriptive about elements (i.e., 'the subheading located under the "General" section').

{prompt__end()}"""

prompt__td_kf_intro: str = lambda task_descrip, ui_name: f"""{prompt__start(task_descrip, ui_name)}

# Workflow Demonstration

You are given the following sequence of screenshots which were sourced from a demonstration of the workflow. 
The screenshots are presented in chronological order.

Here are the screenshots of the workflow:"""

prompt__td_kf_close: str = lambda : f"""
# Instructions

Given what you observe in the screenshots, write an SOP for completing the workflow on this website. The SOP should simply contain an enumerated list of actions taken by the user to complete the given workflow.
In your SOP, list all of the actions taken (i.e., buttons clicked, fields entered, mouse scrolls etc.). Be descriptive about elements (i.e., 'the subheading located under the "General" section'). Use the location of the mouse to identify which exact elements were clicked.

{prompt__end()}"""

prompt__td_act_intro: str = lambda task_descrip, ui_name: f"""{prompt__start(task_descrip, ui_name)}

# Workflow Demonstration

You are given the following sequence of actions which were sourced from a demonstration of the workflow. 
The actions are presented in chronological order.
Note that the action is written in a simplified DSL (domain-specific language) that we use to describe actions taken by users. You will need to translate this into a natural language description of the action and add more details about what was happening, why, and what elements were interacted with.

Here are the actions of the workflow:"""

prompt__td_act_close: str = lambda : f"""
# Instructions

Given what you observe in the sequence of DSL actions, write an SOP for completing the workflow on this website. The SOP should simply contain an enumerated list of actions taken by the user to complete the given workflow.
In your SOP, list all of the actions taken (i.e., buttons clicked, fields entered, mouse scrolls etc.). Be descriptive about elements (i.e., 'the subheading located under the "General" section'). Use the location of the mouse to identify which exact elements were clicked.

{prompt__end()}"""

prompt__td_kf_act_intro: str = lambda task_descrip, ui_name: f"""{prompt__start(task_descrip, ui_name)}

# Workflow Demonstration

You are given the following sequence of screenshots which were sourced from a demonstration of the workflow. 
The screenshots are presented in chronological order.

Between each screenshot, you are also provided the action that was taken to transition between screenshots. 
However, the action is written in a simplified DSL (domain-specific language) that we use to describe actions taken by users. You will need to translate this into a natural language description of the action and add more details about what was happening, why, and what elements were interacted with.

Here are the screenshots and actions of the workflow:"""

prompt__td_kf_act_close: str = lambda : f"""
# Instructions

Given what you observe in the previous sequence of screenshots and DSL actions, write an SOP for completing the workflow for this specific interface. The SOP should simply contain an enumerated list of actions taken by the user to complete the given workflow.
In your SOP, list all of the actions taken (i.e., buttons clicked, fields entered, mouse scrolls etc.). Be descriptive about elements (i.e., 'the subheading located under the "General" section'). Use the location of the mouse to identify which exact elements were clicked.

{prompt__end()}
"""

############################################
#
# Pairwise
#
############################################

prompt__start__pairwise: str = lambda task_descrip, ui_name : f"""# Task
Your job is to determine the single action that was taken between these screenshots were taken.

# User Interface

The web application where the screenshots are taken from is called: "{ui_name}"
"""

prompt__end__pairwise: str = lambda : f"""Here is a sample format for what your output should look like:
```
1. Click on the searchbar at the top left of the screen to focus it. The placeholder text in the search bar is "Find a repository...", and it is located directly to the right of the site logo. 
2. Type the name of the repository into the searchbar.
```

Note, the above output is just an example. Use the same format, but the action might be different for your screenshots.
You might have only one item in your output, or you might have multiple items. It depends on the action that took place between the screenshots.
Be as detailed as possible. Each step should be a discrete action that reflects what you see in the screenshots. Don't skip steps. 
Only include the action that took place between the screenshots, and do not make any assumptions about what happened before or after the screenshots were taken.

Please write your output below:"""


prompt__td_kf_intro__pairwise: str = lambda task_descrip, ui_name: f"""{prompt__start__pairwise(task_descrip, ui_name)}

# Workflow Demonstration

You are given the following two screenshots which were sourced from a demonstration of the workflow. 
The screenshots are presented in chronological order.
The first one was taken directly before the action was taken, and the second one was taken directly after the action was executed.
Note that these screenshots could have been taken at any step of the workflow.

Here are the screenshots of this specific step from the larger workflow:"""

prompt__td_kf_close__pairwise: str = lambda : f"""
# Instructions

Given what you observe in the screenshots, write the step(s) corresponding to this action that would go into a larger SOP for completing the workflow on this website. 
Make sure to list all of the actions taken to go from one screenshot to the other (i.e., buttons clicked, fields entered, mouse scrolls etc.). Be descriptive about elements (i.e., 'the subheading located under the "General" section'). Use the location of the mouse to identify which exact elements were clicked.

{prompt__end__pairwise()}
"""


prompt__td_kf_act_intro__pairwise: str = lambda task_descrip, ui_name: f"""{prompt__start__pairwise(task_descrip, ui_name)}

# Workflow Demonstration

You are given the following two screenshots which were sourced from a demonstration of the workflow. 
The screenshots are presented in chronological order.
The first one was taken directly before the action was taken, and the second one was taken directly after the action was executed.
Note that these screenshots could have been taken at any step of the workflow.

Between each screenshot, you are also provided the action that was taken to transition between screenshots. 
However, the action is written in a simplified DSL (domain-specific language) that we use to describe actions taken by users. You will need to translate this into a natural language description of the action and add more details about what was happening, why, and what elements were interacted with.

Here are the screenshots and action of this specific step from the larger workflow:"""

prompt__td_kf_act_close__pairwise: str = lambda : f"""
# Instructions

Given what you observe in the screenshots and DSL action, write the step(s) corresponding to this action that would go into a larger SOP for completing the workflow on this website. 
Make sure to list all of the actions taken to go from one screenshot to the other (i.e., buttons clicked, fields entered, mouse scrolls etc.). Be descriptive about elements (i.e., 'the subheading located under the "General" section'). Use the location of the mouse to identify which exact elements were clicked.

{prompt__end__pairwise()}
"""

prompt__join_pairwise: str = lambda sop, separator : f"""
Your job is to create a standard operating procedure (SOP) for a workflow that outlines each step taken to complete the workflow.

Previously, you were given subsets of consecutive screenshots taken from a longer sequence of screenshots of a workers doing the workflow. You were asked to write the step(s) taken between each screenshot. Our goal is to compile these smaller sets of steps into a larger SOP for completing the entire workflow.

I've copied your responses for this previous pairwise screenshot analysis below. Each pair of screenshots is separated by {separator}. 

```
{sop}
```

Your job now is to combine these steps into a single, coherent SOP for completing the entire workflow.  The steps are ordered chronologically, so you do not need to worry about the ordering of the steps. Instead, you should remove any duplicate steps and ensure that the steps flow logically from one to the next.

Please write your unified SOP below:
"""