"""
demonstration-collection/experiments/eval/eval_prompts.py

This module contains functions that are used to generate prompts for the evaluation
of SOPs. 
"""

from typing import List, Optional


def rubric_evaluation_prompt(sop: str, action_trace: str) -> str:
    rubric = """- Element Specification: Each element referenced in the SOP has a descriptive name and location (i.e., "Accounting tab under the Finances Section")
- Action Type: The only actions referenced in the SOP should be one of the following: Press, Delete, Click, Type, Scroll.
- Edge Case Coverage: the SOP describes any edge cases that the user might encounter, and how to solve them  (i.e., "if you don't see button, scroll down")
- Discrete Action: The SOP only contains one discrete action per step (i.e., the action "click on the text bar and type "hello"" should be converted to two separate steps: (1) click on the text bar and (2) type "hello")
- Action Relevance: Each action should be true to the task  (i.e., if the task is to find the "grey t-shirt" clothing item, then an action which instructs the user to type text in the search bar should type the text "grey t-shirt")
- Faithfulness to Demonstration: The steps in SOP should match the steps taken in the actual recording of the demonstration
- Generality: The steps of the SOP should reflect how to do this task in general and not overfit to the specific window size or screen of the demonstration (i.e., "Scroll until you find the row with your order" rather than "Scroll 130 pixels down")
"""

    prompt = f"""### Instruction: Please evaluate the SOP based on the following rubric. 
- Please generate a score from 1 (best) to 5 (worst) for the SOP based on the rubric. 
- For the "Faithfulness to Demonstration" criteria, use the trace in the conversational history as context. 
- Your output should be a json object with the following structure:

```
{{
  "explanation": str - your thinking for what score the SOP should get based on the rubric,
  "score": int - the score of the SOP based on the rubric (1-5),
}}
```
    
### Rubric
{rubric}

SOP: {sop}

### SOP evaluation:
```json"""

    return prompt


def prep_json_prompt(prompt: str, requested: Optional[str] = None) -> str:
    """
    Adds a request for output in JSON format to the prompt.

    `requested` should be a string that describes the requested output and
    should be in the format of a JSON key-value pair such as:
    `"index" : int - index of the line in the list that best encapsulates the primary`

    Args:
      prompt (str): The prompt to be preprocessed
      requested (str): The requested output item

    Returns:
      str: The preprocessed prompt
    """

    if requested is None:
        requested = '"index" : int - index of the line in the List of Lines that best encapsulates the primary objective of the Query'

    # Append the structure of the output
    prompt += "- Output your response in the following JSON format:\n"
    prompt += "  {\n"
    prompt += '    "scratchpad": str - think step by step to come up with your decision (e.g. "The line with index 2 encapsulates the Query" or "No line encapsulates the Query")\n'
    prompt += f"    {requested}\n"
    prompt += "  }\n"

    return prompt


def map_query_to_one_prompt(query_line: str, list_of_lines: List[str]) -> str:
    """
    Creates a prompt that asks the LLM to map a given query line to one of the lines
    in the provided `list_of_lines`. The prompt is returned as a string.

    Args:
      query_line (str): The query line
      list_of_lines (List[str]): A list of lines to be queried
      log_str (str): A string that will be used to create a file name for logging

    Returns:
      int: The index of the first line in the list that encapsulates similar if not
        the same meaning as the primary objective of the query line. If no such line
        exists, the function will return -1.
    """
    # Add indecies to the list of lines
    list_of_lines = [f"{i} - {line}" for i, line in enumerate(list_of_lines)]
    # Start by joining the list of lines into a single multi-line string.
    list_of_lines_str = "\n".join(list_of_lines)

    # Generate prompt
    prompt = "Which line in the List of Lines best encapsulates the primary objective of the Query?\n"
    prompt += "Instructions:\n"
    prompt += "- Please give the index of the encapsulating line in the List of Lines (0-indexed).\n"
    prompt += "- If multiple lines encapsulate the Query's objective, please return the index of the first line.\n"
    prompt += "- If no line encapsulates the Query, please return -1.\n"
    prompt = prep_json_prompt(prompt, requested=None)
    prompt += "\n"
    prompt += f"Query: {query_line}\n\n"
    prompt += f"List of Lines:\n{list_of_lines_str}\n"

    return prompt


def map_query_to_many_prompt(query_line: str, list_of_lines: List[str]) -> str:
    """
    Creates a prompt that asks the LLM to map a given query line to a contiguous
    sequence of lines in the provided `list_of_lines`. The prompt is returned as a
    string.

    Args:
      query_line (str): The query line
      list_of_lines (List[str]): A list of lines to be queried
      log_str (str): A string that will be used to create a file name for logging

    Returns:
      int: The index of the first line in the list that encapsulates similar if not
        the same meaning as the primary objective of the query line. If no such line
        exists, the function will return -1.
    """
    # Add indecies to the list of lines
    list_of_lines = [f"{i} - {line}" for i, line in enumerate(list_of_lines)]
    # Start by joining the list of lines into a single multi-line string.
    list_of_lines_str = "\n".join(list_of_lines)

    # Generate prompt
    prompt = "Which line in the List of Lines best encapsulates the primary objective of the Query?\n"
    prompt += "Instructions:\n"
    prompt += "- Please give a list of indicies corresponding to the line(s) in the List of Lines (0-indexed).\n"
    prompt += "- If only one line encapsulates the Query's objective, please return a list with a single index.\n"
    prompt += "- If no line encapsulates the Query, please return an empty list.\n"
    prompt = prep_json_prompt(prompt, requested=None)
    prompt += "\n"
    prompt += f"Query: {query_line}\n\n"
    prompt += f"List of Lines:\n{list_of_lines_str}\n"

    return prompt
