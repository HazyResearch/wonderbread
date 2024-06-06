"""
demonstration-collection/experiments/eval/eval_completions.py

This module contains functions that are used to access the OpenAI API and fetch
completions for a given prompt.
"""

from typing import Optional
from wonderbread.helpers import fetch_openai_json_completion
import os
import json

# LLM Completion Cache
SOP_CACHE_DIR = "." + os.path.sep + "sop_cache"
MODEL = "gpt-4-1106-preview" # model to use for eval

def get_completion(
    id: Optional[str],
    prompt_name: str,
    prompt: str,
    force_fetch: bool = False,
    store_fetch_to_cache: bool = True,
) -> str:
    """
    Determines if there is a cached completion for the prompt. If there is,
    the cached completion is returned. If there is not, the completion is
    generated and returned.

    Args:
        id (str): The identifier for the task. Used to name the subdirectory
            for the cache file.
        prompt_name (str): The identifier for the prompt. Used to name the cache
            file.
        prompt (str): The prompt to be used for generating the completion.
        force_fetch (bool): If True, the completion will be generated even if
            it is cached. Defaults to False.
        store_fetch_to_cache (bool): If True, the fetched completion will be
            stored in the cache located in SOP_CACHE_DIR. Defaults to True.

    Returns:
        str: The completion for the prompt
    """
    if id is not None:
        # Build the cache path
        cache_path = os.path.join(SOP_CACHE_DIR, id, f"{prompt_name}.json")
        # Determine if this prompt is cached
        is_cached = os.path.exists(cache_path)
    else:
        is_cached = False

    if is_cached and not force_fetch:
        # Load the cached completion
        with open(cache_path, "r") as f:
            completion = json.load(f)["completion"]
    else:
        # Generate the completion
        completion = generate_completion(prompt)

        # Attempt to load the cached completion as a dictionary
        try:
            test_dict = json.loads(completion)
            test_index = test_dict["index"]
        except (json.JSONDecodeError, KeyError):
            # Attempt to regenerate the completion (only do this once)
            completion = generate_completion(prompt)
            # Just fail (before logging prompt completion) if the completion is still not valid on the second try
            _ = json.loads(completion)

        # Cache the completion
        log_prompt_completion(cache_path, prompt, completion, store_fetch_to_cache)

    # Load the completion as a dictionary
    completion = json.loads(completion)

    # Add the prompt_name to the completion
    completion["prompt_name"] = prompt_name

    # Convert the index to an integer
    completion["index"] = int(completion["index"])

    return completion


def generate_completion(prompt: str) -> str:
    """
    Generates the completion for the prompt using the OpenAI API.

    Returns:
        str: The completion for the prompt
    """
    # Generate the string completion
    return fetch_openai_json_completion(prompt, model=MODEL)


def log_prompt_completion(
    cache_path: str,
    prompt: str,
    completion: str,
    store_fetch_to_cache: bool = True,
) -> None:
    """
    Logs the prompt and completion to a json file

    Args:
        cache_path (str): The path to the cache file
        prompt (str): The prompt to be used for generating the completion
        completion (str): The completion for the prompt
        store_fetch_to_cache (bool): If True, the fetched completion will be
            stored in the cache located in SOP_CACHE_DIR. Defaults to True.
    """
    # If store_fetch_to_cache is False, do not store the completion
    if not store_fetch_to_cache:
        return

    # Get directories along cache path
    cache_path_dir = os.path.dirname(cache_path)

    # Make log directory if it doesn't exist
    if not os.path.exists(cache_path_dir):
        os.makedirs(cache_path_dir)

    # Beutify the prompt
    prompt = prompt.split("\n")
    prompt = tuple([f"{line}\n" for line in prompt])

    # Write a json file containing the prompt and completion
    with open(cache_path, "w") as f:
        json.dump({"prompt": prompt, "completion": completion}, f, indent=4)
