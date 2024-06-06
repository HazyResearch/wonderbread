""" Evaluate SOPs using rubric criteria for generating each sop"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import os

import json
from tqdm import tqdm
from eval_prompts import rubric_evaluation_prompt
import openai
import time
import sys
import argparse


from workflows.helpers import (
    encode_image,
    convert_trace_action_to_dsl,
    add_standard_experiment_args,
)


def rubric_evaluation_prompt(sop: str, gold_sop: str) -> str:
    rubric = """- Element Specification: Each element referenced in the SOP has a descriptive name and location (i.e., "Accounting tab under the Finances Section")
- Action Type: The only actions referenced in the SOP should be one of the following: Press, Delete, Click, Type, Scroll.
- Edge Case Coverage: the SOP describes any edge cases that the user might encounter, and how to solve them  (i.e., "if you don't see button, scroll down")
- Discrete Action: The SOP only contains one discrete action per step (i.e., the action "click on the text bar and type "hello"" should be converted to two separate steps: (1) click on the text bar and (2) type "hello")
- Action Relevance: Each action should be true to the task  (i.e., if the task is to find the "grey t-shirt" clothing item, then an action which instructs the user to type text in the search bar should type the text "grey t-shirt")
- Generality: The steps of the SOP should reflect how to do this task in general and not overfit to the specific window size or screen of the demonstration (i.e., "Scroll until you find the row with your order" rather than "Scroll 130 pixels down")
"""

    prompt = f"""### Instruction: Please evaluate the SOP based on the following rubric. 
- Please generate a score from 1 (worse) to 5 (best) for the SOP based on the rubric. 
- Be strict in your evaluation.
- Your output should be a json object with the following structure:

```
{{
  "explanation": str - your explanation for what score the SOP should get based on the rubric. For each rubric item, provide a score 0 or 1. Use these to provide a justification for the overall score."
  "score": float - the score of the SOP based on the rubric (1.0-5.0),
}}
```

### Sample High Quality SOP:
```
What is the top-1 best-selling product in 2022?

1. Click on the "Reports" button on the far lefthand sidebar. It has an icon which looks like a chart. It should be located directly above the "Stores" button and below the "Content" button.
2. In the popup menu that appears, click on the "Bestsellers" link to go to the "Bestsellers Report" page. The link should be located under the "Products" section.
3. Locate the field labeled "Period" and click on the dropdown menu to reveal our time options.
4. Click on the "Year" option to set the reporting period to Year.
5. Click on the "From" textbox to focus it. It should be located directly underneath the "Period" field.
6. Type in the first day of our desired time period, which in this case is "01/01/2022". Make sure the textbox is empty before you type into it.
7. Click on the "To" textbox to focus it. It should be located directly underneath the "From" field.
8. Type in the last day of our desired time period, which in this case is "12/31/2022". Make sure the textbox is empty before you type into it.
9. Click on the orange "Show Report" button, which can be found on the top right of the page, in order to generate our best-selling product report.
10. The best-selling products will appear in a table at the bottom of the page. Scroll down if you cannot see the full table. The results should be sorted in descending order by Order Quantity, so the top-1 best-selling product will simply be the first row in the results.
```

Note the detailed localization for each UI element _("'Reports' button on the far lefthand sidebar. It has an icon which looks like a chart...")_ the edge case coverage (_"Make sure the textbox is empty before you type into it."_ and _"Scroll down if you cannot see the full table"_) and generalizability (_"Type in the first day of our desired time period, which in this case is "01/01/2022"."_).
    
### Rubric
{rubric}

### SOP: 
{sop}

### SOP evaluation:"""

    return prompt


def evaluate_sops_all(
    path_to_demos: str,
    output_path: str,
    path_to_results: str,
) -> pd.DataFrame:
    """
    Given a dataframe of sops were the text for sops is in the column "sop", for any row in the dataframe were gt_rankings == 1, evaluate the sop, and add the evaluation metrics to the dataframe.
    """
    # add columns to the dataframe

    entries = []
    client = openai.OpenAI()
    # read in past outputs =
    data = pd.read_csv(path_to_results)

    ### SELECT THE MODEL YOU WANT TO EVALUATE
    # data = data[data["ablation--model"] == "GPT4"]
    # data = data[data["ablation"] == "1--True--True--True--GPT4"]
    # data = data[data["ablation--model"] == "GeminiPro"]
    # data = data[data["ablation"] == "1--True--True--True--GeminiPro"]

    model_name = "Claude3"

    data = data[data["ablation--model"] == model_name]
    data = data[data["ablation"] == f"1--True--True--True--{model_name}"]

    demos = []

    for folder in os.listdir(path_to_demos):
        base_sop = os.path.join(path_to_demos, folder, f"[orig] SOP - {folder}.txt")

        # get all demos for this particular folderd
        demo_name = folder

        sample_demos = data[data["demo_name"] == folder]

        if len(sample_demos) > 0 and demo_name not in demos:
            base_sop = sample_demos.iloc[0]["orig_sop"]

            gold_sop = None

            demo_name = folder
            gold_sop = os.path.join(path_to_demos, demo_name, f"SOP - {demo_name}.txt")

            with open(gold_sop, "r") as f:
                gold_sop = f.read()

            entries.append(
                {
                    "demo_name": demo_name,
                    "gold_sop": gold_sop,
                    "base_sop": base_sop,
                    "sop_1": sample_demos[sample_demos["n_self_reflection"] == 1][
                        "new_sop"
                    ].values[0],
                }
            )

    populated_entries = []

    for i, row in tqdm(enumerate(entries)):

        entry = {
            "demo_name": row["demo_name"],
            "gold_sop": row["gold_sop"],
        }
        rubrics = {}
        for s in ["0", "1"]:

            if s == "0":
                id_str = "base_sop"
            else:
                id_str = f"sop_{s}"

            prompt = rubric_evaluation_prompt(row[id_str], row["gold_sop"])
            messages: List[str] = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a steps of procedure (SOP) evaluator. Please evaluate the following SOP based on the rubric provided. Do not add quotes around the explanation text",
                        }
                    ],
                }
            ] + [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]
            response = client.chat.completions.create(
                response_format={"type": "json_object"},
                messages=messages,
                max_tokens=4096,
                model="gpt-4-turbo",
                temperature=0,
            )
            response: str = response.choices[0].message.content
            response = response.replace("```json", "").replace("```", "")
            try:
                response = eval(response)
                entry[f"{s}_rubric_eval_explanation"] = response["explanation"]
                entry[f"{s}_rubric_eval_score"] = response["score"]
                rubrics[s] = response["score"]

                # print the response score
                print(f"Score for {id_str} is {response['score']}")

            except:
                print("Error in response")

        entry["rubric"] = rubrics
        populated_entries.append(entry)

        sops = pd.DataFrame(populated_entries)

        sops.to_csv(
            f"{output_path}/self_improving_all_results_rubric_{model_name}.csv",
            index=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--is_kf",
        action="store_true",
        default=False,
        help="If TRUE, then include screenshots as key frames into prompts",
    )
    parser.add_argument(
        "--is_act",
        action="store_true",
        default=False,
        help="If TRUE, then include action traces into prompts",
    )
    parser.add_argument(
        "--path_to_recordings",
        type=str,
        help="Path to the recordings",
    )
    parser.add_argument(
        "--path_to_sops",
        type=str,
        help="Path to the SOPs",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the results",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    is_kf: bool = args.is_kf
    is_act: bool = args.is_act

    sops: str = args.path_to_sops
    sops = pd.read_csv(sops)
    # read in the sops
    path_to_recordings: str = args.path_to_recordings

    evaluate_sops_all(path_to_recordings, args.output_path)
