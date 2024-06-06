"""
demonstration-collection/experiments/eval/metrics.py

This file contains functions that are used to evaluate the performance of models
using various metrics.
"""
from typing import Dict, List, Any, Optional, Tuple
from wonderbread.benchmark.tasks.documentation.sop_generation.eval_prompts import map_query_to_one_prompt
from wonderbread.benchmark.tasks.documentation.sop_generation.eval_completion import get_completion

class PairwiseComparison:
    """
    A class for calculating pairwise metrics between two SOPs.
    """

    def __init__(self, pred_sop: List[str], ref_sop: List[str], cache_id: Optional[str] = None):
        """
        Initializes the PairwiseComparison object.

        Args:
        pred_sop (List[str]): The generated SOP
        ref_sop (List[str]): The reference/standard SOP
        """

        # Save the inputs
        self.pred_sop_len = len(pred_sop)
        self.pred_sop = pred_sop

        self.ref_sop_len = len(ref_sop)
        self.ref_sop = ref_sop

        self.cache_id = cache_id

        # Create pairwise inclusion prompts
        self.prompts = self._generate_pairwise_inclusion_prompts()

        # Fetch the completions
        self.completions = self._fetch_completions()

    def _fetch_completions(self) -> Dict[str, Any]:
        """
        Runs the prompts through the OpenAI API (or a pulls a cached completion)
        and returns the completions.

        Returns:
        Dict[str, Any]: A dictionary of completions
        """

        # Create a dictionary to store completions
        completions: Dict[str, Any] = {}

        # Iterate over each prompt
        for prompt_id, prompt in self.prompts.items():
            # Fetch the completion
            completions[prompt_id] = get_completion(
                id=self.cache_id,
                prompt_name=f"{self.cache_id}_{prompt_id}",
                prompt=prompt,
                force_fetch=False,
            )

        return completions

    def _generate_pairwise_inclusion_prompts(self) -> Dict[str, str]:
        """
        Generates the LLM prompts used to determine if the predicted SOP's statements are entailed in the reference SOP,
        and vice versa. Utilizes the `map_query_to_one` function to do so.

        Args:
        pred_sop (List[str]): The generated SOP
        ref_sop (List[str]): The gold standard SOP

        Returns:
        Dict[str, str]: A dictionary that contains prompts
            [key] - The name of the prompt. Format: "{source}_line_{index}" where 
                - `source` is either "pred" or "gold" 
                - `index` is the index of the line in the SOP
            [value] - The prompt itself
        """

        # Create a dictionary to store prompts
        prompts: Dict[str, str] = {}

        # First, let's generate prompts for each line in the predicted SOP
        for pred_ind, pred_line in enumerate(self.pred_sop):
            prompts[f"pred_line_{pred_ind}"] = map_query_to_one_prompt(
                pred_line, self.ref_sop
            )

        # Now, let's get a prompt for each line in the reference SOP
        for ref_ind, ref_line in enumerate(self.ref_sop):
            prompts[f"gold_line_{ref_ind}"] = map_query_to_one_prompt(
                ref_line, self.pred_sop
            )

        return prompts

    def precision(self) -> float:
        """
        Determines the precision of the generated SOP with respect to the gold
        standard SOP.

        In this context, precision is defined as the number of lines in the predicted
        SOP that are entailed in ANY lines in the reference SOP, divided by the total
        number of lines in the predicted SOP.

        Args:
        metrics (List[str]): The metrics dictionary
        pred_sop (List[str]): The generated SOP

        Returns:
        float: The precision of the generated SOP with respect to the gold
            standard SOP
        """
        # Count the number of lines in the predicted SOP that are entailed in ANY
        # lines in the reference SOP
        total_included = 0

        # Iterate over each line in the predicted SOP
        for pred_ind in range(self.pred_sop_len):
            # If the line is included in the reference SOP
            if self.completions[f"pred_line_{pred_ind}"]["index"] != -1:
                total_included += 1

        # Return the precision
        return total_included / self.pred_sop_len

    def recall(self) -> float:
        """
        Determines the recall of the generated SOP with respect to the gold
        standard SOP.

        Args:
        metrics (List[str]): The metrics dictionary
        ref_sop (List[str]): The reference/standard SOP

        Returns:
        float: The recall of the generated SOP with respect to the gold
            standard SOP
        """

        # Count the number of lines in the reference SOP that are entailed in ANY
        # lines in the predicted SOP
        total_included = 0

        # Iterate over each line in the reference SOP
        for ref_ind in range(self.ref_sop_len):
            # If the line is included in the predicted SOP
            if self.completions[f"gold_line_{ref_ind}"]["index"] != -1:
                total_included += 1

        # Return the recall
        return total_included / self.ref_sop_len

    def ordering(self) -> float:
        """
        Determines if the ordering of the predicted SOP is consistent with the ordering
        of the reference SOP.

        Args:
        metrics (List[str]): The metrics dictionary
        pred_sop (List[str]): The generated SOP
        ref_sop (List[str]): The reference/standard SOP
        """
        # Save the index of the last line included in the reference SOP
        last_included = 0

        # Total number of lines in order
        total_in_order = 0

        # Iterate over each line in the predicted SOP
        for pred_ind in range(self.pred_sop_len):
            # If the line is included in the reference SOP
            if self.completions[f"pred_line_{pred_ind}"]["index"] != -1:
                # If the index of the line in the reference SOP is greater than or
                # equal to the index of the last line included
                if self.completions[f"pred_line_{pred_ind}"]["index"] >= last_included:
                    total_in_order += 1
                    last_included = self.completions[f"pred_line_{pred_ind}"]["index"]

        # Return the ordering
        return total_in_order / self.pred_sop_len
