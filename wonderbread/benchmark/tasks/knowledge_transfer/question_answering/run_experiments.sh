#!/bin/bash

# Usage: bash run_experiments.sh "../../../data/demos" "../../../data/qa_dataset.csv"

BASE_DIR=$1
QNA_INPUT_CSV=$2

python generate_responses.py "$BASE_DIR" --path_to_input_csv "$QNA_INPUT_CSV" --model "GeminiPro"
python generate_responses.py "$BASE_DIR" --path_to_input_csv "$QNA_INPUT_CSV" --model "Human"
python generate_responses.py "$BASE_DIR" --path_to_input_csv "$QNA_INPUT_CSV" --model "GPT4"
python generate_responses.py "$BASE_DIR" --path_to_input_csv "$QNA_INPUT_CSV" --model "Claude3"

python eval.py "outputs/qna_Human.csv"
python eval.py "outputs/qna_GeminiPro.csv"
python eval.py "outputs/qna_GPT4.csv"
python eval.py "outputs/qna_Claude3.csv"

python collect_results.py ./outputs/