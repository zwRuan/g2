import argparse
import os
import re
import json
import random
import evaluate
from datasets import load_dataset, Dataset
import sys
# Add your path here sys.path.append("/your_path/g2")
sys.path.append("/workspace/CODE/g2")
from eval.utils import (
    generate_completions,
    load_lm_and_tokenizer,
    load_dexperts_model_and_tokenizer,
    EDT_generate_completions,
    load_EDT,
    dynamic_import_function,
    ensure_dir,
    dexperts_generate_completions
)
import transformers 
from fraction import Fraction
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from comet import download_model, load_from_checkpoint
import torch
import numpy as np


def get_templated_prompt(
    prompt: str,
    generation_tokenizer: transformers.PreTrainedTokenizerFast,
    system_prompt: str=None,
) -> str:
    if system_prompt is None:
        conversation = [
            {"role": "user", "content": prompt},
        ]
        templated_prompt: str = generation_tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
    else:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        templated_prompt: str = generation_tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

    return templated_prompt


def main(args):
    test_data = load_dataset("yimingzhang/novelty-bench", split=args.data)
    eval_dir = (
        args.eval_dir if args.eval_dir else os.path.join(f"{args.data}-evals", args.model)
    )
    os.makedirs(eval_dir, exist_ok=True)
    output_file = os.path.join(eval_dir, "generations.jsonl")
    all_predictions = {}
    for num_predictions in range(args.iter_num):
        all_predictions[num_predictions] = None
    results = []
    embedss = []
    avg_critic_tokens_ratioss = []
    outputss = []
    for j in range(1):
        for i in range(args.iter_num):
            if i == 0:
                print("Loading model and tokenizer...")
                model, tokenizer = load_lm_and_tokenizer(
                    model_name_or_path=args.model_name_or_path,
                    tokenizer_name_or_path=args.tokenizer_name_or_path,
                    load_in_8bit=args.load_in_8bit,
                    use_fast_tokenizer=not args.use_slow_tokenizer,
                )
            else:
                print(f"Iteration {i} is in progress, no need to reload the model")
                print(f"model: {model.__class__.__name__}")
                print(f"tokenizer: {tokenizer.__class__.__name__}")

            prompts = []

            for num_example, example in enumerate(test_data):
                prompt = example['prompt']
                prompt = get_templated_prompt(prompt, tokenizer)
                prompts.append(prompt)

            if i == 0:
                outputs = generate_completions(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.eval_batch_size,
                    pad_token_id = tokenizer.eos_token_id,
                    temperature=args.temperature,
                    do_sample=False,
                )
            else:
                outputs = generate_completions(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.eval_batch_size,
                    pad_token_id = tokenizer.eos_token_id,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    do_sample=True,
                )
                      
            if i == 0:
                for example, output in zip(test_data, outputs):
                    results.append({
                        "id": example["id"],
                        "prompt": example["prompt"],
                        "model": args.model_name_or_path,
                        "generations": [output],
                    })  
            else:
                assert len(results) == len(test_data) , "results and test_data must have the same length"
                for index, output in enumerate(outputs):
                    results[index]["generations"].append(output)
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
            













if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--eval-dir", help="Directory to save evaluation results", required=True
    )
    parser.add_argument(
        "--data",
        default="curated",
        choices=["curated", "wildchat"],
        help="Source of prompts",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--iter_num",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="The model revision to load.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="The model revision to load.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1,
        help="The model revision to load.",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=0.0,
        help="The model revision to load.",
    )
    args = parser.parse_args()

    main(args)