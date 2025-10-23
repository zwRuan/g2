import argparse
import os
import re
import json
import random
import evaluate
from datasets import load_dataset, Dataset
from eval.utils import (
    generate_completions,
    load_lm_and_tokenizer,
    load_dexperts_model_and_tokenizer,
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


def get_input_encoding(
    questions: list[str],
    generation_model: transformers.LlamaForCausalLM,
    generation_tokenizer: transformers.PreTrainedTokenizerFast,
) -> transformers.BatchEncoding:
    input_encoding = generation_tokenizer(
        questions, padding=True, add_special_tokens=False, return_tensors="pt"
    ).to(generation_model.device)
    return input_encoding


def calculate_sentence_bleu(reference, hypothesis):
    """BLEU"""
    reference = [word_tokenize(reference.lower())]
    hypothesis = word_tokenize(hypothesis.lower())
    weights = (0.25, 0.25, 0.25, 0.25)  # BLEU-4
    return sentence_bleu(reference, hypothesis, weights=weights)

def calculate_rouge(reference, hypothesis):
    """ROUGE"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rougeL'].fmeasure

def calculate_comet_score(model, source, reference, hypothesis):
    """COMET"""
    data = {"src": source, "ref": reference, "mt": hypothesis}
    predictions = model.predict([data], batch_size=1, gpus=1 if torch.cuda.is_available() else 0)
    return predictions.scores[0]
def calculate_batch_comet_score(comet_model, sources, references, hypotheses):
    """COMET"""
    data = [
        {"src": src, "ref": ref, "mt": hyp} 
        for src, ref, hyp in zip(sources, references, hypotheses)
    ]
    predictions = comet_model.predict(data, batch_size=320, gpus=1 if torch.cuda.is_available() else 0)
    return np.mean(predictions.scores)



def get_summary_prompt(text):
    prompt = f"""Please generate a concise summary of the following text. Focus on the key points and main ideas. Provide only the summary without any additional explanation.

Text: {text}
Summary:"""
    return prompt

def main(args):
    random.seed(42)
    prefix_outputs = []
    print("Loading data...")
    # Load WMT14 dataset
    print("Loading xlsum dataset...")
    dataset = load_dataset("csebuetnlp/xlsum", "english")
    test_data = dataset["test"].select(range(1000))  
    

    if args.max_examples and len(test_data) > args.max_examples:
        test_data = random.sample(test_data, args.max_examples)

    ensure_dir(args.save_dir)
    all_predictions = {}
    for num_predictions in range(args.iter_num):
        all_predictions[num_predictions] = None
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
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
                if i == 0:
                    prompt = get_summary_prompt(example['text'])
                    prompt = get_templated_prompt(prompt, tokenizer)
                else:
                    if args.pos_or_neg == "base":
                        prompt = get_summary_prompt(example['text'])
                        prompt = get_templated_prompt(prompt, tokenizer)
                    else:
                        raise ValueError("pos_or_neg must be specified")
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
                    top_k=args.top_k,
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
                    top_k=args.top_k,
                    top_p=args.top_p,
                    min_p=args.min_p,
                    do_sample=True,
                )
                      

            if len(prefix_outputs) ==  0:
                prefix_outputs = [f"Output {i}: " + outputs[index] + "\n" for index in range(len(outputs))]
            else:
                assert len(prefix_outputs) == len(outputs), "prefix_outputs and outputs must have the same length"
                prefix_outputs = [prefix_outputs[index] + f"Output {i}: " + outputs[index] + "\n" for index in range(len(outputs))]
            
            
            predictions = []
            individual_bleu_scores = []
            individual_rouge_scores = []


            sources = [example['text'] for example in test_data]
            references = [example['summary'] for example in test_data]

            for example, output in zip(test_data, outputs):
                sentence_bleu_score = calculate_sentence_bleu(example['summary'], output)
                individual_bleu_scores.append(sentence_bleu_score)

                sentence_rouge_score = calculate_rouge(example['summary'], output)
                individual_rouge_scores.append(sentence_rouge_score)

                predictions.append({
                    "source_de": example['text'],
                    "reference_en": example['summary'],
                    "predicted_en": output,
                    "sentence_bleu": sentence_bleu_score,
                    "sentence_rouge": sentence_rouge_score,
                })


            print(f"Calculating COMET scores for iteration {i}...")
            batch_comet_score = calculate_batch_comet_score(comet_model, sources, references, outputs)
            print(f"Iteration {i} COMET score: {batch_comet_score:.4f}")

            print("Individual BLEU scores:", np.mean(individual_bleu_scores))
            print("Individual ROUGE-1 scores:", np.mean(individual_rouge_scores))
            print("Batch COMET score:", batch_comet_score)

            all_predictions[i] = {
                "AVG_BLEU": np.mean(individual_bleu_scores),
                "AVG_ROUGEL": np.mean(individual_rouge_scores),
                "AVG_COMET": batch_comet_score
            }

            with open(os.path.join(args.save_dir, f"predictions_{2*j+i}.jsonl"), "w") as fout:
                for prediction in predictions:
                    fout.write(json.dumps(prediction) + "\n")


    with open(os.path.join(args.save_dir, f"all_metrics.json"), "w") as fout:
        json.dump(all_predictions, fout, indent=4)

    with open(os.path.join(args.save_dir, f"all_answer.jsonl"), "w") as outfile:
        for iter in range(5):
            filename = os.path.join(args.save_dir, f'predictions_{iter}.jsonl')
            if os.path.exists(filename):
                with open(filename, 'r') as infile:
                    outfile.write(infile.read())
            













if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/gsm"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/alpaca_farm")
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="The number of instances to evaluate. If not given, we will evaluate all instances."
    )
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
        default=2048,
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
        "--base_model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-13b-hf',
    )
    parser.add_argument(
        "--expert_model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-7b-chat-hf',
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--method",
        type=float,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--weight_method",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--use_threshold",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--first_n_tokens",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--iter_num",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--pos_or_neg",
        type=str,
        default=None
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
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
        default=None,
        help="The model revision to load.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="The model revision to load.",
    )
    args = parser.parse_args()

    main(args)