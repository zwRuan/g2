from vllm import LLM, SamplingParams
from datasets import load_dataset
import os
import argparse
import json
import transformers 

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
#python src/inference.py --mode vllm --model meta-llama/Meta-Llama-3-8B-Instruct --data curated --eval-dir results/curated/Llama3-8B --num-generations 10
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model to run inference with")
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
        "--sampling",
        choices=["regenerate", "in-context", "paraphrase", "system-prompt"],
        default="regenerate",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=10,
        help="Number of generations per prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="generation temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="top",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=None,
        help="top",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="top",
    )
    args = parser.parse_args()
    max_token_length = 512
    if args.data == "wildchat":
        max_token_length = 1024
    dataset = load_dataset("yimingzhang/novelty-bench", split=args.data)
    eval_dir = (
        args.eval_dir if args.eval_dir else os.path.join(f"{args.data}-evals", args.model)
    )
    os.makedirs(eval_dir, exist_ok=True)
    output_file = os.path.join(eval_dir, "generations.jsonl")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tensor_parallel_size=1, trust_remote_code=True)
    print("min_p:",args.min_p)
    #stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    #sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,max_tokens=max_token_length,n=args.num_generations) 
    questions = [get_templated_prompt(item["prompt"], tokenizer) for item in dataset]

    # First generation with greedy decoding (temperature=0)
    greedy_params = SamplingParams(temperature=0, max_tokens=max_token_length, n=1)
    greedy_outputs = llm.generate(questions, greedy_params)
    # Remaining n-1 generations with sampling
    if args.num_generations > 1:
        sampling_params = SamplingParams(
            temperature=args.temperature, 
            top_p=args.top_p, 
            top_k=args.top_k,
            min_p=args.min_p, 
            max_tokens=max_token_length, 
            n=args.num_generations - 1
        )
        sampled_outputs = llm.generate(questions, sampling_params)


    with open(output_file, "w") as f:
        for i, example in enumerate(dataset):
            generations = []
            
            # Add the greedy generation
            generations.append(greedy_outputs[i].outputs[0].text)
            
            # Add the sampled generations
            if args.num_generations > 1:
                for completion in sampled_outputs[i].outputs:
                    generations.append(completion.text)
            
            result = {
                "id": example["id"],
                "prompt": example["prompt"],
                "model": args.model,
                "generations": generations,
            }
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    main()
