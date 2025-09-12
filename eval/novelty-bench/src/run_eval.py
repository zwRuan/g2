from vllm import LLM, SamplingParams

import os
import argparse
import json
import argparse
import re
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
    load_analysis_dexperts_model_and_tokenizer,
    dynamic_import_function,
    ensure_dir,
    dexperts_generate_completions
)
import transformers 
from fraction import Fraction
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

def select_diverse_samples(embeddings, texts, k=3):
    if len(embeddings) <= k:
        return list(range(len(embeddings)))
    

    similarity = cosine_similarity(embeddings)
    

    selected = [0]
    
    while len(selected) < k:
        unselected = list(set(range(len(embeddings))) - set(selected))
        max_similarities = np.max([similarity[s, unselected] for s in selected], axis=0)
        
        next_idx = unselected[np.argmin(max_similarities)]
        selected.append(next_idx)
    
    return selected

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

def get_compress_prompt(question,response):
    compress_prompt =f'''
Compress the following response into a single sentence that captures the core problem-solving approach. Focus only on the essential methodology and conceptual framework, not implementation details.

Original question: {question}

Original response:
{response}

Provide a one-sentence summary (maximum 30 words) that describes the solution methodology:'''
    return compress_prompt


def get_positive_observer_prompt(question, compressed_responses_text):
    """
    Creates a prompt for the positive observer that encourages diversity.
    
    Args:
        question: The original question
        compressed_responses_text: String containing multiple compressed responses,
                                  formatted as "Output N: [response]\n"
    
    Returns:
        A prompt encouraging generation of responses different from all previous ones
    """
    positive_prompt = f'''
{question}

Previous responses to this question used the following approaches:
{compressed_responses_text}

Generate a NEW solution that uses different methods, concepts, or approaches from ALL previous solutions.
'''
    return positive_prompt

#for qwen2.5,llama3
def get_negative_observer_prompt(instruction, original_output):
    """
    Creates a universal prompt that maintains high logits for all tokens in previous examples,
    regardless of content type (numbers, names, text, etc.) with no pattern-specific processing.
    
    Args:
        instruction: The original question/instruction
        original_output: The previously generated output(s) that should be reproduced
    """
    # Clean the output and format for presentation
    clean_output = original_output.strip()
    
    near_prompt_template = f'''
SYSTEM: You are operating in EXACT REPLICATION mode. You must reproduce outputs IDENTICAL to the reference examples.

QUERY: {instruction}

REFERENCE EXAMPLES (ALL EQUALLY VALID):
{clean_output}

CRITICAL INSTRUCTIONS:
1. The examples above are the ONLY CORRECT responses
2. ALL examples are EQUALLY VALID and must have EQUAL PROBABILITY of being chosen
3. Your task is to PRECISELY REPLICATE one of these examples - exact words, numbers, and patterns
4. This is a MEMORY TEST - you must use the EXACT SAME values from the examples
5. Deviation from these examples is a CRITICAL ERROR
6. When continuing any pattern, you MUST use a value ALREADY SEEN in the examples
7. Every token in your response must match tokens from the reference examples

Memory test beginning. Reproduce one of the examples with perfect accuracy:

'''
    return near_prompt_template
#python src/inference.py --mode vllm --model meta-llama/Meta-Llama-3-8B-Instruct --data curated --eval-dir results/curated/Llama3-8B --num-generations 10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True, help="Model to run inference with")
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
        "--iter_num",
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
        "--top_k",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--bottom",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--weight_method",
        type=str,
        default="entropy",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    
    args = parser.parse_args()
    random.seed(42)
    max_new_tokens=512
    if args.data == "wildchat":
        max_new_tokens=1024
    prefix_outputs = []
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
    for i in range(args.iter_num):
        if i == 0:
            print("Loading model and tokenizer...")
            model, tokenizer = load_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path,
                tokenizer_name_or_path=args.model_name_or_path,
                load_in_8bit=args.load_in_8bit,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )
        elif i == 1:

            model, tokenizer = load_dexperts_model_and_tokenizer(
                model_name_or_path=args.model_name_or_path,
                threshold=args.threshold,
                alpha=args.alpha,
                bottom=args.bottom,
                chat_response_prefix="Answer:",
                load_in_8bit=False,
                use_fast_tokenizer=not False,
            )
        else:
            print(f"正在进行第{i}次迭代, 无需重新加载模型")
            print(f"模型: {model.__class__.__name__}")
            print(f"tokenizer: {tokenizer.__class__.__name__}")

        prompts = []
        # if i != 0:
        pos_prompts = []
        neg_prompts = []
        for num_example, example in enumerate(test_data):
            prompt = example['prompt']
            prompt = get_templated_prompt(prompt, tokenizer)
            if i != 0:
                pos_prompt = get_positive_observer_prompt(example['prompt'],prefix_outputs[num_example])
                neg_prompt = get_negative_observer_prompt(example['prompt'],prefix_outputs[num_example])
                pos_prompt = get_templated_prompt(pos_prompt, tokenizer)
                neg_prompt = get_templated_prompt(neg_prompt, tokenizer)
            prompts.append(prompt)
            if i != 0:
                pos_prompts.append(pos_prompt)
                neg_prompts.append(neg_prompt)
        if i == 0:
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                batch_size=args.eval_batch_size,
                pad_token_id = tokenizer.eos_token_id,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=False,
            )
        else:
            outputs,avg_critic_tokens_ratios = dexperts_generate_completions(
                model=model,
                tokenizer=tokenizer,
                base_prompts=prompts,
                pos_prompts=pos_prompts,
                neg_prompts=neg_prompts,
                theta=args.theta,
                weight_method=args.weight_method,
                first_n_tokens=4096,
                max_new_tokens=max_new_tokens,
                batch_size=args.eval_batch_size,
                pad_token_id = tokenizer.eos_token_id,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=True,
            )

        ###插入代码的位置
        # Process each output
        for idx, output in enumerate(tqdm(outputs, desc=f"Compressing outputs (iteration {i})")):
            # For the first iteration or if we need to refresh compressions
            # Create compression prompt
            question = test_data[idx]['prompt']
            #template = f'After thinking step by step, summry this sentence: {question}: '
            template = 'This_sentence_:_"A_jockey_riding_a_horse."_means_in_one_word:"Equestrian".This_sentence_:_"*sent_0*"_means_in_one_word:"'
            template.replace('*sent_0*', output).replace('_', ' ')
            #compress_prompt_text = get_compress_prompt(question, output)
            compress_prompt_text = get_templated_prompt(template, tokenizer)
            
            # Get the compressed description
            with torch.no_grad():
                # Tokenize the compression prompt
                inputs = tokenizer(compress_prompt_text, return_tensors="pt").to(model.device)
                
                # Generate the compressed description
                if i == 0:
                    compression_outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,  # Short summary
                        temperature=0.2,    # Low temperature for more deterministic output
                        do_sample=False,
                        output_hidden_states=True,
                        return_dict_in_generate=True
                    )
                else:
                    compression_outputs = model.model.generate(
                        **inputs,
                        max_new_tokens=50,  # Short summary
                        temperature=0.2,    # Low temperature for more deterministic output
                        do_sample=False,
                        output_hidden_states=True,
                        return_dict_in_generate=True
                    )
                # Extract the generated text
                compression_tokens = compression_outputs.sequences[0][inputs['input_ids'].shape[1]:]
                compressed_text = tokenizer.decode(compression_tokens, skip_special_tokens=True).strip()
                num_generated_tokens = len(compression_tokens)
                    
                if num_generated_tokens > 0:  # 确保生成了至少一个token
                    # 获取最后一层的hidden states
                    # 注意：hidden_states的形状通常为[层数，batch，序列长度，隐藏维度]
                    # 我们需要所有生成token的最后一层表示
                    
                    # 创建一个列表，存储每个生成步骤的hidden states
                    all_hidden_states = []
                    
                    # 获取每个生成步骤的hidden states（针对生成的token）
                    for step_idx in range(num_generated_tokens):
                        # 获取当前步骤的最后一层hidden state
                        # 注意索引：hidden_states[step_idx][-1][0][-1]表示
                        # 第step_idx步生成的token，最后一层(-1)，第一个样本(0)，最后一个token位置(-1)

                        current_hidden = compression_outputs.hidden_states[step_idx][-1][-1][-1].detach().cpu().numpy()

                        all_hidden_states.append(current_hidden)
                    
                    # 将列表转换为numpy数组并计算平均值
                    if all_hidden_states:
                        all_hidden_array = np.array(all_hidden_states)
                        mean_hidden = np.mean(all_hidden_states, axis=0)
                        
                        # 归一化平均嵌入向量
                        normalized_hidden = mean_hidden / np.linalg.norm(mean_hidden)
                    else:
                        # 防御性编程：如果没有有效的隐藏状态（极少情况），使用零向量
                        normalized_hidden = np.zeros(compression_outputs.hidden_states[0][-1][0][0].shape[0])
                else:
                    # 如果没有生成任何token，使用输出的最后一个token的表示
                    normalized_hidden = compression_outputs.hidden_states[-1][-1][-1][-1].detach().cpu().numpy()
                    normalized_hidden = normalized_hidden / np.linalg.norm(normalized_hidden)
            if i == 0:
                embedss.append([normalized_hidden])
                outputss.append([output])
            else: 
                embedss[idx].append(normalized_hidden)
                outputss[idx].append(output)

        if i >= 3:
            representative_responsess = []
            for embeds, resps in zip(embedss, outputss):  
                arr_embeds = np.array(embeds)
                n_samples = len(arr_embeds) 
                # 使用此函数替代聚类
                diverse_indices = select_diverse_samples(arr_embeds, resps, k=min(3, n_samples))
                print("Original diverse indices:", diverse_indices)
        
                # 随机打乱 diverse_indices 的顺序
                shuffled_diverse_indices = diverse_indices.copy()
                random.shuffle(shuffled_diverse_indices)
                print("Shuffled diverse indices:", shuffled_diverse_indices)
                representative_responses = [resps[idx] for idx in shuffled_diverse_indices]
                representative_responsess.append(representative_responses)
        if i < 3:
            if len(prefix_outputs) ==  0:
                prefix_outputs = [f"Output {i}: " + outputs[index] + "\n" for index in range(len(outputs))]
            else:
                assert len(prefix_outputs) == len(outputs), "prefix_outputs and outputs must have the same length"
                prefix_outputs = [prefix_outputs[index] + f"Output {i}: " + outputs[index] + "\n" for index in range(len(outputs))]
        else:
            assert len(prefix_outputs) == len(representative_responsess)
            for data_index, representative_responses in enumerate(representative_responsess):
                for pre_index,representative_response in enumerate(representative_responses):
                    if pre_index == 0:
                        prefix_outputs[data_index] = f"Output {pre_index}: " + representative_response + "\n"
                    else:
                        prefix_outputs[data_index] = prefix_outputs[data_index] + f"Output {pre_index}: " + representative_response + "\n"
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
            print("avg_critic_tokens_ratios:",sum(avg_critic_tokens_ratios) / len(avg_critic_tokens_ratios))
            avg_critic_tokens_ratioss.append(sum(avg_critic_tokens_ratios) / len(avg_critic_tokens_ratios))
    print("####10-item-avg_critic_tokens_ratioss########:",sum(avg_critic_tokens_ratioss) / len(avg_critic_tokens_ratioss))
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
            


if __name__ == "__main__":
    main()