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
    load_analysis_dexperts_model_and_tokenizer,
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
from tqdm import tqdm
import transformers 
from fraction import Fraction
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

def select_diverse_samples(embeddings, texts, k=3):
    """选择k个最具差异性的样本"""
    if len(embeddings) <= k:
        return list(range(len(embeddings)))
    
    # 计算余弦相似度矩阵
    similarity = cosine_similarity(embeddings)
    
    # 选择第一个样本(可以是任意样本)
    selected = [0]
    
    # 贪心选择剩余的样本
    while len(selected) < k:
        # 计算未选择样本与已选样本的最大相似度
        unselected = list(set(range(len(embeddings))) - set(selected))
        max_similarities = np.max([similarity[s, unselected] for s in selected], axis=0)
        
        # 选择与已选样本最不相似的样本
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
    """计算单个句子的BLEU分数"""
    reference = [word_tokenize(reference.lower())]
    hypothesis = word_tokenize(hypothesis.lower())
    weights = (0.25, 0.25, 0.25, 0.25)  # BLEU-4
    return sentence_bleu(reference, hypothesis, weights=weights)

def calculate_rouge(reference, hypothesis):
    """计算单个句子的ROUGE分数"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rougeL'].fmeasure

def calculate_comet_score(model, source, reference, hypothesis):
    """计算单个句子的COMET分数"""
    data = {"src": source, "ref": reference, "mt": hypothesis}
    predictions = model.predict([data], batch_size=1, gpus=1 if torch.cuda.is_available() else 0)
    return predictions.scores[0]


def calculate_batch_comet_score(comet_model, sources, references, hypotheses):
    """批量计算COMET分数"""
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

def pos_summary_prompt_v1(text, previous_summaries):
    prompt = f"""Generate a concise summary of the following text. Create a new summary that is significantly different from the previous summaries while maintaining accuracy and capturing the key information.

Text: {text}

Previous summaries:
{previous_summaries}

Please provide a new, alternative summary that:
1. Uses different wording and structure than the above summaries
2. Maintains the same key information and meaning
3. Focuses on different aspects or perspectives of the text
4. Remains accurate and coherent

Summary:"""
    return prompt


def neg_summary_prompt(text, previous_list):
    prompt = f"""Generate a summary of the following text. Create a new summary that is very similar to the previous summaries, maintaining the same word choices and structure whenever possible.

Text: {text}

Previous summaries:
{previous_list}

Instructions:
- Study the patterns and word choices in the previous summaries carefully
- Use the same vocabulary and phrasing as much as possible
- Keep the structure and organization highly similar to previous summaries
- Focus on the same key points that were emphasized in previous summaries
- Only make minimal necessary adjustments for clarity and coherence
- Maintain the same level of detail and length as previous summaries

Please provide a new summary that closely aligns with the previous versions:"""
    return prompt

def main(args):
    random.seed(42)
    prefix_outputs = []
    print("Loading data...")
    # Load WMT14 dataset
    print("Loading xlsum dataset...")
    dataset = load_dataset("csebuetnlp/xlsum", "english")
    test_data = dataset["test"].select(range(1000))  # 只取前1000条数据

    

    if args.max_examples and len(test_data) > args.max_examples:
        test_data = random.sample(test_data, args.max_examples)

    ensure_dir(args.save_dir)
    all_predictions = {}
    for num_predictions in range(args.iter_num):
        all_predictions[num_predictions] = None
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
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

                prompt = get_summary_prompt(example['text'])
                prompt = get_templated_prompt(prompt, tokenizer)
                if i!=0:
                    for index in range(1):
                        pos_prompt = pos_summary_prompt_v1(example['text'],prefix_outputs[num_example])
                        neg_prompt = neg_summary_prompt(example['text'],prefix_outputs[num_example])
                    pos_prompt = get_templated_prompt(pos_prompt, tokenizer)
                    neg_prompt = get_templated_prompt(neg_prompt, tokenizer)

                prompts.append(prompt)
                if i != 0:
                    pos_prompts.append(pos_prompt)
                    neg_prompts.append(neg_prompt)


            # with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
            #     fout.write(prompts[0][0]['content'])
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
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.eval_batch_size,
                    pad_token_id = tokenizer.eos_token_id,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=True,
                )
                print("avg_critic_tokens_ratios:",sum(avg_critic_tokens_ratios) / len(avg_critic_tokens_ratios))
            ###插入代码的位置
            # Process each output
            for idx, output in enumerate(tqdm(outputs, desc=f"Compressing outputs (iteration {i})")):
                # For the first iteration or if we need to refresh compressions
                # Create compression prompt
            
                template = 'This_sentence_:_"A_jockey_riding_a_horse."_means_in_one_word:"Equestrian".This_sentence_:_"*sent_0*"_means_in_one_word:"'
                template.replace('*sent_0*', output.strip()).replace('_', ' ')
                if idx==0:
                    print(output)
                    print(output[0])
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
            predictions = []
            individual_bleu_scores = []
            individual_rouge_scores = []

            # 收集所有源文本、参考文本和预测文本用于批量COMET评估
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

            # 批量计算COMET分数
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

    # 保存最终结果，包含每次迭代的平均分数
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
        default=0.5,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--bottom",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--weight_method",
        type=str,
        default="entropy",
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
        default=4096,
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
        "--top_k",
        type=int,
        default=50,
        help="The model revision to load.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="The model revision to load.",
    )
    args = parser.parse_args()

    main(args)