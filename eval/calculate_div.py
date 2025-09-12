"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""Calculate diversity metrics for wandb runs and accompanying text tables."""
import random
import json
import os
import typing as t
from pprint import pprint

import click
import pandas as pd
import os
import sys
sys.path.append('/mnt/workspace/junyue/CODE/regen/eval')
from diversity import DEFAULT_CONFIGS, calculate_diversity_metrics
import argparse



import nltk
from nltk.translate import bleu_score
from nltk.util import ngrams
#nltk.download('punkt_tab')
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 16236))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
import json
def load_json(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def calculate_self_bleu(sentences, n=2):
    """
    计算给定句子列表的 self-BLEU。
    :param sentences: 句子列表
    :param n: n-gram 的数量，默认为4
    :return: self-BLEU 值
    """
    references = [nltk.word_tokenize(sentence) for sentence in sentences]
    hypotheses = [references[i] for i in range(len(references))]
    bleu_scores = []

    for i in range(len(hypotheses)):
        # 排除自身句子
        ref = [references[j] for j in range(len(references)) if j != i]
        bleu = bleu_score.sentence_bleu(ref, hypotheses[i], weights=(1/n,)*n)
        bleu_scores.append(bleu)
    
    self_bleu = sum(bleu_scores) / len(bleu_scores)
    return self_bleu

# def test_our_method(filename,n=2):
#     file_list = [filename]
#     for file in file_list:
#         print(file)
#         data = load_json(file)
#         scores = []
#         # 确保数据格式正确，将数据转换为需要的格式
#         for i in range(100):
#             responses = []
#             for j in range(5):
#                 responses.append(data[100*j+i]["output"])
#             score = calculate_self_bleu(responses,n=n)
#             scores.append(score)
#         print("n:",n)
#         print(len(scores))
#         avg_score = sum(scores) / len(scores)
#         print(avg_score)
#         return avg_score
    

def test_mt_bench(filename,n=4):
    file_list = [filename]
    qs1 = []
    for file in file_list:
        print(file)
        data = load_json(file)
        qs1_scores = []
        data_num = int(len(data)/5)
        #print(data_num)
        # 确保数据格式正确，将数据转换为需要的格式
        for i in range(data_num):
            responses = []
            for j in range(5):
                responses.append(data[data_num*j+i]["choices"][0]["turns"][0])
            score = calculate_self_bleu(responses,n=n)
            qs1_scores.append(score)
        #print("n:",n)
        #print(len(qs1_scores))
        qs1_avg_score = sum(qs1_scores) / len(qs1_scores)
        #print("qs1 avg scores: {:.2%}".format(qs1_avg_score))
        qs2_scores = []
        # 确保数据格式正确，将数据转换为需要的格式
        for i in range(data_num):
            responses = []
            for j in range(5):
                responses.append(data[data_num*j+i]["choices"][0]["turns"][1])
            score = calculate_self_bleu(responses,n=n)
            qs2_scores.append(score)
        #print("n:",n)
        #print(len(qs2_scores))
        qs2_avg_score = sum(qs2_scores) / len(qs2_scores)
        total_avg_score = (qs1_avg_score+qs2_avg_score)/2

        #print("qs2 avg scores: {:.2%}".format(qs2_avg_score))
        print("all avg scores: {:.2%}".format(1-total_avg_score))
        return qs1_scores

def test_wmt(filename,n=4):
    file_list = [filename]
    qs1 = []
    for file in file_list:
        print(file)
        data = load_json(file)
        qs1_scores = []
        data_num = int(len(data)/5)
        print(data_num)
        # 确保数据格式正确，将数据转换为需要的格式
        for i in range(data_num):
            responses = []
            for j in range(5):
                responses.append(data[data_num*j+i]["predicted_en"])
            score = calculate_self_bleu(responses,n=n)
            qs1_scores.append(score)
        print("n:",n)
        print(len(qs1_scores))
        qs1_avg_score = sum(qs1_scores) / len(qs1_scores)
        print("qs avg scores: {:.2%}".format(1-qs1_avg_score))
        return qs1_scores


def test_xlsum(filename,n=4):
    file_list = [filename]
    qs1 = []
    for file in file_list:
        print(file)
        data = load_json(file)
        qs1_scores = []
        data_num = int(len(data)/5)
        print(data_num)
        # 确保数据格式正确，将数据转换为需要的格式
        for i in range(data_num):
            responses = []
            for j in range(5):
                responses.append(data[data_num*j+i]["predicted_en"])
            score = calculate_self_bleu(responses,n=n)
            qs1_scores.append(score)
        print("n:",n)
        print(len(qs1_scores))
        qs1_avg_score = sum(qs1_scores) / len(qs1_scores)
        print("qs avg scores: {:.2%}".format(1-qs1_avg_score))
        return qs1_scores











def test_alpaca(filename,n=4):
    file_list = [filename]
    for file in file_list:
        print(file)
        data = load_json(file)
        data_num = int(len(data)/5)
        scores = []
        # 确保数据格式正确，将数据转换为需要的格式
        for i in range(data_num):
            responses = []
            for j in range(5):
                responses.append(data[data_num*j+i]["output"])
            score = calculate_self_bleu(responses,n=n)
            scores.append(score)
        print("n:",n)
        print(len(scores))
        avg_score = sum(scores) / len(scores)
        print("qs avg scores: {:.2%}".format(1-avg_score))
        return 1-avg_score




def test_curated(filename,n=4):
    file_list = [filename]
    qs1 = []
    for file in file_list:
        print(file)
        data = load_json(file)
        qs1_scores = []
        data_num = int(len(data)/5)
        print(data_num)
        # 确保数据格式正确，将数据转换为需要的格式
        for i in range(data_num):
            responses = data[i]["generations"]
            score = calculate_self_bleu(responses,n=n)
            qs1_scores.append(score)
        print("n:",n)
        print(len(qs1_scores))
        qs1_avg_score = sum(qs1_scores) / len(qs1_scores)
        print("qs avg scores: {:.2%}".format(1-qs1_avg_score))
        return qs1_scores













def load_json(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data








def evaluate_mt_bench_diversity(filename: str):
    """
    评估MT-Bench数据的回答多样性
    较高的分数表示更高的多样性
    """
    #print(f"Evaluating diversity for {filename}")

    data = load_json(filename)
    data_num = int(len(data)/5)
    
    
    # 收集所有回答
    responsess = []
    #qs2_all_responses = []
    
    # 收集所有问题的回答
    for i in range(data_num):
        qs1_responses = []
        qs2_responses = []
        for j in range(5):
            qs1_responses.append(data[data_num*j+i]["choices"][0]["turns"][0])
            qs2_responses.append(data[data_num*j+i]["choices"][0]["turns"][1])
        
        # 收集用于批量COMET计算的回答
        responsess.append(qs1_responses)
        responsess.append(qs2_responses)
    
    diversity_metrics_config = DEFAULT_CONFIGS.copy()
    #ead_averaged_distinct_ngrams,nli_sample_from_sim,sent_bert_from_sim
    diversity_metrics = "ead_averaged_distinct_ngrams,sent_bert_from_sim"
    if diversity_metrics != "all":
        diversity_metrics_config = {
            k: v for k, v in diversity_metrics_config.items() if k in diversity_metrics
        }
    print("diversity_metrics_config")
    print(diversity_metrics_config)
    run_results = calculate_diversity_metrics(
        responsess, diversity_metrics_config
    )
    #rint("Outputs for", log_prefix)
    print("mt-bench eval:",filename)
    print(run_results)

    return run_results

def evaluate_alpaca_diversity(filename: str):
    """
    评估MT-Bench数据的回答多样性
    较高的分数表示更高的多样性
    """
    #print(f"Evaluating diversity for {filename}")

    data = load_json(filename)
    len_data = int(len(data)/5)
    responsess = []
    # 确保数据格式正确，将数据转换为需要的格式
    for i in range(len_data):
        responses = []
        for j in range(5):
            responses.append(data[len_data*j+i]["output"])
        responsess.append(responses)


    diversity_metrics_config = DEFAULT_CONFIGS.copy()
    #ead_averaged_distinct_ngrams,nli_sample_from_sim,sent_bert_from_sim
    diversity_metrics = "ead_averaged_distinct_ngrams,sent_bert_from_sim"
    if diversity_metrics != "all":
        diversity_metrics_config = {
            k: v for k, v in diversity_metrics_config.items() if k in diversity_metrics
        }
    print("diversity_metrics_config")

    print(diversity_metrics_config)

    run_results = calculate_diversity_metrics(
        responsess, diversity_metrics_config
    )
    print("alpaca eval:",filename)
    print(run_results)

    return run_results


def evaluate_curated_diversity(filename: str):
    """
    评估MT-Bench数据的回答多样性
    较高的分数表示更高的多样性
    """
    #print(f"Evaluating diversity for {filename}")

    data = load_json(filename)
    len_data = int(len(data)/5)
    responsess = []
    # 确保数据格式正确，将数据转换为需要的格式
    for i in range(len_data):
        responses = data[i]["generations"]
        
        responsess.append(responses)


    diversity_metrics_config = DEFAULT_CONFIGS.copy()
    #ead_averaged_distinct_ngrams,nli_sample_from_sim,sent_bert_from_sim
    diversity_metrics = "ead_averaged_distinct_ngrams,sent_bert_from_sim"
    if diversity_metrics != "all":
        diversity_metrics_config = {
            k: v for k, v in diversity_metrics_config.items() if k in diversity_metrics
        }
    print("diversity_metrics_config")

    print(diversity_metrics_config)

    run_results = calculate_diversity_metrics(
        responsess, diversity_metrics_config
    )
    print("curated eval:",filename)
    print(run_results)

    return run_results






def evaluate_wmt_diversity(filename: str):
    """
    评估MT-Bench数据的回答多样性
    较高的分数表示更高的多样性
    """
    #print(f"Evaluating diversity for {filename}")

    data = load_json(filename)
    data_num = int(len(data)/5)
    
    
    # 收集所有回答
    responsess = []
    #qs2_all_responses = []
    
    # 收集所有问题的回答
    for i in range(data_num):
        qs1_responses = []
        for j in range(5):
            qs1_responses.append(data[data_num*j+i]["predicted_en"])
        
        # 收集用于批量COMET计算的回答
        responsess.append(qs1_responses)

    
    diversity_metrics_config = DEFAULT_CONFIGS.copy()
    #ead_averaged_distinct_ngrams,nli_sample_from_sim,sent_bert_from_sim
    diversity_metrics = "ead_averaged_distinct_ngrams,sent_bert_from_sim"
    if diversity_metrics != "all":
        diversity_metrics_config = {
            k: v for k, v in diversity_metrics_config.items() if k in diversity_metrics
        }
    print("diversity_metrics_config")
    print(diversity_metrics_config)
    run_results = calculate_diversity_metrics(
        responsess, diversity_metrics_config
    )
    #rint("Outputs for", log_prefix)
    print("wmt eval:",filename)
    print(run_results)

    return run_results


def evaluate_xlsum_diversity(filename: str):
    """
    评估MT-Bench数据的回答多样性
    较高的分数表示更高的多样性
    """
    #print(f"Evaluating diversity for {filename}")

    data = load_json(filename)
    data_num = int(len(data)/5)
    
    
    # 收集所有回答
    responsess = []
    #qs2_all_responses = []
    
    # 收集所有问题的回答
    for i in range(data_num):
        qs1_responses = []
        for j in range(5):
            qs1_responses.append(data[data_num*j+i]["predicted_en"])
        
        # 收集用于批量COMET计算的回答
        responsess.append(qs1_responses)

    
    diversity_metrics_config = DEFAULT_CONFIGS.copy()
    #ead_averaged_distinct_ngrams,nli_sample_from_sim,sent_bert_from_sim
    diversity_metrics = "ead_averaged_distinct_ngrams,sent_bert_from_sim"
    if diversity_metrics != "all":
        diversity_metrics_config = {
            k: v for k, v in diversity_metrics_config.items() if k in diversity_metrics
        }
    print("diversity_metrics_config")
    print(diversity_metrics_config)
    run_results = calculate_diversity_metrics(
        responsess, diversity_metrics_config
    )
    #rint("Outputs for", log_prefix)
    print("xlsum eval:",filename)
    print(run_results)

    return run_results


def evaluate_gsm8k_diversity(filename: str):
    """
    评估MT-Bench数据的回答多样性
    较高的分数表示更高的多样性
    """
    #print(f"Evaluating diversity for {filename}")

    data = load_json(filename)
    data_num = int(len(data)/5)
    
    
    # 收集所有回答
    responsess = []
    #qs2_all_responses = []
    
    # 收集所有问题的回答
    for i in range(data_num):
        qs1_responses = []
        for j in range(5):
            qs1_responses.append(data[data_num*j+i]["prediction"])
        
        # 收集用于批量COMET计算的回答
        responsess.append(qs1_responses)

    
    diversity_metrics_config = DEFAULT_CONFIGS.copy()
    #ead_averaged_distinct_ngrams,nli_sample_from_sim,sent_bert_from_sim
    diversity_metrics = "ead_averaged_distinct_ngrams,sent_bert_from_sim"
    if diversity_metrics != "all":
        diversity_metrics_config = {
            k: v for k, v in diversity_metrics_config.items() if k in diversity_metrics
        }
    print("diversity_metrics_config")
    print(diversity_metrics_config)
    run_results = calculate_diversity_metrics(
        responsess, diversity_metrics_config
    )
    #rint("Outputs for", log_prefix)
    print("gsm8k eval:",filename)
    print(run_results)

    return run_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate response diversity using ROUGE and COMET metrics')
    parser.add_argument('--file', type=str,
                      help='Path to input jsonl file(s). Multiple files can be specified for batch processing')
    parser.add_argument('--task', type=str,
                      help='Path to input jsonl file(s). Multiple files can be specified for batch processing')
    args = parser.parse_args()
    #filename = f"/mnt/workspace/junyue/CODE/regen/results/curated/baseline/regen_cos_avg/regen-one-word-avg-1/threthold0.1_bottom0.1_theta0.5/generations.jsonl"
    filename = args.file
    
    if args.task == "curated":
        try:
            metrics = evaluate_curated_diversity(filename)
            metrics = test_curated(filename,n=4)
        except FileNotFoundError as e:
            print(f"File not found: {filename}")
    elif args.task == "alpaca":
        try:
            metrics = evaluate_alpaca_diversity(filename)
            metrics = test_alpaca(filename,n=4)
        except FileNotFoundError as e:
            print(f"File not found: {filename}")
    elif args.task == "mt_bench":
        try:
            metrics = evaluate_mt_bench_diversity(filename)
            metrics = test_mt_bench(filename,n=4)
        except FileNotFoundError as e:
            print(f"File not found: {filename}")
    elif args.task == "wmt":
        try:
            metrics = evaluate_wmt_diversity(filename)
            metrics = test_wmt(filename,n=4)
        except FileNotFoundError as e:
            print(f"File not found: {filename}")
    elif args.task == "xlsum":
        try:
            metrics = evaluate_xlsum_diversity(filename)
            metrics = test_xlsum(filename,n=4)
        except FileNotFoundError as e:
            print(f"File not found: {filename}")
    else:
        raise  FileNotFoundError
    

