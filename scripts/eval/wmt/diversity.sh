dir_file=results/wmt/theta0.3_temp1.0/all_answer.jsonl
CUDA_VISIBLE_DEVICES=0 python eval/calculate_div.py --file ${dir_file} --task wmt
