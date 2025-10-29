dir_file=results/novelty/g2_theta0.3_temp1.0
CUDA_VISIBLE_DEVICES=0 python eval/calculate_div.py --file ${dir_file} --task curated
