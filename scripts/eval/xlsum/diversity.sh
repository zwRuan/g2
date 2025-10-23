source activate eval_g2

dir_file=results/xlsum/theta0.3_temp1.0
CUDA_VISIBLE_DEVICES=0 python eval/calculate_div.py --file ${dir_file} --task xlsum
