# this scripts use vllm to run_sample
# if you want use transformers, please change "eval/novelty-bench/src/inference.py" to "eval/novelty-bench/src/run_sample.py"

source activate g2
top_p=1.0
top_k=50
min_p=0
iter_num=10
temperature=1

outputfile=results/novelty/g2_theta${theta}_temp${temperature}
model=meta-llama/Meta-Llama-3-8B-Instruct

python eval/novelty-bench/src/inference.py --model $model --data curated --eval-dir $outputfile --num-generations 10 --temperature $temperature --top_p $top_p --top_k $top_k --min_p $min_p

source activate eval_g2
python eval/novelty-bench/src/partition.py --eval-dir $outputfile --alg classifier
python eval/novelty-bench/src/score.py --eval-dir $outputfile --patience 0.8
python eval/novelty-bench/src/summarize.py --eval-dir $outputfile