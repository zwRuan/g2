# G2: Guided Generation for Enhanced Output Diversity in LLMs

![Overview](figure/main.png)

> **🎉 Accepted at EMNLP 2025 Main Conference**

## Abstract

Large Language Models (LLMs) have demonstrated exceptional performance across diverse natural language processing tasks. However, these models exhibit a critical limitation in output diversity, often generating highly similar content across multiple attempts. This limitation significantly affects tasks requiring diverse outputs, from creative writing to reasoning. Existing solutions, like temperature scaling, enhance diversity by modifying probability distributions but compromise output quality. We propose **Guide-to-Generation** (G2), a training-free plug-and-play method that enhances output diversity while preserving generation quality. G2 employs a base generator alongside dual Guides, which guide the generation process through decoding-based interventions to encourage more diverse outputs conditioned on the original query. Comprehensive experiments demonstrate that G2 effectively improves output diversity while maintaining an optimal balance between diversity and quality.


## Project Structure

```
g2/
├── modeling/
│   └── dexperts_entropy.py      # Core G2 implementation
├── eval/
│   ├── novelty-bench/           # NoveltyBench evaluation
│   ├── wmt/                     # WMT translation evaluation
│   ├── xlsum/                   # XLSum summarization evaluation
│   ├── diversity/               # Diversity metrics
│   └── utils.py                 # Evaluation utilities
├── scripts/                     # Evaluation scripts
├── figure/                      # Paper figures
└── data/                        # Dataset configurations
```

# Environment Setup

We run the experiments on 8 NVIDIA A100-80G GPU.

Due to library conflicts between diversity evaluation and inference dependencies, we have separated the inference and diversity evaluation environments.
## Inference Environment
```
conda create -n g2 python=3.10
conda activate g2 
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r g2_requirements.txt
```
## Evaluation Environment
```
conda create -n g2_eval python=3.10
conda activate g2_eval 
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r g2_eval_requirements.txt
```

# Evaluation
We provide generation and evaluation scripts for each benchmark under `scripts`.  
Taking WMT as an example: `run_eval.sh` evaluates our method; `run_sample.sh` evaluates sampling-based methods (temperature, top-k, top-p, min-p); `diversity.sh` evaluates the diversity of the model’s outputs.  

👉 The datasets used in each benchmark can be found in the corresponding `run_eval.py`.  
For instance, in `eval/wmt/run_eval.py` (line 172–173):  
```python
dataset = load_dataset("wmt14", "de-en") 
test_data = dataset["test"]
```

## WMT'14 GE->EN

For diversity and quality scores on WMT'14 GE->EN, you can use the following scripts:
```
# g2
conda activate g2
bash scripts/eval/wmt/run_eval.sh

# sample methods
conda activate g2
bash scripts/eval/wmt/run_sample.sh
```
This code will create a directory containing JSONL files for the outputs from five sampling runs, as well as an all_metrics.json file with the BLEU and COMET for each run.

## Additional Diversity Metrics

If you want to evaluate diversity metrics such as sentence-BERT, self-BLEU, and EAD (Expectation-Adjusted Distinct Ngrams), please execute:
```
conda activate g2_eval
bash scripts/eval/wmt/diversity.sh
```


## Acknowledgments

We thank the excellent open-source libraries including:
- [Proxy-tuning](https://github.com/alisawuffles/proxy-tuning)
- [NoveltyBench](https://github.com/novelty-bench/novelty-bench)

And other outstanding works that contributed to this research.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{g2-guided-generation-2025,
    title={G2: Guided Generation for Enhanced Output Diversity in LLMs},
    author={Zhiwen Ruan and Yixia Li and Yefeng Liu and Yun Chen and Weihua Luo and Peng Li and Yang Liu and Guanhua Chen},
    booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
    year={2025},
    publisher={Association for Computational Linguistics}
}
```