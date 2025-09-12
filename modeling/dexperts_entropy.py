from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from transformers import TopKLogitsWarper, TopPLogitsWarper, LogitsProcessorList 
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
    #top_k_top_p_filtering,
    StoppingCriteriaList,
    LogitsProcessorList
)
from collections import defaultdict
from tqdm import tqdm
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
def compute_entropy(logits):
    # 将 logits 转换为概率分布
    probs = F.softmax(logits, dim=-1)
    # 计算熵
    if torch.all(probs == 0):
        entropy = torch.tensor(0.0, device=logits.device)
        print("entropy is 0")
    else:
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return entropy

class DExpertsLlama:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None,
        alpha: float = 1.0,
        threshold: float = 0.1,
        bottom: float = None,
        chat_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None
    ):
        """
        chat_response_prefix: For llama chat models, it can be helpful for the response
        to start with a certain prefix to constrain the generation to directly answer
        the question. This makes evaluation on MC datasets easier.
        """
        print("dexperts_entropy")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **model_kwargs
        )
        #self.model.bfloat16()
        self.model.eval()
        self.threshold = threshold
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.bottom = bottom
        self.device = self.model.device
        self.chat_response_prefix = chat_response_prefix

    def forward(
        self,
        base_inputs,
        pos_inputs=None,
        neg_inputs=None,
        return_dict=None
    ):
        base_outputs = self.model(**base_inputs, return_dict=return_dict)
        if pos_inputs != None:
            pos_outputs = self.model(**pos_inputs, return_dict=return_dict)
            neg_outputs = self.model(**neg_inputs, return_dict=return_dict)

            return base_outputs, pos_outputs, neg_outputs
        return base_outputs

    def _get_tokenized_chat_inputs(self, input_ids):
        """Decode input_ids and encode again to insert chat formatting"""

        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # remove response_prefix (e.g., "Answer:") from the prompt if it's already there
        if self.chat_response_prefix:
            cleaned_prompts = []
            for p in prompts:
                if self.chat_response_prefix in p:
                    p = p.replace(self.chat_response_prefix, '').rstrip()
                cleaned_prompts.append(p)
        else:
            cleaned_prompts = prompts

        chat_prompts = [f'{self.chat_prefix} {p} {self.chat_suffix}' for p in cleaned_prompts]
        # print('DExperts expert prompt', flush=True)
        # print(chat_prompts[0], flush=True)
        chat_inputs = self.tokenizer(
            chat_prompts, padding="longest", return_tensors="pt",
            add_special_tokens=True
        )
        chat_inputs.input_ids = chat_inputs.input_ids.to(self.device)
        chat_inputs.attention_mask = chat_inputs.attention_mask.to(self.device)

        return chat_inputs

    def update_analysis_data(self, analysis_data, next_tokens, next_token_logits_dict):
        analysis_data['tokens'].append([self.tokenizer.decode(t) for t in next_tokens])
        analysis_data['token_ids'].append(next_tokens)

        # logits from each model for the next token
        for model in next_token_logits_dict.keys():
            analysis_data[f'logits_{model}'].append(next_token_logits_dict[model].unsqueeze(dim=1))

        return analysis_data

    def generate(
        self,
        base_input_ids: Optional[torch.Tensor] = None,
        pos_input_ids: Optional[torch.Tensor] = None,
        neg_input_ids: Optional[torch.Tensor] = None,
        base_attention_mask: Optional[torch.Tensor] = None,
        pos_attention_mask: Optional[torch.Tensor] = None,
        neg_attention_mask: Optional[torch.Tensor] = None,
        theta = None,
        weight_method = None,
        first_n_tokens: Optional[int] = 500,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        return_logits_for_analysis: bool = False,
        top_k: Optional[int] = None,  # 新增top_k参数
        **kwargs
    ):
        # 处理top_k参数
        if top_k is not None and top_k > 0:
            if logits_processor is None:
                logits_processor = LogitsProcessorList()
            logits_processor.append(TopKLogitsWarper(top_k=top_k))
        else:
            print("top_k is None")

        if top_p < 1.0:
            if logits_processor is None:
                logits_processor = LogitsProcessorList()
            logits_processor.append(TopPLogitsWarper(top_p=top_p))

        base_kwargs = kwargs.copy()
        pos_kwargs = kwargs.copy() 
        neg_kwargs = kwargs.copy()
        # prepare inputs for expert model
        base_input_ids = base_input_ids.to(base_input_ids.device)
        base_kwargs['attention_mask'] = base_attention_mask
        pos_input_ids = pos_input_ids.to(pos_input_ids.device)
        pos_kwargs['attention_mask'] = pos_attention_mask
        neg_input_ids = neg_input_ids.to(neg_input_ids.device)
        neg_kwargs['attention_mask'] = neg_attention_mask

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(base_input_ids.shape[0], dtype=torch.long, device=base_input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(base_input_ids.device)

        if return_logits_for_analysis:
            analysis_data = defaultdict(list)
        
        input_length = len(base_input_ids[0])
        import matplotlib.pyplot as plt

        all_base = []
        all_max_diff = []
        all_max_base = []
        cal = True
        total_non_zero_entropy = 0
        # 记录batch中生成的总token数
        generated_tokens_count = 0
        for step in range(max_new_tokens):
            # prepare model inputs with past_key_values and attention_mask

            base_inputs = self.model.prepare_inputs_for_generation(base_input_ids, **base_kwargs)
            
            
            if cal:
                pos_inputs = self.model.prepare_inputs_for_generation(pos_input_ids, **pos_kwargs)
                neg_inputs = self.model.prepare_inputs_for_generation(neg_input_ids, **neg_kwargs)
            # DExperts
                base_outputs, pos_outputs, neg_outputs = self.forward(
                    base_inputs, pos_inputs, neg_inputs, return_dict=True
                )
                base_next_token_logits = base_outputs.logits[..., -1, :]
                pos_next_token_logits = pos_outputs.logits[..., -1, :]
                neg_next_token_logits = neg_outputs.logits[..., -1, :]

                # sometimes our experts have extra (irrelevant) tokens at the end of the normal vocabulary
                pos_next_token_logits = pos_next_token_logits[:, :base_next_token_logits.shape[-1]]
                neg_next_token_logits = neg_next_token_logits[:, :base_next_token_logits.shape[-1]]
                if weight_method == "entropy":
                    entropy_base = compute_entropy(base_next_token_logits).unsqueeze(dim=1)
                    entropy_base = torch.where(entropy_base < self.threshold, torch.tensor(0.0).to(entropy_base.device), entropy_base)
                    entropy_base = torch.where(entropy_base >= self.bottom, torch.tensor(theta).to(entropy_base.device), entropy_base)
                    # 计算非零的entropy_base元素数量 (只考虑未完成的序列)
                    masked_entropy_base = entropy_base * unfinished_sequences.unsqueeze(1).float()
                    critic_token_num = torch.count_nonzero(masked_entropy_base).item()
                    unfinished_token_num = torch.count_nonzero(unfinished_sequences).item()
                    # 累加到总数中
                    total_non_zero_entropy += critic_token_num
                    # 更新当前步生成的token数量（只计算未完成的序列）
                    generated_tokens_count += unfinished_token_num

                if weight_method == "entropy":
                    next_token_logits = (
                        base_next_token_logits +
                        entropy_base * (pos_next_token_logits - neg_next_token_logits)
                    )
                elif weight_method == "alpha":
                    next_token_logits = (
                        base_next_token_logits +
                        self.alpha * (pos_next_token_logits - neg_next_token_logits)
                    )
                    print("alpha")
                else:
                    raise ValueError("weight_method must be 'entropy' or 'alpha'")
            else:
                base_outputs = self.forward(
                    base_inputs, return_dict=True)
                pos_outputs, neg_outputs = None, None
                base_next_token_logits = base_outputs.logits[..., -1, :]
                
                # DExperts!
                next_token_logits = (
                    base_next_token_logits
                )
           
            if temperature != 1.0 and temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            if logits_processor is not None:
                next_token_logits = logits_processor(base_input_ids, next_token_logits)
            
            if step == 0:
                top_k_values, top_k_indices = torch.topk(next_token_logits, k=60, dim=-1)
                #print("top_k_values", top_k_values)
            # decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = (
                next_tokens * unfinished_sequences +
                self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )
            
            if return_logits_for_analysis:
                next_token_logits_dict = {
                    'dexperts': next_token_logits,
                    'base': base_next_token_logits,
                    'pos': pos_next_token_logits,
                    'neg': neg_next_token_logits
                }
                analysis_data = self.update_analysis_data(analysis_data, next_tokens, next_token_logits_dict)

            # update model inputs for next step
            base_input_ids = torch.cat([base_input_ids, next_tokens[:, None]], dim=-1)
            pos_input_ids = torch.cat([pos_input_ids, next_tokens[:, None]], dim=-1)
            neg_input_ids = torch.cat([neg_input_ids, next_tokens[:, None]], dim=-1)

            # update kwargs
            base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
            pos_kwargs = self._update_model_kwargs_for_generation(pos_outputs, pos_kwargs)
            neg_kwargs = self._update_model_kwargs_for_generation(neg_outputs, neg_kwargs)

            # stopping criteria
            if stopping_criteria and stopping_criteria(base_input_ids, None):
                break

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break
        # 计算平均每个批次需要处理的tokens比例
        if weight_method == "entropy":
            avg_critic_tokens_ratio = total_non_zero_entropy / generated_tokens_count if generated_tokens_count > 0 else 0
        else:
            avg_critic_tokens_ratio = 1
        if return_logits_for_analysis:
            for k in analysis_data.keys():
                if k.startswith('logits'):
                    analysis_data[k] = torch.cat(analysis_data[k], dim=1)
            return base_input_ids, analysis_data,avg_critic_tokens_ratio
        
        return base_input_ids,avg_critic_tokens_ratio

    def _update_model_kwargs_for_generation(
        self,
        outputs: Optional[ModelOutput] = None,
        kwargs: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        # update past_key_values
        if outputs is not None:
            kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return kwargs