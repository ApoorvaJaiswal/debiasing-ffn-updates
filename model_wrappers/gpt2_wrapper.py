from typing import List, Optional, Tuple, Dict, Union
from torch.nn import CrossEntropyLoss
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import scipy
import logging

import numpy as np
import torch.nn.functional as F

class GPT2Wrapper():
    def __init__(self, model_name: str = "gpt2-medium", use_cuda: bool = False):
        """
        :param model_name: the name of the pretrained GPT2 model (default: "gpt2-medium")
        :param use_cuda: whether to use CUDA
        """
        self._device = "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"

        self._tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self._model = GPT2LMHeadModel.from_pretrained(model_name)  
        if use_cuda:
            self._model.parallelize()
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.config.pad_token_id = self._tokenizer.eos_token_id
        
        self.hooks = None

    def query_model_tok_dist(self, prompt):
        tokens = self._tokenizer.encode_plus(prompt, return_tensors = 'pt').to(self._device)
        output = self._model(**tokens)
        logits = output['logits']
        probs = F.softmax(logits[0][tokens['input_ids'].shape[1] - 1], dim=-1) #gets probs after last tok in seq

        probs = torch.reshape(probs, (-1,)).detach().cpu().numpy()

        #assert probs add to 1
        assert np.abs(np.sum(probs) - 1) <= 0.01, str(np.abs(np.sum(probs)-1)) 

        probs_ = []
        for index, prob in enumerate(probs):
            probs_.append((index, prob))

        top_k = sorted(probs_, key = lambda x: x[1], reverse = True)[:10]
        top_k = [(t[1].item(), self._tokenizer.decode(t[0])) for t in top_k]
        
        return top_k

    def generate(self, input_text: List[str], word_filter: bool = False, min_length: int = 20, max_length: int = 20, **kwargs):
        inputs = self._tokenizer.batch_encode_plus(input_text, padding=True, return_tensors='pt')
        inputs['attention_mask'] = torch.flip(inputs['attention_mask'], dims=[1])
        shifts = inputs['attention_mask'].shape[-1] - inputs['attention_mask'].sum(dim=-1)
        for batch_idx in range(inputs['input_ids'].shape[0]):
            inputs['input_ids'][batch_idx] = inputs['input_ids'][batch_idx].roll(shifts[batch_idx].item())

        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_length = inputs['input_ids'].shape[1]
        if min_length is not None:
            min_length = min_length + input_length
        if max_length is not None:
            max_length = max_length + input_length

        output_ids = self._model.generate(**inputs, min_length=min_length, max_length=max_length, **kwargs)
        
        #only return the continuation text
        batch_size = output_ids.shape[0]
        output_ids = output_ids[:batch_size, inputs['input_ids'].shape[1]:]
        
        return self._tokenizer.batch_decode(output_ids)

    def project_value_to_vocab(self, layer, value_idx, top_k = 10):        
        normed = self._model.transformer.ln_f(self._model.transformer.h[layer].mlp.c_proj.weight.data[value_idx]).to(self._device)

        logits = torch.matmul(self._model.lm_head.weight, normed.T).to(self._device)
        probs = F.softmax(logits, dim=-1)
        probs = torch.reshape(probs, (-1,)).detach().cpu().numpy()

        probs_ = []
        for index, prob in enumerate(probs):
            probs_.append((index, prob))

        top_k = sorted(probs_, key = lambda x: x[1], reverse = True)[:top_k]
        value_preds = [(self._tokenizer.decode(t[0]), t[1]) for t in top_k]
        
        return value_preds
    
    def generate_word_filter(self,
                 prompt: Union[str, List[str]],
                 min_length: int = 20, 
                 max_length: int = 20, 
                 **model_kwargs) -> List[str]:
        
        bad_words = open("word_filter_words.txt").read().split("\n")
        bad_words_ids = [
            self._tokenizer.encode(bad_word, add_prefix_space=True) 
            for bad_word in bad_words
        ]
        
        if isinstance(prompt, str):
            prompt = [prompt]
        
        inputs = self._tokenizer.batch_encode_plus(prompt, padding=True, return_tensors='pt')
        inputs['attention_mask'] = torch.flip(inputs['attention_mask'], dims=[1])
        shifts = inputs['attention_mask'].shape[-1] - inputs['attention_mask'].sum(dim=-1)
        for batch_idx in range(inputs['input_ids'].shape[0]):
            inputs['input_ids'][batch_idx] = inputs['input_ids'][batch_idx].roll(shifts[batch_idx].item())

        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_length = inputs['input_ids'].shape[1]
        if min_length is not None:
            min_length = min_length + input_length
        if max_length is not None:
            max_length = max_length + input_length
        
        output_ids = self._model.generate(**inputs, 
                                          min_length=min_length, max_length=max_length, 
                                          bad_words_ids=bad_words_ids,
                                          **model_kwargs)

        # only return the continuation text
        batch_size = output_ids.shape[0]
        output_ids = output_ids[:batch_size, inputs['input_ids'].shape[1]:]

        return self._tokenizer.batch_decode(output_ids)

    def compute_loss(self, input_ids: torch.LongTensor, labels: torch.LongTensor) -> torch.Tensor:
        outputs = self._model(input_ids, labels=labels)
        lm_logits = outputs[1]

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss

    def set_value_activations(self, 
                              values_per_layer: Dict[int, List[int]], 
                              coef_value: int = 3):
        """
        Uses PyTorch hooks to set the activations of each value in values_per_layer to coef_value
        Only works on GPT2 from HF.
        
        :params values_per_layer: dictionary with keys that correspond to layers and values that correspond to indices
        :params coef_value: number to set the values' activations to
        """
        
        def value_activation_replacement_hook(values, coef_val):
            def hook(module, input, output):
                output[:, :, values] = coef_val
              
            return hook
        
        hooks = []
        
        for layer in range(self._model.config.n_layer):
            if layer in values_per_layer:
                values = values_per_layer[layer]
            else:
                values = []
            
            hook = self._model.transformer.h[layer].mlp.c_fc.register_forward_hook(
                    value_activation_replacement_hook(values, coef_value)
                )
                
            hooks.append(hook)
        
        self.hooks = hooks
        
        return hooks   
    
    def remove_all_hooks(self):
        if self.hooks is not None:
            for hook in self.hooks:
                hook.remove()
            
            self.hooks = None
        else:
            print("No hooks to remove")


    ## From Big Bench ##
    def score(
        self,
        inputs: Union[List[str], str],
        targets: Union[List[str], str],
        mask_token_id=-100,
    ) -> List[float]:
        """Scores one or a batch of example targets given their inputs.
        Args:
        inputs: input context
        targets:  targets to be scored
        Returns:
        list of log probabilities for each target given the input.
        """

        if isinstance(inputs, str):
            input_list = [inputs]
            target_list = [targets]
        else:
            input_list = inputs
            target_list = targets

        tokenized_ids = _gpt_batch_tokenize(
            tokenizer=self._tokenizer,
            batch_inputs=input_list,
            batch_targets=target_list,
        )

        inputs_and_targets_ids = tf.constant(tokenized_ids["inputs_and_targets_ids"])
        targets_ids = tf.constant(tokenized_ids["targets_ids"])
        attention_mask = tf.constant(tokenized_ids["attention_mask"])

        inputs_and_targets_ids = self._maybe_truncate_input(
            inputs_and_targets_ids, verbose=True
        )
        targets_ids = self._maybe_truncate_input(targets_ids, verbose=False)
        attention_mask = self._maybe_truncate_input(attention_mask, verbose=False)
        # Calculating position ids, since they might be changed by truncation
        position_ids = tf.maximum(tf.cumsum(attention_mask, axis=-1) - 1, 0)

        logits = self._model(
            inputs_and_targets_ids,
            labels=targets_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).logits

        return self.compute_loss(targets_ids, logits)

    def cond_log_prob(
        self,
        inputs: Union[str, List[str]],
        targets: Union[List[str], List[List[str]]],
        batch_size: int = 64,
        absolute_normalization: Optional[bool] = False,
    ) -> Union[List[float], List[List[float]]]:
        """Computes conditional log probabilities of targets given inputs.

        Args:
          `inputs`: A single string input or a list of string inputs.

          `targets`: Possible string outputs for each input. If input is a
             string, this is a list `[t_1, t_2, ..., t_n]` of possible string
             outputs. If input is a list of strings, then this is a nested
             list `[[t_1, t_2, ..., t_n], ...]` with length equal to `len(inputs)`.

           `absolute_normalization`: When True, the function returns the log
             probability of unconstrained generation or the target sequence. When
             False (default), log probabilities are normalized so that the probabilities
             of generating `targets` sum to 1. Note that setting `absolute_normalization`
             to True restricts the class of models that can be evaluated to those that
             can assign absolute probabilities to sequences.

           Returns:
             If a single string input is provided, returns a list of
             log-probabilities `[lp_1, lp_2, ..., lp_n]` predicted by the model,
             where  `lp_i = log(prob(t_i | input)`  is the conditional log-prob
             to generate target `t_i` given input. If a list of string inputs
             was provided, returns a list of such elements of the form
             `[[lp_1, lp_2, ..., lp_n], ...]`, where each element contains the
             log-probabilities for the corresponding input and targets.
             In this case, the length of the returned list is `len(input)`.
        """

        if isinstance(inputs, str):
            input_list = [inputs]
            target_list = [targets]
        else:
            input_list = inputs
            target_list = targets

        flat_idx = []
        flat_inputs = []
        flat_choices = []
        for example_id, (example_input, choices) in enumerate(zip(input_list, target_list)):
            for choice_id, choice in enumerate(choices):
                flat_idx.append((example_id, choice_id))
                flat_inputs.append(example_input)
                flat_choices.append(choice)

        flat_idx, flat_inputs, flat_choices
        print(flat_inputs[0])
        print(flat_choices[0])
    
        num_examples = len(flat_idx)
        flat_scores = []
        for idx in range(0, num_examples, batch_size):
            batch_idx = flat_idx[idx : min(idx + batch_size, num_examples)]
            batch_inputs = flat_inputs[idx : min(idx + batch_size, num_examples)]
            batch_choices = flat_choices[idx : min(idx + batch_size, num_examples)]

            print("BATCH INPUTS")
            print(batch_inputs)

            print("BATCH CHOICES")
            print(batch_choices)
        #     batch_scores = self._model.score(batch_inputs, batch_choices)
        #     flat_scores += batch_scores

        # scores = [[] for _ in range(len(input_list))]

        # for idx, score in zip(flat_idx, flat_scores):
        #     if score == 0:
        #       # all tokens were masked. Setting score to -inf.
        #       logging.warning('Found score identical to zero. Probably from empty target. '
        #                      'Setting score to -inf.'
        #                     )
        #       scores[idx[0]].append(-np.inf)
        #     else:
        #       scores[idx[0]].append(score)

        # if not absolute_normalization:
        #     scores = [
        #         list(score_row - scipy.special.logsumexp(score_row))
        #         for score_row in scores
        #     ]

        # if isinstance(inputs, str):
        #     scores = scores[0]

        # return scores
    