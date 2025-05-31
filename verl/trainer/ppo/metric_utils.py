# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""

import torch
from typing import Any, Callable
import numpy as np
from verl import DataProto
from collections import defaultdict
from functools import partial
import wandb
from collections import defaultdict



def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    reduced_metrics = {}
    for key, val in metrics.items():
        reduced_metrics[key] = np.mean(val)
    return reduced_metrics


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(
    batch: DataProto,
    result_dicts: list[dict[str, Any]],
    step: int,
    use_critic: bool = True,
) -> dict[str, Any]:
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    is_correct = [d["accuracy_reward"] == 1.0 for d in result_dicts]
    correction_mask = torch.tensor(is_correct, dtype=torch.bool)
    correct_response_length = torch.masked_select(response_length, correction_mask)
    wrong_response_length = torch.masked_select(response_length, ~correction_mask)

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    reward_values = {}
    for result in result_dicts:
        for key, value in result.items():
            if "reward" in key and value is not None:
                reward_values.setdefault(key, []).append(value)

    rew_metric_dict = {}
    for reward_name, values in reward_values.items():
        key = f'rews/{reward_name}'
        rew_metric_dict[key] = np.mean(values)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'rews/total_score':
            torch.mean(sequence_score).detach().item(),
        # reward
        'rews/total_rew':
            torch.mean(sequence_reward).detach().item(),
        # adv
        'rews/advantage':
            torch.mean(valid_adv).detach().item(),
        # returns
        'rews/return':
            torch.mean(valid_returns).detach().item(),
        # separate rews
        **rew_metric_dict,
        **({
            # values
            'training/vf_value': torch.mean(valid_values).detach().item(),
            # vf explained var
            'training/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean_length':
            torch.mean(response_length).detach().item(),
        'response_length/length_clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),

        # correct response length
        'response_length/mean_correct_length':
            torch.mean(correct_response_length).detach().item(),
        'response_length/length_correct_clip_ratio':
            torch.mean(torch.eq(correct_response_length, max_response_length).float()).detach().item(),

        # wrong response length
        'response_length/mean_wrong_length':
            torch.mean(wrong_response_length).detach().item(),
        'response_length/length_wrong_clip_ratio':
            torch.mean(torch.eq(wrong_response_length, max_response_length).float()).detach().item(),

        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    by_source, idx_by_source = defaultdict(list), defaultdict(list)
    for i, d in enumerate(result_dicts):
        data_source = d["data_source"].replace("/", "_")
        by_source[data_source].append(d)
        idx_by_source[data_source].append(i)

    data_source_metrics = {}
    for data_source in idx_by_source:
        idxs = idx_by_source[data_source]
        data_source_result_dicts = by_source[data_source]

        data_source_reward_values = {}
        for result in data_source_result_dicts:
            for key, value in result.items():
                if "reward" in key and value is not None:
                    data_source_reward_values.setdefault(key, []).append(value)

        data_source_rew_metric_dict = {}
        for reward_name, values in data_source_reward_values.items():
            key = f'rews_{reward_name}/{data_source}'
            data_source_rew_metric_dict[key] = np.mean(values)

        data_source_response_length = response_length[idxs]

        data_source_prompt_texts = [d["prompt"] for d in data_source_result_dicts]
        data_source_response_texts = [d["response"] for d in data_source_result_dicts]
        data_source_is_correct = [d["accuracy_reward"] == 1.0 for d in data_source_result_dicts]
        data_source_ids = [d["id"] for d in data_source_result_dicts]

        words = ["re-check", "re-evaluate", "re-examine", "re-think", "recheck", "reevaluate", "reexamine", "reevaluation", "rethink", "check again", "think again", "try again", "verify", "wait", "yet"]
        reflection_metrics = _compute_words_metrics_and_tables(
            words=words,
            ids=data_source_ids,
            prompt_texts=data_source_prompt_texts,
            response_texts=data_source_response_texts,
            is_correct=data_source_is_correct,
            step=step,
            metric_name="reflection",
            data_source=data_source
        )

        data_source_correction_mask = torch.tensor(data_source_is_correct, dtype=torch.bool)
        data_source_correct_response_length = torch.masked_select(data_source_response_length, data_source_correction_mask)
        data_source_wrong_response_length = torch.masked_select(data_source_response_length, ~data_source_correction_mask)

        data_source_metrics.update({
            # response length
            f'response_length_{data_source}/mean_length':
                torch.mean(data_source_response_length).detach().item(),
            f'response_length_{data_source}/length_clip_ratio':
                torch.mean(torch.eq(data_source_response_length, max_response_length).float()).detach().item(),
            # correct response length
            f'response_length_{data_source}/mean_correct_length':
                torch.mean(data_source_correct_response_length).detach().item(),
            f'response_length_{data_source}/length_correct_clip_ratio':
                torch.mean(torch.eq(data_source_correct_response_length, max_response_length).float()).detach().item(),
            # wrong response length
            f'response_length_{data_source}/mean_wrong_length':
                torch.mean(data_source_wrong_response_length).detach().item(),
            f'response_length_{data_source}/length_wrong_clip_ratio':
                torch.mean(torch.eq(data_source_wrong_response_length, max_response_length).float()).detach().item(),
            **reflection_metrics,
            **data_source_rew_metric_dict
        })


    metrics.update(data_source_metrics)
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    total_num_tokens = sum(batch.meta_info['global_token_num'])
    time = timing_raw['step']
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        'perf/total_num_tokens': total_num_tokens,
        'perf/time_per_step': time,
        'perf/throughput': total_num_tokens / (time * n_gpus),
    }


def _compute_words_metrics_and_tables(
    words: list[str],
    ids: list[int],
    prompt_texts: list[str],
    response_texts: list[str],
    is_correct: list[bool],
    step: int,
    metric_name: str,
    data_source: str,
) -> dict[str, Any]:

    # Convert all text to lowercase for easier matching
    texts_lower = [t.lower() for t in response_texts]
    total_count = len(texts_lower)

    # Identify whether each text contains reflection words for ratio computation
    has_words = [
        any(word in text for word in words) for text in texts_lower
    ]

    has_word_ids = [
        ids[i] for i in range(len(ids)) if has_words[i] and is_correct[i]
    ]
    has_word_correct_responses = [
        response_texts[i] for i in range(len(response_texts)) if has_words[i] and is_correct[i]
    ]
    has_word_correct_prompts = [
        prompt_texts[i] for i in range(len(prompt_texts)) if has_words[i] and is_correct[i]
    ]
    no_word_ids = [
        ids[i] for i in range(len(ids)) if (not has_words[i]) and is_correct[i]
    ]
    no_word_correct_responses = [
        response_texts[i] for i in range(len(response_texts)) if (not has_words[i]) and is_correct[i]
    ]
    no_word_correct_prompts = [
        prompt_texts[i] for i in range(len(prompt_texts)) if (not has_words[i]) and is_correct[i]
    ]

    has_word_example_table = wandb.Table(columns=["step", "id", "prompt", "response"])
    if len(has_word_correct_responses) > 0 and len(has_word_correct_prompts) > 0:
        has_word_example_table.add_data(step, has_word_ids[0], has_word_correct_prompts[0], has_word_correct_responses[0])

    no_word_example_table = wandb.Table(columns=["step", "id", "prompt", "response"])
    if len(no_word_correct_responses) > 0 and len(no_word_correct_prompts) > 0:
        no_word_example_table.add_data(step, no_word_ids[0], no_word_correct_prompts[0], no_word_correct_responses[0])

    example_dict = {
        f"has_{metric_name}_correct_examples_{data_source}/step_{step}": has_word_example_table,
        f"no_{metric_name}_correct_examples_{data_source}/step_{step}": no_word_example_table,
    }

    # Count total, correct, incorrect, and reflection-included samples
    total_correct = sum(is_correct)
    total_incorrect = total_count - total_correct
    word_count = sum(has_words)

    # 1. Ratio of responses that contain at least one reflection word
    word_ratio = word_count / total_count if total_count else 0.0

    # 2. Among correct responses, ratio that contain reflection words
    if total_correct > 0:
        correct_with_word_count = sum(
            has_words[i] for i in range(total_count) if is_correct[i]
        )
        word_ratio_in_correct_answers = (
            correct_with_word_count / total_correct
        )
    else:
        word_ratio_in_correct_answers = 0.0

    # 3. Among incorrect responses, ratio that contain reflection words
    if total_incorrect > 0:
        incorrect_with_word_count = sum(
            has_words[i] for i in range(total_count) if not is_correct[i]
        )
        word_ratio_in_incorrect_answers = (
            incorrect_with_word_count / total_incorrect
        )
    else:
        word_ratio_in_incorrect_answers = 0.0

    # 4. Among responses with reflection words, ratio that are correct
    if word_count > 0:
        correct_in_word_texts_count = sum(
            is_correct[i] for i in range(total_count) if has_words[i]
        )
        correct_ratio_in_word_texts = (
            correct_in_word_texts_count / word_count
        )
    else:
        correct_ratio_in_word_texts = 0.0

    # 5. Among responses without reflection words, ratio that are correct
    no_word_count = total_count - word_count
    if no_word_count > 0:
        correct_in_no_word_texts_count = sum(
            is_correct[i] for i in range(total_count) if not has_words[i]
        )
        correct_ratio_in_no_word_texts = (
            correct_in_no_word_texts_count / no_word_count
        )
    else:
        correct_ratio_in_no_word_texts = 0.0

    # (A) Aggregate all computed statistics
    word_ratio_dict = {
        f"{metric_name}_ratios_{data_source}/{metric_name}_ratio": word_ratio,
        f"{metric_name}_ratios_{data_source}/{metric_name}_ratio_in_correct_answers": word_ratio_in_correct_answers,
        f"{metric_name}_ratios_{data_source}/{metric_name}_ratio_in_incorrect_answers": word_ratio_in_incorrect_answers,
        f"{metric_name}_ratios_{data_source}/correct_ratio_in_{metric_name}_texts": correct_ratio_in_word_texts,
        f"{metric_name}_ratios_{data_source}/correct_ratio_in_no_{metric_name}_texts": correct_ratio_in_no_word_texts,
    }
    # (B) Count total occurrences of each reflection word (accumulated across texts)
    word_frequency = {
        f"{metric_name}_words_{data_source}/{word}": sum(text.count(word) for text in texts_lower)
        for word in words
    }

    return_dict = {
        **word_ratio_dict,
        **word_frequency,
        **example_dict,
    }
    return return_dict

def bootstrap_metric(data: list[Any],
                     subset_size: int,
                     reduce_fns: list[Callable[[np.ndarray], float]],
                     n_bootstrap: int = 1000,
                     seed: int = 42) -> list[tuple[float, float]]:
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(data_sources: list[str],
                               sample_inputs: list[str],
                               infos_dict: dict[str, list[Any]],
                               seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """Process validation metrics into a structured format.
    
    Args:
        data_sources: Array of data source identifiers for each sample
        sample_inputs: List of input prompts
        infos_dict: variable name -> list of values for each sample
        
    Returns:
        dict[str, dict[str, dict[str, float]]]: data source -> variable name -> metric value
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)
                metric[f"std@{n_resps}"] = np.std(var_vals)

                ns = []
                n = 2
                while n < n_resps:
                    ns.append(n)
                    n *= 2
                ns.append(n_resps)

                for n in ns:
                    # Best/Worst-of-N
                    [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals,
                                                                                  subset_size=n,
                                                                                  reduce_fns=[np.max, np.min],
                                                                                  seed=seed)
                    metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                    metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                    # Majority voting
                    if var2vals.get("pred", None) is not None:
                        vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                        [(maj_n_mean, maj_n_std)
                        ] = bootstrap_metric(data=vote_data,
                                             subset_size=n,
                                             reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                             seed=seed)
                        metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val
