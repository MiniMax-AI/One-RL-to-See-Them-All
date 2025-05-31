from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time

import numpy as np
import torch

from verl import DataProto

from .remote_proxy import SingleStepRemoteProxyManager


class RewardWorker:

    def __init__(self, tokenizer, reward_server_params, is_training, step, total_steps):
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.step = step
        self.total_steps = total_steps
        # Initialize reward server proxy within the actor
        self.reward_server_proxy = SingleStepRemoteProxyManager(
            rm_job=reward_server_params.get("rm_job"),
            rm_fun=reward_server_params.get("rm_fun")
        )

    def process_item(self, idx, data_item):

        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            else:
                return obj

        # Convert tensor to numpy if it exists
        image_grid_thw = None
        if 'multi_modal_inputs' in data_item.non_tensor_batch and 'image_grid_thw' in data_item.non_tensor_batch[
                'multi_modal_inputs']:
            image_grid_thw = data_item.non_tensor_batch['multi_modal_inputs']['image_grid_thw'].numpy()
            image_grid_thw = [(int(t), int(h * 14), int(w * 14)) for t, h, w in image_grid_thw]

        prompt_ids = data_item.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        data_source = data_item.non_tensor_batch['data_source']
        reward_verifier = data_item.non_tensor_batch['reward_model']['verifier']
        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        answer = data_item.non_tensor_batch['reward_model']['answer']
        accuracy_reward_ratio = data_item.non_tensor_batch['reward_model'].get('accuracy_ratio', None)
        format_reward_ratio = data_item.non_tensor_batch['reward_model'].get('format_ratio', None)
        reward_verifier_parm = data_item.non_tensor_batch['reward_model'].get('verifier_parm', {})
        if reward_verifier_parm is None:
            reward_verifier_parm = {}

        extra_info = data_item.non_tensor_batch['extra_info']
        image_path = extra_info.get('image_path', None)
        if isinstance(image_path, str):
            image_path = [image_path]

        # prepare reward verifier parm
        reward_verifier_parm['is_training'] = self.is_training
        reward_verifier_parm['step'] = int(self.step)
        reward_verifier_parm['total_steps'] = int(self.total_steps)
        reward_verifier_parm['image_grid_thw'] = image_grid_thw
        reward_verifier_parm['image_path'] = image_path

        # Convert any numpy types to Python native types for JSON serialization
        # serializable_parm = convert_to_serializable(reward_verifier_parm)

        payload = {
            "response": response_str,
            "answer": answer,
            "prompt": prompt_str,
            "solution": ground_truth,
            "data_source": data_source,
            "reward_verifier": reward_verifier,
            "reward_verifier_parm": json.dumps(reward_verifier_parm),
        }

        if format_reward_ratio is not None:
            payload['format_ratio'] = format_reward_ratio
        if accuracy_reward_ratio is not None:
            payload['accuracy_ratio'] = accuracy_reward_ratio

        rewards = self.reward_server_proxy.get_reward([payload])
        try:
            gather_rewards = rewards[0]['rewards']
            score = float(gather_rewards['final_reward'])
        except Exception as e:
            print(f"Error in get_reward: {e}")
            gather_rewards = {}
            score = 0.0

        result_dict = {
            "id": data_item.non_tensor_batch.get('index', 0),
            "data_source": data_source,
            "prompt": prompt_str,
            "response": response_str,
            "ground_truth": ground_truth,
            "answer": answer,
        }
        for reward_key, reward_value in gather_rewards.items():
            result_dict[reward_key] = reward_value

        return idx, int(valid_response_length), score, result_dict


class RemoteRewardManager:

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key='data_source',
        max_workers=32,
        is_training=True,
        remote_reward_job_id=None,
    ) -> None:
        self.tokenizer = tokenizer
        # 需要打印的解码结果批次数
        self.num_examine = num_examine
        # 最大线程数
        self.max_workers = max_workers
        self.is_training = is_training
        self.remote_reward_job_id = remote_reward_job_id

    def __call__(self, data: DataProto, step: int, total_steps: int, log_sample: bool = False):
        """
        计算数据中每个样本的奖励分数。
        """
        # 如果数据中已经包含 rm_scores，则直接返回
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # 从环境变量中构造 reward server 参数
        reward_server_params = {
            "rm_job": self.remote_reward_job_id,
            "rm_fun": "/judge"
        }

        start_time = time.time()
        print(f"=======Start {len(data)} reward, max_workers: {self.max_workers}=======")

        # Create a worker instance
        worker = RewardWorker(self.tokenizer, reward_server_params, self.is_training, step, total_steps)

        # Use ThreadPoolExecutor for parallel processing
        num_workers = min(self.max_workers, len(data))
        results = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(worker.process_item, i, data_item) for i, data_item in enumerate(data)]

            # Gather results as they complete
            for future in as_completed(futures):
                results.append(future.result())

        # Sort results by idx (in-place)
        results.sort(key=lambda x: x[0])

        # 处理返回的结果，更新 reward_tensor 并进行部分结果打印
        data_source_counts = {}
        all_result_dicts = []
        for idx, valid_response_length, score, result_dict in results:
            reward_tensor[idx, valid_response_length - 1] = score
            all_result_dicts.append(result_dict)

            data_source = result_dict['data_source']
            if data_source not in data_source_counts:
                data_source_counts[data_source] = 0
            if data_source_counts[data_source] < self.num_examine:
                if log_sample:
                    print(json.dumps(result_dict, indent=4))
                    data_source_counts[data_source] += 1

        end_time = time.time()
        print(f"=======Complete {len(data)} reward, takes: {round(end_time - start_time, 2)} seconds=======")
        return reward_tensor, all_result_dicts
