from abc import ABC, abstractmethod
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Tuple
from latex2sympy2_extended import NormalizationConfig
import math_verify
from utils.det_helper import (
    convert_bbox_to_coco_format,
    extract_bbox,
    normalize_bbox_by_real_size,
    greedy_match_by_iou_max_iou_first,
    greedy_match_by_iou_max_label_first
)
from utils.pycocotools import COCO, COCOeval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
VerifyResult = Dict[str, Any]


class Verifier:
    """Registry for all verifier classes."""
    _verifiers: Dict[str, Type['BaseVerifier']] = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        """Decorator to register a verifier class."""

        def decorator(verifier_cls: Type['BaseVerifier']) -> Type['BaseVerifier']:
            verifier_name = name or verifier_cls.__name__
            cls._verifiers[verifier_name] = verifier_cls
            return verifier_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type['BaseVerifier']]:
        """Get a verifier class by name."""
        return cls._verifiers.get(name)

    @classmethod
    def list_verifiers(cls) -> List[str]:
        """List all registered verifiers."""
        return list(cls._verifiers.keys())


class BaseVerifier(ABC):
    """Base class for all verifiers."""

    def __init__(self,
                 is_training: bool,
                 step: int,
                 total_steps: int,
                 image_path: Optional[List[str]] = None,
                 image_grid_thw: Optional[List[Tuple[int, int, int]]] = None,
                 verifier_style: str = 'rule',
                 det_verifier_normalized: bool = False,
                 det_reward_ratio: Dict[str, float] = {}):

        self.is_training = is_training
        self.step = step
        self.total_steps = total_steps
        self.step_ratio = float(step) / float(total_steps)
        self.image_path = image_path
        self.image_grid_thw = image_grid_thw
        self.verifier_style = verifier_style
        self.det_verifier_normalized = det_verifier_normalized
        self.det_reward_ratio = det_reward_ratio

    @abstractmethod
    def verify_format(self, predict_str: Any) -> VerifyResult:
        """
        Verify the format of the input.
        
        Args:
            predict_str: The input to verify

        Returns:
            Dict containing verification results
        """
        pass

    @abstractmethod
    def verify_accuracy(self, predict_str: Any, solution: Any) -> VerifyResult:
        """
        Verify the accuracy of the input against the solution.
        
        Args:
            predict_str: The input to verify
            solution: The solution to verify against
            
        Returns:
            Dict containing verification results
        """
        pass




def extract_answer_content(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else ""


@Verifier.register(name="mathverify")
class MathVerifyVerifier(BaseVerifier):

    def verify_format(self, predict_str: str) -> float:
        """@yema | Reward function that checks if the reasoning process
        is enclosed within <think> and </think> tags, while the final answer
        is enclosed within <answer> and </answer> tags.
        referenced this part of the code from
        https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py"""

        def count_tags(text: str) -> float:
            count = 0.0
            if text.count("<think>\n") == 1:
                count += 0.25
            if text.count("\n</think>\n") == 1:
                count += 0.25
            if text.count("\n<answer>\n") == 1:
                count += 0.25
            if text.count("\n</answer>") == 1:
                count += 0.25
            return count

        return count_tags(predict_str)

    def verify_accuracy(self, predict_str: str, solution: str) -> float:
        """@yema | Reward function that checks if the completion is the same as the ground truth.
        referenced this part of the code from 
        https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py"""
        predict_str, solution = predict_str.lower(), solution.lower()
        gold_parsed = math_verify.parse(
            solution,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = math_verify.parse(
                predict_str,
                extraction_config=[
                    math_verify.LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(math_verify.verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                #TODO: @yema the return should be none, and we need mask these "None" samples
                reward = 0.
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            #TODO: @yema the return should be none, and we need mask these "None" samples
            reward = 0.
            print("Failed to parse gold solution: ", solution)

        return reward

@Verifier.register(name="detection")
class DetectionVerifier(BaseVerifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        iou_threshold = os.environ.get("DET_IOU_THRESHOLD", None)
        if iou_threshold is None:
            self.iou_threshold = 'average'
        elif iou_threshold == 'average':
            self.iou_threshold = 'average'
        elif iou_threshold == 'dynamic':
            self.iou_threshold = 'dynamic'
        else:
            try:
                self.iou_threshold = float(iou_threshold)
            except ValueError:
                self.iou_threshold = 'average'

        if self.iou_threshold not in [
                0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 'average', 'dynamic'
        ]:
            logger.error("DET_IOU_THRESHOLD is not set, assign 'average' to it")
            self.iou_threshold = 'average'
        else:
            logger.info(f"DET_IOU_THRESHOLD is set to {self.iou_threshold}")

        for key in ['iou_max_label_first', 'iou_max_iou_first', 'iou_completeness', 'map', 'map50', 'map75']:
            if key not in self.det_reward_ratio:
                logger.error(f"Key {key} not found in det_reward_ratio, assign 0.0 to it")
                self.det_reward_ratio[key] = 0.0
            elif self.det_reward_ratio[key] < 0 or self.det_reward_ratio[key] > 1:
                logger.warning(f"Value for key {key} must be between 0 and 1, assign 0.0 to it")
                self.det_reward_ratio[key] = 0.0


    def verify_format(self, predict_str: str) -> float:
        try:
            # Only one answer tag is allowed
            if predict_str.count("\n<answer>\n") and predict_str.count("\n</answer>") == 1:
                # continue to match the format
                predict_extract = extract_answer_content(predict_str.strip()).lower()
                predict_bbox = extract_bbox(predict_extract)

                # extract bbox from answer
                if predict_bbox is None or len(predict_bbox) == 0:
                    return 0.0
                else:
                    return 1.0
            else:
                return 0.0
        except Exception:
            return 0.0

    def pack_for_map_score(self, predict_bbox, answer_bbox):
        """
        predict_bbox: [{'bbox_2d': [x1, y1, x2, y2], 'label': 'label_name'}, ...]
        answer_bbox: [{'bbox_2d': [x1, y1, x2, y2], 'label': 'label_name'}, ...]
        return dict with mAP and mAP50 scores

        advice: pure map should be used with length penalty
        """

        # Initialize COCO object for ground truth
        gt_json = {"annotations": [], "images": [], "categories": []}
        gt_json["images"] = [{"id": 0, "file_name": "fake_image_0.jpg"}]

        gt_json["categories"] = []

        cats2id, cat_count = {}, 0
        for idx, gt_bbox in enumerate(answer_bbox):
            if gt_bbox["label"] not in cats2id:
                cats2id[gt_bbox["label"]] = cat_count
                gt_json["categories"].append({"id": cat_count, "name": gt_bbox["label"]})
                cat_count += 1

            bbox_coco_format, bbox_area_coco_format = convert_bbox_to_coco_format(gt_bbox["bbox_2d"])
            gt_json["annotations"].append({
                "id": idx + 1,
                "image_id": 0,
                "category_id": cats2id[gt_bbox["label"]],
                "bbox": bbox_coco_format,
                "area": bbox_area_coco_format,
                "iscrowd": 0
            })
        # Initialize COCO object for ground truth
        coco_gt = COCO(gt_json)

        dt_json = []
        for idx, pred_bbox in enumerate(predict_bbox):
            if pred_bbox["label"] not in cats2id:
                continue
            bbox_coco_format, bbox_area_coco_format = convert_bbox_to_coco_format(pred_bbox["bbox_2d"])
            dt_json.append({
                "image_id": 0,
                "category_id": cats2id[pred_bbox["label"]],
                "bbox": bbox_coco_format,
                "score": 1.0,  # no confidence score in predict_bbox right now
                "area": bbox_area_coco_format
            })
        if len(dt_json) == 0:
            return {'map': 0.0, 'map50': 0.0, 'map75': 0.0}

        coco_dt = coco_gt.loadRes(dt_json)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        map_score = float(coco_eval.stats[0])
        map50_score = float(coco_eval.stats[1])
        map75_score = float(coco_eval.stats[2])
        return {'map': map_score, 'map50': map50_score, 'map75': map75_score}

    @staticmethod
    def calculate_iou_score(predict_bbox, answer_bbox, match_strategy, iou_threshold, completeness_weight, iou_weight):
        """
        predict_bbox: [{'bbox_2d': [x1, y1, x2, y2], 'label': 'label_name'}, ...]
        answer_bbox: [{'bbox_2d': [x1, y1, x2, y2], 'label': 'label_name'}, ...]
        return dict with iou scores

        advice: pure iou should be used with length penalty
        """

        if match_strategy == "greedy_match_by_iou_max_iou_first":
            matches = greedy_match_by_iou_max_iou_first(predict_bbox, answer_bbox, iou_threshold)
        elif match_strategy == "greedy_match_by_iou_max_label_first":
            matches = greedy_match_by_iou_max_label_first(predict_bbox, answer_bbox, iou_threshold)
        else:
            raise ValueError(f"Invalid match strategy: {match_strategy}")

        # pure iou score
        mean_iou_score = sum(match["iou"] for match in matches) / len(answer_bbox) if matches else 0.0

        # miss penalty score
        miss_rate = (len(answer_bbox) - len(matches)) / len(answer_bbox)
        # Avoid division by zero when predict_bbox is empty
        false_alarm_rate = 0.0 if len(predict_bbox) == 0 else (len(predict_bbox) - len(matches)) / len(predict_bbox)
        completeness_score = 1.0 - (miss_rate + false_alarm_rate) / 2.0

        weighted_iou_score = mean_iou_score * iou_weight + completeness_score * completeness_weight

        return {
            'mean_iou_score': mean_iou_score,
            'completeness_score': completeness_score,
            'precision': 1 - false_alarm_rate,
            'recall': 1 - miss_rate,
            'weighted_iou_score': weighted_iou_score
        }

    def pack_for_iou_score(self, predict_bbox, answer_bbox):
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        results = {
            'greedy_match_by_iou_max_iou_first': {},
            'greedy_match_by_iou_max_label_first': {},
        }

        completeness_weight = self.det_reward_ratio['iou_completeness']
        iou_weight = 1.0 - completeness_weight

        for strategy, strategy_results in results.items():
            for iou_threshold in iou_thresholds:
                iou_scores = self.calculate_iou_score(predict_bbox, answer_bbox, strategy, iou_threshold,
                                                      completeness_weight, iou_weight)
                strategy_results[iou_threshold] = iou_scores

        # calculate mean iou score for each strategy
        for _, strategy_results in results.items():
            strategy_results['average'] = {
                'weighted_iou_score':
                sum(strategy_results[iou_threshold]['weighted_iou_score']
                    for iou_threshold in iou_thresholds) / len(iou_thresholds)
            }
        return results

    def verify_accuracy(self, predict_str: str, solution: str, return_dict=True) -> Dict[str, float]:
        if return_dict:
            result_dict = {
                'final_score': 0.0,
                'size_penalty': 0.0,
                'pure_iou_score_max_iou': 0.0,
                'pure_iou_score_max_label': 0.0,
                'pure_map_score': 0.0,
                'pure_map50_score': 0.0,
                'pure_map75_score': 0.0,
                'weighted_iou_score_max_iou': 0.0,
                'weighted_iou_score_max_label': 0.0,
                'weighted_map_score': 0.0,
                'weighted_map50_score': 0.0,
                'weighted_map75_score': 0.0,
                'report_iou_50_iou_match_pure': 0.0,
                'report_iou_50_iou_match_precision': 0.0,
                'report_iou_50_iou_match_recall': 0.0,
                'report_iou_95_iou_match_pure': 0.0,
                'report_iou_95_iou_match_precision': 0.0,
                'report_iou_95_iou_match_recall': 0.0,
                'report_iou_50_label_match_pure': 0.0,
                'report_iou_50_label_match_precision': 0.0,
                'report_iou_50_label_match_recall': 0.0,
                'report_iou_75_label_match_pure': 0.0,
                'report_iou_75_label_match_precision': 0.0,
                'report_iou_75_label_match_recall': 0.0,
                'report_iou_95_label_match_pure': 0.0,
                'report_iou_95_label_match_precision': 0.0,
                'report_iou_95_label_match_recall': 0.0,
                'report_iou_99_label_match_pure': 0.0,
                'report_iou_99_label_match_precision': 0.0,
                'report_iou_99_label_match_recall': 0.0,
            }

        predict_extract = extract_answer_content(predict_str.strip()).lower()
        answer = extract_answer_content(solution.strip()).lower()

        # for both predict and answer, ignore confidence
        predict_bbox = extract_bbox(predict_extract, ignore_confidence=True)
        answer_bbox = extract_bbox(answer, ignore_confidence=True)

        if answer_bbox is None:
            logger.warning(f"Check GT! No bbox found in ground truth: {solution}")
            return result_dict if return_dict else 0.0

        if predict_bbox is None:
            format_score = self.verify_format(predict_str)
            if format_score == 0.0:
                return result_dict if return_dict else 0.0
            else:
                logger.warning(
                    f"Potential format error! Format score: {format_score}, but no bbox found in predict_str: {predict_str}"
                )
                return result_dict if return_dict else 0.0

        if self.det_verifier_normalized:
            if self.image_grid_thw is not None and len(self.image_grid_thw) > 0:
                predict_bbox = normalize_bbox_by_real_size(
                    pred_bboxes=predict_bbox,
                    input_width=self.image_grid_thw[0][2],
                    input_height=self.image_grid_thw[0][1],
                    normalize_size=1000.0,
                )
            else:
                logger.warning("No image grid thw found in verifier_parm")

        # Handle empty predict_bbox after normalization
        if len(predict_bbox) == 0:
            return result_dict if return_dict else 0.0

        # size_penalty ranges from (0, 1.0], 1.0 when lengths match, approaches 0 as difference increases
        size_penalty_ratio = 0.6
        size_penalty = size_penalty_ratio**(abs(len(predict_bbox) - len(answer_bbox)))

        # ==================iou score=================
        # {'mean_iou_score': mean_iou_score, 'completeness_score': completeness_score, 'precision': 1- false_alarm_rate, 'recall': 1- miss_rate, 'weighted_iou_score': weighted_iou_score}}
        gathered_iou_scores = self.pack_for_iou_score(predict_bbox, answer_bbox)

        report_iou_50_iou_match = gathered_iou_scores['greedy_match_by_iou_max_iou_first'][0.5]
        report_iou_50_iou_match_pure, report_iou_50_iou_match_precision, report_iou_50_iou_match_recall = report_iou_50_iou_match[
            'mean_iou_score'], report_iou_50_iou_match['precision'], report_iou_50_iou_match['recall']
        report_iou_95_iou_match = gathered_iou_scores['greedy_match_by_iou_max_iou_first'][0.95]
        report_iou_95_iou_match_pure, report_iou_95_iou_match_precision, report_iou_95_iou_match_recall = report_iou_95_iou_match[
            'mean_iou_score'], report_iou_95_iou_match['precision'], report_iou_95_iou_match['recall']

        report_iou_50_label_match = gathered_iou_scores['greedy_match_by_iou_max_label_first'][0.5]
        report_iou_50_label_match_pure, report_iou_50_label_match_precision, report_iou_50_label_match_recall = report_iou_50_label_match[
            'mean_iou_score'], report_iou_50_label_match['precision'], report_iou_50_label_match['recall']
        report_iou_75_label_match = gathered_iou_scores['greedy_match_by_iou_max_label_first'][0.75]
        report_iou_75_label_match_pure, report_iou_75_label_match_precision, report_iou_75_label_match_recall = report_iou_75_label_match[
            'mean_iou_score'], report_iou_75_label_match['precision'], report_iou_75_label_match['recall']
        report_iou_95_label_match = gathered_iou_scores['greedy_match_by_iou_max_label_first'][0.95]
        report_iou_95_label_match_pure, report_iou_95_label_match_precision, report_iou_95_label_match_recall = report_iou_95_label_match[
            'mean_iou_score'], report_iou_95_label_match['precision'], report_iou_95_label_match['recall']
        report_iou_99_label_match = gathered_iou_scores['greedy_match_by_iou_max_label_first'][0.99]
        report_iou_99_label_match_pure, report_iou_99_label_match_precision, report_iou_99_label_match_recall = report_iou_99_label_match[
            'mean_iou_score'], report_iou_99_label_match['precision'], report_iou_99_label_match['recall']

        # calculate the report score
        if self.iou_threshold == 'dynamic':
            if self.step_ratio <= 0.1:
                select_iou = 0.85
            elif self.step_ratio <= 0.25:
                select_iou = 0.95
            else:
                select_iou = 0.99
        else:
            select_iou = self.iou_threshold

        pure_iou_score_max_iou = gathered_iou_scores['greedy_match_by_iou_max_iou_first'][select_iou][
            'weighted_iou_score']
        pure_iou_score_max_label = gathered_iou_scores['greedy_match_by_iou_max_label_first'][select_iou][
            'weighted_iou_score']
        weighted_iou_score_max_iou = pure_iou_score_max_iou * size_penalty
        weighted_iou_score_max_label = pure_iou_score_max_label * size_penalty

        # ==================map score=================
        # {'map': map_score, 'map50': map50_score, 'map75': map75_score}
        map_score = self.pack_for_map_score(predict_bbox, answer_bbox)

        pure_map_score = map_score['map']
        pure_map50_score = map_score['map50']
        pure_map75_score = map_score['map75']

        weighted_map_score = pure_map_score * size_penalty
        weighted_map50_score = pure_map50_score * size_penalty
        weighted_map75_score = pure_map75_score * size_penalty

        # Check for zero normalization factor to avoid division by zero
        normalization_factor = (float(self.det_reward_ratio['iou_max_iou_first'])
                                + float(self.det_reward_ratio['iou_max_label_first'])
                                + float(self.det_reward_ratio['map']) + float(self.det_reward_ratio['map50'])
                                + float(self.det_reward_ratio['map75']))

        logger.info('det_reward_ratio: %s', self.det_reward_ratio)
        # map75 is not used in the final score
        if normalization_factor > 0:
            final_det_score = (weighted_iou_score_max_iou * float(self.det_reward_ratio['iou_max_iou_first'])
                               + weighted_iou_score_max_label * float(self.det_reward_ratio['iou_max_label_first'])
                               + weighted_map_score * float(self.det_reward_ratio['map'])
                               + weighted_map50_score * float(self.det_reward_ratio['map50'])
                               + weighted_map75_score * float(self.det_reward_ratio['map75']))
            final_det_score /= normalization_factor
        else:
            logger.error("Normalization factor is zero, set final score to 0.0")
            final_det_score = 0.0

        return {
            'final_score': final_det_score,
            'size_penalty': size_penalty,
            'pure_iou_score_max_iou': pure_iou_score_max_iou,
            'pure_iou_score_max_label': pure_iou_score_max_label,
            'pure_map_score': pure_map_score,
            'pure_map50_score': pure_map50_score,
            'pure_map75_score': pure_map75_score,
            'weighted_iou_score_max_iou': weighted_iou_score_max_iou,
            'weighted_iou_score_max_label': weighted_iou_score_max_label,
            'weighted_map_score': weighted_map_score,
            'weighted_map50_score': weighted_map50_score,
            'weighted_map75_score': weighted_map75_score,
            'report_iou_50_iou_match_pure': report_iou_50_iou_match_pure,
            'report_iou_50_iou_match_precision': report_iou_50_iou_match_precision,
            'report_iou_50_iou_match_recall': report_iou_50_iou_match_recall,
            'report_iou_95_iou_match_pure': report_iou_95_iou_match_pure,
            'report_iou_95_iou_match_precision': report_iou_95_iou_match_precision,
            'report_iou_95_iou_match_recall': report_iou_95_iou_match_recall,
            'report_iou_50_label_match_pure': report_iou_50_label_match_pure,
            'report_iou_50_label_match_precision': report_iou_50_label_match_precision,
            'report_iou_50_label_match_recall': report_iou_50_label_match_recall,
            'report_iou_75_label_match_pure': report_iou_75_label_match_pure,
            'report_iou_75_label_match_precision': report_iou_75_label_match_precision,
            'report_iou_75_label_match_recall': report_iou_75_label_match_recall,
            'report_iou_95_label_match_pure': report_iou_95_label_match_pure,
            'report_iou_95_label_match_precision': report_iou_95_label_match_precision,
            'report_iou_95_label_match_recall': report_iou_95_label_match_recall,
            'report_iou_99_label_match_pure': report_iou_99_label_match_pure,
            'report_iou_99_label_match_precision': report_iou_99_label_match_precision,
            'report_iou_99_label_match_recall': report_iou_99_label_match_recall,
        } if return_dict else final_det_score


def verify_for_test(predict_str: Any, solution: Any, verifier_name: str, **kwargs) -> VerifyResult:
    """
    Convenience function to run verification using a specified verifier.
    
    Args:
        predict_str: The input to verify
        solution: The solution to verify against
        verifier_name: Name of the verifier to use
        **kwargs: Additional arguments to pass to the verifier
        
    Returns:
        Verification results
    """
    if verifier_name in set(Verifier.list_verifiers()):
        verifier_cls = Verifier.get(verifier_name)
        if not verifier_cls:
            msg = f"Verifier '{verifier_name}' not found"
            return {"success": False, "score": 0.0, "details": msg}
    else:
        msg = f"Verifier '{verifier_name}' not found"
        raise ValueError(msg)

    verifier = verifier_cls(**kwargs)
    return verifier.verify_accuracy(predict_str, solution)


if __name__ == "__main__":
    # Example of how to use the verification system

    # Test math verifier
    math_result = verify_for_test("42", "42", "mathverify")
    assert math_result == 0.0

    math_result = verify_for_test("42", "\\boxed{42}", "mathverify")
    assert math_result == 0.0

    math_result = verify_for_test("\\boxed{42}", "42", "mathverify")
    assert math_result == 1.0

    math_result = verify_for_test("\\boxed{42}", "\\boxed{42}", "mathverify")
    assert math_result == 1.0

    # test for detection iou
    solution = "<answer> [{'bbox_2d': [386, 183, 973, 758], 'label': 'dog'}] </answer>"
    prediction = """<think> The image shows a dog lying on a bed next to a laptop. The dog appears to be the main subject of the image, and it is clearly visible. The bounding box should be drawn around the dog, capturing its entire body. The laptop is also present, but it is not the focus of the image. The bed and other objects in the background are not relevant to the detection of the dog. The dog's position and size suggest that the bounding box should be drawn around its entire body, from its head to its tail. The confidence score is high because the dog is clearly visible and the bounding box can be drawn with a high degree of accuracy. </think>\n<answer> [{'bbox_2d': [240,76,600,320], 'label': 'dog', 'confidence': 0.95}] </answer>"""
    kwargs = {'image_grid_thw': [[1, 420, 616]], 'det_verifier_normalized': True}
    print(verify_for_test(prediction, solution, "detection_iou_acc", **kwargs))
