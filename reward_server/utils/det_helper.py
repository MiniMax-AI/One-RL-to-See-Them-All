import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_bbox(bbox_str, ignore_confidence=True):
    """Extract bounding box information from a string.
            
            Parses a string containing bbox data in the format:
            [{'bbox_2d':[x1,y1,x2,y2],'confidence':number,'label':label_name},...]
            
            Returns:
                List of dictionaries containing bbox information or None if parsing fails
            """
    if not bbox_str or not bbox_str.strip():
        return None

    bbox_str = bbox_str.replace("[[", '[').replace("]]", ']')
    if not bbox_str.endswith("]"):
        bbox_str = bbox_str.rsplit("},", 1)[0] + "}]"
    bbox_str = bbox_str.replace("'", '"').replace("\\", '')
    try:
        bbox_list = json.loads(bbox_str)
        if not isinstance(bbox_list, list):
            return None

        # Validate that each item has the required fields and correct types
        filtered_bbox_list = []
        seen = set()
        for item in bbox_list:

            if (isinstance(item, dict) and 'bbox_2d' in item and (ignore_confidence or 'confidence' in item)
                    and 'label' in item and isinstance(item['bbox_2d'], list) and len(item['bbox_2d']) == 4
                    and all(isinstance(coord, (int, float)) for coord in item['bbox_2d'])
                    and item['bbox_2d'][0] <= item['bbox_2d'][2] and item['bbox_2d'][1] <= item['bbox_2d'][3]
                    and (ignore_confidence or isinstance(item['confidence'],
                                                         (int, float))) and isinstance(item['label'], str)):

                # Remove duplicates based on bbox position
                position_tuple = tuple(item['bbox_2d'])
                if position_tuple not in seen:
                    seen.add(position_tuple)
                    filtered_bbox_list.append(item)

        if len(filtered_bbox_list) == 0:
            return None
        return filtered_bbox_list

    except Exception:
        return None


def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: List or tuple of 4 values [x1, y1, x2, y2] representing first box
        box2: List or tuple of 4 values [x1, y1, x2, y2] representing second box
        
    Returns:
        float: IoU value between the two boxes (0.0 to 1.0)
    """
    # Find coordinates of intersection
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    # Check if boxes overlap
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    # Calculate intersection area
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area
    union_area = box1_area + box2_area - inter_area

    # Return IoU
    return float(inter_area) / union_area


def iou_score_by_greedy_match_by_confidence(gt_bboxes, pred_bboxes, iou_threshold):
    pred_bboxes_sorted = sorted(pred_bboxes, key=lambda x: x['confidence'], reverse=True)
    iou_results = []
    matched_gt_bbox_indices = set()
    for pred_bbox in pred_bboxes_sorted:
        matched_gt_bbox = -1
        best_iou = 0
        for i, gt_bbox in enumerate(gt_bboxes):
            if i not in matched_gt_bbox_indices:
                iou = compute_iou(gt_bbox['bbox_2d'], pred_bbox['bbox_2d'])
                if iou > best_iou:
                    best_iou = iou
                    matched_gt_bbox = i
        if best_iou > iou_threshold:
            iou_results.append(best_iou)
            matched_gt_bbox_indices.add(matched_gt_bbox)
    return iou_results


def convert_bbox_to_coco_format(bbox):
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1], (x2 - x1) * (y2 - y1)


def normalize_bbox_by_real_size(pred_bboxes, input_width, input_height, normalize_size=1000.0):
    if pred_bboxes is None:
        return None

    for idx, pred_bbox in enumerate(pred_bboxes):
        try:
            x1, y1, x2, y2 = pred_bbox['bbox_2d']

            # Calculate normalized coordinates
            x1_norm = int(x1 / input_width * normalize_size)
            y1_norm = int(y1 / input_height * normalize_size)
            x2_norm = int(x2 / input_width * normalize_size)
            y2_norm = int(y2 / input_height * normalize_size)

            pred_bbox['bbox_2d'] = [x1_norm, y1_norm, x2_norm, y2_norm]
        except (KeyError, ValueError, TypeError) as e:
            # Handle case where bbox_2d is missing or malformed
            logger.warning(f"Error normalizing bbox:  {e}")
            continue

    return pred_bboxes


def greedy_match_by_iou_max_iou_first(predict_bbox, answer_bbox, iou_threshold):
    """
    Use IoU as metric to perform greedy match.
    Find the maximum IoU in predict bbox for each solution (answer) bbox.
    Check if the label is correct and add IoU to the final score if correct.
    """
    iou_matrix = np.zeros((len(predict_bbox), len(answer_bbox)))

    for i in range(len(predict_bbox)):
        for j in range(len(answer_bbox)):
            iou_matrix[i][j] = compute_iou(predict_bbox[i]['bbox_2d'], answer_bbox[j]['bbox_2d'])

    # Find the maximum IoU in predict bbox for each solution bbox globally
    matches = []  # Store matched pairs of predicted and ground truth boxes
    unmatched_pred = list(range(len(predict_bbox)))  # Unmatched predicted boxes
    unmatched_gt = list(range(len(answer_bbox)))  # Unmatched ground truth boxes

    # Greedy matching: find the best match for each predicted box
    while unmatched_pred and unmatched_gt:
        # Find the maximum IoU
        max_iou, pred_idx, gt_idx = -1, -1, -1

        for pred_idx in unmatched_pred:
            for gt_idx in unmatched_gt:
                curr_iou = iou_matrix[pred_idx][gt_idx]
                #  find the largest iou in the unmatched list
                if curr_iou > max_iou:
                    max_iou, pred_idx, gt_idx = curr_iou, pred_idx, gt_idx

        # Stop matching if the maximum IoU is below the threshold
        if max_iou < iou_threshold:
            break

        # Record matching results
        pred_label = predict_bbox[pred_idx]["label"].lower()
        gt_label = answer_bbox[gt_idx]["label"].lower()
        iou_score = max_iou if pred_label == gt_label else 0.0

        matches.append({"pred_idx": pred_idx, "gt_idx": gt_idx, "iou": iou_score})

        # Remove matched boxes from the unmatched list
        unmatched_pred.remove(pred_idx)
        unmatched_gt.remove(gt_idx)

    return matches


def greedy_match_by_iou_max_label_first(predict_bbox, answer_bbox, iou_threshold):
    """
    Use IoU as metric to perform greedy match.
    First find the matched labels for both predict and answer.
    Then use max IoU to find the best match.
    """
    matches = []
    matched_pred_idx = set()
    for gt_idx, gt_bbox in enumerate(answer_bbox):
        label = gt_bbox["label"].lower()

        # Find the potential matches and IoU scores
        potential_matches, iou_scores = [], []
        for pred_idx, pred_bbox in enumerate(predict_bbox):
            if pred_idx in matched_pred_idx:
                continue

            pred_label = pred_bbox["label"].lower()
            if pred_label == label:
                iou = compute_iou(pred_bbox["bbox_2d"], gt_bbox["bbox_2d"])
                potential_matches.append(pred_idx)
                iou_scores.append(iou)

        if len(potential_matches) == 0:
            continue

        max_iou, max_iou_idx = np.max(iou_scores), np.argmax(iou_scores)
        if max_iou >= iou_threshold:
            matches.append({"pred_idx": potential_matches[max_iou_idx], "gt_idx": gt_idx, "iou": max_iou})
            matched_pred_idx.add(potential_matches[max_iou_idx])

    return matches


if __name__ == "__main__":

    def test_greedy_match_by_iou_max_iou_first():
        # Test cases for greedy_match_by_iou_max_iou_first function
        test_cases = [
            # Case 1: Perfect match - same labels, high IoU
            {
                "predict_bbox": [{
                    "label": "car",
                    "bbox_2d": [100, 100, 200, 200]
                }, {
                    "label": "person",
                    "bbox_2d": [300, 300, 400, 400]
                }],
                "answer_bbox": [{
                    "label": "car",
                    "bbox_2d": [110, 110, 210, 210]
                }, {
                    "label": "person",
                    "bbox_2d": [305, 305, 405, 405]
                }],
                "iou_threshold":
                0.5
            },
            # Case 2: No matches - IoU below threshold
            {
                "predict_bbox": [{
                    "label": "car",
                    "bbox_2d": [100, 100, 200, 200]
                }],
                "answer_bbox": [{
                    "label": "car",
                    "bbox_2d": [250, 250, 350, 350]
                }],
                "iou_threshold": 0.5
            },
            # Case 3: Label mismatch but high IoU
            {
                "predict_bbox": [{
                    "label": "car",
                    "bbox_2d": [100, 100, 200, 200]
                }],
                "answer_bbox": [{
                    "label": "truck",
                    "bbox_2d": [110, 110, 210, 210]
                }],
                "iou_threshold": 0.5
            },
            # Case 4: Multiple potential matches with different IoUs
            {
                "predict_bbox": [{
                    "label": "car",
                    "bbox_2d": [100, 100, 200, 200]
                }, {
                    "label": "car",
                    "bbox_2d": [150, 150, 250, 250]
                }],
                "answer_bbox": [{
                    "label": "car",
                    "bbox_2d": [120, 120, 220, 220]
                }],
                "iou_threshold":
                0.3
            },
            # Case 5: Empty lists
            {
                "predict_bbox": [],
                "answer_bbox": [{
                    "label": "car",
                    "bbox_2d": [100, 100, 200, 200]
                }],
                "iou_threshold": 0.5
            }
        ]

        print("\nTesting greedy_match_by_iou_max_iou_first function:")
        for i, test_case in enumerate(test_cases):
            predict_bbox = test_case["predict_bbox"]
            answer_bbox = test_case["answer_bbox"]
            iou_threshold = test_case["iou_threshold"]

            result = greedy_match_by_iou_max_iou_first(predict_bbox, answer_bbox, iou_threshold)

            print(f"Test case {i+1}:")
            print(f"  Predict bbox: {predict_bbox}")
            print(f"  Answer bbox: {answer_bbox}")
            print(f"  IoU threshold: {iou_threshold}")
            print(f"  Result: {result}")
            print()

    def test_greedy_match_by_iou_max_label_first():
        # Test cases for greedy_match_by_iou_max_label_first function
        test_cases = [
            # Case 1: Perfect match - same labels, high IoU
            {
                "predict_bbox": [{
                    "label": "car",
                    "bbox_2d": [100, 100, 200, 200]
                }, {
                    "label": "person",
                    "bbox_2d": [300, 300, 400, 400]
                }],
                "answer_bbox": [{
                    "label": "car",
                    "bbox_2d": [110, 110, 210, 210]
                }, {
                    "label": "person",
                    "bbox_2d": [305, 305, 405, 405]
                }],
                "iou_threshold":
                0.5
            },
            # Case 2: No matches - IoU below threshold
            {
                "predict_bbox": [{
                    "label": "car",
                    "bbox_2d": [100, 100, 200, 200]
                }],
                "answer_bbox": [{
                    "label": "car",
                    "bbox_2d": [250, 250, 350, 350]
                }],
                "iou_threshold": 0.5
            },
            # Case 3: Label mismatch but high IoU
            {
                "predict_bbox": [{
                    "label": "car",
                    "bbox_2d": [100, 100, 200, 200]
                }],
                "answer_bbox": [{
                    "label": "truck",
                    "bbox_2d": [110, 110, 210, 210]
                }],
                "iou_threshold": 0.5
            },
            # Case 4: Multiple potential matches with same label
            {
                "predict_bbox": [{
                    "label": "car",
                    "bbox_2d": [100, 100, 200, 200]
                }, {
                    "label": "car",
                    "bbox_2d": [150, 150, 250, 250]
                }],
                "answer_bbox": [{
                    "label": "car",
                    "bbox_2d": [120, 120, 220, 220]
                }],
                "iou_threshold":
                0.3
            },
            # Case 5: Case highlighting the difference between the two algorithms
            {
                "predict_bbox": [{
                    "label": "car",
                    "bbox_2d": [100, 100, 200, 200]
                }, {
                    "label": "person",
                    "bbox_2d": [300, 300, 400, 400]
                }, {
                    "label": "person",
                    "bbox_2d": [310, 310, 410, 410]
                }],
                "answer_bbox": [{
                    "label": "car",
                    "bbox_2d": [150, 150, 250, 250]
                }, {
                    "label": "person",
                    "bbox_2d": [305, 305, 405, 405]
                }],
                "iou_threshold":
                0.1
            }
        ]

        print("\nTesting greedy_match_by_iou_max_label_first function:")
        for i, test_case in enumerate(test_cases):
            predict_bbox = test_case["predict_bbox"]
            answer_bbox = test_case["answer_bbox"]
            iou_threshold = test_case["iou_threshold"]

            result = greedy_match_by_iou_max_label_first(predict_bbox, answer_bbox, iou_threshold)

            print(f"Test case {i+1}:")
            print(f"  Predict bbox: {predict_bbox}")
            print(f"  Answer bbox: {answer_bbox}")
            print(f"  IoU threshold: {iou_threshold}")
            print(f"  Result: {result}")
            print()

    def test_iou():
        # Test cases for compute_iou function
        test_cases = [
            # Identical boxes
            ([100, 100, 200, 200], [100, 100, 200, 200]),
            # Completely separate boxes
            ([100, 100, 200, 200], [300, 300, 400, 400]),
            # Partial overlap
            ([100, 100, 300, 300], [200, 200, 400, 400]),
            # One box inside another
            ([100, 100, 400, 400], [150, 150, 350, 350]),
            # Overlap on edge
            ([100, 100, 200, 200], [200, 100, 300, 200]),
            # Overlap on corner
            ([100, 100, 200, 200], [150, 150, 250, 250])
        ]

        print("Testing compute_iou function with various bounding box configurations:")
        for i, (box1, box2) in enumerate(test_cases):
            compute_iou_result = compute_iou(box1, box2)
            print(f"Test case {i+1}:")
            print(f"  Box 1: {box1}")
            print(f"  Box 2: {box2}")
            print(f"  compute_iou result: {compute_iou_result:.4f}")
            print()

    # test_iou()
    test_greedy_match_by_iou_max_iou_first()
    test_greedy_match_by_iou_max_label_first()
