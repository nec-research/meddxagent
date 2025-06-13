"""
Metrics module for evaluating differential diagnosis predictions.

Two matching strategies are available:

1. **Strict Matching (default)**: Exact string matching
   - "Pneumonia" matches only "Pneumonia"
   
2. **Weak Matching**: Substring matching (case-insensitive)
   - "Pneumonia" matches "Bacterial Pneumonia", "Viral Pneumonia", etc.
   
To change strategy, modify `WEAK_METRICS` in `ddxdriver/run_ddxdriver.py`.
"""

from typing import List, Dict, Union, Callable

from ddxdriver.utils import Patient
from ddxdriver.logger import log

def strict_match(a: str, b: str) -> bool:
    """Exact string matching between predicted and ground truth diagnosis."""
    return a == b

def weak_match(a: str, b: str) -> bool:
    """Substring matching where ground truth is found within predicted diagnosis (case-insensitive)."""
    return b.lower() in a.lower()

def _calculate_ddr(pred_ddx: List[str], gt_ddx: List[str], match_fn: Callable[[str, str], bool]) -> float:
    """Calculates the Differential Diagnosis Recall (DDR)."""
    true_positives = sum(1 for gt in gt_ddx if any(match_fn(pred, gt) for pred in pred_ddx))
    total_ground_truth = len(gt_ddx)
    return true_positives / total_ground_truth if total_ground_truth else 0.0


def _calculate_ddp(pred_ddx: List[str], gt_ddx: List[str], match_fn: Callable[[str, str], bool]) -> float:
    """Calculates the Differential Diagnosis Precision (DDP)."""
    true_positives = sum(1 for pred in pred_ddx if any(match_fn(pred, gt) for gt in gt_ddx))
    total_predicted = len(pred_ddx)
    return true_positives / total_predicted if total_predicted else 0.0


def _calculate_ddf1(ddr: float, ddp: float) -> float:
    """Calculates the Differential Diagnosis F1 (DDF1) score."""
    if ddr + ddp == 0:
        return 0.0
    return 2 * (ddr * ddp) / (ddr + ddp)


def _calculate_gtpa_k_ddx(pred_ddx: List[str], gt_pathology: str, k: int, match_fn: Callable[[str, str], bool]) -> float:
    """
    Calculates the Ground Truth Pathology Accuracy at k (GTPA@k).
    """
    if gt_pathology is None:
        return 0.0
    return 1.0 if any(match_fn(pred, gt_pathology) for pred in pred_ddx[:k]) else 0.0


def _calculate_gt_pathology_rank(
    ddx: List[str], gt_pathology: str, ddx_fixed_length: int, match_fn: Callable[[str, str], bool]
) -> Union[int, str]:
    """
    Returns the 1-based index of the target string in the list of strings.
    The ddx should be a fixed length (ddx_fixed_length)
    For non-fixed lengths, will either predict the max (min(1 + length of list, 20)), to be more consistent with fixed lengths of 10 (used for Rarebench and iCraftMD)
    """
    if not ddx or not gt_pathology:
        raise ValueError("One of ddx or gt_pathology is None when it shouldn't be")
    if ddx_fixed_length:
        if len(ddx) != ddx_fixed_length:
            raise ValueError("Length of predicted ddx doesn't match ddx_fixed_length")
        default_rank = min(ddx_fixed_length + 1, 20)
    else:
        default_rank = min(max(len(ddx) + 1, 11), 20)
    for i, diagnosis in enumerate(ddx):
        if match_fn(diagnosis, gt_pathology):
            return min(i + 1, 20)
    return default_rank


def get_progress(intermediate_ddx_predictions: List[List[str]], gt_pathology, match_fn: Callable[[str, str], bool]) -> List[int]:
    """
    Calculate the progress of a ground truth pathology in a list of ranked differential diagnoses.

    Parameters:
    gt_pathology (str): The ground truth pathology to track.
    intermediate_ddx_predictions (List[List[str]]): A list of ranked differential diagnosis lists at successive steps.

    Returns:
    List[int]: A list of integers representing the number of positions the ground truth moved up (positive) or down (negative) between successive lists.
    """
    if not intermediate_ddx_predictions or not gt_pathology:
        raise ValueError("One of the parameters is None when it shouldn't be")

    def find_rank(ddx_list: List[str], pathology: str) -> int:
        """
        Find the rank of a pathology in a differential diagnosis list.
        If the pathology is not present, return a rank one greater than the length of the list (0-index being the length of the list)
        """
        for i, diagnosis in enumerate(ddx_list):
            if match_fn(diagnosis, pathology):
                return i
        return len(ddx_list)

    if not intermediate_ddx_predictions:
        return []

    # Initialize the previous rank for the first list
    previous_rank = find_rank(intermediate_ddx_predictions[0], gt_pathology)
    progress = []

    # Iterate through each successive list and calculate the movement
    for current_ddx in intermediate_ddx_predictions[1:]:
        current_rank = find_rank(current_ddx, gt_pathology)
        movement = previous_rank - current_rank  # Positive if it moved up, negative if down
        progress.append(movement)
        previous_rank = current_rank  # Update the previous rank to the current one

    return progress


def get_metrics(results: dict, weak: bool = False) -> Dict[str, float]:
    """
    Calculate and return the metrics for this benchmark.
    Set weak=True for weak (substring) matching, False for strict (exact) matching.
    Params:
        results: a dictionary of results in this format
         {
                "patient" (dict): dictionary of patient packed attributes (from Patient in ddxdriver.utils)
                "ddx_fixed_length" (Union[int, None]): fixed length for ddx
                "final_ddx_prediction" (List[str]): list of ddx prediction,
                "intermediate_ddx_predictions" (List[List[str]]): list of intermediate ddx predictions,
        }
        Each patient dict is formatted as:
        {
            "patient_id": self.patient_id,
            "patient_initial_info": self.patient_initial_info,
            "gt_pathology": self.gt_pathology,
            "patient_profile": self.patient_profile,
            "gt_ddx": self.gt_ddx,
        }
    Returns:
        Dictionary with keys as metrics and float values
    """
    match_fn = weak_match if weak else strict_match
    if not results:
        log.warning("Warning, trying to call get_metrics with no results, returning empty dict...")
        return {}

    metrics = {
        "GTPA@1": [],
        "GTPA@3": [],
        "GTPA@5": [],
        "GTPA@10": [],
        "Average Rank": [],
        "DDR": [],
        "DDP": [],
        "DDF1": [],
    }

    for i, result in enumerate(results):
        if not (patient := result.get("patient")) or not (
            pred_ddx := result.get("final_ddx_prediction")
        ):
            log.warning(f"Patient {i} metrics could not be generated")
            continue

        ddx_fixed_length = result.get("ddx_fixed_length")
        gt_pathology = patient.get("gt_pathology")
        gt_ddx = patient.get("gt_ddx")

        metrics["GTPA@1"].append(_calculate_gtpa_k_ddx(pred_ddx, gt_pathology, 1, match_fn))
        metrics["GTPA@3"].append(_calculate_gtpa_k_ddx(pred_ddx, gt_pathology, 3, match_fn))
        metrics["GTPA@5"].append(_calculate_gtpa_k_ddx(pred_ddx, gt_pathology, 5, match_fn))
        metrics["GTPA@10"].append(_calculate_gtpa_k_ddx(pred_ddx, gt_pathology, 10, match_fn))

        metrics["Average Rank"].append(
            _calculate_gt_pathology_rank(pred_ddx, gt_pathology, ddx_fixed_length, match_fn)
        )

        if gt_ddx:
            ddr = _calculate_ddr(pred_ddx, gt_ddx, match_fn)
            ddp = _calculate_ddp(pred_ddx, gt_ddx, match_fn)
            metrics["DDR"].append(ddr)
            metrics["DDP"].append(ddp)
            metrics["DDF1"].append(_calculate_ddf1(ddr, ddp))

    # Filter out empty scores
    metrics = {key: scores for key, scores in metrics.items() if len(scores) > 0}

    metrics = {key: round(sum(scores) / len(scores), 2) for key, scores in metrics.items()}
    return metrics


def get_intermediate_metrics(results: dict, weak: bool = False) -> Dict[str, Union[List, float]]:
    """
    Given a list of the intermediate ddx predictions from different examples,
    calculates each of the examples progress (movements of ground truth pathology), and the average pathology
    Set weak=True for weak (substring) matching, False for strict (exact) matching.
    Params:
        results: a dictionary of results in this format
         {
                "patient" (dict): dictionary of patient packed attributes (from Patient in ddxdriver.utils)
                "ddx_fixed_length" (Union[int, None]): fixed length for ddx
                "final_ddx_prediction" (List[str]): list of ddx prediction,
                "intermediate_ddx_predictions" (List[List[str]]): list of intermediate ddx predictions,
        }
        Each patient dict is formatted as:
        {
            "patient_id": self.patient_id,
            "patient_initial_info": self.patient_initial_info,
            "gt_pathology": self.gt_pathology,
            "patient_profile": self.patient_profile,
            "gt_ddx": self.gt_ddx,

    Returns:
        Dictionary:
        {
         "Progress": a list of progress lists for each example
        "Average Progress": the average progress for all the examples
        }
    """
    match_fn = weak_match if weak else strict_match
    if not results:
        log.warning("Warning, trying to call get_metrics with no results, returning empty dict...")
        return {}

    metrics = {
        "Progress": [],
        "Average Progress": [],
    }

    for i, result in enumerate(results):
        if not (patient := result.get("patient")) or not (
            intermediate_ddxs := result.get("intermediate_ddx_predictions")
        ):
            log.warning(f"Patient {i} metrics could not be generated")
            continue

        gt_pathology = patient.get("gt_pathology")

        progress = get_progress(
            intermediate_ddx_predictions=intermediate_ddxs, gt_pathology=gt_pathology, match_fn=match_fn
        )
        if progress:  # Avoid division by zero if progress is empty
            average_progress = sum(progress) / len(progress)
        else:
            average_progress = 0
        metrics["Progress"].append(progress)
        metrics["Average Progress"].append(average_progress)

    if metrics["Average Progress"]:  # Check if the list is not empty
        overall_average_progress = sum(metrics["Average Progress"]) / len(
            metrics["Average Progress"]
        )
    else:
        overall_average_progress = 0
    metrics["Average Progress"] = overall_average_progress

    return metrics
