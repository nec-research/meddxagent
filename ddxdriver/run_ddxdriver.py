from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Dict, Tuple, Any, Union
import yaml
import random
from datasets import Dataset
from copy import deepcopy
import traceback

from ddxdriver.ddxdrivers import DDxDriver, init_ddxdriver
from ddxdriver.benchmarks import Bench, init_bench
from ddxdriver.benchmarks.metrics import get_metrics, get_intermediate_metrics
from ddxdriver.diagnosis_agents import init_diagnosis_agent
from ddxdriver.history_taking_agents import init_history_taking_agent
from ddxdriver.patient_agents import init_patient_agent
from ddxdriver.rag_agents import init_rag_agent
from ddxdriver.utils import ddx_list_to_string, Patient, Constants, find_project_root
from ddxdriver.logger import log, enable_logging, set_file_handler, log_json_data

DEFAULT_EXPERIMENT_PATH = find_project_root() / "experiments/example_experiment"
WEAK_METRICS = False

def setup_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--bench_cfg",
        type=Path,
        required=True,
        help="Path to the bench's config yaml file.",
    )
    parser.add_argument(
        "--ddxdriver_cfg",
        type=Path,
        required=True,
        help="Path to the driver's config yaml file",
    )
    parser.add_argument(
        "--diagnosis_agent_cfg",
        type=Path,
        required=False,
        help="Path to the diagnosis agent's config yaml file",
    )
    parser.add_argument(
        "--history_taking_agent_cfg",
        type=Path,
        required=False,
        help="Path to the history taking agent's config yaml file",
    )
    parser.add_argument(
        "--patient_agent_cfg",
        type=Path,
        required=False,
        help="Path to the patient agent's config yaml file",
    )
    parser.add_argument(
        "--rag_agent_cfg",
        type=Path,
        required=False,
        help="Path to the rag agent's config yaml file",
    )

    return parser.parse_args()


def run_ddxdriver(
    bench_cfg: dict,
    ddxdriver_cfg: dict,
    diagnosis_agent_cfg: dict | None = None,
    history_taking_agent_cfg: dict | None = None,
    patient_agent_cfg: dict | None = None,
    rag_agent_cfg: dict | None = None,
    experiment_folder: Union[str, Path] = DEFAULT_EXPERIMENT_PATH,
):
    """
    Reads configuration files specified in scripts/run.sh, and runs the ddxdriver given these configuration files
    Logs results to medicalagent/experiments/{output_filename}.json
    Will catch all setup errors for functions in the try block
    """
    experiment_folder = Path(experiment_folder)
    run_file_logging_path = experiment_folder / "run.log"
    set_file_handler(run_file_logging_path)
    enable_logging(console_logging=True, file_logging=True)

    if not bench_cfg or not ddxdriver_cfg:
        raise ValueError("bench_cfg or ddxdriver_cfg is None, cannot run ddxdriver without.")
    try:
        seed = Constants.SEED.value
        ddxdriver_cfg["config"]["seed"] = seed

        # Load benchmark
        bench = init_bench(bench_cfg=bench_cfg)
        if not bench:
            raise ValueError("No benchmark, can't run ddxdriver. Returning early...")

        num_patients = bench_cfg.get("num_patients", 1)
        patients = sample_dataset(bench=bench, num_patients=num_patients, seed=seed)
        if not patients:
            raise ValueError("Could not sample patients from benchmark. Returning early...")

        # print(diagnosis_agent_cfg)
        # exit()
        diagnosis_agent = (
            init_diagnosis_agent(
                class_name=diagnosis_agent_cfg["class_name"],
                diagnosis_agent_cfg=diagnosis_agent_cfg["config"],
            )
            if diagnosis_agent_cfg
            else None
        )

        history_taking_agent = (
            init_history_taking_agent(
                history_taking_agent_cfg["class_name"],
                history_taking_agent_cfg=history_taking_agent_cfg["config"],
            )
            if history_taking_agent_cfg
            else None
        )
        patient_agent = (
            init_patient_agent(
                patient_agent_cfg["class_name"], patient_agent_cfg=patient_agent_cfg["config"]
            )
            if patient_agent_cfg
            else None
        )
        rag_agent = (
            init_rag_agent(rag_agent_cfg["class_name"], rag_agent_cfg["config"])
            if rag_agent_cfg
            else None
        )

        ddxdriver_args = {
            "ddxdriver_cfg": ddxdriver_cfg["config"],
            "bench": bench,
            "diagnosis_agent": diagnosis_agent,
            "history_taking_agent": history_taking_agent,
            "patient_agent": patient_agent,
            "rag_agent": rag_agent,
        }

        # Initialize ddxdriver and pass in agents
        ddxdriver = init_ddxdriver(ddxdriver_cfg["class_name"], **ddxdriver_args)

        # Running experiment, which will log results
        results, num_successful_examples = run_experiment(
            patients=patients,
            ddxdriver=ddxdriver,
            patient_logs_folder=experiment_folder / "patient_logs",
        )

        # Resetting because was overwritten in experiment
        set_file_handler(run_file_logging_path)

        average_metrics = get_metrics(results=results, weak=WEAK_METRICS    )
        average_intermediate_metrics = get_intermediate_metrics(results=results, weak=WEAK_METRICS)
        
        json_data = {
            "Percent Successful Examples": num_successful_examples / len(patients),
            "Average Metrics": average_metrics,
            "Average Intermediate Metrics": average_intermediate_metrics,
            "Results": results,
        }
        log_json_data(json_data=json_data, file_path=experiment_folder / "results.json")

    except Exception as e:
        set_file_handler(run_file_logging_path)
        tb = traceback.format_exc()
        # Log exception (full traceback)
        log.error(f"run_ddxdriver() failed due following error:\n{e}\nTraceback:\n{tb}")


def sample_dataset(
    bench: Bench,
    num_patients: int = 0,
    seed: int | None = None,
) -> List[Patient]:
    patients = bench.patients

    if not patients:
        raise ValueError("Trying to sample but bench.patients is None")
    if seed is not None:  # Shuffle datasets
        random.seed(seed)
        random.shuffle(patients)
    if num_patients:
        patients_sample = patients[:num_patients]
        if len(patients_sample) < num_patients:
            log.warning(
                f"Warning, tried to sample {num_patients} patients but only found {len(patients_sample)}"
            )

    return patients_sample


def run_experiment(
    patients: List[Patient],
    ddxdriver: DDxDriver,
    patient_logs_folder: Union[str, Path] = DEFAULT_EXPERIMENT_PATH / "patient_logs",
):

    if not patients or not ddxdriver:
        raise ValueError("Trying to run experiment with no input patients or no ddxdriver")
    if not all(isinstance(patient, Patient) for patient in patients):
        raise TypeError("Not all elements in patients are Patient objects")

    results = []
    num_successful_examples = 0
    for time_step, patient in enumerate(patients, start=1):
        set_file_handler(patient_logs_folder / f"patient_{time_step}.log")
        log.info(f"\nPATIENT {time_step}\n\nGround truth pathology: {patient.gt_pathology}")

        if patient.patient_profile:
            log.info(f"\nPatient Profile (inputted):\n{patient.patient_profile}\n")

        # Call driver to run and predict diagnosis
        try:
            ddxdriver(patient)
        except Exception as e:
            tb = traceback.format_exc()
            # Log exception (full traceback)
            log.error(
                f"ddxdriver() failed due following error, skipping patient example in evaluation:{e}\nTraceback:\n{tb}"
            )
            continue

        # Gathering data
        gt_pathology = patient.gt_pathology
        gt_ddx = patient.gt_ddx
        final_ddx_rationale = ddxdriver.get_final_ddx_rationale()
        final_ddx = ddxdriver.get_final_ddx()
        intermediate_ddxs = ddxdriver.pred_ddxs
        dialogue_history = ddxdriver.get_dialogue_history()
        final_rag_content = ddxdriver.get_final_rag_content()

        if not final_ddx:
            log.error("Example did not generate a final differential diagnosis...")
        else:
            num_successful_examples += 1
            log.info("Patient example successfully generated, printing results...\n")

            if dialogue_history:
                log.info("Dialogue history\n" + dialogue_history + "\n")

            if final_rag_content:
                log.info("Final rag content\n" + final_rag_content + "\n")

            log.info("Ground truth pathology:\n" + gt_pathology + "\n")
            if gt_ddx:
                log.info("Ground truth ddx:\n" + ddx_list_to_string(gt_ddx) + "\n")

            log.info("Final Ranked Differential Diagnosis:\n" + ddx_list_to_string(final_ddx) + "\n")

            if final_ddx_rationale:
                log.info("Final ddx rationale\n" + final_ddx_rationale + "\n")

        results.append(
            {
                "patient": patient.pack_attributes(),
                "ddx_fixed_length": ddxdriver.bench.DDX_LENGTH,
                "final_ddx_prediction": deepcopy(final_ddx),
                "intermediate_ddx_predictions": deepcopy(intermediate_ddxs),
            }
        )

    return results, num_successful_examples


if __name__ == "__main__":
    # Load configuration files
    try:
        args = setup_args()
        bench_cfg = yaml.safe_load(args.bench_cfg.read_text())
        ddxdriver_cfg = yaml.safe_load(args.ddxdriver_cfg.read_text())
        diagnosis_agent_cfg = yaml.safe_load(args.diagnosis_agent_cfg.read_text())
        history_taking_agent_cfg = yaml.safe_load(args.history_taking_agent_cfg.read_text())
        patient_agent_cfg = yaml.safe_load(args.patient_agent_cfg.read_text())
        rag_agent_cfg = yaml.safe_load(args.rag_agent_cfg.read_text())
    except Exception as e:
        raise Exception(f"Error setting up config files:\n{e}")

    run_ddxdriver(
        bench_cfg=bench_cfg,
        ddxdriver_cfg=ddxdriver_cfg,
        diagnosis_agent_cfg=diagnosis_agent_cfg,
        history_taking_agent_cfg=history_taking_agent_cfg,
        patient_agent_cfg=patient_agent_cfg,
        rag_agent_cfg=rag_agent_cfg,
        experiment_folder=find_project_root() / "experiments/test",
    )
