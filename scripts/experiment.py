import itertools
from typing import Union, List, Tuple
from pathlib import Path
import yaml
import traceback
import time
import argparse

from ddxdriver.run_ddxdriver import run_ddxdriver
from ddxdriver.utils import find_project_root
from ddxdriver.logger import log, enable_logging, set_file_handler, log_json_data
from ddxdriver.rag_agents._searchrag_utils import Corpus


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Adjust as needed
print(f"Project root is: {PROJECT_ROOT}")

# All available models
# Select which models to run (modify this list based on which GPU/machine you're using)
# llama3-70B model (i.e., llama31instruct, Llama-3-70B-UltraMedical) are deployed with vllm
ACTIVE_MODELS = [
#    {"class_name": "oai_azure_chat.OpenAIAzureChat", "model_name": "gpt-4o"},
   {"class_name": "oai_chat.OpenAIChat", "model_name": "gpt-4o"},
#    {"class_name": "llama3_instruct.Llama3Instruct", "model_name": "llama31instruct"},
#    {"class_name": "llama3_ultramedical.Llama370BUltraMedical", "model_name": "Llama-3-70B-UltraMedical"},
#    {"class_name": "llama31_8b.Llama318B", "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
#    {"class_name": "llama31_8b.Llama318B", "model_name": "TsinghuaC3I/Llama-3.1-8B-UltraMedical"}
]
NUM_PATIENTS = 1

def diagnosis_experiment(diagnosis_experiment_folder: Union[str, Path]):
    diagnosis_experiment_folder = Path(diagnosis_experiment_folder)

    experiment_logging_path = diagnosis_experiment_folder / "experiment_logs.log"
    set_file_handler(experiment_logging_path, mode="a")
    enable_logging(console_logging=True, file_logging=True)
    # Bench settings
    datasets = ["ddxplus.DDxPlus", "icraftmd.ICraftMD", "rarebench.RareBench"]

    # Model settings
    models = ACTIVE_MODELS

    # Diagnosis agent + fewshot hyperparameters
    # Ordering by embedding type to reduce calculating embeddings to only once
    diagnosis_agents = [
        {"class_name": "single_llm_standard.SingleLLMStandard", "type": "none", "num_shots": 0},
        {"class_name": "single_llm_cot.SingleLLMCOT", "type": "none", "num_shots": 0},
        {"class_name": "single_llm_standard.SingleLLMStandard", "type": "static", "num_shots": 5},
        {
            "class_name": "single_llm_cot.SingleLLMCOT",
            "type": "static",
            "num_shots": 5,
            "self_generated_fewshot_cot": False,
        },
        {
            "class_name": "single_llm_cot.SingleLLMCOT",
            "type": "static",
            "num_shots": 5,
            "self_generated_fewshot_cot": True,
        },
        {
            "class_name": "single_llm_standard.SingleLLMStandard",
            "type": "dynamic",
            "num_shots": 5,
            "embedding_model": "emilyalsentzer/Bio_ClinicalBERT",
            "pooling": "average",
        },
        {
            "class_name": "single_llm_cot.SingleLLMCOT",
            "type": "dynamic",
            "num_shots": 5,
            "embedding_model": "emilyalsentzer/Bio_ClinicalBERT",
            "pooling": "average",
            "self_generated_fewshot_cot": False,
        },
        {
            "class_name": "single_llm_cot.SingleLLMCOT",
            "type": "dynamic",
            "num_shots": 5,
            "embedding_model": "emilyalsentzer/Bio_ClinicalBERT",
            "pooling": "average",
            "self_generated_fewshot_cot": True,
        },
        {
            "class_name": "single_llm_standard.SingleLLMStandard",
            "type": "dynamic",
            "num_shots": 5,
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "pooling": "cls",
        },
        {
            "class_name": "single_llm_cot.SingleLLMCOT",
            "type": "dynamic",
            "num_shots": 5,
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "pooling": "cls",
            "self_generated_fewshot_cot": False,
        },
        {
            "class_name": "single_llm_cot.SingleLLMCOT",
            "type": "dynamic",
            "num_shots": 5,
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "pooling": "cls",
            "self_generated_fewshot_cot": True,
        },
    ]

    num_combinations = len(list(itertools.product(diagnosis_agents, datasets, models)))

    # Defining ddxdriver_cfg (model won't be used)
    ddxdriver_cfg = {
        "class_name": "ddxdriver.ddxdrivers.fixed_choice.FixedChoice",
        "config": {
            "seed": 42,
            "agent_prompt_length": 0,
            "agent_order": ["diagnosis"],
            "iterations": 1,
            "model": {
                "class_name": "ddxdriver.models.llama3_instruct.Llama3Instruct",
                "config": {"model_name": "llama31instruct"},
                # "class_name": "ddxdriver.models.oai_azure_chat.OpenAIAzureChat",
                # "class_name": "ddxdriver.models.oai_chat.OpenAIChat",
                # "config": {"model_name": "gpt-4o"},
            },
        },
    }

    # Running data to determine whether to precompute new embeddings
    covered_embedding_combinations: List[Tuple[str, str]] = []
    precompute_new_embeddings = False

    # Create bench and diagnosis_agent configs
    cfgs = []
    for diagnosis_agent, dataset, model in itertools.product(diagnosis_agents, datasets, models):
        # Load constant configs
        try:
            embedding_model = diagnosis_agent.get("embedding_model")

            embedding_combination = (dataset, embedding_model)
            if embedding_model and embedding_combination not in covered_embedding_combinations:
                covered_embedding_combinations.append(embedding_combination)
                precompute_new_embeddings = False
            else:
                precompute_new_embeddings = False

            bench_cfg = yaml.safe_load((PROJECT_ROOT / "configs/bench.yml").read_text())
            bench_cfg["config"]["knn_search_cfg"][
                "precompute_new_embeddings"
            ] = precompute_new_embeddings
            bench_cfg["class_name"] = f"ddxdriver.benchmarks.{dataset}"
            bench_cfg["config"]["knn_search_cfg"]["embedding_model"] = embedding_model
            bench_cfg["config"]["knn_search_cfg"]["pooling"] = diagnosis_agent.get("pooling")
            bench_cfg["num_patients"] = NUM_PATIENTS
            bench_cfg["enforce_diagnosis_options"] = True

            diagnosis_cfg = {
                "class_name": f"ddxdriver.diagnosis_agents.{diagnosis_agent['class_name']}",
                "config": {
                    "model": {
                        "class_name": f"ddxdriver.models.{model['class_name']}",
                        "config": {"model_name": model["model_name"]},
                    },
                    "fewshot": {
                        "type": diagnosis_agent["type"],
                        "num_shots": diagnosis_agent["num_shots"],
                        "self_generated_fewshot_cot": diagnosis_agent.get(
                            "self_generated_fewshot_cot", False
                        ),
                    },
                },
            }
            cfgs.append((diagnosis_cfg, bench_cfg))
        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"Error with setting up diagnosis experiments:\n{e}\nTraceback:\n{tb}")

    if len(cfgs) != num_combinations:
        log.error(f"Did not correctly generate {num_combinations} pairs\n")
        raise Exception
    else:
        log.info(f"Correctly generated {num_combinations} different experiments\n")

    log.info("Starting to run entire diagnosis experiment...\n")
    complete_start_time = time.time()
    for experiment_number, (diagnosis_cfg, bench_cfg) in enumerate(cfgs, start=1):
        # if experiment_number >= 10:
        #     continue
        set_file_handler(experiment_logging_path, mode="a")
        log.info(f"STARTING EXPERIMENT {experiment_number}...\n")
        start_time = time.time()
        try:
            experiment_folder = (
                PROJECT_ROOT / diagnosis_experiment_folder / str(experiment_number)
            )

            # Log config
            set_file_handler(experiment_logging_path, mode="a")
            json_data = {"diagnosis_cfg": diagnosis_cfg, "bench_cfg": bench_cfg}
            log_json_data(json_data=json_data, file_path=experiment_folder / "configs.json")

            # Run example
            run_ddxdriver(
                bench_cfg=bench_cfg,
                ddxdriver_cfg=ddxdriver_cfg,
                diagnosis_agent_cfg=diagnosis_cfg,
                experiment_folder=experiment_folder,
            )

            # Reset file handler
            set_file_handler(experiment_logging_path, mode="a")
            log.info(f"FINISHED EXPERIMENT {experiment_number}.\n")
        except Exception as e:
            set_file_handler(experiment_logging_path, mode="a")
            tb = traceback.format_exc()
            log.error(
                f"Error with running diagnosis experiment {experiment_number}:\n{e}\nTraceback:\n{tb}"
            )

        log.info(
            f"Finished running experiment {experiment_number} in {time.time()-start_time} seconds.\n"
        )

    log.info(
        f"Finished running entire diagnosis experiment in {time.time()-complete_start_time} seconds.\n"
    )


def history_taking_experiment(history_taking_experiment_folder: Union[str, Path]):
    history_taking_experiment_folder = Path(history_taking_experiment_folder)
    
    experiment_logging_path = history_taking_experiment_folder / "experiment_logs.log"
    set_file_handler(experiment_logging_path, mode="a")
    enable_logging(console_logging=True, file_logging=True)

    # Bench settings
    datasets = ["ddxplus.DDxPlus", "icraftmd.ICraftMD", "rarebench.RareBench"]

    # Model settings
    models = ACTIVE_MODELS

    # Max questions
    max_questions_list = [5, 10, 15]

    num_combinations = len(list(itertools.product(models, datasets, max_questions_list)))

    # Create bench and diagnosis configs
    cfgs = []
    for max_questions, dataset, model in itertools.product(max_questions_list, datasets, models):
        # Load constant configs
        try:
            bench_cfg = yaml.safe_load((PROJECT_ROOT / "configs/bench.yml").read_text())

            bench_cfg["class_name"] = f"ddxdriver.benchmarks.{dataset}"
            bench_cfg["num_patients"] = NUM_PATIENTS
            bench_cfg["enforce_diagnosis_options"] = True

            # Defining ddxdriver_cfg (using the same model as specified by user)
            ddxdriver_cfg = {
                "class_name": "ddxdriver.ddxdrivers.fixed_choice.FixedChoice",
                "config": {
                    "seed": 42,
                    "agent_prompt_length": 0,
                    "agent_order": ["history_taking", "diagnosis"],
                    "iterations": 1,
                    "model": {
                        "class_name": f"ddxdriver.models.{model['class_name']}",
                        "config": {"model_name": model["model_name"]},
                    },
                },
            }

            # Diagnosis agent config
            diagnosis_cfg = {
                "class_name": "ddxdriver.diagnosis_agents.single_llm_standard.SingleLLMStandard",
                "config": {
                    "model": {
                        "class_name": f"ddxdriver.models.{model['class_name']}",
                        "config": {"model_name": model["model_name"]},
                    },
                    "fewshot": {"type": "none", "num_shots": 0},
                },
            }

            history_taking_cfg = yaml.safe_load(
                (PROJECT_ROOT / "configs/history_taking_agents/llm_history_taking.yml").read_text()
            )
            history_taking_cfg["config"]["max_questions"] = max_questions
            history_taking_cfg["config"]["model"] = {
                "class_name": f"ddxdriver.models.{model['class_name']}",
                "config": {"model_name": model["model_name"]},
            }

            patient_cfg = yaml.safe_load((PROJECT_ROOT / "configs/patient_agents/llm_patient.yml").read_text())
            patient_cfg["config"]["model"] = {
                "class_name": f"ddxdriver.models.{model['class_name']}",
                "config": {"model_name": model["model_name"]},
            }
            cfgs.append((ddxdriver_cfg, history_taking_cfg, patient_cfg, diagnosis_cfg, bench_cfg))
        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"Error with setting up diagnosis experiments:\n{e}\nTraceback:\n{tb}")

    if len(cfgs) != num_combinations:
        log.error(f"Did not correctly generate {num_combinations} pairs\n")
        raise Exception
    else:
        log.info(f"Correctly generated {num_combinations} different experiments\n")

    log.info("Starting to run entire history taking experiment...\n")
    complete_start_time = time.time()
    for experiment_number, (
        ddxdriver_cfg,
        history_taking_cfg,
        patient_cfg,
        diagnosis_cfg,
        bench_cfg,
    ) in enumerate(cfgs, start=1):
        # if experiment_number >= 10:
        #     continue
        set_file_handler(experiment_logging_path, mode="a")
        log.info(f"STARTING EXPERIMENT {experiment_number}...\n")
        start_time = time.time()
        try:
            experiment_folder = (
                PROJECT_ROOT / history_taking_experiment_folder / str(experiment_number)
            )

            # Log config
            set_file_handler(experiment_logging_path, mode="a")
            json_data = {
                "ddxdriver_cfg": ddxdriver_cfg,
                "history_taking_cfg": history_taking_cfg,
                "bench_cfg": bench_cfg,
                "patient_cfg": patient_cfg,
                "diagnosis_cfg": diagnosis_cfg,
            }
            log_json_data(json_data=json_data, file_path=experiment_folder / "configs.json")

            # Run example
            run_ddxdriver(
                bench_cfg=bench_cfg,
                ddxdriver_cfg=ddxdriver_cfg,
                diagnosis_agent_cfg=diagnosis_cfg,
                history_taking_agent_cfg=history_taking_cfg,
                patient_agent_cfg=patient_cfg,
                experiment_folder=experiment_folder,
            )

            # Reset file handler
            set_file_handler(experiment_logging_path, mode="a")
            log.info(f"FINISHED EXPERIMENT {experiment_number}.\n")
        except Exception as e:
            set_file_handler(experiment_logging_path, mode="a")
            tb = traceback.format_exc()
            log.error(
                f"Error with running history taking experiment {experiment_number}:\n{e}\nTraceback:\n{tb}"
            )

        log.info(
            f"Finished running experiment {experiment_number} in {time.time()-start_time} seconds.\n"
        )

    log.info(
        f"Finished running entire history taking experiment in {time.time()-complete_start_time} seconds.\n"
    )


def rag_experiment(rag_experiment_folder: Union[str, Path]):
    rag_experiment_folder = Path(rag_experiment_folder)
    
    experiment_logging_path = rag_experiment_folder / "experiment_logs.log"
    set_file_handler(experiment_logging_path, mode="a")
    enable_logging(console_logging=True, file_logging=True)

    # Bench settings
    datasets = ["ddxplus.DDxPlus", "icraftmd.ICraftMD", "rarebench.RareBench"]

    # Model settings
    models = ACTIVE_MODELS

    # Corpus name: PubMed and Wikipedia
    corpus_names = [Corpus.PUBMED.value, Corpus.WIKIPEDIA.value]
    
    num_combinations = len(list(itertools.product(models, datasets, corpus_names)))

    # Create bench and diagnosis configs
    cfgs = []
    for corpus_name, dataset, model in itertools.product(corpus_names, datasets, models):
        # Load constant configs
        try:
            bench_cfg = yaml.safe_load((PROJECT_ROOT / "configs/bench.yml").read_text())

            bench_cfg["class_name"] = f"ddxdriver.benchmarks.{dataset}"
            bench_cfg["num_patients"] = NUM_PATIENTS
            bench_cfg["enforce_diagnosis_options"] = True

            # Defining ddxdriver_cfg (model will be used to provide RAG search to ddxdriver + give diagnosis instructions to diagnosis agent)
            ddxdriver_cfg = {
                "class_name": "ddxdriver.ddxdrivers.fixed_choice.FixedChoice",
                "config": {
                    "seed": 42,
                    "agent_prompt_length": 0,
                    "agent_order": ["rag", "diagnosis"],
                    "iterations": 1,
                    "only_patient_initial_information": True, #ONLY PATIENT INFO
                    "model": {
                        "class_name": f"ddxdriver.models.{model['class_name']}",
                        "config": {"model_name": model["model_name"]},
                    },
                },
            }

            # Diagnosis agent config
            diagnosis_cfg = {
                "class_name": "ddxdriver.diagnosis_agents.single_llm_standard.SingleLLMStandard",
                "config": {
                    "model": {
                        "class_name": f"ddxdriver.models.{model['class_name']}",
                        "config": {"model_name": model["model_name"]},
                    },
                    "fewshot": {"type": "none", "num_shots": 0},
                },
            }

            rag_cfg = {
                "class_name": "ddxdriver.rag_agents.searchrag_standard.SearchRAGStandard",
                "config": {
                    "corpus_name": corpus_name,
                    "top_k_search": 2,
                    "max_keyword_searches": 3,
                    "model": {
                        "class_name": f"ddxdriver.models.{model['class_name']}",
                        "config": {"model_name": model["model_name"]},
                    },
                },
            }
            cfgs.append((rag_cfg, ddxdriver_cfg, diagnosis_cfg, bench_cfg))
        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"Error with setting up diagnosis experiments:\n{e}\nTraceback:\n{tb}")

    if len(cfgs) != num_combinations:
        log.error(f"Did not correctly generate {num_combinations} pairs\n")
        raise Exception
    else:
        log.info(f"Correctly generated {num_combinations} different experiments\n")

    log.info("Starting to run entire history taking experiment...\n")
    complete_start_time = time.time()
    for experiment_number, (
        rag_cfg,
        ddxdriver_cfg,
        diagnosis_cfg,
        bench_cfg,
    ) in enumerate(cfgs, start=1):
        # if experiment_number >= 10:
        #     continue
        set_file_handler(experiment_logging_path, mode="a")
        log.info(f"STARTING EXPERIMENT {experiment_number}...\n")
        start_time = time.time()
        try:
            experiment_folder = PROJECT_ROOT / rag_experiment_folder / str(experiment_number)
            # Log config
            set_file_handler(experiment_logging_path, mode="a")
            json_data = {
                "rag_cfg": rag_cfg,
                "bench_cfg": bench_cfg,
                "ddxdriver_cfg": ddxdriver_cfg,
                "diagnosis_cfg": diagnosis_cfg,
            }
            log_json_data(json_data=json_data, file_path=experiment_folder / "configs.json")

            # Run example
            run_ddxdriver(
                bench_cfg=bench_cfg,
                ddxdriver_cfg=ddxdriver_cfg,
                diagnosis_agent_cfg=diagnosis_cfg,
                rag_agent_cfg=rag_cfg,
                experiment_folder=experiment_folder,
            )

            # Reset file handler
            set_file_handler(experiment_logging_path, mode="a")
            log.info(f"FINISHED EXPERIMENT {experiment_number}.\n")
        except Exception as e:
            set_file_handler(experiment_logging_path, mode="a")
            tb = traceback.format_exc()
            log.error(
                f"Error with running history taking experiment {experiment_number}:\n{e}\nTraceback:\n{tb}"
            )

        log.info(
            f"Finished running experiment {experiment_number} in {time.time()-start_time} seconds.\n"
        )

    log.info(
        f"Finished running entire rag experiment in {time.time()-complete_start_time} seconds.\n"
    )


def iterative_experiment(iterative_experiment_folder: Union[str, Path]):
    iterative_experiment_folder = Path(iterative_experiment_folder)

    experiment_logging_path = iterative_experiment_folder / "experiment_logs.log"
    set_file_handler(experiment_logging_path, mode="a")
    enable_logging(console_logging=True, file_logging=True)

    # Bench settings
    datasets = [
        {
            "dataset": "ddxplus.DDxPlus",
            "diagnosis_agent_class_name": "single_llm_standard.SingleLLMStandard",
            "fewshot_type": "dynamic",
            "fewshot_embedding_model": "BAAI/bge-base-en-v1.5",
        },
        {
            "dataset": "icraftmd.ICraftMD",
            "diagnosis_agent_class_name": "single_llm_cot.SingleLLMCOT",
            "fewshot_type": "none",
        },
        {
            "dataset": "rarebench.RareBench",
            "diagnosis_agent_class_name": "single_llm_standard.SingleLLMStandard",
            "fewshot_type": "dynamic",
            "fewshot_embedding_model": "BAAI/bge-base-en-v1.5",
            # "self_generated_fewshot_cot": False,
        },
    ]

    # Model settings
    models = ACTIVE_MODELS

    # DDxDrivers
    ddxdrivers = [
        {
            "class_name": "fixed_choice.FixedChoice",
            "agents": ["history_taking", "rag", "diagnosis"],
        },
        {"class_name": "open_choice.OpenChoice", "agents": ["history_taking", "rag", "diagnosis"]},
    ]

    iterations_list = [1, 2, 3]

    num_combinations = len(list(itertools.product(iterations_list, ddxdrivers, datasets, models)))

    # Running data to determine whether to precompute new embeddings
    covered_embeddings: List[str] = []
    precompute_new_embeddings = False

    # Create bench and diagnosis_cfg configs
    cfgs = []
    for iterations, ddxdriver, dataset, model in itertools.product(
        iterations_list, ddxdrivers, datasets, models
    ):
        try:
            # Define agents, which are fixed besides the model they use
            embedding_model = dataset.get("fewshot_embedding_model")

            diagnosis_cfg = {
                "class_name": f"ddxdriver.diagnosis_agents.{dataset['diagnosis_agent_class_name']}",
                "config": {
                    "model": {
                        "class_name": f"ddxdriver.models.{model['class_name']}",
                        "config": {"model_name": model["model_name"]},
                    },
                    "fewshot": {
                        "type": f"{dataset['fewshot_type']}",
                        "num_shots": 5,
                        "embedding_model": embedding_model,
                        "self_generated_fewshot_cot": dataset.get(
                            "self_generated_fewshot_cot", False
                        ),
                    },
                },
            }

            history_taking_cfg = yaml.safe_load(
                (PROJECT_ROOT / "configs/history_taking_agents/llm_history_taking.yml").read_text()
            )
            history_taking_cfg["config"]["max_questions"] = 5
            history_taking_cfg["config"]["model"] = {
                "class_name": f"ddxdriver.models.{model['class_name']}",
                "config": {"model_name": model["model_name"]},
            }

            patient_cfg = yaml.safe_load((PROJECT_ROOT / "configs/patient_agents/llm_patient.yml").read_text())
            patient_cfg["config"]["model"] = {
                "class_name": f"ddxdriver.models.{model['class_name']}",
                "config": {"model_name": model["model_name"]},
            }

            rag_cfg = {
                "class_name": "ddxdriver.rag_agents.searchrag_standard.SearchRAGStandard",
                "config": {
                    "corpus_name": "PubMed",
                    "top_k_search": 2,
                    "max_keyword_searches": 3,
                    "model": {
                        "class_name": f"ddxdriver.models.{model['class_name']}",
                        "config": {"model_name": model["model_name"]},
                    },
                },
            }

            if embedding_model and embedding_model not in covered_embeddings:
                covered_embeddings.append(embedding_model)
                precompute_new_embeddings = False
            else:
                precompute_new_embeddings = False

            bench_cfg = yaml.safe_load((PROJECT_ROOT / "configs/bench.yml").read_text())
            bench_cfg["config"]["knn_search_cfg"][
                "precompute_new_embeddings"
            ] = precompute_new_embeddings
            bench_cfg["class_name"] = f"ddxdriver.benchmarks.{dataset['dataset']}"
            bench_cfg["config"]["knn_search_cfg"]["embedding_model"] = embedding_model
            bench_cfg["config"]["knn_search_cfg"]["pooling"] = "average"
            bench_cfg["num_patients"] = NUM_PATIENTS
            bench_cfg["enforce_diagnosis_options"] = True

            # Create ddxdriver config
            ddxdriver_cfg = {
                "class_name": f"ddxdriver.ddxdrivers.{ddxdriver['class_name']}",
                "config": {
                    "agent_prompt_length": 10,
                    "model": {
                        "class_name": f"ddxdriver.models.{model['class_name']}",
                        "config": {"model_name": model["model_name"]},
                    },
                },
            }

            if ddxdriver["class_name"] == "fixed_choice.FixedChoice":
                ddxdriver_cfg["config"]["agent_order"] = ["history_taking", "rag", "diagnosis"]
                ddxdriver_cfg["config"]["iterations"] = iterations
            elif ddxdriver["class_name"] == "open_choice.OpenChoice":
                ddxdriver_cfg["config"]["available_agents"] = ["history_taking", "rag", "diagnosis"]
                ddxdriver_cfg["config"]["max_turns"] = iterations * 3
            else:
                raise ValueError(
                    "DDxDriver class name is invalid, should be either fixed_choice.FixedChoice or open_choice.OpenChoice"
                )

            cfgs.append(
                (ddxdriver_cfg, diagnosis_cfg, history_taking_cfg, patient_cfg, rag_cfg, bench_cfg)
            )
        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"Error with setting up diagnosis experiments:\n{e}\nTraceback:\n{tb}")

    if len(cfgs) != num_combinations:
        log.error(f"Did not correctly generate {num_combinations} pairs\n")
        raise Exception
    else:
        log.info(f"Correctly generated {num_combinations} different experiments\n")

    log.info("Starting to run entire iterative experiment...\n")
    complete_start_time = time.time()
    for experiment_number, (
        ddxdriver_cfg,
        diagnosis_cfg,
        history_taking_cfg,
        patient_cfg,
        rag_cfg,
        bench_cfg,
    ) in enumerate(cfgs, start=1):
        # if experiment_number != 1:
        #     continue
        set_file_handler(experiment_logging_path, mode="a")
        log.info(f"STARTING EXPERIMENT {experiment_number}...\n")
        start_time = time.time()
        try:
            experiment_folder = (
                PROJECT_ROOT / iterative_experiment_folder / str(experiment_number)
            )

            # Log confi
            set_file_handler(experiment_logging_path, mode="a")
            json_data = {
                "ddxdriver_cfg": ddxdriver_cfg,
                "bench_cfg": bench_cfg,
                "diagnosis_cfg": diagnosis_cfg,
                "history_taking_cfg": history_taking_cfg,
                "patient_cfg": patient_cfg,
                "rag_cfg": rag_cfg,
            }
            log_json_data(json_data=json_data, file_path=experiment_folder / "configs.json")

            # Run example
            run_ddxdriver(
                bench_cfg=bench_cfg,
                ddxdriver_cfg=ddxdriver_cfg,
                diagnosis_agent_cfg=diagnosis_cfg,
                history_taking_agent_cfg=history_taking_cfg,
                patient_agent_cfg=patient_cfg,
                rag_agent_cfg=rag_cfg,
                experiment_folder=experiment_folder,
            )

            # Reset file handler
            set_file_handler(experiment_logging_path, mode="a")
            log.info(f"FINISHED EXPERIMENT {experiment_number}.\n")
        except Exception as e:
            set_file_handler(experiment_logging_path, mode="a")
            tb = traceback.format_exc()
            log.error(
                f"Error with running diagnosis experiment {experiment_number}:\n{e}\nTraceback:\n{tb}"
            )

        log.info(
            f"Finished running experiment {experiment_number} in {time.time()-start_time} seconds.\n"
        )

    log.info(
        f"Finished running entire iterative experiment in {time.time()-complete_start_time} seconds.\n"
    )


def main():
    # Update the global NUM_PATIENTS with command line argument
    global NUM_PATIENTS
    
    parser = argparse.ArgumentParser(description="Run MEDDxAgent experiments")
    parser.add_argument(
        "--experiment_type", 
        choices=["diagnosis", "history_taking", "rag", "iterative", "all"],
        default="all",
        help="Type of experiment to run"
    )
    parser.add_argument(
        "--experiment_folder",
        type=str,
        help="Custom experiment folder path (optional, will auto-generate if not provided)"
    )
    parser.add_argument(
        "--num_patients",
        type=int,
        default=NUM_PATIENTS,
        help=f"Number of patients to run experiments on (default: {NUM_PATIENTS})"
    )
    
    args = parser.parse_args()
    
    # Update NUM_PATIENTS with the command line argument
    NUM_PATIENTS = args.num_patients
    
    # Create model-specific folder names based on active models
    model_name = "_".join(m["model_name"].split("/")[-1].lower() for m in ACTIVE_MODELS)
    
    # Base paths for each experiment type
    base_folders = {
        "diagnosis": f"experiments/diagnosis/diagnosis_{model_name}",
        "history_taking": f"experiments/history_taking/history_taking_{model_name}",
        "rag": f"experiments/rag/rag_{model_name}",
        "iterative": f"experiments/iterative/iterative_{model_name}",
    }
    
    # Run specific experiment type
    if args.experiment_type == "diagnosis":
        experiment_folder = PROJECT_ROOT / (args.experiment_folder if args.experiment_folder else base_folders["diagnosis"])
        diagnosis_experiment(experiment_folder)
    
    elif args.experiment_type == "history_taking":
        experiment_folder = PROJECT_ROOT / (args.experiment_folder if args.experiment_folder else base_folders["history_taking"])
        history_taking_experiment(experiment_folder)
    
    elif args.experiment_type == "rag":
        experiment_folder = PROJECT_ROOT / (args.experiment_folder if args.experiment_folder else base_folders["rag"])
        rag_experiment(experiment_folder)
    
    elif args.experiment_type == "iterative":
        experiment_folder = PROJECT_ROOT / (args.experiment_folder if args.experiment_folder else base_folders["iterative"])
        iterative_experiment(experiment_folder)
    
    elif args.experiment_type == "all":
        # Run all experiment types with current models (original behavior)
        diagnosis_experiment(PROJECT_ROOT / base_folders["diagnosis"])
        history_taking_experiment(PROJECT_ROOT / base_folders["history_taking"])
        rag_experiment(PROJECT_ROOT / base_folders["rag"])
        iterative_experiment(PROJECT_ROOT / base_folders["iterative"])


if __name__ == "__main__":
    main()