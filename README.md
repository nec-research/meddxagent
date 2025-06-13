# MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis

Authors: Daniel Rose, Chia-Chien Hung, Marco Lepri, Israa Alqassem, Kiril Gashteovski, and Carolin Lawrence 

Paper: [https://arxiv.org/pdf/2502.19175](https://arxiv.org/pdf/2502.19175)

## Introduction
Differential Diagnosis (DDx) is a fundamental yet complex aspect of clinical decision-making, in which physicians iteratively refine a ranked list of possible diseases based on symptoms, antecedents, and medical knowledge. While recent advances in large language models (LLMs) have shown promise in supporting DDx, existing approaches face key limitations, including single-dataset evaluations, isolated optimization of components, unrealistic assumptions about complete patient profiles, and single-attempt diagnosis. We introduce a **M**odular **E**xplainable **DDx** **Agent** (**MEDDxAgent**) framework designed for interactive DDx, where diagnostic reasoning evolves through *iterative learning*, rather than assuming a complete patient profile is accessible. MEDDxAgent integrates three modular components: (1) an orchestrator (DDxDriver), (2) a history taking simulator, and (3) two specialized agents for knowledge retrieval and diagnosis strategy. To ensure robust evaluation, we introduce a comprehensive DDx benchmark covering respiratory, skin, and rare diseases.  We analyze single-turn diagnostic approaches and demonstrate the importance of iterative refinement when patient profiles are not available at the outset. Our broad evaluation demonstrates that MEDDxAgent achieves over 10% accuracy improvements in interactive DDx across both large and small LLMs, while offering critical explainability into its diagnostic reasoning process.


## Citation
If you use any source codes, or DDx benchmark included in this repo in your work, please cite the following paper:
<pre>
@article{rose2025meddxagent,
  title={MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis},
  author={Rose, Daniel and Hung, Chia-Chien and Lepri, Marco and Alqassem, Israa and Gashteovski, Kiril and Lawrence, Carolin},
  journal={arXiv preprint arXiv:2502.19175},
  year={2025}
}
</pre>

## Installation

**Recommended: Using uv (fastest)**

Install uv if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If your system doesn't have curl, you can use wget:
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Then install the project:
```bash
git clone https://github.com/nec-research/meddxagent.git
cd meddxagent
uv sync
uv pip install torch==2.3.1
```

To activate the environment:
```bash
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
```

**Alternative: Using conda/pip**
```bash
conda create -n meddxagent python=3.13
conda activate meddxagent
pip install torch==2.3.1
git clone https://github.com/nec-research/meddxagent.git
cd meddxagent
pip install -e .
```

If you'd like to use openai models/huggingface models here, add a `.env` file in `ddxdriver/models` with keys formatted as below.

```
OAI_KEY=(Insert your open ai key)
AZURE_ENDPOINT=(Insert your azure endpoint)
```

This repo relies on Microsoft Azure for OpenAI, a vllm deployment of [Llama3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), and huggingface/transformers for [Llama3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).
If you'd like to use a different model in a different setup, feel free to add a different way under `ddxdriver/models` folder. 
To load llama-* models, you will need to provide `HUGGINGFACE_HUB_TOKEN` with permissions.


## Datasets
Three datasets are included in this repo, where we convert into the format: (1) Initial Patient Profile, (2) Complete Patient Profile, and (3) Ground Truth Pathology/DDx.
- [DDxPlus](https://huggingface.co/datasets/appier-ai-research/StreamBench) (orig: CC-BY)
- [ICraftMD](https://github.com/stellalisy/mediQ/tree/main/data) (orig: MIT)
- [RareBench](https://huggingface.co/datasets/chenxz/RareBench) (orig: Apache-2.0)

Some codes are sourced from the StreamBench project, which you can find [here](https://github.com/stream-bench/stream-bench).

The ICraftMD and RareBench datasets are already provided in data/icrafmd and data/rarebench folders respectively. The DDxPlus dataset will be downloaded from huggingface.

## Evaluation Metrics Configuration

MEDDxAgent supports two evaluation strategies:

- **Strict Matching (Default)**: Exact string matching between predicted and ground truth diagnoses
- **Weak Matching**: Substring matching where ground truth is found within predicted diagnosis

To change the evaluation strategy, edit `WEAK_METRICS` in `ddxdriver/run_ddxdriver.py`:

```python
WEAK_METRICS = False  # Strict matching (default)
WEAK_METRICS = True   # Weak matching
```

## How to run?

First, make sure you have activated the conda environment:
```
conda activate meddxagent
```

### Run Individual Experiments

**By default, all experiments run on 100 patients.** You can customize the number of patients and model using command line arguments:

```bash
# Run diagnosis agent experiments (default: 100 patients, gpt-4o)
bash scripts/run_diagnosis_agent.sh

# Run with custom number of patients
bash scripts/run_diagnosis_agent.sh --num_patients 50

# Run with custom model
bash scripts/run_diagnosis_agent.sh --model_name gpt-4o

# Run with both custom patients and model
bash scripts/run_diagnosis_agent.sh --num_patients 10 --model_name gpt-4o
```

The same pattern applies to all experiment scripts:

```bash
# History taking simulator experiments
bash scripts/run_history_taking_simulator.sh --num_patients 50

# Retrieval agent experiments  
bash scripts/run_retrieval_agent.sh --num_patients 50

# Iterative experiments
bash scripts/run_iterative.sh --num_patients 50
```

### Run All Experiments
Select the model in ACTIVE_MODELS from `scripts/experiment.py`, and comment out the others
To run the full experiment suite:
```bash
# Run all experiments with default settings (100 patients)
CUDA_VISIBLE_DEVICES=0 python scripts/experiment.py

# Run all experiments with custom number of patients
CUDA_VISIBLE_DEVICES=0 python scripts/experiment.py --num_patients 50

# Run specific experiment type
CUDA_VISIBLE_DEVICES=0 python scripts/experiment.py --experiment_type rag --num_patients 50
```

### Direct DDxDriver Usage
For direct usage of DDxDriver:
```
bash scripts/run.sh
```

### Experiment Output and Logging

Experiments create organized output in `experiments/`:

```
experiments/
├── diagnosis/diagnosis_gpt-4o/
│   ├── experiment_logs.log          # Main experiment log
│   └── 1/, 2/, 3/...               # Individual experiment runs
├── history_taking/history_taking_gpt-4o/
├── rag/rag_gpt-4o/
└── iterative/iterative_gpt-4o/
```

- **experiment_logs.log**: High-level experiment progress and errors
- **Individual run folders**: Detailed results for each configuration combination
- **configs.json**: Complete configuration used for each run
- **run_logs.log**: Detailed execution logs for each run

## Changing Configs/Settings
If you'd like to change configurations (i.e. datasets, fewshot settings), go to the configs folder. 
There are configurations for the different benchmarks, ddxdriver, history taking simulator, knowledge retrieval agent, and diagnosis strategy agent.
We provide an example for iterative learning (fixed iteration, n=5, GPT-4o), in `examples` folder.

