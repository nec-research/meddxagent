import json
import re
from pathlib import Path
from typing import List, Set, Dict, Any
from datasets import Dataset, load_dataset

from ddxdriver.models import Model
from ddxdriver.models.oai_azure_chat import OpenAIAzureChat
from ddxdriver.utils import strip_all_lines

'''
This script is used to generate the disease options for the rarebench dataset and extract the disease options for the icraftmd dataset.
This does not need to be run again, as the disease options have already been generated in ddxdriver/benchmarks/data
'''

def rare_diseases_to_single_rare_disease(
    diseases: str, diagnosis_options: Set[str], model: Model
) -> str:
    system_prompt = strip_all_lines(
        """\
        You are a rare disease specialist. 
        Your task is to synthesize a list of diseases into a single rare disease name in English. 
        This single rare disease name should be an accepted name for the rare disease.
        
        You will receive these inputs:
        1) Diseases List: A comma-separated list of diseases. 
        - The list may include disease names/terms in both English and Chinese languages. 
        2) Current Diagnosis Options: A list of the current rare disease names. 
        - You may either chose a disease name from this or generate your own if you don't find the disease here.
        - The Current Diagnosis Options may include a disease name which is a synonym to those in the Diseases List.

        Follow these steps when generating your rare disease name:
        1. Look over the current Current Diagnosis Options. 
        - You will either classify the diseases as one of these options (if they are synonyms), or respond with a new disease option. 
        2. Consider the Diseases List and what single rare disease name this Diseases List corresponds to
        - This name should not consider specific numbers or types, just the single rare disease name
        3. Determine whether these diseases are already present in Current Diagnosis Options as a synonym disease name
        - If the disease name is already in Current Diagnosis Options, simply respond with this disease name
        - If the disease name is not yet present, generate a new disease name which covers the Diseases list
        4. Respond with the single rare disease name representative of the Disease List, nothing else
        - Respond in English 
        - Provide only one name for the disease. 
        - Make the first word or acronyms capitalized, and the rest lower case.
        - Ensure that the name you create is consistent with the naming conventions of rare diseases and retains unique information
        from the provided list
        - Choose a rare disease name 
        - Respond with just the skin disease name, nothing else

        Notes on choosing a single rare disease name vs synonym rare disease names:
        - These rare disease names should be distinct
        - Disease names with different specified symptoms should be considered as one disease, unless they are standardly accepted to be different rare diseases.
        - For example, Methylmalonic acidemia with homocystinuria and Methylmalonic acidemia are the same disease: Methylmalonic acidemia

        Here is an example of how to select a name. Do not overfit to the content in this example.

        Disease List: Vitamin B12-unresponsive methylmalonic acidemia/Methylmalonic aciduria
        due to methylmalonyl-coa mutase deficiency, Vitamin B12-unresponsive methylmalonic
        acidemia/Methylmalonic aciduria due to methylmalonyl-coa mutase deficiency 2, 甲基丙二酸血症;
        甲基丙二酸尿症/Methylmalonic acidemia; MMA; Methylmalonic aciduria type 1
        
        Current Diagnosis Options: 
        - Thyroid ectopia
        - Athyreosis

        Single rare disease name: 
        Methylmalonic acidemia
        """
    )
    diagnosis_options = "\n".join("- " + disease_name for disease_name in diagnosis_options)
    user_prompt = strip_all_lines(
        f"""\
            Disease List: {diseases}

            Current Diagnosis Options: 
            {diagnosis_options}
            
            Single rare disease name:
        """
    )
    return strip_all_lines(model(user_prompt=user_prompt, system_prompt=system_prompt))


def open_dictionary_json_file_helper(json_path: Path) -> Dict[str, Any]:
    if json_path.exists():
        print(f"Successfully found file: {json_path}, continuing...")
        try:
            with open(json_path, "r") as f:
                file = json.load(f)
            # Assert that the loaded file is a dictionary
            if not isinstance(file, dict):
                raise ValueError(f"The file at {json_path} does not contain a dictionary.")
        except json.JSONDecodeError:
            raise ValueError(f"The file at {json_path} is not a valid JSON.")
        except Exception as e:
            raise RuntimeError(f"Exception while trying to open {json_path}:\n{e}")
    else:
        print(
            f"Warning: No file found at {json_path}, creating a new file with an empty dictionary..."
        )
        file = {}
        with open(json_path, "w") as f:
            json.dump(file, f, indent=4)

    return file


def generate_disease_options_rarebench(
    dataset_subset: List[str],
    diagnosis_options_path: Path,
    disease_mapping_path: Path,
    model: Model,
):
    # File checking
    # Check if diagnosis_options_path file exists and load or initialize it
    diagnosis_options_file = open_dictionary_json_file_helper(json_path=diagnosis_options_path)
    disease_mapping_file = open_dictionary_json_file_helper(json_path=disease_mapping_path)

    # Going over subsets to generate options
    for subset in dataset_subset:
        # A set of the disease names
        diagnosis_options_set = set()
        # A mapping to map the possible disease lists to diagnosis options
        disease_mapping = {}

        patient_entries = _make_subsets_patient_entries(dataset_subset=dataset_subset)

        for subset_key in patient_entries:
            for patient_entry in patient_entries[subset_key]:
                disease_list = patient_entry["pathology"]
                if disease_list not in disease_mapping:
                    print(f"Generating single disease name for disease list: {disease_list}")
                    single_rare_disease = rare_diseases_to_single_rare_disease(
                        disease_list, diagnosis_options_set, model
                    )
                    # Add disease_list to mapping
                    disease_mapping[disease_list] = single_rare_disease
                    # If disease is new, add to diagnosis options
                    if single_rare_disease not in diagnosis_options_set:
                        print(f"Adding new disease to diagnosis options: {single_rare_disease}")
                        diagnosis_options_set.add(single_rare_disease)

        sorted_diagnosis_options = sorted(diagnosis_options_set, key=str.lower)
        diagnosis_options_file[subset] = sorted_diagnosis_options

        sorted_disease_mapping = {k: disease_mapping[k] for k in sorted(disease_mapping)}
        disease_mapping_file[subset] = sorted_disease_mapping

    with open(diagnosis_options_path, "w") as f:
        json.dump(diagnosis_options_file, f, indent=4)

    with open(disease_mapping_path, "w") as f:
        json.dump(disease_mapping_file, f, indent=4)


def normalize_diagnosis(diagnosis: str) -> str:
    # Remove any non-alphabetic characters except spaces, then split by spaces
    parts = [part.capitalize() for part in re.split(r"\s+", re.sub(r"[^A-Za-z\s]", "", diagnosis))]

    # Join parts with no separator and return
    return "".join(parts)


def extract_disease_mapping_icraftmd(
    icraftmd_path: str,
    disease_mapping_path: Path,
):
    """
    Loading disease mapping for icraftmd into a json file, and creating a temporary mapping where each diagnosis maps to itself.
    This mapping should then be manually edited to remove diseases which are synonyms
    """
    # Check if disease_mapping_path file exists and load or initialize it
    open_dictionary_json_file_helper(json_path=disease_mapping_path)

    # Starts off as just mapping the first normalized disease as the key
    disease_mapping = {}

    with open(icraftmd_path) as f:
        for line in f:
            json_obj = json.loads(line.strip())

            options = json_obj.get("options", {})
            for _, disease_name in options.items():
                disease_mapping[disease_name] = disease_name

    sorted_disease_mapping = {k: disease_mapping[k] for k in sorted(disease_mapping)}

    with open(disease_mapping_path, "w") as f:
        json.dump(sorted_disease_mapping, f, indent=4)

    print(f"Wrote to {disease_mapping_path}")


def extract_disease_options_icraftmd(
    disease_mapping_path: Path,
    diagnosis_options_path: Path,
):
    """
    Loading diagnosis options for icraftmd into a text file, based on disease mapping
    """

    # Starts off as just mapping the first normalized disease as the key
    with open(disease_mapping_path) as file:
        disease_mapping_file = json.load(file)

    diagnosis_options_set = set()
    for disease_code, disease in disease_mapping_file.items():
        diagnosis_options_set.add(disease)

    sorted_diagnosis_options_set = sorted(diagnosis_options_set, key=str.lower)

    with open(diagnosis_options_path, "w") as f:
        f.write("\n".join(sorted_diagnosis_options_set))

    print(f"Wrote to {diagnosis_options_path}")


def _make_subsets_patient_entries(dataset_subset: List[str]):
    """Borrowed from rarebench.py"""
    DATASET_FILE = "chenxz/RareBench"
    _subsets_patient_entries = {}
    for subset in dataset_subset:
        dataset_subset = load_dataset(DATASET_FILE, subset, split="test")
        patient_entries = _load_ehr_phenotype_data(dataset_subset)
        for i, patient_entry in enumerate(patient_entries):
            patient_entry["id"] = f"{subset}_{i}"
        _subsets_patient_entries[subset] = patient_entries
    return _subsets_patient_entries


def _load_ehr_phenotype_data(dataset: Dataset) -> List[Dict[str, str]]:
    """Borrowed from rarebench.py"""
    # Loads phenotype and disease mappings
    _dataset_path = Path(__file__).parent / "data/rarebench"

    phenotype_map_file = _dataset_path / "rarebench_phenotype_mapping.json"
    disease_map_file = _dataset_path / "rarebench_disease_mapping.json"
    phenotype_mapping = json.load(phenotype_map_file.open(encoding="utf-8-sig"))
    disease_mapping = json.load(disease_map_file.open(encoding="utf-8-sig"))

    patient_entries = []
    for patient_entry in dataset:
        phenotype_list = [
            phenotype_mapping[phenotype]
            for phenotype in patient_entry["Phenotype"]
            if phenotype in phenotype_mapping
        ]
        disease_list = [
            disease_mapping[disease]
            for disease in patient_entry["RareDisease"]
            if disease in disease_mapping
        ]
        phenotype = ", ".join(phenotype_list)
        disease = ", ".join(disease_list)
        if not phenotype or not disease:
            continue
        patient_entries.append({"phenotypes": phenotype, "pathology": disease})

    return patient_entries


def main_diagnosis_options():
    # Rarebench, generating disease names from list of accepted disease names
    # Get the directory of the current file

    current_dir = Path(__file__).parent
    gpt4 = OpenAIAzureChat(model_name="gpt-4o")

    # Code for rarebench
    rarebench_diagnosis_options_path = current_dir / "data/rarebench/diagnosis_options.json"
    rarebench_disease_mapping_path = current_dir / "data/rarebench/disease_mapping.json"
    dataset_names = ["HMS", "LIRICAL"]
    dataset_subset = [dataset_names[1]]
    generate_disease_options_rarebench(
        dataset_subset=dataset_subset,
        diagnosis_options_path=rarebench_diagnosis_options_path,
        disease_mapping_path=rarebench_disease_mapping_path,
        model=gpt4,
    )

    # # Code for icraftmd
    # icraftmd_diagnosis_options_path = current_dir / "data/icraftmd/diagnosis_options.txt"
    # icraftmd_disease_mapping_path = current_dir / "data/icraftmd/disease_mapping.json"
    # icraft_normalized_disease_mapping_path = (
    #     current_dir / "data/icraftmd/normalized_disease_mapping.json"
    # )
    # icraftmd_path = current_dir / "data/icraftmd/all_craft_md.jsonl"
    # # extract_disease_mapping_icraftmd(
    # #     icraftmd_path=icraftmd_path,
    # #     disease_mapping_path=icraftmd_disease_mapping_path,
    # # )
    # extract_disease_options_icraftmd(
    #     disease_mapping_path=icraftmd_disease_mapping_path,
    #     diagnosis_options_path=icraftmd_diagnosis_options_path,
    # )

if __name__ == "__main__":
    # Paths in below function expects you to run the file from within the ddxdriver directory
    main_diagnosis_options()
