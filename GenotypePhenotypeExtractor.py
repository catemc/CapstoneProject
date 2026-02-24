from clients.OpenAIBase import OpenAIStructuredOutputClient
from models.models import MutationList, MutationObject, OptimizedSystemPrompt, EvaluationResult, AnnotationMatch
import json
import configparser
import re
import copy

config = configparser.ConfigParser()
config.read("config.ini")

PATHS = config["paths"]
RUN = config["run"]

INVALID = {
    "",
    "none",
    "none stated",
    "not stated",
    "not specified",
    "unknown",
    "unspecified",
    "null",
    "nan",
    "n/a",
    "na"
}

MUTATION_FIXES = {
    'Q226L': 'Q222L',
    'Q227P': 'Q223P',
    'R292K': 'R293K',
    'N294S': 'N295S',
    'H274Y': 'H275Y',
    'P186L': 'P182L',
    'G228S': 'G224S',
    'K193T': 'K189T',
    'V186K': 'V182K',
    'V186G': 'V182G',
    'V186N': 'V182N',
    'N224K': 'N220K'
}

def is_valid(x):
    if x is None:
        return False
    s = str(x).strip().lower()
    return s not in INVALID

def normalize_mutation(mut_string: str) -> str:
    if not is_valid(mut_string):
        return ""

    raw_parts = [m.strip() for m in mut_string.split(",") if m.strip()]

    cleaned = []
    for part in raw_parts:
        part = re.sub(r"[A-Za-z0-9\-]+:", "", part).upper().strip()
        part = MUTATION_FIXES.get(part, part)
        cleaned.append(part)

    return ", ".join(sorted(set(cleaned)))

def normalize_effect(effect):
    if not is_valid(effect):
        return ""

    text = str(effect).lower().strip()

    replace_map = {
        'zanamiriv': 'zanamivir'
    }
    for bad, good in replace_map.items():
        text = re.sub(bad, good, text)

    # Normalize phrases
    patterns = [
        (r'highly reduced inhibition', 'reduced inhibition'),
        (r'from reduced to highly reduced inhibition to ([\w\s-]+)', r'reduced inhibition to \1'),
        (r'from reduced to reduced inhibition to ([\w\s-]+)', r'reduced inhibition to \1'),
        (r'from normal to reduced inhibition to ([\w\s-]+)', r'reduced inhibition to \1'),
        (r'from normal to highly reduced inhibition to ([\w\s-]+)', r'reduced inhibition to \1'),
        (r'enhanced contact transmission in ([\w\s-]+)', r'contact transmission in \1'),
        (r'contributes to contact transmission in ([\w\s-]+)', r'contact transmission in \1'),
        (r'enhanced replication in ([\w\s-]+)', r'increased replication in \1'),
        (r'enhanced virulence in ([\w\s-]+)', r'increased virulence in \1')
    ]

    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)

    return text.strip()
    
def normalize_subtype(subtype: str) -> str:
    """
    Normalize subtype strings:
    - Uppercase
    - Strip leading/trailing whitespace
    - Preserve full subtype string (do NOT remove 'A/' or parentheses)
    - Collapse pandemic H1N1 to 'H1N1PDM09'
    """
    if not is_valid(subtype):
        return ""

    s = str(subtype).upper().strip()

    # Preserve pandemic lineage
    if "H1N1" in s and "PDM09" in s:
        return "H1N1PDM09"

    return s

class GenotypePhenotypeExtractor:

    def __init__(self, api_key: str, full_text: str, expected_annotations: list[dict], model: str="gpt-4.1"):
        self.openai_structured_output_client = OpenAIStructuredOutputClient(api_key, model)
        self.full_text = full_text
        self.expected_annotations = expected_annotations
        self.prompt_paths = prompt_paths

        self.conversation = []
        self.annotations = []
        self.iteration_history = []
        self.conversation_history = []

        # Load system prompts
        with open(self.prompt_paths["extract_mutations"], encoding="utf-8") as f:
            self.system_prompt_extract_mutations = f.read()
            self.current_system_prompt = self.system_prompt_extract_mutations

    def extract_from_page(self, page_text: str):
        conversation = [
            {"role": "system", "content": self.current_system_prompt},
            {"role": "user", "content": page_text}
        ]

        response = self.openai_structured_output_client.call(
            conversation,
            MutationList
        )
        return response.mutations
        
    def extract(self):
        self.annotations = []

        for page in self.pages:
            # 1️. Extract from narrative text
            if page["text"].strip():
                muts = self.extract_from_page(page["text"])
                self.annotations.extend(muts)

            # 2️. Extract from tables (separately!)
            if page["tables"]:
                muts = self.extract_from_page(page["tables"])
                self.annotations.extend(muts)

    def deduplicate_annotations(self, annotations: list[MutationObject]) -> list[MutationObject]:
        seen = {}
        for m in annotations:
            key = (m.mutation, m.effect, m.subtype)
            if key not in seen:
                seen[key] = m
        return list(seen.values())
    
    def filter_invalid_annotations(self, annotations: list[MutationObject]) -> list[MutationObject]:
        cleaned = []
        for m in annotations:
            if (
                is_valid(m.mutation) and
                is_valid(m.effect) and
                is_valid(m.subtype)
            ):
                cleaned.append(m)
        return cleaned

    def iteratively_extract(self):
        self.extract()

        # Normalize
        for m in self.annotations:
            m.mutation = normalize_mutation(m.mutation)
            m.effect = normalize_effect(m.effect)
            m.subtype = normalize_subtype(m.subtype)
            m.type = "combination" if "," in m.mutation else "single"

        # 🧹 REMOVE EMPTY / INVALID ANNOTATIONS
        self.annotations = self.filter_invalid_annotations(self.annotations)

        # 🧬 DEDUPE (mutation + effect + subtype)
        self.annotations = self.deduplicate_annotations(self.annotations)

        print("\n--- Final Extracted Annotations ---")
        for m in self.annotations:
            print(m.model_dump())
        print("--- End of Extraction ---\n")

    def write_conversation_to_file(self, out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)

    def write_annotations_to_file(self, out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            annotations_to_dump = [m.model_dump() for m in self.annotations]
            json.dump(annotations_to_dump, f, indent=4, ensure_ascii=False)
    
    def write_iteration_history(self, out_path):
        """Save detailed iteration info to JSON"""
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.iteration_history, f, indent=2, ensure_ascii=False)
