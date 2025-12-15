from clients.OpenAIBase import OpenAIStructuredOutputClient
from models.models import MutationList, OptimizedSystemPrompt, EvaluationResult
import json
import configparser
import re

config = configparser.ConfigParser()
config.read("config.ini")

PATHS = config["paths"]
PROMPTS = config["prompts"]
RUN = config["run"]

INVALID = {"", "none", "null", "nan", "n/a", "na", "unknown"}

def is_valid(x):
    if x is None:
        return False
    s = str(x).strip().lower()
    return s not in INVALID


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

class GenotypePhenotypeExtractor:

    def __init__(
        self,
        api_key: str,
        full_text: str,
        expected_annotations: list[dict],
        prompt_paths: dict,
        model: str = "gpt-4.1",
    ):
        self.prompt_paths = prompt_paths
        self.openai_structured_output_client = OpenAIStructuredOutputClient(api_key, model)
        self.full_text = full_text
        self.expected_annotations = expected_annotations
        self.conversation = []
        self.annotations = []
        self.iteration_history = []

        # Load system prompts
        with open(self.prompt_paths["extract_mutations"], encoding="utf-8") as f:
            self.system_prompt_extract_mutations = f.read()
            self.current_system_prompt = self.system_prompt_extract_mutations
        with open(self.prompt_paths["evaluate_mutations"], encoding="utf-8") as f:
            self.system_prompt_extract_mutations_evaluation = f.read()

    def extract(self):
        if len(self.conversation) == 0:
            self.conversation = [
                {"role": "system", "content": self.current_system_prompt},
                {"role": "user", "content": self.full_text}
            ]
        response = self.openai_structured_output_client.call(
            self.conversation,
            MutationList
        )
        self.annotations = response.mutations

    def evaluate(self):
        evaluation_payload = {
            "paper": "Baek Y. et al., 2015",
            "current_system_prompt": self.current_system_prompt,
            "input_text": self.full_text,
            "expected_annotations": self.expected_annotations,
            "extracted_annotations": [m.model_dump() for m in self.annotations],
            "instructions": (
                "Compare extracted_annotations against expected_annotations. "
                "Match using mutation identity, subtype equivalence, and phenotype meaning. "
                "Return:\n"
                "1) matched annotations\n"
                "2) missed annotations (expected but not extracted)\n"
                "3) hallucinated annotations (extracted but unsupported)\n"
                "Explain what was missed and why."
            )
        }

        response = self.openai_structured_output_client.call(
            [
                {"role": "system", "content": self.system_prompt_extract_mutations_evaluation},
                {"role": "user", "content": json.dumps(evaluation_payload, indent=2)}
            ],
            EvaluationResult
        )

        # Append instructions from evaluator if not converged
        if not response.converged:
            self.current_system_prompt += (
                "\n\n# Additional Extraction Instructions\n"
                + response.system_prompt
            )
            # Reset conversation cleanly
            self.conversation = [
                {"role": "system", "content": self.current_system_prompt},
                {"role": "user", "content": self.full_text}
            ]

        return response

    def iteratively_extract(self, max_steps: int = 10):
        previous_matches = 0
        self.iteration_history = []

        for step in range(max_steps):
            # Extract annotations
            self.extract()

            for m in self.annotations:
                # Strip protein prefixes consistently if needed
                m.mutation = re.sub(r"^[A-Za-z0-9\-]+:", "", m.mutation).strip()

                # Normalize effect
                m.effect = normalize_effect(m.effect).lower().strip()

                # Determine type
                m.type = "combination" if '+' in m.mutation else "single"

            # Print raw extracted objects
            print(f"\n--- Raw Extraction (Iteration {step+1}) ---")
            for m in self.annotations:
                print(m.model_dump())
            print("--- End of Extraction ---\n")

            # Run evaluation
            evaluation_response = self.evaluate()
            matched = evaluation_response.matched_annotations
            missed = evaluation_response.missed_annotations
            hallucinated = evaluation_response.hallucinated_annotations

            matched_count = len(matched)
            improvement = matched_count - previous_matches
            print(f"Iteration {step+1}: matched={matched_count}, improvement={improvement}")

            self.iteration_history.append({
                "iteration": step + 1,
                "matched_count": matched_count,
                "improvement": improvement,
                "matched_annotations": matched,
                "missed_annotations": missed,
                "hallucinated_annotations": hallucinated
            })

            previous_matches = matched_count
            if evaluation_response.converged:
                print("Converged!")
                break

    def write_conversation_to_file(self, out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.conversation, f, indent=2, ensure_ascii=False)

    def write_annotations_to_file(self, out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            annotations_to_dump = [m.model_dump() for m in self.annotations]
            json.dump(annotations_to_dump, f, indent=4, ensure_ascii=False)
    
    def write_iteration_history(self, out_path):
        """Save detailed iteration info to JSON"""
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.iteration_history, f, indent=2, ensure_ascii=False)
