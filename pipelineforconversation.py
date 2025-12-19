from pathlib import Path
from PdfToTextConverter import PdfToTextConverter
from GenotypePhenotypeExtractorforConversation import GenotypePhenotypeExtractorforConversation
import configparser
import re

from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

config = configparser.ConfigParser()
config.read("config.ini")

PATHS = config["paths"]
PROMPTS = config["prompts"]
RUN = config["run"]

INVALID = {"", "none", "null", "nan", "n/a", "na"}

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


def normalize_effect(effect: str) -> str:
    """
    Normalize effect strings:
    - Lowercase & strip
    - Replace known typos
    - Collapse redundant phrases
    """
    if not is_valid(effect):
        return ""

    text = str(effect).lower().strip()

    # Known typos
    text = text.replace("zanamiriv", "zanamivir")

    # Standard phrase replacements
    patterns = [
        (r"highly reduced inhibition", "reduced inhibition"),
        (r"from reduced to highly reduced inhibition to ([\w\s-]+)", r"reduced inhibition to \1"),
        (r"from reduced to reduced inhibition to ([\w\s-]+)", r"reduced inhibition to \1"),
        (r"from normal to reduced inhibition to ([\w\s-]+)", r"reduced inhibition to \1"),
        (r"from normal to highly reduced inhibition to ([\w\s-]+)", r"reduced inhibition to \1"),
        (r"enhanced contact transmission in ([\w\s-]+)", r"contact transmission in \1"),
        (r"contributes to contact transmission in ([\w\s-]+)", r"contact transmission in \1"),
        (r"enhanced replication in ([\w\s-]+)", r"increased replication in \1"),
        (r"enhanced virulence in ([\w\s-]+)", r"increased virulence in \1")
    ]

    for pat, repl in patterns:
        text = re.sub(pat, repl, text)

    return text.strip().lower()

# Normalize mutations

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
    
# Expected-annotation builder with normalization

def build_expected_annotations_from_lloren(csv_path: str) -> list[dict]:
    import pandas as pd
    import re

    df = pd.read_csv(csv_path)
    subset = df[df["paper_id"].str.strip().replace('"', '') == "Baek Y. et al., 2015"].copy()

    expected = []
    for _, row in subset.iterrows():
        mutation = normalize_mutation(str(row.get("combined_mutations", "")).strip().replace('"', ''))
        effect = normalize_effect(row.get("effect_name", ""))
        subtype = str(row.get("subtype", "")).strip()

        expected.append({
            "mutation": mutation,
            "effect": effect,
            "subtype": subtype
        })

    return expected

if __name__ == "__main__":
    pdf_folder = Path(PATHS["input_papers"])
    results_folder = Path(PATHS["results"])
    results_folder.mkdir(parents=True, exist_ok=True)

    expected_annotations = []
    expected_annotations += build_expected_annotations_from_lloren(
        PATHS["single_mutations_csv"]
    )
    expected_annotations += build_expected_annotations_from_lloren(
        PATHS["combined_mutations_csv"]
    )

    # === RUN ONE PDF (from config.ini) ===
    pdf_file = Path(RUN["pdf_file"])

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    pdf_to_text_converter = PdfToTextConverter(
        str(pdf_file),
        api_key
    )
    pdf_to_text_converter.convert()
    pdf_to_text_converter.write_full_paper_text()

    with open(pdf_to_text_converter.full_paper_text_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    genotype_phenotype_extractor = GenotypePhenotypeExtractor(
        api_key,
        full_text,
        expected_annotations
    )

    genotype_phenotype_extractor.extract()
    for m in extractor.annotations:
            m.mutation = normalize_mutation(m.mutation)
            m.effect = normalize_effect(m.effect)
            m.subtype = normalize_subtype(m.subtype)
            m.type = "combination" if "," in m.mutation else "single"

    # Write annotations to annotations.txt
    genotype_phenotype_extractor.write_annotations_to_file(results_folder / "annotations1.txt")

    # Print summary
    evaluation = genotype_phenotype_extractor.evaluate()
    print(f"Matched: {len(evaluation.matched_annotations)}")
    print(f"Missed: {len(evaluation.missed_annotations)}")
    print(f"Hallucinated: {len(evaluation.hallucinated_annotations)}")
