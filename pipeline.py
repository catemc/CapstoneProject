from pathlib import Path
from PdfToTextConverter import PdfToTextConverter
from GenotypePhenotypeExtractor import GenotypePhenotypeExtractor
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
    
# Expected-annotation builder with normalization

def build_expected_annotations_from_lloren(csv_path: str) -> list[dict]:
    import pandas as pd
    import re

    df = pd.read_csv(csv_path)
    subset = df[df["paper_id"].str.strip().replace('"', '') == "Baek Y. et al., 2015"].copy()

    expected = []
    for _, row in subset.iterrows():
        mutation = str(row.get("combined_mutations", "")).strip().replace('"', '')
        effect = normalize_effect(str(row.get("effect_name", ""))).lower().strip()
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
        expected_annotations,
        prompt_paths=PROMPTS,
    )

    genotype_phenotype_extractor.iteratively_extract()
    genotype_phenotype_extractor.write_annotations_to_file(
        results_folder / "annotations.json"
    )
    genotype_phenotype_extractor.write_conversation_to_file(
        results_folder / "conversation.json"
    )
