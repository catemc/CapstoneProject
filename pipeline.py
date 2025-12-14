from pathlib import Path
from PdfToTextConverter import PdfToTextConverter
from GenotypePhenotypeExtractor import GenotypePhenotypeExtractor

from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

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

# =========================
# Expected-annotation builder with normalization
# =========================

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
    # Folders
    pdf_folder = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/2015+papers")
    results_folder = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results")
    results_folder.mkdir(parents=True, exist_ok=True)

    expected_annotations = []
    expected_annotations += build_expected_annotations_from_lloren(
        "C:/Users/catem/OneDrive/Desktop/CapstoneProject/Single_Mutations_Only.csv"
    )
    expected_annotations += build_expected_annotations_from_lloren(
        "C:/Users/catem/OneDrive/Desktop/CapstoneProject/Combined_Mutations_Only.csv"
    )

    # Testing with one file
    for pdf_file in [Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/2015+papers/zjv287.pdf")]:
        # Extract text
        pdf_to_text_converter = PdfToTextConverter(
            str(pdf_file),
            api_key
        )
        pdf_to_text_converter.convert()
        pdf_to_text_converter.write_full_paper_text()

        # Extract annotations
        with open(pdf_to_text_converter.full_paper_text_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        genotype_phenotype_extractor = GenotypePhenotypeExtractor(
            api_key,
            full_text,
            expected_annotations
        )
        genotype_phenotype_extractor.iteratively_extract()
        genotype_phenotype_extractor.write_annotations_to_file(results_folder / "annotations.txt")
        # Writing conversation to file with iteratively adding new instructions
        genotype_phenotype_extractor.write_conversation_to_file(results_folder / "conversation.txt")
