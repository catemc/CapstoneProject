from pathlib import Path
from PdfToTextConverter import PdfToTextConverter
from GenotypePhenotypeExtractor import GenotypePhenotypeExtractor

from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

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
