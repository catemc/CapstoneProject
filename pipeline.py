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
    pdf_folder = Path("./data/input_papers")
    results_folder = Path("./test_results/")
    results_folder.mkdir(parents=True, exist_ok=True)

    # Testing with one file
    for pdf_file in [Path("./data/input_papers/2015+papers/srep19474.pdf")]:
        # Extract text
        pdf_to_text_converter = PdfToTextConverter(
            str(pdf_file),
            api_key
        )
        pdf_to_text_converter.convert()
        pdf_to_text_converter.write_full_paper_text()

        # Extract annotations
        f = open("tmp/tmptvr4m022/full_paper_text.txt")
        full_text = f.read()
        f.close()
        genotype_phenotype_extractor = GenotypePhenotypeExtractor(
            api_key,
            full_text
        )
        genotype_phenotype_extractor.iteratively_extract()
        genotype_phenotype_extractor.write_annotations_to_file(results_folder / "annotations.txt")
        # Writing conversation to file with iteratively adding new instructions
        genotype_phenotype_extractor.write_conversation_to_file(results_folder / "conversation.txt")