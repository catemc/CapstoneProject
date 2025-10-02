from openai import OpenAI
from PyPDF2 import PdfReader
from pathlib import Path
import json
from json_repair import repair_json

client = OpenAI(api_key="")

pdf_folder = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/pdfs")
results_folder = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results")
results_folder.mkdir(parents=True, exist_ok=True)

def genotype_phenotype(term, client):
    system_prompt = (
    "Read the text and extract all the markers mentioned. " 
    "For each marker, extract the protein, marker format, subtype, direct quote, and citation in 'Author et al., Year' format. " 
    "Group markers that are mentioned together as having combined effects. " 
    "Return a list of unique effects (the key outcome, effect, or observation described in the quote as a short string),  based on the quotes in the papers to create an \"Effects Bucket\". " 
    "Assign one or more effects to each annotation based on direct quotes. "
    )
    try:
        response = client.chat.completions.create( 
            model="gpt-4.1", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": term}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "marker_extraction",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "Effects Bucket": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "annotations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "protein": {"type": "string"},
                                        "marker": {"type": "string"},
                                        "subtype": {"type": "string"},
                                        "effect": {"type": "array", "items": {"type": "string"}},
                                        "quote": {"type": "string"},
                                        "citation": {"type": "string"}
                                    },
                                    "required": ["protein", "marker"]
                                }
                            }
                        },
                        "required": ["annotations", "Effects Bucket"]
                    }
                }
            },
            max_completion_tokens=6000
        )
        content = response.choices[0].message.content
        if not content:
            return None
        return json.loads(content)
    except Exception as e:
        print(f"Error generating JSON: {e}")
        return None


def chunk_text(text, max_words=4000, overlap=200):
    words = text.split()
    for i in range(0, len(words), max_words - overlap):
        yield " ".join(words[i:i + max_words])


for pdf_path in pdf_folder.glob("*.pdf"):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() or ""

    # Use same key name as schema ("annotations")
    all_annotations = {"annotations": [], "Effects Bucket": []}

    for chunk in chunk_text(text):
        output = genotype_phenotype(chunk, client)
        if output and "annotations" in output:
            all_annotations["annotations"].extend(output["annotations"])
        if output and "Effects Bucket" in output:
            all_annotations["Effects Bucket"].extend(output["Effects Bucket"])

    # Deduplicate Effects Bucket
    all_annotations["Effects Bucket"] = list(set(all_annotations["Effects Bucket"]))

    out_file = results_folder / f"{pdf_path.stem}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, indent=4, ensure_ascii=False)

    print(f"Saved results for {pdf_path.name} → {out_file}")
