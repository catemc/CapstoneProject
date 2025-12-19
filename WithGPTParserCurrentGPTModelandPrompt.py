from __future__ import annotations
import re
import json
import time
from pydantic import BaseModel, ValidationError
from pathlib import Path
from openai import OpenAI
from json import dumps
from pandas import DataFrame
from pathlib import Path
from base64 import b64encode
from typing import List, Optional

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

class MutationObject(BaseModel):
    protein: str
    mutation: str
    subtype: str
    effect: str
    quote: str
    citation: str
    numbering: str | None = None  # optional
    type: str | None = None

# The full return is an array of these
class MutationList(BaseModel):
    mutations: list[MutationObject]


# Folders
pdf_folder = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/2015+papers")
results_folder = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results")
results_folder.mkdir(parents=True, exist_ok=True)

# Extraction prompt
extractionPrompt = """You are an expert virologist that is focused on scientific curation.
Perform the following steps exactly as described for each mutation marker.

# Tables
Some sections of text may contain tables, indicated by [START_TABLE] and [END_TABLE].
Each table row represents structured data — typically including mutation, effect, subtype, and citation.
Interpret each row as a complete sentence describing a relationship, and extract markers and effects the same way you would from normal text.

# Number Conversion
Evaluate mutations in each paper as you get to a new paper from a PDF page and determine which numbering scheme is being used, putting it in the 'numbering' section of the JSON for each query.
Here are potential numbering schemes that can be found in papers: 'H1', 'H3', 'H5', 'N1', 'N2', etc.

# Definition of marker
A marker is an amino acid mutation in the format 'LetterNumberLetter' (e.g., E627K, Q226L) or 'NumberLetter' (e.g., 234G, 101A).
Ignore any mutation that does not include both the original and changed amino acid or a position (e.g., 'del', '316').
If the mutation text includes additional numbering systems, parentheses, or extra notes (e.g., 'E119V (H3 numbering, H5: E117V)'), only extract the canonical mutation (e.g., 'E119V') and discard everything after the first space, parenthesis, or slash.
Mutation markers can appear as V587T, 587V→T, V587T (H5 numbering), or V587T mutant. Treat all as equivalent.
If a mutation appears in parentheses, tables, or numbering explanations, you MUST still extract the mutation string itself ('R152K') and record the numbering separately.
Never discard a mutation solely because it is followed by parentheses or numbering notes.

# *** NEW COMBINED-MUTATION LOGIC ***
A marker can also refer to a group of amino acids. For example, E119D/H275Y, H275Y-I436N, V186K,K193T,G228S, and 195D/627E.
When such strings appear, DO NOT split into individual mutations.
Treat the entire combination as a single marker. List each mutation in the standard 'LetterNumberLetter' format, separated by commas, without including any proteins. For example, if the text contains multiple mutations like 'E190G and G228E', output them as 'E190G, G228E'.
All associated effects should be listed as separate objects, but linked to the same combined marker.
Do not include entries like 'Human-H6N1 HA' or 'Duck-H6N1 HA' in the <mutation> field.

If a group of mutations are listed as 'E119A/D/G', they must be expanded into fully separate mutations unless they are explicitly part of a single combined mutant construct.
When a group of mutations are paired with an additional mutation (e.g., 'E119A/D/G-H274Y'), expand this into separate combination mutations, one for each possibility.
For example, 'E119A/D/G-H274Y' must output: 'E119A, H274Y', 'E119D, H274Y', and 'E119G, H274Y' as three separate combination mutation objects.
Never output a collapsed mutation string such as 'E119A, E119D, E119G, H274Y'. Always expand into separate objects.

A single mutation such as 'E119D' must always be treated as a single marker, not part of a combination unless explicitly written as such.
Never collapse unrelated mutations appearing in the same sentence. Each mutation must remain its own marker unless explicitly given as a combined construct.

Do not allow multiple proteins in a single object. If the text produces something like 'HA1-5, HA2-5', select the protein explicitly associated with the mutation, otherwise choose the dominant protein in the sentence.
Never output more than one protein in the 'protein' field.

Do not allow multiple subtypes in the subtype field. Only one subtype may be output for every mutation object.
If the sentence contains multiple subtypes, select the subtype directly linked to the mutation or effect. If none is linked, use 'none stated'.
Never output multi-subtype strings such as 'H10N8, H7N9, H9N2'.

Subtype must never be 'none', 'None', 'null', or empty. Use only 'none stated' when a subtype is not provided.

Collapsed mutation shorthand such as 'D151E/V' must be expanded into 'D151E' and 'D151V' as separate mutations.
Likewise, 'I223R/V/T' must be expanded into 'I223R', 'I223V', and 'I223T'.
Never output the collapsed form. Always expand multi-option mutations into separate objects unless they are part of a deliberately constructed combination mutant.

If PDF parsing merges characters or words together (e.g., 'Ontheglycanarray,thisV186K...' or missing spaces), treat this as a formatting artifact.
You must still extract the correct mutation string, restoring normal spacing and mutation boundaries.

Comma-separated mutation lists such as 'V182K,K189T,G224S' must only be treated as a combination if the text explicitly describes them as a combined mutant.
If the text does not explicitly describe the mutations as a single engineered multi-mutation construct, treat each mutation independently.
For example, 'V182K,K189T,G224S' becomes three separate mutation markers unless the paper explicitly states this was a constructed triple mutant.

When identifying combination mutations, the key requirement is that the mutations must be engineered or tested together as a single construct. Do not infer combinations unless stated.
Use 'type': 'combination' only when the mutations were explicitly combined experimentally or described as a combined mutant.

Mutations such as 'K220K' (no amino acid change) represent PDF or OCR extraction errors.
If a mutation contains identical amino acids (e.g., 'K220K'), attempt to correct using context and known mappings. If correction is impossible, skip the mutation.
Never output uncorrectable same→same mutations.

# Drug-related extraction rules
When a list of drugs appears (e.g., zanamivir, oseltamivir, peramivir, laninamivir),
 you must extract and process EVERY drug explicitly.
 Always generate one complete effect object per drug, even when they appear in the
 same sentence, clause, or table row.

If multiple drugs are listed together, treat this as a loop:
     "zanamivir, oseltamivir, peramivir, and laninamivir"
and extract four separate effect entries.

If fold-change or inhibition values are listed in parentheses or separated by commas, assume they map in order to the listed drugs and still output one entry per drug.

Never stop after the first or second drug. Do not skip any drug in a list.

 Example requirement:
  From the sentence:
    “E119D mutant exhibited a marked increase in the IC50 against all NAIs
     (827-, 25-, 286-, and 702-fold for zanamivir, oseltamivir, peramivir,
     and laninamivir, respectively)”
  you must output FOUR entries:
      – Reduced inhibition to Zanamivir
      – Reduced inhibition to Oseltamivir
      – Reduced inhibition to Peramivir
      – Reduced inhibition to Laninamivir
  All four entries share the same mutation, subtype, quote, and citation fields.

Whenever you see any of these drugs — zanamivir, oseltamivir, peramivir, laninamivir - double-check that you extract every corresponding mutation-drug effect pair.

# Effect extraction rules
Each output object must contain exactly one effect.
 If a sentence contains multiple effects OR multiple drugs, generate multiple objects,
 one per effect per drug.

If a mutation has multiple effects, create a separate object for each effect, duplicating all other fields (mutation, subtype, quote, citation).

If multiple mutations are listed together with one shared effect, assign that effect to ALL mutations individually.

Mutations mentioned together but described independently must each be extracted as valid, separate entries.

# Definitions and distinctions
 "Reduced susceptibility"
  – Indicates decreased virus responsiveness in whole-virus or infectivity assays.
  – Example quote:
      “Drug susceptibility profiling revealed that E119 mutations conferred reduced
       susceptibility mainly to zanamivir…”
 “Reduced inhibition”
  – Indicates decreased inhibition of an enzyme or protein in biochemical assays.
  – Example quote:
      “E119D conferred reduced inhibition by oseltamivir (95-fold increase in IC50).”
 These terms represent different biological phenomena.
  Both may appear in the same paper.
  Extract both when present — do not collapse them into one.

IMPORTANT CLARIFICATION:
If a paper explicitly reports "reduced susceptibility" for a mutation–drug pair,
you MUST extract it even if:
- the subtype is mentioned earlier in the paragraph but not repeated in the sentence
- the effect is described at a group level (e.g., "these mutants showed reduced susceptibility")
- the effect is inferred from whole-virus or infectivity assays

If subtype is not restated in the same sentence, inherit the most recently stated subtype
within the same section or table.

# Summary behavior
Process every drug listed.
Process every effect listed.
Process every mutation listed.
Output one object per (mutation × drug × effect).
Duplicate all other fields (subtype, quote, citation) as needed.
Never skip a drug, mutation, or effect even if they appear together in one sentence.

# Fields
For each marker, extract exactly the following fields:
- protein,
- mutation,
- subtype (use 'none stated' if not provided, but always in H[number]N[number] format if subtype-specific),
- effect (one effect per object, matched to the provided list of standard effect strings),
- quote supporting this effect,
- citation in 'Author et al., Year' format (do not include numeric IDs or brackets).
- numbering denoting which numbering scheme is used for the mutation mentioned
Do not create an object if any required field is missing, empty, or 'none'/'none stated'.
Deduplicate strictly by (normalized mutation, subtype, effect) and keep only the first occurrence.

# Minimal text
Use only the minimal portion of text that directly supports the effect, not the full paragraph or citation.
Assign exactly one effect per annotation and match it to the closest string from the provided unique effects list.
If a mutation effect is not subtype-specific, do not create multiple objects for different subtypes.

Use this list of proteins to validate mutation proteins. Only create an object if the protein matches one in this list.
    "M1", "M2", "NS-1", "NS-2", "NP", "PB2", "PB1", "PB1-F2", "PA", "HA1-5", "HA2-5", "NA-1"
For each marker, identify the a quote that states the effect of the mutation.
Use only the minimal portion of the text that directly supports the effect, not the full paragraph or citation.
Using this list of unique effects match one to each marker based on direct quotes from the text:
['Increased virulence in mice' 'Increased virulence in chickens'
'Increased virulence in ducks' 'Increased resistance to amantadine'
'Increased resistance to rimantadine'
'Decreased antiviral response in mice' 'Enhanced replication in mammalian cells'
'Enhanced pathogenicity in mice'
'Decreased replication in mammalian cells' 'Enhanced interferon response'
'Increased virulence in swine'
'Increased viral replication in mammalian cells' 'Decreased interferon response'
'Decreased interferon response in chickens'
'Decreased antiviral response in ferrets' 'Decreased replication in avian cells'
'Increased polymerase activity in mammalian cells'
'Decreased polymerase activity in mammalian cells'
'Increased polymerase activity in chickens (but not ducks)'
'Increased replication in chickens (but not ducks)'
'Increased replication in avian cells'
'Increased replication in mammalian cells'
'Decreased pathogenicity in mice'
'Increased polymerase activity in avian cells'
'Decreased virulence in mice' 'Decreased polymerase activity in mice'
'Decreased virulence in ducks' 'Decreased virulence in ferrets'
'Increased polymerase activity in mice' 'Decreased replication in ferrets'
'Enhanced replication in mice' 'Enhanced virulence in mice'
'Enhanced antiviral response in mice' 'Decreased polymerase activity in ducks'
'Decreased replication in ducks' 'Enhanced replication in duck cells'
'Increased polymerase activity in duck cells'
'Increased polymerase activity in mice cells' 'Enhanced replication in mice cells'
'Increased pseudovirus binding to α2-6' 'Increased virus binding to α2-6'
'Increased transmission in guinea pigs' 'Increased virus binding to α2-3'
'Decreased virus binding to α2-3' 'Maintained virus binding to α2-3'
'Enhanced binding affinity to mammalian cells' 'Dual receptor specificity'
'Decreased pH of fusion' 'Increased HA stability'
'Increased viral replication efficiency' 'Increased pH of fusion'
'Decreased HA stability' 'Transmitted via aerosol among ferrets'
'Transmissible among ferrets' 'Reduced lethality in mice' 'Systemic spread in mice'
'Loss of binding to α2–3' 'No binding to α2–6' 'Decreased antiviral response in host'
'Reduced tissue tropism in guinea pigs' 'Reduced susceptibility to Oseltamivir'
'Reduced susceptibility to Zanamivir' 'Reduced inhibition to Oseltamivir'
'Reduced inhibition to Zanamiriv' 'From normal to reduced inhibition to Oseltamivir'
'From normal to reduced inhibition to Peramivir' 'Reduced inhibition to Laninamivir'
'Highly reduced inhibition to Zanamivir' 'Reduced susceptibility to Peramivir'
'Highly reduced inhibition to Peramivir' 'Highly reduced inhibition to Laninamivir'
'From reduced to highly reduced inhibition to Peramivir' 'Highly reduced inhibition to Oseltamivir'
'From reduced to highly reduced inhibition to Oseltamivir' 'Reduced inhibition to Peramivir'
'From normal to reduced inhibition to Zanamivir' 'From normal to highly reduced inhibition to Peramivir'
'Dual α2–3 and α2–6 binding' 'From reduced to highly reduced inhibition to Laninamivir'
'Resistance to Favipiravir' 'PA inhibitor (PAI) baloxavir susceptibility'
'Evade human BTN3A3 (inhibitor of avian influenza A viruses replication)'
'Distruption of the second sialic acid binding site (2SBS)'
'Conferred Amantidine resistance' 'Reduced susceptibility to Laninamivir'
'Resistance to human interferon-induced antiviral factor MxA'
'Increased viral replication in mice lungs' 'Increased virus thermostability'
'Decreased virus binding to α2-6' 'Contact transmission in guinea pigs'
'Contact transmission in ferrets' 'Enhanced replication in guinea pigs'
'Increased infectivity in mammalian cells' 'Prevents airborne transmission in ferrets'
'Transmitted via aerosol among guinea pigs' 'Enhanced contact transmission in ferrets'
'Enhanced replication in ferrets' 'Contributes to contact transmission in guinea pigs'
'Enhanced polymerase activity' 'Increased virulence in ferrets'
'Contributes to airborne pathogenicity in ferrets' 'Decreases virulence in chickens'
'Decreased polymerase activity in avian cells' 'Increased polymerase activity in guinea pigs'
'Increased virulence in guinea pigs'
'Increased binding breadth to glycans bearing terminal α2-3 sialic acids'
'Enhanced contact transmission in guinea pigs']

#Quotes and validation
For each marker, identify a quote that directly supports its effect.
Only create objects if all required fields are valid. Do not output empty or placeholder objects."""

# Encode PDF page to base64
def _encode_pdf(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return b64encode(f.read()).decode("utf-8")

# GPT input builders
def text_or_table(file_path: str) -> list[dict]:
    """GPT determines if page contains text, tables, mutations, references."""
    base64_pdf = _encode_pdf(file_path)

    return [
        {
            "role": "system",
            "content": """
You are an expert in virology.

Your task is to examine a PDF page and determine:

- Is this page part of the references?
- Does it contain any mutation patterns (E627K, 234G, etc.)?
- Does it contain a table?
- Do tables contain mutations?
- Does surrounding text contain mutations?

Output JSON ONLY:
{"references": bool,
 "mutation_information": bool,
 "contains_table": bool,
 "tables_contain_mutations": bool,
 "text_contain_mutations": bool,
 "examples": str}
"""
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "filename": "PDF Page",
                    "file_data": f"data:application/pdf;base64,{base64_pdf}"
                }
            ]
        }
    ]


def qpage_to_annotations(file_path: str) -> list[dict]:
    """Extract mutation annotations from full-page text."""
    base64_pdf = _encode_pdf(file_path)

    return [
        {
            "role": "system",
            "content": f"""
You are an expert in influenza virology.

Extract mutation annotations strictly following the JSON schema.

{extractionPrompt}
"""
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "filename": "Page",
                    "file_data": f"data:application/pdf;base64,{base64_pdf}"
                }
            ]
        }
    ]


def page_to_table(file_path: str) -> list[dict]:
    """Reconstruct tables from PDF via GPT (Markdown format)."""
    base64_pdf = _encode_pdf(file_path)

    return [
        {
            "role": "system",
            "content": """
You will reconstruct any tables found on this PDF page.

Return:
- success: true/false
- markdown: reconstructed table in Markdown
- description: short summary
- error: message if failed
"""
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "filename": "Page",
                    "file_data": f"data:application/pdf;base64,{base64_pdf}"
                }
            ]
        }
    ]


def qtable_to_annotations(table_md: str) -> list[dict]:
    """Extract mutation annotations from a Markdown table."""
    return [
        {
            "role": "system",
            "content": f"""
You are an expert in influenza virology.

The following is a table. Extract all mutation annotations.

{extractionPrompt}
"""
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"Process this table:\n{table_md}"
                }
            ]
        }
    ]



# Full extraction for one PDF
def extract_full_page_mutations(pdf_path: Path, client):
    """
    Fully GPT-based extraction for a single PDF.
    - GPT decides if text/tables/mutations exist
    - GPT extracts text mutations
    - GPT reconstructs and extracts table mutations
    """

    all_annotations = []

    print(f"\n=== Processing PDF: {pdf_path.name} ===")

    # 1. Ask GPT to classify the page
    classify_prompt = text_or_table(str(pdf_path))
    classify_resp = client.responses.create(
        model="gpt-4.1",
        input=classify_prompt
    )

    try:
        info = json.loads(classify_resp.output[0].content[0].text)
    except Exception:
        print("Failed to classify page — skipping.")
        return all_annotations

    if info.get("references"):
        print("Page is references — skipping.")
        return all_annotations

# Helper: normalize effects
    def normalize_effect(effect: str) -> str:
        if not effect:
            return effect

        # Only decode known unicode escapes safely
        try:
            # Replace literal \uXXXX sequences with actual characters
            import re
            effect = re.sub(
                r'\\u([0-9a-fA-F]{4})',
                lambda m: chr(int(m.group(1), 16)),
                effect
            )
        except Exception:
            pass

        # Fix dashes and whitespace
        effect = effect.replace("\u2013", "–")  # en-dash
        effect = effect.replace("\u2014", "–")  # em-dash
        effect = " ".join(effect.split())
        return effect.strip()


    # 2. Extract text-based mutations
    if info.get("text_contain_mutations"):
        print("Extracting TEXT mutations…")

        prompt = qpage_to_annotations(str(pdf_path))
        text_resp = client.responses.create(
            model="gpt-4.1",
            input=prompt
        )

        try:
            text_output = text_resp.output[0].content[0].text
            parsed_json = json.loads(text_output)
            if isinstance(parsed_json, list):
                parsed_json = {"mutations": parsed_json}

            for entry in parsed_json["mutations"]:
                # Normalize effect text
                if "effect" in entry and entry["effect"]:
                    entry["effect"] = normalize_effect(entry["effect"])
                # Determine single vs combination mutation
                if "mutation" in entry and entry["mutation"]:
                    entry["type"] = "combination" if ',' in entry["mutation"] else "single"

            muts = MutationList(**parsed_json)
            all_annotations.extend([m.model_dump() for m in muts.mutations])
        except (json.JSONDecodeError, ValidationError) as e:
            print("Failed to parse or validate GPT output:", e)

        time.sleep(2)

    # 3. Extract table mutations
    if info.get("contains_table"):
        print("Reconstructing TABLE(S)…")

        prompt = page_to_table(str(pdf_path))
        table_resp = client.responses.create(model="gpt-4.1", input=prompt)

        table_out = table_resp.output[0].content[0].text
        table_md = table_out.strip()

        if info.get("tables_contain_mutations"):
            print("Extracting TABLE mutations…")

            prompt = qtable_to_annotations(table_md)
            table_extract_resp = client.responses.create(
                model="gpt-4.1",
                input=prompt
            )

            try:
                table_output = table_extract_resp.output[0].content[0].text
                parsed_json = json.loads(table_output)
                if isinstance(parsed_json, list):
                    parsed_json = {"mutations": parsed_json}

                for entry in parsed_json["mutations"]:
                    if "effect" in entry and entry["effect"]:
                        entry["effect"] = normalize_effect(entry["effect"])
                    if "mutation" in entry and entry["mutation"]:
                        entry["type"] = "combination" if ',' in entry["mutation"] else "single"

                muts = MutationList(**parsed_json)
                all_annotations.extend([m.model_dump() for m in muts.mutations])
            except (json.JSONDecodeError, ValidationError) as e:
                print("Failed to parse or validate GPT output:", e)

            time.sleep(3)

    return all_annotations

# Pipeline process for folder

def extract_all_pdfs(client):
    """
    Processes every PDF in the directory and writes JSON outputs.
    """
    for pdf in pdf_folder.glob("*.pdf"):
        annotations = extract_full_page_mutations(pdf, client)

        out_path = results_folder / f"{pdf.stem}_mutations.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        print(f"Saved: {out_path}")

# Run code

if __name__ == "__main__":
    print("Starting full GPT-based PDF extraction pipeline...")
    total = len(list(pdf_folder.glob("*.pdf")))
    print(f"Found {total} PDFs to process.")

    extract_all_pdfs(client)

    print("Pipeline finished.")
