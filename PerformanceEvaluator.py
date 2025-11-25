import json
import pathlib
import re
import time
from typing import List, Dict
from pathlib import Path

import pdfplumber
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(api_key="")


# Pydantic Models

class MutationObject(BaseModel):
    protein: str
    mutation: str
    subtype: str
    effect: str
    quote: str
    citation: str
    numbering: str | None = None

class MutationList(BaseModel):
    mutations: list[MutationObject]


# Folders

pdf_folder = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/2015+papers")
results_folder = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results")
results_folder.mkdir(parents=True, exist_ok=True)


# Utilities

SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₐₑₒₓₕₖₗₘₙₚₛₜ", "0123456789aeoxhklmnpst")
def normalize_subscripts(text):
    return text.translate(SUBSCRIPT_MAP)

def clean_mutation_label(mutation):
    if not isinstance(mutation, str):
        return mutation
    return re.sub(r"\s*\(H[35]\)\s*", "", mutation).strip()



# Mutation Extraction LLM

def genotype_phenotype(term, client):
    system_prompt = (
        "You are an expert in virology and antiviral resistance. "
        "Perform the following steps exactly as described for each mutation marker. Do not skip any fields. Only output valid, complete objects. "
        
        # Tables
        "Some sections of text may contain tables, indicated by [START_TABLE] and [END_TABLE]. "
        "Each table row represents structured data — typically including mutation, effect, subtype, and citation. "
        "Interpret each row as a complete sentence describing a relationship, and extract markers and effects the same way you would from normal text. "

        # Number Conversion
        "Evaluate each paper's mutations as you get to a new PDF and determine which numbering scheme is being used, putting it in the 'numbering' section of the JSON for each query."
        "Here are potential numbering schemes that can be found in papers: 'H1', 'H3', 'H5', 'N1', 'N2', etc. "

        # Definition of marker
        "A marker is a unique amino acid mutation in the format 'LetterNumberLetter' (e.g., E627K, Q226L) or 'NumberLetter' (e.g., 234G, 101A). "
        "Normalize partial mutations (e.g., '31N') to the full form ('S31N') if the reference protein shows 'S' at that position. "
        "Ignore any mutation that does not include both the original and changed amino acid or a position (e.g., 'del', '316'). "
        "If the mutation text includes additional numbering systems, parentheses, or extra notes (e.g., 'E119V (H3 numbering, H5: E117V)'), only extract the canonical mutation (e.g., 'E119V') and discard everything after the first space, parenthesis, or slash."
        "“Mutation markers can appear as V587T, 587V→T, V587T (H5 numbering), or V587T mutant. Treat all as equivalent.”"
        
        # Strict mutation-format rules
        "Only extract a mutation if it is a SINGLE mutation in the format LetterNumberLetter or NumberLetter."

        "Never extract composite, combined, or multi-mutation strings such as:"
        "  - E119D/H275Y"
        "  - H275Y-I436N"
        "  - V186K,K193T,G228S"
        "  - 195D/627E"
        "  - any entry containing commas, slashes, hyphens, en-dashes, or multiple numbers/letters."

        "Whenever such multi-mutation strings appear, you MUST split them into individual mutations and create ONE object per mutation:"
        "  Example: 'E119D/H275Y' → 'E119D' and 'H275Y' (two separate entries)."

        "If ANY segment in the composite string does not match a valid mutation format (LetterNumberLetter or NumberLetter), ignore ONLY that segment."

        "Never output the composite string itself. Only output the normalized individual mutations."

        # Mutation Numbering Conversion Rules
        "For HA proteins, convert all mutation numbering to H5 numbering. "
        "For NA proteins, convert all mutation numbering to N1 numbering. "
        "If a mutation is provided in another numbering scheme (such as H3 numbering for HA or N2 numbering for NA), "
        "you must translate it into the corresponding H5 or N1 position before producing the final mutation string. "
        "Use domain knowledge, alignment clues, or explicit mappings provided in the text when available. "
        "If a mutation is listed in multiple numbering systems, always output the H5 (for HA) or N1 (for NA) version. "
        "If the text provides a direct mapping (e.g., 'Q226L (H3 numbering, H5: Q222L)'), ALWAYS use the H5/N1 form. "
        "If the paper uses a consistent alternative numbering scheme, infer the positional shift (e.g., H3→H5 often subtracts 4) "
        "and apply it to all HA mutations in that paper. "
        "These rules must always be applied before extracting effects. "
        "Always output the corrected H5 or N1 mutation in the 'mutation' field. "

        # Drug-related extraction rules
        " When a list of drugs appears (e.g., zanamivir, oseltamivir, peramivir, laninamivir),"
        "  you must extract and process EVERY drug explicitly."
        "  Always generate one complete effect object per drug, even when they appear in the"
        "  same sentence, clause, or table row."

        " If multiple drugs are listed together, treat this as a loop:"
        "      “zanamivir, oseltamivir, peramivir, and laninamivir”"
        "  → extract four separate effect entries."

        " If fold-change or inhibition values are listed in parentheses or separated by commas,"
        "  assume they map in order to the listed drugs and still output one entry per drug."

        " Never stop after the first or second drug. Do not skip any drug in a list."

        " Example requirement:"
        "  From the sentence:"
        "    “E119D mutant exhibited a marked increase in the IC50 against all NAIs"
        "     (827-, 25-, 286-, and 702-fold for zanamivir, oseltamivir, peramivir,"
        "     and laninamivir, respectively)”"
        "  you must output FOUR entries:"
        "      – Reduced inhibition to Zanamivir"
        "      – Reduced inhibition to Oseltamivir"
        "      – Reduced inhibition to Peramivir"
        "      – Reduced inhibition to Laninamivir"
        "  All four entries share the same mutation, subtype, quote, and citation fields."

        " Whenever you see any of these drugs — zanamivir, oseltamivir, peramivir,"
        "  laninamivir — double-check that you extract every corresponding mutation-drug effect pair."

        # Effect extraction rules
        " Each output object must contain exactly one effect."
        "  If a sentence contains multiple effects OR multiple drugs, generate multiple objects,"
        "  one per effect per drug."

        " If a mutation has multiple effects, create a separate object for each effect,"
        "  duplicating all other fields (mutation, subtype, quote, citation)."

        " If multiple mutations are listed together with one shared effect,"
        "  assign that effect to ALL mutations individually."

        " Mutations mentioned together but described independently must each be extracted as valid,"
        "  separate entries."

        # Definitions and distinctions
        " “Reduced susceptibility”"
        "  – Indicates decreased virus responsiveness in whole-virus or infectivity assays."
        "  – Example quote:"
        "      “Drug susceptibility profiling revealed that E119 mutations conferred reduced"
        "       susceptibility mainly to zanamivir…”"
        " “Reduced inhibition”"
        "  – Indicates decreased inhibition of an enzyme or protein in biochemical assays."
        "  – Example quote:"
        "      “E119D conferred reduced inhibition by oseltamivir (95-fold increase in IC50).”"

        " These terms represent different biological phenomena."
        "  Both may appear in the same paper."
        "  Extract both when present — do not collapse them into one."

        # Summary behavior
        " Process every drug listed."
        " Process every effect listed."
        " Process every mutation listed."
        " Output one object per (mutation × drug × effect)."
        " Duplicate all other fields (subtype, quote, citation) as needed."
        " Never skip a drug, mutation, or effect even if they appear together in one sentence."

        # Fields
        "For each marker, extract exactly the following fields: "
        "- protein, "
        "- mutation, "
        "- subtype (use 'none stated' if not provided, but always in H[number]N[number] format if subtype-specific), "
        "- effect (one effect per object, matched to the provided list of standard effect strings), "
        "- quote supporting this effect, "
        "- citation in 'Author et al., Year' format (do not include numeric IDs or brackets). "
        "- numbering denoting which numbering scheme is used for the mutation mentioned"
        "Do not create an object if any required field is missing, empty, or 'none'/'none stated'. "
        "Deduplicate strictly by (normalized mutation, subtype, effect) and keep only the first occurrence."

        # Minimal text
        "Use only the minimal portion of text that directly supports the effect, not the full paragraph or citation. "
        "Assign exactly one effect per annotation and match it to the closest string from the provided unique effects list. "
        "If a mutation effect is not subtype-specific, do not create multiple objects for different subtypes."

        "Use this list of proteins to validate mutation proteins. Only create an object if the protein matches one in this list. "
        + str([
            "M1", "M2", "NS-1", "NS-2", "NP", "PB2", "PB1", "PB1-F2", "PA",
            "HA1-5", "HA2-5", "NA-1"
            ])
        + ". "
        "For each marker, identify the a quote that states the effect of the mutation."
        "Use only the minimal portion of the text that directly supports the effect, not the full paragraph or citation."
        "Using this list of unique effects match one to each marker based on direct quotes from the text: "
        "['Increased virulence in mice' 'Increased virulence in chickens' "
        "'Increased virulence in ducks' 'Increased resistance to amantadine' "
        "'Increased resistance to rimantadine' "
        "'Decreased antiviral response in mice' 'Enhanced replication in mammalian cells' "
        "'Enhanced pathogenicity in mice' "
        "'Decreased replication in mammalian cells' 'Enhanced interferon response' "
        "'Increased virulence in swine' "
        "'Increased viral replication in mammalian cells' 'Decreased interferon response' "
        "'Decreased interferon response in chickens' "
        "'Decreased antiviral response in ferrets' 'Decreased replication in avian cells' "
        "'Increased polymerase activity in mammalian cells' "
        "'Decreased polymerase activity in mammalian cells' "
        "'Increased polymerase activity in chickens (but not ducks)' "
        "'Increased replication in chickens (but not ducks)' "
        "'Increased replication in avian cells' "
        "'Increased replication in mammalian cells' "
        "'Decreased pathogenicity in mice' "
        "'Increased polymerase activity in avian cells' "
        "'Decreased virulence in mice' 'Decreased polymerase activity in mice' "
        "'Decreased virulence in ducks' 'Decreased virulence in ferrets' "
        "'Increased polymerase activity in mice' 'Decreased replication in ferrets' "
        "'Enhanced replication in mice' 'Enhanced virulence in mice' "
        "'Enhanced antiviral response in mice' 'Decreased polymerase activity in ducks' "
        "'Decreased replication in ducks' 'Enhanced replication in duck cells' "
        "'Increased polymerase activity in duck cells' "
        "'Increased polymerase activity in mice cells' 'Enhanced replication in mice cells' "
        "'Increased pseudovirus binding to α2-6' 'Increased virus binding to α2-6' "
        "'Increased transmission in guinea pigs' 'Increased virus binding to α2-3' "
        "'Decreased virus binding to α2-3' 'Maintained virus binding to α2-3' "
        "'Enhanced binding affinity to mammalian cells' 'Dual receptor specificity' "
        "'Decreased pH of fusion' 'Increased HA stability' "
        "'Increased viral replication efficiency' 'Increased pH of fusion' "
        "'Decreased HA stability' 'Transmitted via aerosol among ferrets' "
        "'Transmissible among ferrets' 'Reduced lethality in mice' 'Systemic spread in mice' "
        "'Loss of binding to α2–3' 'No binding to α2–6' 'Decreased antiviral response in host' "
        "'Reduced tissue tropism in guinea pigs' 'Reduced susceptibility to Oseltamivir' "
        "'Reduced susceptibility to Zanamivir' 'Reduced inhibition to Oseltamivir' "
        "'Reduced inhibition to Zanamiriv' 'From normal to reduced inhibition to Oseltamivir' "
        "'From normal to reduced inhibition to Peramivir' 'Reduced inhibition to Laninamivir' "
        "'Highly reduced inhibition to Zanamivir' 'Reduced susceptibility to Peramivir' "
        "'Highly reduced inhibition to Peramivir' 'Highly reduced inhibition to Laninamivir' "
        "'From reduced to highly reduced inhibition to Peramivir' 'Highly reduced inhibition to Oseltamivir' "
        "'From reduced to highly reduced inhibition to Oseltamivir' 'Reduced inhibition to Peramivir' "
        "'From normal to reduced inhibition to Zanamivir' 'From normal to highly reduced inhibition to Peramivir' "
        "'Dual α2–3 and α2–6 binding' 'From reduced to highly reduced inhibition to Laninamivir' "
        "'Resistance to Favipiravir' 'PA inhibitor (PAI) baloxavir susceptibility' "
        "'Evade human BTN3A3 (inhibitor of avian influenza A viruses replication)' "
        "'Distruption of the second sialic acid binding site (2SBS)' "
        "'Conferred Amantidine resistance' 'Reduced susceptibility to Laninamivir' "
        "'Resistance to human interferon-induced antiviral factor MxA' "
        "'Increased viral replication in mice lungs' 'Increased virus thermostability' "
        "'Decreased virus binding to α2-6' 'Contact transmission in guinea pigs' "
        "'Contact transmission in ferrets' 'Enhanced replication in guinea pigs' "
        "'Increased infectivity in mammalian cells' 'Prevents airborne transmission in ferrets' "
        "'Transmitted via aerosol among guinea pigs' 'Enhanced contact transmission in ferrets' "
        "'Enhanced replication in ferrets' 'Contributes to contact transmission in guinea pigs' "
        "'Enhanced polymerase activity' 'Increased virulence in ferrets' "
        "'Contributes to airborne pathogenicity in ferrets' 'Decreases virulence in chickens' "
        "'Decreased polymerase activity in avian cells' 'Increased polymerase activity in guinea pigs' "
        "'Increased virulence in guinea pigs' "
        "'Increased binding breadth to glycans bearing terminal α2-3 sialic acids' "
        "'Enhanced contact transmission in guinea pigs'] "
    
    #Quotes and validation
    "For each marker, identify a quote that directly supports its effect. "
    "Only create objects if all required fields are valid. Do not output empty or placeholder objects."
)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "mutation_extraction",
            "schema": {
                "type": "array",                      # <-- IMPORTANT: LIST, NOT SINGLE OBJECT
                "items": {
                    "type": "object",
                    "properties": {
                        "protein": {"type": "string"},
                        "mutation": {"type": "string"},
                        "subtype": {"type": "string"},
                        "effect": {"type": "string"},
                        "quote": {"type": "string"},
                        "citation": {"type": "string"},
                        "numbering": {"type": "string"}
                    },
                    "required": [
                        "protein",
                        "mutation",
                        "subtype",
                        "effect",
                        "quote",
                        "citation"
                    ]
                }
            }
        }
    }

    try:
        response = client.responses.parse(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": term}
            ],
            max_output_tokens=8000,
            text_format=MutationList  
        )

        # Parse into validated Python objects
        parsed: MutationList = response.output_parsed

        # Convert the list back to a normal Python list of dicts
        return parsed.mutations

    except Exception as e:
        print(f"Error generating structured response: {e}")
        return None

def safe_genotype_phenotype(term, client, max_retries=5):
    for attempt in range(max_retries):
        try:
            return genotype_phenotype(term, client)
        except Exception as e:
            if "rate_limit" in str(e).lower():
                wait_time = (2 ** attempt) + 1
                time.sleep(wait_time)
            else:
                return None
    return None



# LLM Evaluation of mutation–phenotype pairs

def evaluate_annotation(annotation: dict, pdf_excerpt: str):
    prompt = f"""
You are given a PDF excerpt and a mutation–phenotype annotation.

PDF excerpt:
\"\"\"{pdf_excerpt[:2500]}\"\"\"

Annotation:
{json.dumps(annotation)}

Does the PDF excerpt support this mutation–phenotype pair?

Return JSON only:
{{"correct": true/false, "reason": "short explanation"}}    
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=500
    )
    try:
        return json.loads(response.output_text)
    except:
        return {"correct": False, "reason": "Parsing error"}



# PDF Parsing (ONE pass)

def extract_annotations_from_pdf(pdf_path):
    all_annotations = []
    page_texts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            tables = page.extract_tables()
            table_text = ""

            for t_idx, table in enumerate(tables, start=1):
                table_text += f"[START_TABLE]\n"
                for row in table:
                    row_text = "\t".join([cell.strip() if cell else "" for cell in row])
                    table_text += row_text + "\n"
                table_text += "[END_TABLE]\n"

            full_page_text = raw_text + "\n" + table_text
            page_texts.append(full_page_text.strip())

            page_annotations = safe_genotype_phenotype(full_page_text, client)
            if page_annotations:
                all_annotations.extend(page_annotations)

            time.sleep(0.7)

    normalized = []
    for a in all_annotations:
        if isinstance(a, MutationObject):
            normalized.append(a.model_dump())
        elif isinstance(a, dict):
            normalized.append(a)
    return normalized, page_texts

# MAIN LOOP
for pdf_file in pdf_folder.glob("*.pdf"):
    print(f"\n=== Processing {pdf_file.name} ===")

    annotations, page_texts = extract_annotations_from_pdf(pdf_file)

    # Deduplicate
    seen = set()
    unique_annotations = []
    for ann in annotations:
        key = (ann["mutation"].upper(), ann["effect"].lower(), ann["subtype"].lower())
        if key not in seen:
            seen.add(key)
            unique_annotations.append(ann)

    # Evaluate correctness
    evaluation_results = []
    for ann in unique_annotations:
        for page_text in page_texts:
            res = evaluate_annotation(ann, page_text)
            if res["correct"]:
                break
        evaluation_results.append({
            "mutation": ann["mutation"],
            "effect": ann["effect"],
            "subtype": ann["subtype"],
            "AI_says_correct": res["correct"],
            "reason": res["reason"]
        })

class MutationObject(BaseModel):
    protein: str | None = None
    mutation: str
    subtype: str
    effect: str
    quote: str | None = None
    citation: str | None = None
    numbering: str | None = None

class MutationList(BaseModel):
    mutations: list[MutationObject]

# DETECT MISSING MUTATION–EFFECT PAIRS (includes subtype)

full_text = "\n".join(page_texts)

# Known triples normalized exactly like the rest of the pipeline
known_triples = {
    (
        a["mutation"].upper().strip(),
        a["effect"].lower().strip(),
        a["subtype"].lower().strip()
    )
    for a in unique_annotations
}

missing_prompt = f"""
You are analyzing the complete PDF of a scientific paper.

Here is the full combined text:
\"\"\"{full_text[:100000]}\"\"\"

Extract EVERY mutation–effect–subtype relationship mentioned anywhere.
Return ONLY valid JSON in this exact format:

[
  {{
    "protein": "NA",
    "mutation": "E119V",
    "subtype": "H5N1",
    "effect": "Reduced inhibition to Oseltamivir",
    "quote": "short quote from text",
    "citation": "page or section",
    "numbering": "N2"
  }}
]

Rules:
- DO NOT include duplicates.
- mutation MUST be normalized (E119V, H275Y, Q226L).
- subtype MUST be exactly as written (H5N1, H1N1, H7N9, none stated).
- effect MUST be the text’s description (not your own invention).
- quote MUST be a minimal snippet from the text.
- If protein is not stated, use "NA".
- If subtype not stated, use "none stated".
- Return ONLY the JSON list. No explanations.
"""

missing_response = client.responses.create(
    model="gpt-4.1",
    input=missing_prompt,
    max_output_tokens=6000,
)

raw_output = missing_response.output_text


# Attempt to parse LLM output into JSON list

try:
    parsed = json.loads(raw_output)
except Exception as e:
    print("Error parsing missing-pair JSON:", e)
    parsed = []


# Validate with Pydantic so all fields exist & are consistent

validated_pairs = []
for item in parsed:
    try:
        obj = MutationObject(**item)
        validated_pairs.append(obj)
    except ValidationError as ve:
        print("Skipping invalid extracted item:", ve)

# Compare validated extracted items to known triples
missing_items = []
for obj in validated_pairs:
    key = (
        obj.mutation.upper().strip(),
        obj.effect.lower().strip(),
        obj.subtype.lower().strip()
    )
    if key not in known_triples:
        missing_items.append(obj.model_dump())

# Save missing items per paper
missing_outfile = results_folder / f"{pdf_file.stem}_missing_pairs.json"
with open(missing_outfile, "w", encoding="utf-8") as f:
    json.dump(missing_items, f, indent=4, ensure_ascii=False)

print(f"Saved missing mutation–effect–subtype items → {missing_outfile}")
