import re
import json
import time
import fitz
import pdfplumber
import tiktoken
from pydantic import BaseModel
from pathlib import Path
from typing import List
from openai import OpenAI
import OpenAIBase
from base64 import b64encode

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

# Normalization helpers
def pdf_page_to_text(file: str) -> list[dict]:
    """Returns PDF page text ONLY—no extraction, no analysis."""
    with open(file, "rb") as fin:
        base64_pdf = b64encode(fin.read()).decode("utf-8")

    return [
        {
            "role": "system",
            "content": "Extract and return ONLY the plain text from the provided PDF page. Do not summarize, analyze, or interpret."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_data": f"data:application/pdf;base64,{base64_pdf}",
                    "filename": str(file),   # ✅ convert to string
                }
            ],
        },
    ]


def pdf_page_to_table(file: str) -> list[dict]:
    """Returns PDF page tables as markdown ONLY—no extraction, no interpretation."""
    with open(file, "rb") as fin:
        base64_pdf = b64encode(fin.read()).decode("utf-8")

    return [
        {
            "role": "system",
            "content": "Extract and return ONLY tables from the PDF page in markdown format. Do not analyze or interpret."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_data": f"data:application/pdf;base64,{base64_pdf}",
                    "filename": str(file)  # ✅ convert to string
                }
            ]
        }
    ]

def text_or_table(content: str) -> list[dict]:
    """Returns whether content is table-like—nothing else."""
    return [
        {
            "role": "system",
            "content": "Determine whether the provided text contains a table. Reply with 'table' or 'text'. No extraction."
        },
        {
            "role": "user",
            "content": content
        }
    ]

SUBSCRIPT_MAP = str.maketrans(
    "₀₁₂₃₄₅₆₇₈₉ₐₑₒₓₕₖₗₘₙₚₛₜ",
    "0123456789aeoxhklmnpst"
)

def normalize_subscripts(text):
    return text.translate(SUBSCRIPT_MAP)

# GPT structured page parser
def gpt_parse_page(text, page_num):
    """Return: { paragraphs: [...], tables: [...] }"""

    prompt = f"""
You are reconstructing the structure of a scientific PDF page (page {page_num}).

You will receive raw messy extracted text.

Your job:

- Identify REAL PARAGRAPHS
- Identify REAL TABLES (structured data)
- Clean line breaks, merge split sentences
- DO NOT invent information
- Return VALID JSON ONLY

Schema (must follow exactly):

{{
  "page": {page_num},
  "paragraphs": [
    {{ "id": "P{page_num}-1", "text": "..." }},
    {{ "id": "P{page_num}-2", "text": "..." }}
  ],
  "tables": [
    {{ "id": "T{page_num}-1", "text": "full table content as text" }}
  ]
}}

Raw text:
\"\"\"{text}\"\"\""""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=10000
    )

    return json.loads(response.output_text)

# Clean mutation labels
def clean_mutation_label(mutation):
    if not isinstance(mutation, str):
        return mutation
    return re.sub(r"\s*\(H[35]\)\s*", "", mutation).strip()


# GPT genotype-phenotype extractor

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
        
        # *** NEW COMBINED-MUTATION LOGIC ***
        "Combined or multi-mutation expressions may appear, such as:"
        "  - E119D/H275Y"
        "  - H275Y-I436N"
        "  - V186K,K193T,G228S"
        "  - 195D/627E"
        "When such strings appear, DO NOT split into individual mutations."
        "Treat the entire combination as a single marker. List each mutation in the standard 'LetterNumberLetter' format, separated by commas, without including any proteins. For example, if the text contains multiple mutations like 'E190G and G228E', output them as 'E190G, G228E'."
        "All associated effects should be listed as separate objects, but linked to the same combined marker."
        "Do not include entries like 'Human-H6N1 HA' or 'Duck-H6N1 HA' in the <mutation> field."

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
        "Apply the following built-in normalization corrections if the mapping is known: "
        + str({
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
        })
        + "Multi-option mutation patterns such as 'E119A/D/G' must be expanded into fully separate mutations unless they are explicitly part of a single combined mutant construct."
        "When a multi-option mutation is paired with an additional mutation (e.g., 'E119A/D/G-H274Y'), expand this into separate combination mutations, one for each possibility."
        "For example, 'E119A/D/G-H274Y' must output: 'E119A, H274Y', 'E119D, H274Y', and 'E119G, H274Y' as three separate combination mutation objects."
        "Never output a collapsed mutation string such as 'E119A, E119D, E119G, H274Y'. Always expand into separate objects."

        "A single mutation such as 'E119D' must always be treated as a single marker, not part of a combination unless explicitly written as such."
        "Never collapse unrelated mutations appearing in the same sentence. Each mutation must remain its own marker unless explicitly given as a combined construct."

        "Do not allow multiple proteins in a single object. If the text produces something like 'HA1-5, HA2-5', select the protein explicitly associated with the mutation, otherwise choose the dominant protein in the sentence."
        "Never output more than one protein in the 'protein' field."

        "Do not allow multiple subtypes in the subtype field. Only one subtype may be output for every mutation object."
        "If the sentence contains multiple subtypes, select the subtype directly linked to the mutation or effect. If none is linked, use 'none stated'."
        "Never output multi-subtype strings such as 'H10N8, H7N9, H9N2'."

        "Subtype must never be 'none', 'None', 'null', or empty. Use only 'none stated' when a subtype is not provided."

        "Collapsed mutation shorthand such as 'D151E/V' must be expanded into 'D151E' and 'D151V' as separate mutations."
        "Likewise, 'I223R/V/T' must be expanded into 'I223R', 'I223V', and 'I223T'."
        "Never output the collapsed form. Always expand multi-option mutations into separate objects unless they are part of a deliberately constructed combination mutant."

        "If PDF parsing merges characters or words together (e.g., 'Ontheglycanarray,thisV186K...' or missing spaces), treat this as a formatting artifact."
        "You must still extract the correct mutation string, restoring normal spacing and mutation boundaries."

        "Comma-separated mutation lists such as 'V182K,K189T,G224S' must only be treated as a combination if the text explicitly describes them as a combined mutant."
        "If the text does not explicitly describe the mutations as a single engineered multi-mutation construct, treat each mutation independently."
        "For example, 'V182K,K189T,G224S' becomes three separate mutation markers unless the paper explicitly states this was a constructed triple mutant."

        "When identifying combination mutations, the key requirement is that the mutations must be engineered or tested together as a single construct. Do not infer combinations unless stated."
        "Use 'type': 'combination' only when the mutations were explicitly combined experimentally or described as a combined mutant."

        "Mutations such as 'K220K' (no amino acid change) represent PDF or OCR extraction errors."
        "If a mutation contains identical amino acids (e.g., 'K220K'), attempt to correct using context and known mappings. If correction is impossible, skip the mutation."
        "Never output uncorrectable same→same mutations."

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
                "type": "array",                      
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
            max_output_tokens=12000,
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
            return genotype_phenotype(term, client)  # Your existing function
        except Exception as e:
            if "rate_limit" in str(e).lower():
                wait_time = (2 ** attempt) + 1
                print(f"Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Error: {e}")
                return None
    print("Max retries reached; skipping chunk.")
    return None

# Full-page extraction
def extract_full_page_mutations_gpt(pdf_path, client):
    """
    Fully GPT-driven extraction: fetch text and tables via GPT prompts,
    then extract mutation annotations using GPT.
    Returns a list of MutationObject annotations.
    """
    all_annotations = []

    # Convert Path to string
    pdf_file_str = str(pdf_path)

    # 1️⃣ Extract raw text from PDF via GPT prompt
    text_prompt = pdf_page_to_text(pdf_file_str)
    response = client.responses.create(
        model="gpt-4.1",
        input=text_prompt,
        max_output_tokens=10000
    )
    page_text = response.output_text.strip()
    if page_text:
        annotations = safe_genotype_phenotype(page_text, client)
        if annotations:
            all_annotations.extend(annotations)
        time.sleep(0.7)

    # 2️⃣ Extract tables from PDF via GPT prompt
    table_prompt = pdf_page_to_table(pdf_file_str)
    response = client.responses.create(
        model="gpt-4.1",
        input=table_prompt,
        max_output_tokens=10000
    )
    table_text = response.output_text.strip()
    if table_text:
        annotations = safe_genotype_phenotype(table_text, client)
        if annotations:
            all_annotations.extend(annotations)
        time.sleep(0.7)

    # 3️⃣ Clean mutation labels and assign type
    for ann in all_annotations:
        ann.mutation = clean_mutation_label(ann.mutation)
        ann.type = "combination" if "," in ann.mutation else "single"

    return all_annotations

def append_annotations(all_annotations, output):
    """Flatten GPT output and clean mutations."""
    if isinstance(output, dict):
        if "mutation" in output:
            output["mutation"] = clean_mutation_label((output["mutation"]))
        all_annotations.append(output)
    elif isinstance(output, list):
        for o in output:
            if "mutation" in o:
                o["mutation"] = clean_mutation_label((o["mutation"]))
        all_annotations.extend(output)
    return all_annotations

# Main loop

pdf_folder = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/2015+papers")
results_folder = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results")
results_folder.mkdir(parents=True, exist_ok=True)

for pdf_file in pdf_folder.glob("*.pdf"):
    annotations = extract_full_page_mutations_gpt(pdf_file, client)

    out_file = results_folder / f"{pdf_file.stem}_fullpage_annotations.json"
    with open(out_file, "w", encoding="utf-8") as f:
        annotations_to_dump = [m.model_dump() for m in annotations]
        json.dump(annotations_to_dump, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(annotations)} annotations → {out_file}")
