import json
from pathlib import Path
import pdfplumber
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import List


# Data models
class MutationObject(BaseModel):
    protein: str
    mutation: str
    subtype: str
    effect: str
    quote: str
    citation: str
    numbering: str | None = None


class MutationList(BaseModel):
    mutations: List[MutationObject]


# PDF parsing
def merge_mutations_per_effect(json_input) -> list:
    """
    Merge mutations that share the same effect, subtype, quote, citation, and numbering
    into a single annotation with multiple mutations separated by commas.

    Accepts:
      - A list of mutation dicts (from GPT output or previous JSON)
      - Or a dict with key "mutations"
    Returns:
      - A list of merged mutation dicts (Python objects, NOT JSON string)
    """
    # Handle both list and dict input
    if isinstance(json_input, dict) and "mutations" in json_input:
        data = json_input
    elif isinstance(json_input, list):
        data = {"mutations": json_input}
    else:
        raise ValueError("Input must be a list of dicts or a dict with key 'mutations'")

    merged = {}

    for m in data["mutations"]:
        key = (m["effect"], m["subtype"], m["quote"], m["citation"], m.get("numbering"))
        if key not in merged:
            merged[key] = m.copy()
        else:
            # Combine mutation strings, avoid duplicates
            existing_mutations = set(x.strip() for x in merged[key]["mutation"].split(","))
            new_mutations = set(x.strip() for x in m["mutation"].split(","))
            combined = sorted(existing_mutations | new_mutations)
            merged[key]["mutation"] = ", ".join(combined)

    # Return Python list of dicts 
    return list(merged.values())

def normalize_subscripts(text: str) -> str:
    sub_map = {
        "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
        "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9"
    }
    for k, v in sub_map.items():
        text = text.replace(k, v)
    return text


def extract_text(pdf_path: Path) -> str:
    """Extract full plain text from a PDF using pdfplumber."""
    output = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            cleaned = normalize_subscripts(raw)
            if cleaned.strip():
                output.append(cleaned)

    return "\n\n".join(output)


# GPT prompt

def update_annotations(existing_json: dict, full_text: str, client) -> str:
    """
    Receives:
      - existing annotations (first-pass file)
      - extracted full text
    Asks GPT to add ONLY missing mutations, ensuring full schema compliance.
    """

    first_pass_json = json.dumps(existing_json, indent=2)

    system_prompt = (
        "You are an expert in virology and antiviral resistance. "
        "Perform the following steps exactly as described for each mutation marker. Do not skip any fields. Only output valid, complete objects. "
        
        # Tables
        "Some sections of text may contain tables, indicated by [START_TABLE] and [END_TABLE]. "
        "Each table row represents structured data — typically including mutation, effect, subtype, and citation. "
        "Interpret each row as a complete sentence describing a relationship, and extract markers and effects the same way you would from normal text. "

        # Number Conversion
        "Evaluate each paper's mutations as you get to a new PDF and determine which numbering scheme is being used, putting it in the 'numbering' section of the JSON for each query. "
        "Here are potential numbering schemes that can be found in papers: 'H1', 'H3', 'H5', 'N1', 'N2', etc. "

        # Definition of marker
        "A marker is a unique amino acid mutation in the format 'LetterNumberLetter' (e.g., E627K, Q226L) or 'NumberLetter' (e.g., 234G, 101A). "
        "Normalize partial mutations (e.g., '31N') to the full form ('S31N') if the reference protein shows 'S' at that position. "
        "Ignore any mutation that does not include both the original and changed amino acid or a position (e.g., 'del', '316'). "
        "If the mutation text includes additional numbering systems, parentheses, or extra notes (e.g., 'E119V (H3 numbering, H5: E117V)'), only extract the canonical mutation (e.g., 'E119V') and discard everything after the first space, parenthesis, or slash. "
        "Mutation markers can appear as V587T, 587V→T, V587T (H5 numbering), or V587T mutant. Treat all as equivalent. "
        
        # Combined-mutation logic
        "Combined, composite, or multi-mutation expressions may appear, such as: "
        "  - E119D/H275Y "
        "  - H275Y-I436N "
        "  - V186K,K193T,G228S "
        "  - 195D/627E "
        "When such strings appear, you MUST: "
        "  1. Split the expression into individual valid mutations. "
        "  2. Validate each piece as a mutation using the strict rules. "
        "  3. Recombine ONLY the valid mutations into a single combined marker string formatted EXACTLY as: "
        "       '<protein>:MUT1, <protein>:MUT2, <protein>:MUT3' "
        "Never output the original raw combined string from the paper. "
        "Never include invalid fragments; simply omit invalid entries. "
        "The combined marker string represents the mutation set for this effect; treat the entire set as ONE marker. "
        "Only combine mutations into a single 'mutation' field if they are explicitly listed together in the source text with the same effect and quote. "
        "Mutations mentioned separately, even in the same sentence or paragraph, must each have their own annotation object. "

        # Mutation Numbering Conversion Rules
        "For HA proteins, convert all mutation numbering to H5 numbering. "
        "For NA proteins, convert all mutation numbering to N1 numbering. "
        "If a mutation is provided in another numbering scheme (such as H3 numbering for HA or N2 numbering for NA), "
        "you must translate it into the corresponding H5 or N1 position before producing the final mutation string. "
        "If the text provides a direct mapping (e.g., 'Q226L (H3 numbering, H5: Q222L)'), ALWAYS use the H5/N1 form. "
        "If the paper uses a consistent alternative numbering scheme, infer the positional shift (e.g., H3→H5 often subtracts 4) "
        "and apply it to all HA mutations in that paper. "
        "These rules must always be applied before extracting effects. "
        "Always output the corrected H5 or N1 mutation in the 'mutation' field. "

        # Drug-related extraction rules
        "When a list of drugs appears (e.g., zanamivir, oseltamivir, peramivir, laninamivir), "
        "you must extract and process EVERY drug explicitly. "
        "Always generate one complete effect object per drug, even when they appear in the same sentence, clause, or table row. "
        "If multiple drugs are listed together, treat this as a loop: 'zanamivir, oseltamivir, peramivir, and laninamivir' → extract four separate effect entries. "
        "If fold-change or inhibition values are listed in parentheses or separated by commas, assume they map in order to the listed drugs and still output one entry per drug. "
        "Never stop after the first or second drug. Do not skip any drug in a list. "

        # Effect extraction rules
        "Each output object must contain exactly one effect. "
        "If a sentence contains multiple effects OR multiple drugs, generate multiple objects, one per effect per drug. "
        "If a mutation has multiple effects, create a separate object for each effect, duplicating all other fields (mutation, subtype, quote, citation). "
        "If multiple mutations are mentioned together in the same sentence or table row with one shared effect and quote, combine them into a single 'mutation' field formatted as:"
            "<protein>:MUT1, <protein>:MUT2, <protein>:MUT3"
        "Mutations mentioned together but described independently must each be extracted as valid, separate entries. "

        # Phenotypes list
        "Use only the following list of phenotypes when extracting effects: "
        "['Increased virulence in mice', 'Increased virulence in chickens', "
        "'Increased virulence in ducks', 'Increased resistance to amantadine', "
        "'Increased resistance to rimantadine', 'Decreased antiviral response in mice', "
        "'Enhanced replication in mammalian cells', 'Enhanced pathogenicity in mice', "
        "'Decreased replication in mammalian cells', 'Enhanced interferon response', "
        "'Increased virulence in swine', 'Increased viral replication in mammalian cells', "
        "'Decreased interferon response', 'Decreased interferon response in chickens', "
        "'Decreased antiviral response in ferrets', 'Decreased replication in avian cells', "
        "'Increased polymerase activity in mammalian cells', 'Decreased polymerase activity in mammalian cells', "
        "'Increased polymerase activity in chickens (but not ducks)', 'Increased replication in chickens (but not ducks)', "
        "'Increased replication in avian cells', 'Increased replication in mammalian cells', "
        "'Decreased pathogenicity in mice', 'Increased polymerase activity in avian cells', "
        "'Decreased virulence in mice', 'Decreased polymerase activity in mice', "
        "'Decreased virulence in ducks', 'Decreased virulence in ferrets', "
        "'Increased polymerase activity in mice', 'Decreased replication in ferrets', "
        "'Enhanced replication in mice', 'Enhanced virulence in mice', "
        "'Enhanced antiviral response in mice', 'Decreased polymerase activity in ducks', "
        "'Decreased replication in ducks', 'Enhanced replication in duck cells', "
        "'Increased polymerase activity in duck cells', 'Increased polymerase activity in mice cells', "
        "'Enhanced replication in mice cells', 'Increased pseudovirus binding to α2-6', "
        "'Increased virus binding to α2-6', 'Increased transmission in guinea pigs', "
        "'Increased virus binding to α2-3', 'Decreased virus binding to α2-3', "
        "'Maintained virus binding to α2-3', 'Enhanced binding affinity to mammalian cells', "
        "'Dual receptor specificity', 'Decreased pH of fusion', 'Increased HA stability', "
        "'Increased viral replication efficiency', 'Increased pH of fusion', "
        "'Decreased HA stability', 'Transmitted via aerosol among ferrets', "
        "'Transmissible among ferrets', 'Reduced lethality in mice', 'Systemic spread in mice', "
        "'Loss of binding to α2–3', 'No binding to α2–6', 'Decreased antiviral response in host', "
        "'Reduced tissue tropism in guinea pigs', 'Reduced susceptibility to Oseltamivir', "
        "'Reduced susceptibility to Zanamivir', 'Reduced inhibition to Oseltamivir', "
        "'Reduced inhibition to Zanamiriv', 'From normal to reduced inhibition to Oseltamivir', "
        "'From normal to reduced inhibition to Peramivir', 'Reduced inhibition to Laninamivir', "
        "'Highly reduced inhibition to Zanamivir', 'Reduced susceptibility to Peramivir', "
        "'Highly reduced inhibition to Peramivir', 'Highly reduced inhibition to Laninamivir', "
        "'From reduced to highly reduced inhibition to Peramivir', 'Highly reduced inhibition to Oseltamivir', "
        "'From reduced to highly reduced inhibition to Oseltamivir', 'Reduced inhibition to Peramivir', "
        "'From normal to reduced inhibition to Zanamivir', 'From normal to highly reduced inhibition to Peramivir', "
        "'Dual α2–3 and α2–6 binding', 'From reduced to highly reduced inhibition to Laninamivir', "
        "'Resistance to Favipiravir', 'PA inhibitor (PAI) baloxavir susceptibility', "
        "'Evade human BTN3A3 (inhibitor of avian influenza A viruses replication)', "
        "'Distruption of the second sialic acid binding site (2SBS)', "
        "'Conferred Amantidine resistance', 'Reduced susceptibility to Laninamivir', "
        "'Resistance to human interferon-induced antiviral factor MxA', "
        "'Increased viral replication in mice lungs', 'Increased virus thermostability', "
        "'Decreased virus binding to α2-6', 'Contact transmission in guinea pigs', "
        "'Contact transmission in ferrets', 'Enhanced replication in guinea pigs', "
        "'Increased infectivity in mammalian cells', 'Prevents airborne transmission in ferrets', "
        "'Transmitted via aerosol among guinea pigs', 'Enhanced contact transmission in ferrets', "
        "'Enhanced replication in ferrets', 'Contributes to contact transmission in guinea pigs', "
        "'Enhanced polymerase activity', 'Increased virulence in ferrets', "
        "'Contributes to airborne pathogenicity in ferrets', 'Decreases virulence in chickens', "
        "'Decreased polymerase activity in avian cells', 'Increased polymerase activity in guinea pigs', "
        "'Increased virulence in guinea pigs', "
        "'Increased binding breadth to glycans bearing terminal α2-3 sialic acids', "
        "'Enhanced contact transmission in guinea pigs'] "
        
        # Quotes and validation
        "For each marker, identify a quote that directly supports its effect. "
        "Only create objects if all required fields are valid. Do not output empty or placeholder objects. "
    )

    # Combine with dynamic content
    prompt = f"""
{system_prompt}

EXISTING JSON:
{first_pass_json}

FULL PAPER TEXT:
{full_text}
"""

    resp = client.responses.parse(
        model="gpt-5.1",
        input=prompt,
        text_format=MutationList
    )

    # Convert model objects → dicts → JSON string
    updated_list = [m.model_dump() for m in resp.output_parsed.mutations]
    return updated_list


# Main pipeline

def process_updates():
    client = OpenAI(api_key="")
    PDF_FOLDER = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/2015+papers")
    ANNOT_FOLDER = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results")
    OUTPUT_FOLDER = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/Updated_Results")
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Iterate only over existing first-pass annotation files
    for annot_file in ANNOT_FOLDER.glob("*_fullpage_annotations.json"):
        print(f"\n=== Updating {annot_file.name} ===")

        # Determine which PDF corresponds to this annotation file
        paper_id = annot_file.stem.replace("_fullpage_annotations", "")
        pdf_path = PDF_FOLDER / f"{paper_id}.pdf"

        if not pdf_path.exists():
            print(f"WARNING: PDF not found: {pdf_path}")
            continue

        # Load existing JSON
        existing = json.loads(annot_file.read_text())

        # Extract full text from PDF
        full_text = extract_text(pdf_path)

        # Run GPT update
        updated_json = update_annotations(existing, full_text, client)

        # Merge mutations that share the same effect
        updated_json = merge_mutations_per_effect(updated_json)

        # Save updated file with NEW NAME
        new_name = annot_file.name.replace(
            "_fullpage_annotations",
            "_updated_fullpage_annotations"
        )
        out_path = OUTPUT_FOLDER / new_name

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(updated_json, f, indent=4, ensure_ascii=False)

        print(f"Saved updated file → {out_path}")

# Run this line

if __name__ == "__main__":
    process_updates()
