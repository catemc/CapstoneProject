import json
from pathlib import Path
import pdfplumber
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import List


# ============================================================
#  DATA MODELS
# ============================================================

class MutationObject(BaseModel):
    protein: str
    mutation: str
    subtype: str
    effect: str
    quote: str
    citation: str
    numbering: str | None = None
    type: str | None = None


class MutationList(BaseModel):
    mutations: List[MutationObject]


# ============================================================
#  PDF PARSING
# ============================================================
def merge_mutations_per_effect(json_input) -> list:
    """
    Merge mutations that share the same effect, subtype, quote, citation, numbering,
    and type into a single annotation with multiple mutations separated by commas.
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
        key = (
            m["effect"],
            m["subtype"],
            m["quote"],
            m["citation"],
            m.get("numbering"),
            m.get("type"),  # <-- include type in the merge key
        )
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


# ============================================================
#  UPDATE USING GPT — compares JSON and PDF text
# ============================================================

def update_annotations(existing_json: dict, full_text: str, client) -> list:
    """
    Receives:
      - existing annotations (first-pass JSON)
      - full paper text
    Returns:
      - A complete, validated list of mutation objects:
        * keeps valid entries,
        * removes incorrect ones,
        * adds missing correct pairs.
    """

    first_pass_json = json.dumps(existing_json, indent=2)

    system_prompt = (
        "You are an expert virologist tasked with reviewing and validating influenza mutation "
        "annotations. Do NOT perform a blind extraction. Instead, perform the following steps:\n\n"

        "1. Review every mutation object in the existing JSON:\n"
        "   - Keep the object if it is fully correct, supported by the paper text, and conforms to the schema.\n"
        "   - Remove the object if it is invalid, unsupported, or formatted incorrectly.\n\n"

        "2. Read the full paper text and identify any missing genotype-phenotype pairs.\n"
        "   - Add new valid mutation objects for any missing pairs.\n\n"

        "3. Validate all mutations:\n"
        "   - Apply HA→H5 and NA→N1 numbering rules.\n"
        "   - Split combined mutations (e.g., E119D/H275Y) into individual valid mutations and recombine into '<protein>:MUT1, <protein>:MUT2' format.\n"
        "   - Only merge mutations if explicitly listed together in the source text with the same effect and quote.\n\n"

        "4. Validate effects using ONLY this list:\n"
        "   ['Increased virulence in mice', 'Increased virulence in chickens', "
        "'Increased virulence in ducks', 'Increased resistance to amantadine', "
        "'Increased resistance to rimantadine', 'Decreased antiviral response in mice', "
        "'Enhanced replication in mammalian cells', 'Enhanced pathogenicity in mice', "
        "'Decreased replication in mammalian cells', 'Enhanced interferon response', "
        "'Increased virulence in swine', 'Increased viral replication in mammalian cells', "
        "'Decreased interferon response', 'Decreased interferon response in chickens', "
        "'Decreased antiviral response in ferrets', 'Decreased replication in avian cells', "
        "'Increased polymerase activity in mammalian cells', 'Decreased polymerase activity in mammalian cells', "
        "'Increased polymerase activity in chickens (but not ducks)', "
        "'Increased replication in chickens (but not ducks)', 'Increased replication in avian cells', "
        "'Increased replication in mammalian cells', 'Decreased pathogenicity in mice', "
        "'Increased polymerase activity in avian cells', 'Decreased virulence in mice', "
        "'Decreased polymerase activity in mice', 'Decreased virulence in ducks', "
        "'Decreased virulence in ferrets', 'Increased polymerase activity in mice', "
        "'Decreased replication in ferrets', 'Enhanced replication in mice', 'Enhanced virulence in mice', "
        "'Enhanced antiviral response in mice', 'Decreased polymerase activity in ducks', "
        "'Decreased replication in ducks', 'Enhanced replication in duck cells', "
        "'Increased polymerase activity in duck cells', 'Increased polymerase activity in mice cells', "
        "'Enhanced replication in mice cells', 'Increased pseudovirus binding to α2-6', "
        "'Increased virus binding to α2-6', 'Increased transmission in guinea pigs', "
        "'Increased virus binding to α2-3', 'Decreased virus binding to α2-3', "
        "'Maintained virus binding to α2-3', 'Enhanced binding affinity to mammalian cells', "
        "'Dual receptor specificity', 'Decreased pH of fusion', 'Increased HA stability', "
        "'Increased viral replication efficiency', 'Increased pH of fusion', 'Decreased HA stability', "
        "'Transmitted via aerosol among ferrets', 'Transmissible among ferrets', 'Reduced lethality in mice', "
        "'Systemic spread in mice', 'Loss of binding to α2–3', 'No binding to α2–6', "
        "'Decreased antiviral response in host', 'Reduced tissue tropism in guinea pigs', "
        "'Reduced susceptibility to Oseltamivir', 'Reduced susceptibility to Zanamivir', "
        "'Reduced inhibition to Oseltamivir', 'Reduced inhibition to Zanamiriv', "
        "'From normal to reduced inhibition to Oseltamivir', 'From normal to reduced inhibition to Peramivir', "
        "'Reduced inhibition to Laninamivir', 'Highly reduced inhibition to Zanamivir', "
        "'Reduced susceptibility to Peramivir', 'Highly reduced inhibition to Peramivir', "
        "'Highly reduced inhibition to Laninamivir', 'From reduced to highly reduced inhibition to Peramivir', "
        "'Highly reduced inhibition to Oseltamivir', 'From reduced to highly reduced inhibition to Oseltamivir', "
        "'Reduced inhibition to Peramivir', 'From normal to reduced inhibition to Zanamivir', "
        "'From normal to highly reduced inhibition to Peramivir', 'Dual α2–3 and α2–6 binding', "
        "'From reduced to highly reduced inhibition to Laninamivir', 'Resistance to Favipiravir', "
        "'PA inhibitor (PAI) baloxavir susceptibility', "
        "'Evade human BTN3A3 (inhibitor of avian influenza A viruses replication)', "
        "'Distruption of the second sialic acid binding site (2SBS)', 'Conferred Amantidine resistance', "
        "'Reduced susceptibility to Laninamivir', 'Resistance to human interferon-induced antiviral factor MxA', "
        "'Increased viral replication in mice lungs', 'Increased virus thermostability', "
        "'Decreased virus binding to α2-6', 'Contact transmission in guinea pigs', "
        "'Contact transmission in ferrets', 'Enhanced replication in guinea pigs', "
        "'Increased infectivity in mammalian cells', 'Prevents airborne transmission in ferrets', "
        "'Transmitted via aerosol among guinea pigs', 'Enhanced contact transmission in ferrets', "
        "'Enhanced replication in ferrets', 'Contributes to contact transmission in guinea pigs', "
        "'Enhanced polymerase activity', 'Increased virulence in ferrets', "
        "'Contributes to airborne pathogenicity in ferrets', 'Decreases virulence in chickens', "
        "'Decreased polymerase activity in avian cells', 'Increased polymerase activity in guinea pigs', "
        "'Increased virulence in guinea pigs', 'Increased binding breadth to glycans bearing terminal α2-3 sialic acids', "
        "'Enhanced contact transmission in guinea pigs']\n\n"

        "Using this list of unique effects, match **exactly one effect to each mutation marker** "
        "based only on direct quotes from the paper text. Do not invent or extrapolate effects.\n\n"

        "5. Quotes and citations:\n"
        "   - Every object must have a supporting quote that matches the paper text.\n"
        "   - Citation must match the paper.\n\n"

        "6. Drugs:\n"
        "   - If multiple drugs are mentioned in a sentence, create one object per drug-effect pair.\n\n"

        "7. Output rules:\n"
        "   - Return a complete list of mutation objects, including valid existing objects and newly added ones.\n"
        "   - Exclude invalid or unsupported objects.\n"
        "   - Follow the MutationObject schema strictly (protein, mutation, subtype, effect, quote, citation, numbering).\n"
    )

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

    # Convert model objects → dicts
    validated_list = [m.model_dump() for m in resp.output_parsed.mutations]
    return validated_list

# ============================================================
#  MAIN PIPELINE
# ============================================================

def process_updates():
    client = OpenAI(api_key="")
    PDF_FOLDER = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/2015+papers")
    ANNOT_FOLDER = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results")
    OUTPUT_FOLDER = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/Updated_Results")
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Iterate ONLY over existing first-pass annotation files
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

        updated_json = update_annotations(existing, full_text, client)

        # Normalize type/combined flags
        for m in updated_json:
            if "/" in m["mutation"] or "," in m["mutation"]:
                m["type"] = "combination"
                m["combined"] = True
            else:
                m["type"] = "single"
                m["combined"] = False

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


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    process_updates()

# Run this line

if __name__ == "__main__":
    process_updates()
