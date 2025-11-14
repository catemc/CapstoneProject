import re
import json
import time
import fitz
import pdfplumber
import tiktoken
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key="")

pdf_folder = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/2015+papers")
results_folder = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results")
results_folder.mkdir(parents=True, exist_ok=True)

SUBSCRIPT_MAP = str.maketrans(
    "₀₁₂₃₄₅₆₇₈₉ₐₑₒₓₕₖₗₘₙₚₛₜ",
    "0123456789aeoxhklmnpst"
)

def normalize_subscripts(text):
    return text.translate(SUBSCRIPT_MAP)

def extract_joint_effects(text):
    joint_patterns = [
        r"(?:mutations?|variants?|substitutions?)\s+([A-Z]\d+[A-Z](?:\s*(?:and|,|/|\+|-)\s*[A-Z]\d+[A-Z])*)",
        r"(?:combination|double|triple)\s+mutation[s]?\s+([A-Z]\d+[A-Z](?:[/,]\s*[A-Z]\d+[A-Z])*)"
    ]
    joint_effects = []
    for pattern in joint_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            joint_effects.append(match.group(0))
            text = text.replace(match.group(0), "")
    return text, joint_effects

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
        max_output_tokens=3500
    )

    return json.loads(response.output_text)

def parse_pdf_into_units(pdf_path):
    doc = fitz.open(pdf_path)
    all_units = []   # each unit is a CLEAN paragraph or table
    print(f"\n=== Parsing {pdf_path.name} with GPT page parser ===")

    for i, page in enumerate(doc, start=1):
        print(f" Page {i}")
        raw = page.get_text("text")

        try:
            structured = gpt_parse_page(raw, i)
        except Exception as e:
            print(" GPT parse error:", e)
            continue

        # Append paragraph texts
        for p in structured["paragraphs"]:
            txt = p["text"].strip()
            if txt:
                all_units.append(txt)

        # Append table texts
        for t in structured["tables"]:
            txt = t["text"].strip()
            if txt:
                all_units.append(f"[TABLE]\n{txt}\n[/TABLE]")

    doc.close()
    return all_units

def clean_mutation_label(mutation):
    if not isinstance(mutation, str):
        return mutation
    return re.sub(r"\s*\(H[35]\)\s*", "", mutation).strip()

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
        + ". "
        "These rules must always be applied before extracting effects. "
        "Always output the corrected H5 or N1 mutation in the 'mutation' field. "

        # Drug-related extraction rules
        " When a list of drugs appears (e.g., zanamivir, oseltamivir, peramivir, laninamivir), you must extract and process every drug explicitly."
        "  Always generate one complete effect object per drug, even when they appear in the same sentence, clause, or table row."

        " If multiple drugs are listed together, treat this as a loop: zanamivir, oseltamivir, peramivir, and laninamivir”, and extract four separate effect entries."

        " If fold-change or inhibition values are listed in parentheses or separated by commas, assume they map in order to the listed drugs and still output one entry per drug."
        " Never stop after the first or second drug. Do not skip any drug in a list."

        " Example requirement:"
        "  From the sentence:"
        "    “E119D mutant exhibited a marked increase in the IC50 against all NAIs (827-, 25-, 286-, and 702-fold for zanamivir, oseltamivir, peramivir, and laninamivir, respectively)”"
        "  you must output FOUR entries:"
        "      – Reduced inhibition to Zanamivir"
        "      – Reduced inhibition to Oseltamivir"
        "      – Reduced inhibition to Peramivir"
        "      – Reduced inhibition to Laninamivir"
        "  All four entries share the same mutation, subtype, quote, and citation fields."

        " Whenever you see any of these drugs: zanamivir, oseltamivir, peramivir, laninamivir,  double-check that you extract every corresponding mutation-drug effect pair."

        # Effect extraction rules
        " Each output object must contain exactly one effect."
        "  If a sentence contains multiple effects OR multiple drugs, generate multiple objects, one per effect per drug."
        " If a mutation has multiple effects, create a separate object for each effect, duplicating all other fields (mutation, subtype, quote, citation)."
        " If multiple mutations are listed together with one shared effect, assign that effect to ALL mutations individually."
        " Mutations mentioned together but described independently must each be extracted as valid, separate entries."

        # Definitions and distinctions
        " “Reduced susceptibility”"
        "  – Indicates decreased virus responsiveness in whole-virus or infectivity assays."
        "  – Example quote:"
        "      “Drug susceptibility profiling revealed that E119 mutations conferred reduced susceptibility mainly to zanamivir…”"
        " “Reduced inhibition”"
        "  – Indicates decreased inhibition of an enzyme or protein in biochemical assays."
        "  – Example quote:"
        "      “E119D conferred reduced inhibition by oseltamivir (95-fold increase in IC50).”"

        " These terms represent different biological phenomena."
        "  Both may appear in the same paper."
        "  Extract both when present and do not collapse them into one."

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

        "Use this list of proteins to match a mutation protein to each subtype. Do not create an object if the protein is not listed for that subtype. "
        + str({
            "H5N1": ["M1", "M2", "NS-1", "NS-2", "NP", "PB2", "PB1", "PB1-F2", "PA", "HA1-5", "HA2-5", "NA-1"],
            "H5N2": ["M2", "PA", "PB2"],
            "H7N2": ["M2", "HA1-5"],
            "H9N2": ["M2", "NP", "PB2", "PB1", "PA", "HA1-5", "HA2-5"],
            "H7N1 backbone with H5N1 NS": ["NS-1"],
            "H1N1 backbone with H5N1 NS": ["NS-1"],
            "H1N1 backbone with H5N1 HA, NA and NS": ["NS-1"],
            "H1N1 with all internal genes from H7N9": ["NS-1"],
            "H7N1": ["NS-1", "NP", "M1", "PB2", "HA1-5"],
            "H7N9": ["NP", "PB2", "PA", "HA1-5", "HA2-5", "NA-1", "PB1"],
            "H7N7": ["NP", "PB2", "PB1", "PA"],
            "H5N1 backbone with H1N1 NS": ["PB2"],
            "H4N6": ["PB2", "HA1-5"],
            "H5N1 backbone with pH1N1 PB2": ["PB2"],
            "H10N8": ["PB2"],
            "H7N3": ["PB2", "PA"],
            "H1N1 backbone with H5N1 PB2, PB1, PA and NP": ["PB1"],
            "H6N1": ["PA", "HA1-5", "PB2"],
            "H7N9 (human isolate)": ["PA"],
            "H13N6": ["HA1-5"],
            "H6N2": ["HA1-5", "PB2"],
            "H7N7 (human isolate)": ["HA1-5"],
            "H10N8 (human isolate)": ["HA1-5"],
            "H1N1": ["NA-1", "PA", "HA1-5", "NS-1", "PB2"],
            "A(H1N1)pdm09": ["NA-1", "PA"],
            "Unknown": ["PB1", "NP", "NA-1", "M2"],
            "A(H1N1)": ["PA"],
            "H4N2": ["NA-1"],
            "H10N7": ["PB2"],
            "H1N2": ["HA1-5", "PB2"],
            "H6N6": ["HA1-5", "PA", "PB2"],
            "H5N6": ["HA1-5"],
            "H3N2": ["HA1-5", "PB2"],
            "H2N2": ["HA1-5"],
            "H3N1": ["HA1-5"],
            "H5N9": ["PB2"],
            "H3N2 (avian)": ["PB2"],
            "H3N2 (human isolate) backbone with H9N2 HA": ["HA1-5", "HA2-5"]
        })
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
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": term}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "mutation_extraction",
                    "schema": {
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
                        "required": ["protein", "mutation", "subtype", "effect", "quote", "citation"]
                    }
                }
            },
            max_completion_tokens=2000
        )
        content = response.choices[0].message.content
        if not content:
            return None
        return json.loads(content)
    except Exception as e:
        print(f"Error generating JSON: {e}")
        return None

def safe_genotype_phenotype(term, client, max_retries=5):
    for attempt in range(max_retries):
        try:
            return genotype_phenotype(term, client)
        except Exception as e:
            if "rate_limit" in str(e).lower():
                wait_time = (2 ** attempt) + 1
                print(f"Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Error generating JSON: {e}")
                return None
    print("Max retries reached; skipping chunk.")
    return None

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

# --------------------------------------------------------
# MAIN LOOP — NOW USE PARAGRAPHS (not chunks!)
# --------------------------------------------------------

for pdf_path in pdf_folder.glob("*.pdf"):

    print(f"\n\n=== PROCESSING: {pdf_path.name} ===\n")

    # 1. Get structured paragraphs/tables via GPT
    units = parse_pdf_into_units(pdf_path)

    # 2. Normalize subscripts
    units = [normalize_subscripts(u) for u in units]

    # 3. Remove joint mutation clusters
    cleaned = []
    for u in units:
        u2, _ = extract_joint_effects(u)
        cleaned.append(u2)

    all_annotations = []

    # 4. Feed EACH PARAGRAPH / TABLE ROW individually into your extractor
    for u in cleaned:
        out = safe_genotype_phenotype(u, client)
        if out:
            append_annotations(all_annotations, out)
        time.sleep(0.7)

    # 5. Save
    out_file = results_folder / f"{pdf_path.stem}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(all_annotations)} annotations → {out_file}")
