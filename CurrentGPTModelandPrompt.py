from openai import OpenAI # type: ignore
from PyPDF2 import PdfReader # type: ignore
from pathlib import Path
import re
import tiktoken # type: ignore
import time
import json
from json_repair import repair_json # type: ignore

client = OpenAI(api_key="")

pdf_folder = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/2015+papers")
results_folder = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results")
results_folder.mkdir(parents=True, exist_ok=True)

SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₐₑₒₓₕₖₗₘₙₚₛₜ", "0123456789aeoxhklmnpst")

def normalize_subscripts(text):
    """Convert Unicode subscripts to normal text."""
    text = text.translate(SUBSCRIPT_MAP)
    return text

def extract_joint_effects(text):
    """
    Extract phrases describing joint or combinatorial effects, 
    and remove them from the main text.
    """
    joint_patterns = [
        r"(?:mutations?|variants?|substitutions?)\s+([A-Z]\d+[A-Z](?:\s*(?:and|,|/)\s*[A-Z]\d+[A-Z])+)",  # e.g., "E627K and D701N"
        r"(?:combination|double|triple)\s+mutation[s]?\s+([A-Z]\d+[A-Z](?:[/,]\s*[A-Z]\d+[A-Z])*)",       # "double mutation E627K/D701N"
    ]

    joint_effects = []
    for pattern in joint_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            joint_effects.append(match.group(0))  # full phrase
            text = text.replace(match.group(0), "")  # remove from main text

    return text, joint_effects

def count_tokens(text, model="gpt-4.1"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def chunk_text_by_tokens(text, max_tokens=1000, overlap_tokens=50, model="gpt-4.1"):
    paragraphs = re.split(r'\n\s*\n+', text.strip())
    chunks = []
    current_chunk = []
    current_tokens = 0

    for paragraph in paragraphs:
        para_tokens = count_tokens(paragraph, model)

        # If paragraph alone is bigger than max_tokens, split it by words
        if para_tokens > max_tokens:
            words = paragraph.split()
            start = 0
            while start < len(words):
                sub_words = []
                sub_tokens = 0
                while start < len(words) and sub_tokens < max_tokens:
                    sub_words.append(words[start])
                    sub_tokens = count_tokens(" ".join(sub_words), model)
                    start += 1
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                chunks.append(" ".join(sub_words))

        # If adding paragraph exceeds max_tokens → start new chunk
        elif current_tokens + para_tokens > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            # handle overlap
            if overlap_tokens > 0 and len(current_chunk) > overlap_tokens:
                current_chunk = current_chunk[-overlap_tokens:]
                current_tokens = count_tokens(" ".join(current_chunk), model)
            else:
                current_chunk = []
                current_tokens = 0
            
            current_chunk.extend(paragraph.split())
            current_tokens += para_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def genotype_phenotype(term, client):
    system_prompt = (
        "Perform all of the following steps exactly as described for each marker. Do not skip any fields. Only output valid, complete objects. "
        "You are an expert in virology. "
        "Your task is to extract markers as well as the protein the marker is affecting, its effect, subtype, direct quote, and citation from the given text. "
        "A marker is a unique amino acid mutation. "
        "Only include mutations that exactly match one of the following formats: `LetterNumberLetter` (e.g., E627K, Q226L) or `NumberLetter` (e.g., 234G, 101A). "
        "Normalize partial mutations such as '31N' to the full form 'S31N' if the reference protein shows 'S' at that position. "
        "Ignore any mutation that does not include both the original and changed amino acid or does not include a position (e.g., 'del', '316'). "
        "Each object must contain exactly one effect. "
        "If a single mutation has multiple effects, create a separate object for each effect, duplicating all other fields except for the effect and quote, which should only support that specific effect. "
        "Only extract markers described as having an effect individually. "
        "If two or more markers are described as causing an effect together, do not list those markers. "
        "Markers mentioned together but described independently are valid. "
        "For each marker, extract exactly the following fields: "
            "- protein, "
            "- mutation, "
            "- subtype (use 'none stated' if not provided, but only in 'H[number]N[number]' format if subtype-specific), "
            "- effect (exactly one effect per object, matched to the provided list of standard effect strings), "
            "- quote supporting this effect, "
            "- citation in 'Author et al., Year' format (do not include brackets or numeric IDs). "
        "Do not create an object if mutation, protein, effect, quote, or citation is missing, empty, or listed as 'none'/'none stated'. "
        "Deduplicate entries strictly by the tuple (normalized mutation, subtype, effect) and keep only the first occurrence. Ignore differences in quote or citation when deduplicating."
        "Use only the minimal portion of the text that directly supports the effect. "
        "Assign exactly one effect per annotation and match it to the closest string from the provided unique effects list. "
        "The 'subtype' field should always be a string of the form H[number]N[number]. "
        "If the mutation effect is not subtype-specific, do not create multiple objects for different subtypes. "
        "Use this list of proteins to match a mutation protein to each subtype. Do not create an object if the protein is not listed for that subtype. "
        "Citations must be in the form 'Author et al., Year'. If only numeric IDs are provided in the text, leave citation blank. "
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
    "Identify a quote from the text that supports each mutation's effect. "
    "Only create objects if all required fields are valid. Do not output empty or placeholder objects."
    )
    
    try:
        response = client.chat.completions.create( 
            model="gpt-4.1", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": term}
            ],
            response_format = {
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
                            "citation": {"type": "string"}
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
                print(f"Rate limit hit. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"Error generating JSON: {e}")
                return None
    print("Max retries reached; skipping chunk.")
    return None

for pdf_path in pdf_folder.glob("*.pdf"):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() or ""

    text = normalize_subscripts(text)
    text, joint_effects = extract_joint_effects(text)

    all_annotations = []

    for chunk in chunk_text_by_tokens(text, max_tokens=1000, overlap_tokens=50, model="gpt-4.1"):
        output = safe_genotype_phenotype(chunk, client)
        if output:
            all_annotations.append(output)
        time.sleep(0.5) 

    combined_results = []
    out_file = results_folder / f"{pdf_path.stem}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, indent=4, ensure_ascii=False)
    combined_results.extend(all_annotations)
    print(f"Saved results for {pdf_path.name} → {out_file}")

    combined_file = results_folder / "combined_results.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4, ensure_ascii=False)

    print(f"Combined results saved → {combined_file}")
