from openai import OpenAI # type: ignore
from PyPDF2 import PdfReader # type: ignore
from pathlib import Path
import re
import tiktoken # type: ignore
import time
import json
from json_repair import repair_json # type: ignore
import pdfplumber # type: ignore

client = OpenAI(api_key="")

pdf_folder = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/2015+papers")
results_folder = Path("C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results")
results_folder.mkdir(parents=True, exist_ok=True)

H3_TO_H5_MAP = {
    155: 154, 156: 155, 158: 156, 159: 157, 160: 158,
    186: 185, 190: 189, 193: 190, 194: 191, 196: 193,
    198: 195, 226: 222, 228: 224
}

SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₐₑₒₓₕₖₗₘₙₚₛₜ", "0123456789aeoxhklmnpst")

def normalize_subscripts(text):
    return text.translate(SUBSCRIPT_MAP)

def extract_joint_effects(text):
    """Extract joint/combinatorial mutation phrases and remove from main text."""
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

def extract_text_with_tables(pdf_path):
    """Extract text and tables from PDF; label tables for GPT parsing."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            text += f"\n--- Page {i} ---\n" + page_text + "\n"
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables, start=1):
                text += f"\n[START_TABLE page={i} table={t_idx}]\n"
                for row in table:
                    clean_row = [cell.strip().replace("\n", " ") if cell else "" for cell in row]
                    text += "\t".join(clean_row) + "\n"
                text += "[END_TABLE]\n"
    return text

def clean_mutation_label(mutation: str) -> str:
    if not isinstance(mutation, str):
        return mutation
    return re.sub(r"\s*\(H[35]\)\s*", "", mutation).strip()

def convert_h3_to_h5(text):
    """Convert H3 numbering in text to H5 based on mapping."""
    def replace_match(match):
        aa_from = match.group("from") or ""
        pos = int(match.group("pos"))
        aa_to = match.group("to") or ""
        original = match.group(0)
        if pos in H3_TO_H5_MAP:
            new_pos = H3_TO_H5_MAP[pos]
            if aa_from and aa_to:
                return f"{aa_from}{new_pos}{aa_to} (H5)"
            elif aa_from:
                return f"{aa_from}{new_pos} (H5)"
            else:
                return f"{new_pos} (H5)"
        return original
    pattern = re.compile(r"(?P<from>[A-Z])?(?P<pos>\d{1,3})(?P<to>[A-Z])?(?:\s*\(?H3(?: numbering)?\)?)?", flags=re.IGNORECASE)
    return pattern.sub(replace_match, text)

def count_tokens(text, model="gpt-4.1"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def chunk_text_by_tokens(text, max_tokens=1000, overlap_tokens=50, model="gpt-4.1"):
    """Chunk text into token-limited segments, keeping tables intact."""
    table_pattern = re.compile(r"\[START_TABLE[\s\S]*?\[END_TABLE\]")
    chunks = []
    tables = list(table_pattern.finditer(text))
    segments = []

    last_idx = 0
    for table in tables:
        if table.start() > last_idx:
            segments.append(text[last_idx:table.start()])
        segments.append(text[table.start():table.end()])
        last_idx = table.end()
    if last_idx < len(text):
        segments.append(text[last_idx:])

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        if segment.startswith("[START_TABLE"):
            chunks.append(segment)
            continue
        paragraphs = re.split(r'\n\s*\n+', segment)
        current_chunk, current_tokens = [], 0
        for paragraph in paragraphs:
            para_tokens = count_tokens(paragraph, model)
            if para_tokens > max_tokens:
                words = paragraph.split()
                start = 0
                while start < len(words):
                    sub_words, sub_tokens = [], 0
                    while start < len(words) and sub_tokens < max_tokens:
                        sub_words.append(words[start])
                        sub_tokens = count_tokens(" ".join(sub_words), model)
                        start += 1
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk, current_tokens = [], 0
                    chunks.append(" ".join(sub_words))
            elif current_tokens + para_tokens > max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                if overlap_tokens > 0 and len(current_chunk) > overlap_tokens:
                    current_chunk = current_chunk[-overlap_tokens:]
                    current_tokens = count_tokens(" ".join(current_chunk), model)
                else:
                    current_chunk, current_tokens = [], 0
                current_chunk.extend(paragraph.split())
                current_tokens += para_tokens
            else:
                current_chunk.extend(paragraph.split())
                current_tokens += para_tokens
        if current_chunk:
            chunks.append(" ".join(current_chunk))
    return chunks

def genotype_phenotype(term, client):
    system_prompt = (
        "You are an expert in virology and antiviral resistance. "
        "Perform the following steps exactly as described for each mutation marker. Do not skip any fields. Only output valid, complete objects. "
        
        # Tables
        "Some sections of text may contain tables, indicated by [START_TABLE] and [END_TABLE]. "
        "Each table row represents structured data — typically including mutation, effect, subtype, and citation. "
        "Interpret each row as a complete sentence describing a relationship, and extract markers and effects the same way you would from normal text. "

        # H3->H5
        "Mutations may be reported using H3 numbering. Convert all H3-numbered mutations to H5 numbering using the standard mapping provided: "
        f"{H3_TO_H5_MAP}. Include the converted H5 mutation in the output, and retain the original H3 numbering in parentheses if present. "
        "Do not skip mutations originally labeled with '(H3)'. "

        # Definition of marker
        "A marker is a unique amino acid mutation in the format 'LetterNumberLetter' (e.g., E627K, Q226L) or 'NumberLetter' (e.g., 234G, 101A). "
        "Normalize partial mutations (e.g., '31N') to the full form ('S31N') if the reference protein shows 'S' at that position. "
        "Ignore any mutation that does not include both the original and changed amino acid or a position (e.g., 'del', '316'). "
        "If the mutation text includes additional numbering systems, parentheses, or extra notes (e.g., 'E119V (H3 numbering, H5: E117V)'), only extract the canonical mutation (e.g., 'E119V') and discard everything after the first space, parenthesis, or slash."

        # Effects
        "Each object must contain exactly one effect. "
        "Definitions: 'Reduced susceptibility' means the virus is less sensitive to a drug; "
        "'Reduced inhibition' means how much a viral protein or enzyme is blocked by a compound in vitro. "
        "If a mutation has multiple effects, create separate objects for each effect, duplicating other fields except the effect and quote. "
        "Only extract markers described as having an effect individually; do not extract markers that only cause effects together. "
        "Markers mentioned together but described independently are valid."

        # Fields
        "For each marker, extract exactly the following fields: "
        "- protein, "
        "- mutation (converted to H5 numbering if originally H3), "
        "- subtype (use 'none stated' if not provided, but always in H[number]N[number] format if subtype-specific), "
        "- effect (one effect per object, matched to the provided list of standard effect strings), "
        "- quote supporting this effect, "
        "- citation in 'Author et al., Year' format (do not include numeric IDs or brackets). "
        "Do not create an object if any required field is missing, empty, or 'none'/'none stated'. "
        "Deduplicate strictly by (normalized mutation, subtype, effect) and keep only the first occurrence."

        # Minimal text
        "Use only the minimal portion of text that directly supports the effect, not the full paragraph or citation. "
        "Assign exactly one effect per annotation and match it to the closest string from the provided unique effects list. "
        "If a mutation effect is not subtype-specific, do not create multiple objects for different subtypes."

        # Effects
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
                            "citation": {"type": "string"}
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
            output["mutation"] = clean_mutation_label(convert_h3_to_h5(output["mutation"]))
        all_annotations.append(output)
    elif isinstance(output, list):
        for o in output:
            if "mutation" in o:
                o["mutation"] = clean_mutation_label(convert_h3_to_h5(o["mutation"]))
        all_annotations.extend(output)
    return all_annotations

# ---- Main processing loop ----
for pdf_path in pdf_folder.glob("*.pdf"):
    text = extract_text_with_tables(pdf_path)
    text = normalize_subscripts(text)
    text = convert_h3_to_h5(text)
    text, joint_effects = extract_joint_effects(text)
    
    all_annotations = []
    for chunk in chunk_text_by_tokens(text, max_tokens=1000, overlap_tokens=50):
        output = safe_genotype_phenotype(chunk, client)
        if output:
            append_annotations(all_annotations, output)
        time.sleep(0.5)
    
    out_file = results_folder / f"{pdf_path.stem}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, indent=4, ensure_ascii=False)
    print(f"Saved results for {pdf_path.name} → {out_file}")
