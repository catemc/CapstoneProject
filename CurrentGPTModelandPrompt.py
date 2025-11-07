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

def count_tokens(text, model="gpt-4.1"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def chunk_text_by_tokens(text, model="gpt-4.1", max_tokens=2000, overlap_tokens=200):
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

        # Number Conversion
        "Evaluate each paper's mutations as you get to a new PDF and determine which numbering scheme is being used, putting it in the 'numbering' section of the JSON for each query."
        "Here are potential numbering schemes that can be found in papers: 'H1', 'H3', 'H5', 'N1', 'N2', etc. "

        # Definition of marker
        "A marker is a unique amino acid mutation in the format 'LetterNumberLetter' (e.g., E627K, Q226L) or 'NumberLetter' (e.g., 234G, 101A). "
        "Normalize partial mutations (e.g., '31N') to the full form ('S31N') if the reference protein shows 'S' at that position. "
        "Ignore any mutation that does not include both the original and changed amino acid or a position (e.g., 'del', '316'). "
        "If the mutation text includes additional numbering systems, parentheses, or extra notes (e.g., 'E119V (H3 numbering, H5: E117V)'), only extract the canonical mutation (e.g., 'E119V') and discard everything after the first space, parenthesis, or slash."
        "“Mutation markers can appear as V587T, 587V→T, V587T (H5 numbering), or V587T mutant. Treat all as equivalent.”"
        
        # drug instructions 
        "Whenever a list of drugs appears, read and process each drug name explicitly. "
        "Enumerate all drugs in the list (for example: zanamivir, oseltamivir, peramivir, laninamivir) and create one complete object per drug. "
        "Do not skip the latter drugs even if earlier ones already have effects assigned."
        
        # Effects
        "Each object must contain exactly one effect. "
        "However, if a single sentence or table row contains multiple drugs or effects, you must create multiple separate objects — one per effect — while keeping all other fields the same. "
        "For example: if a sentence lists multiple drugs (e.g., zanamivir, oseltamivir, peramivir, and laninamivir), you must create four separate effect objects — one for each drug — even if the same mutation and quote apply to all of them."
        "For example, for the sentence: 'E119D mutant that exhibited a marked increase in the 50% inhibitory concentrations against all tested NAIs (827-, 25-, 286-, and 702-fold for zanamivir, oseltamivir, peramivir, and laninamivir, respectively)', your output must include FOUR separate entries:"
            "Reduced inhibition to Zanamivir" 
            "Reduced inhibition to Oseltamivir" 
            "Reduced inhibition to Peramivir"
            "Reduced inhibition to Laninamivir" 
        "Each with identical mutation, subtype, quote, and citation fields."
        "If a sentence or clause lists multiple drugs separated by commas or conjunctions (e.g., zanamivir, oseltamivir, peramivir, and laninamivir), you must treat this as a loop: extract one complete object per drug. Do not stop after the first or second drug; always continue until all listed drugs have been processed."
        "From that one sentence, you should extract four separate effects for the E119D mutation: 'Reduced inhibition to Zanamivir', 'Reduced inhibition to Oseltamivir', 'Reduced inhibition to Peramivir', and 'Reduced inhibition to Laninamivir'."
            "Whenever multiple drugs or substrates are listed together (for example: 'zanamivir, oseltamivir, peramivir, and laninamivir'), you **must create one effect object per drug** even if they are in the same sentence. Do not choose only one or skip others."
            "If fold-change or inhibition data are listed in parentheses or separated by commas or semicolons, assume they map in order to the drugs listed, and still extract a separate object for each drug name."
        "Definitions: 'Reduced susceptibility' means the virus is less sensitive to a drug. One example of a quote that shows 'Reduced susceptibility' is 'Moreover, our drug susceptibility profiling studies revealed that E119 mutations conferred reduced susceptibility mainly to zanamivir, suggesting that selective pressure is exerted by zanamivir in the N1 subtype'"
        "'Reduced inhibition' means how much a viral protein or enzyme is blocked by a compound in vitro. One example of a quote that shows 'Reduced inhibition' is 'the E119D mutation conferred reduced inhibition by oseltamivir (a 95-fold increase in the 50% inhibitory concentration'"
        "Another important distinction is that 'reduced susceptibility' refers to virological evidence (virus growth or infectivity assays) while 'reduced inhibition' refers to biochemical evidence (enzyme activity assays)"
        "Make sure to differentiate between 'reduced inhibition' and 'reduced susceptibility', both can appear in the same paper, but they describe distinct effects."
        "Both 'reduced susceptibility' and 'reduced inhibition' can be effects found in one paper, so check to make sure you aren't missing any mutation-phenotype pairs."
        "If the paper that is being analyzed mentions a drug like 'Zanamivir', 'Oseltamivir', 'Peramivir', or 'Laninamivir', take extra time to make sure you're not missing any mutation-drug effect pairs"
        "If a mutation has multiple effects, create separate objects for each effect, duplicating other fields except the effect and quote. "
        "If multiple mutations are mentioned together with a shared effect, assign the effect to all listed mutations."
        "Markers mentioned together but described independently should be treated as separate, valid entries."

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

# ---- Main processing loop ----
for pdf_path in pdf_folder.glob("*.pdf"):
    text = extract_text_with_tables(pdf_path)
    text = normalize_subscripts(text)
    text, joint_effects = extract_joint_effects(text)
    
    all_annotations = []
    for chunk in chunk_text_by_tokens(text, max_tokens=1000, overlap_tokens=50):
        output = safe_genotype_phenotype(chunk, client)
        if output:
            append_annotations(all_annotations, output)
        time.sleep(1)
    
    out_file = results_folder / f"{pdf_path.stem}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, indent=4, ensure_ascii=False)
    print(f"Saved results for {pdf_path.name} → {out_file}")
