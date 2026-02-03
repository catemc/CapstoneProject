from clients.OpenAIBase import OpenAIStructuredOutputClient
from Bio import Entrez
from time import sleep
from openai import OpenAI
from pydantic import BaseModel
import json
import os
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

PATHS = config["paths"]
OPENAI_API_KEY = config["openai"]["api_key"]
OPENAI_MODEL = config["openai"]["model"]

client = OpenAIStructuredOutputClient(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL
)

Entrez.email = "cmcmahon@scripps.edu"
Entrez.tool = "InfluenzaMutationFetcher"

gpt_batch_size = 20
pubmed_batch_size = 200

yes_out_path = PATHS["yes_paper_retrieval_results"]
no_out_path = PATHS["no_paper_retrieval_results"]

# ---------------------------
# GPT Structured Output
# ---------------------------
class RelevanceDecision(BaseModel):
    relevant: bool
    reason: str

def screen_relevance_batch(papers: list[dict]) -> list[RelevanceDecision]:
    paper_entries = "\n\n".join(
        f"{i+1}) Title:\n{p['title']}\nAbstract:\n{p['abstract']}"
        for i, p in enumerate(papers)
    )

    return client.call(
        conversation=[
            {
                "role": "system",
                "content": (
                    "You are a biomedical literature screening assistant. "
                    "Decide whether each paper likely contains explicit links between viral genotypes and phenotypes. "
                    "Return a single JSON array, one object per paper, with keys "
                    "'relevant' (bool) and 'reason' (string)."
                )
            },
            {"role": "user", "content": paper_entries}
        ],
        text_format=list[RelevanceDecision],
        max_output_tokens=300 * len(papers)
    )

# ---------------------------
# PUBMED QUERY
# ---------------------------
mesh_query = """
(
    "Influenza A Virus"[Mesh] OR
    "Influenza A Virus, H1N1 Subtype"[Mesh] OR
    "Influenza A Virus, H3N2 Subtype"[Mesh] OR
    "Influenza A Virus, H5N1 Subtype"[Mesh] OR
    "Influenza A Virus, H7N9 Subtype"[Mesh] OR
    "Influenza A Virus, H9N2 Subtype"[Mesh] OR
    "Influenza A Virus, H6N1 Subtype"[Mesh] OR
    "Influenza A Virus, H10N8 Subtype"[Mesh] OR
    "Influenza A Virus, H10N7 Subtype"[Mesh] OR
    "Influenza A Virus, H7N3 Subtype"[Mesh] OR
    "Influenza A Virus, H13N6 Subtype"[Mesh] OR
    "Influenza A Virus, H3N8 Subtype"[Mesh]
)
AND
(
    "Amino Acid Substitution"[Mesh] OR
    "Adaptation, Biological"[Mesh] OR
    "Biological Evolution"[Mesh] OR
    "Mutagenesis"[Mesh] OR
    "Mutagenesis, Site-Directed"[Mesh] OR
    "Receptors, Virus"[Mesh] OR
    "Influenza A Virus / genetics"[Mesh] OR
    "Influenza in Birds / physiopathology"[Mesh] OR
    "Influenza in Birds / transmission"[Mesh] OR
    "Influenza in Birds / virology"[Mesh] OR
    "Influenza, Human / virology"[Mesh] OR
    "Influenza A Virus / physiology"[Mesh] OR
    "Influenza A Virus / pathogenicity"[Mesh] OR
    "Drug Resistance, Viral"[Mesh] OR
    "Neuraminidase / genetics"[Mesh] OR
    "Viral Proteins / genetics"[Mesh] OR
    "Viral Proteins / metabolism"[Mesh] OR
    "Hemagglutinin Glycoproteins, Influenza Virus / genetics"[Mesh] OR
    "DNA-Directed RNA Polymerases / metabolism"[Mesh] OR
    "Mutation / genetics"[Mesh] OR
    "Virus Replication / genetics"[Mesh] OR
    Mutagenesis[Title/Abstract] OR
    "Amino Acid Substitution"[Title/Abstract]
)
AND free full text[Filter]
"""

search_handle = Entrez.esearch(
    db="pubmed",
    term=mesh_query,
    usehistory="y",
    retmax=0
)
search_results = Entrez.read(search_handle)
count = int(search_results["Count"])
webenv = search_results["WebEnv"]
query_key = search_results["QueryKey"]

print(f"Total records found: {count}")

# ---------------------------
# FETCH + BATCH SCREEN
# ---------------------------
with open(yes_out_path, "w", encoding="utf-8") as yes_out, \
     open(no_out_path, "w", encoding="utf-8") as no_out:

    for start in range(0, count, pubmed_batch_size):
        fetch_handle = Entrez.efetch(
            db="pubmed",
            rettype="abstract",
            retmode="xml",
            retstart=start,
            retmax=pubmed_batch_size,
            webenv=webenv,
            query_key=query_key
        )
        records = Entrez.read(fetch_handle)

        papers = []
        pmids = []
        for article in records["PubmedArticle"]:
            citation = article["MedlineCitation"]
            article_data = citation["Article"]

            pmid = str(citation["PMID"])
            title = article_data.get("ArticleTitle", "")
            abstract = ""
            if "Abstract" in article_data:
                abstract = " ".join(str(t) for t in article_data["Abstract"]["AbstractText"])

            if title and abstract:
                papers.append({"title": title, "abstract": abstract})
                pmids.append(pmid)

        # GPT sub-batching
        for i in range(0, len(papers), gpt_batch_size):
            sub_batch = papers[i:i + gpt_batch_size]
            sub_pmids = pmids[i:i + gpt_batch_size]

            decisions = screen_relevance_batch(sub_batch)

            for pmid, paper, decision in zip(sub_pmids, sub_batch, decisions):
                out_file = yes_out if decision.relevant else no_out
                out_file.write(f"PMID: {pmid}\n")
                out_file.write(f"Title: {paper['title']}\n")
                out_file.write(f"Abstract: {paper['abstract']}\n")
                out_file.write(f"Relevant: {decision.relevant}\n")
                out_file.write(f"Reason: {decision.reason}\n")
                out_file.write("-" * 80 + "\n\n")

        sleep(0.2)

print("✅ Relevance screening complete for all papers.")

