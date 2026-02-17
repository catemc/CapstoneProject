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
retrieval_prompt_path = PATHS["retrieval_prompt"]
with open(retrieval_prompt_path, "r", encoding="utf-8") as f:
    retrieval_system_prompt = f.read()
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

# GPT Structured Output

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
                "content": retrieval_system_prompt  # <-- use prompt from config
            },
            {"role": "user", "content": paper_entries}
        ],
        text_format=list[RelevanceDecision],
        max_output_tokens=300 * len(papers)
    )

# PUBMED QUERY

mesh_query = """
MEDLINE[SB]
AND
(
    "Influenza A Virus"[Mesh]
    OR Influenza A virus[Mesh]
    OR "Influenza A Virus, H1N1 Subtype"[Mesh]
    OR "Influenza A Virus, H3N2 Subtype"[Mesh]
    OR "Influenza A Virus, H5N1 Subtype"[Mesh]
    OR "Influenza A Virus, H7N9 Subtype"[Mesh]
    OR "Influenza A Virus, H9N2 Subtype"[Mesh]
)
AND
(
    "Amino Acid Substitution"[Mesh]
    OR "Mutation"[Mesh]
    OR "Mutation, Missense"[Mesh]
    OR "Mutagenesis"[Mesh]
    OR "Adaptation, Biological"[Mesh]
    OR "Biological Evolution"[Mesh]
    OR "Virulence"[Mesh]
    OR "Drug Resistance, Viral"[Mesh]
    OR "Virus Replication"[Mesh]
    OR "Virus Replication / genetics"[Mesh]

    OR "Receptors, Virus"[Mesh]
    OR "Viral Proteins / genetics"[Mesh]
    OR "Viral Proteins / metabolism"[Mesh]
    OR "Viral Proteins / physiology"[Mesh]

    OR "Influenza A Virus / genetics"[Mesh]
    OR "Influenza A Virus / pathogenicity"[Mesh]
    OR "Influenza A Virus / physiology"[Mesh]
    OR "Influenza A Virus / enzymology"[Mesh]

    OR "Hemagglutinin Glycoproteins, Influenza Virus / genetics"[Mesh]
    OR "Neuraminidase / genetics"[Mesh]
    OR "RNA-Dependent RNA Polymerase / genetics"[Mesh]

    OR (
        influenza A[Title/Abstract]
        AND (mutation*[Title/Abstract] OR substitution*[Title/Abstract])
    )
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



