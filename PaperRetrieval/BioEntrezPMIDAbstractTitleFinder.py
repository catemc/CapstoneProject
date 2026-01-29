from Bio import Entrez
from time import sleep

Entrez.email = "cmcmahon@scripps.edu"
Entrez.tool = "InfluenzaMutationFetcher"

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
    retmax=0  # IMPORTANT: don't pull IDs yet
)
search_results = Entrez.read(search_handle)

count = int(search_results["Count"])
webenv = search_results["WebEnv"]
query_key = search_results["QueryKey"]

print(f"Total records found: {count}")

from time import sleep

file_path = "C:/Users/catem/OneDrive/Desktop/CapstoneProject/PMIDs_with_Abstracts.txt"

batch_size = 200

with open(file_path, "w", encoding="utf-8") as out:
    for start in range(0, count, batch_size):
        fetch_handle = Entrez.efetch(
            db="pubmed",
            rettype="abstract",
            retmode="xml",
            retstart=start,
            retmax=batch_size,
            webenv=webenv,
            query_key=query_key
        )

        records = Entrez.read(fetch_handle)

        for article in records["PubmedArticle"]:
            citation = article["MedlineCitation"]
            article_data = citation["Article"]

            pmid = citation["PMID"]
            title = article_data.get("ArticleTitle", "No title")

            abstract = "No abstract available"
            if "Abstract" in article_data:
                abstract = " ".join(
                    str(t) for t in article_data["Abstract"]["AbstractText"]
                )

            out.write(f"PMID: {pmid}\n")
            out.write(f"Title: {title}\n")
            out.write(f"Abstract: {abstract}\n")
            out.write("-" * 80 + "\n\n")

        sleep(0.4)  