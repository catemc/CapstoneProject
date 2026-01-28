from Bio import Entrez

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

handle = Entrez.esearch(db="pubmed", term=mesh_query, retmax=100000)
records = Entrez.read(handle)
pmids = records["IdList"]
file_path = "C:/Users/catem/OneDrive/Desktop/CapstoneProject/PMIDsList.txt"
with open(file_path, 'w') as file:
    file.write("\n".join(map(str, pmids)) + "\n")

print(pmids)
print(f"Found {len(pmids)} free full-text articles.")