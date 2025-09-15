import sqlite3
import pandas as pd
import requests

# Check if URL is actually a PDF
def is_real_pdf(url):
    try:
        r = requests.head(url, allow_redirects=True, timeout=10)
        ctype = r.headers.get("Content-Type", "").lower()
        return ctype.startswith("application/pdf")
    except:
        return False

# Get PMC PDF from DOI 
def get_pmc_pdf_from_doi(doi):
    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {"ids": doi, "format": "json"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        recs = data.get("records", [])
        if recs and "pmcid" in recs[0]:
            pmcid = recs[0]["pmcid"]
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf"
    except:
        return None
    return None

# Get PDF from Unpaywall
def get_unpaywall_pdf(doi, email="cate.mcmahon@gmail.com"):
    api_url = f"https://api.unpaywall.org/v2/{doi}"
    try:
        r = requests.get(api_url, params={"email": email}, timeout=15)
        r.raise_for_status()
        data = r.json()

        # Best OA location
        best = data.get("best_oa_location")
        if best and best.get("url_for_pdf"):
            if is_real_pdf(best["url_for_pdf"]):
                return best["url_for_pdf"]

        # Other OA locations
        for loc in data.get("oa_locations", []):
            pdf_url = loc.get("url_for_pdf")
            if pdf_url and is_real_pdf(pdf_url):
                return pdf_url
    except:
        return None
    return None

# Patterns for specific publications
def get_publisher_pdf(doi, journal):
    journal = str(journal).lower()

    if "journal of virology" in journal or "j virol" in journal:
        return f"https://journals.asm.org/doi/pdf/{doi}"
    if "journal of medical virology" in journal or "j med virol" in journal:
        return f"https://onlinelibrary.wiley.com/doi/pdf/{doi}"
    if "archives of virology" in journal or "virus genes" in journal:
        return f"https://link.springer.com/content/pdf/{doi}.pdf"

    # Elsevier journals (Virology, Antiviral Research, Virus Research, etc.)
    if "virology" in journal or "antiviral" in journal or "virus research" in journal:
        return f"https://doi.org/{doi}"  # may still be paywalled

    return None

def get_best_pdf_link(doi, journal=None, email="cmcmahon@scripps.edu"):
    #Preferences: go from PMC, Unpaywall, then DOI location
    
    # 1. Try PMC (trust it if PMCID exists)
    pmc_url = get_pmc_pdf_from_doi(doi)
    if pmc_url and pmc_url != "Not Found":
        return pmc_url

    # 2. Try Unpaywall
    try:
        api_url = f"https://api.unpaywall.org/v2/{doi}"
        r = requests.get(api_url, params={"email": email}, timeout=15)
        r.raise_for_status()
        data = r.json()

        # Best OA location
        best = data.get("best_oa_location")
        if best and best.get("url_for_pdf") and is_real_pdf(best["url_for_pdf"]):
            return best["url_for_pdf"]

        # Any OA locations
        for loc in data.get("oa_locations", []):
            pdf_url = loc.get("url_for_pdf")
            if pdf_url and is_real_pdf(pdf_url):
                return pdf_url
    except Exception:
        pass
    return f"https://doi.org/{doi}"


# Database setup
conn = sqlite3.connect('C:/Users/catem/OneDrive/Desktop/CapstoneProject/flumut_db.sqlite')

mutations = pd.read_sql_query("SELECT * FROM markers_mutations", conn)
effects = pd.read_sql_query("SELECT * FROM markers_effects", conn)
effects_dropped = effects.drop(['in_vivo', 'in_vitro'], axis=1)
merged_df = pd.merge(mutations, effects_dropped, on='marker_id')

papers = pd.read_sql_query('SELECT * FROM papers', conn)
papers = papers.rename(columns={'id': 'paper_id'})
papers_dropped = papers.drop(['title','authors','year','web_address'], axis=1)
merged_with_papers = pd.merge(merged_df, papers_dropped, on='paper_id')

# Loop over DOIs and fetch best PDF links
for index, row in merged_with_papers.iterrows():
    doi = str(row['doi']).strip()
    journal = row.get("journal", "")
    pdf_url = get_best_pdf_link(doi, journal)

    merged_with_papers.at[index, 'pdf_url'] = pdf_url
    print(f"[{index}] DOI: {doi} | Journal: {journal} -> {pdf_url}")

# Save results
output_file = 'C:/Users/catem/OneDrive/Desktop/CapstoneProject/CombinedEffectsMutationsandReferences.csv'
merged_with_papers.to_csv(output_file, index=False)
print(f"\n✅ Results saved to {output_file}")

