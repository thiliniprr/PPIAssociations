# -*- coding: utf-8 -*-
"""
@author: pereran
"""

from Bio import Entrez
import requests, time, math, os
from tqdm import tqdm

Entrez.email = "your.email@domain"          # NCBI requirement
N_RESULTS   = 200                           # how many PubMed hits to examine
TOP_N       = 20                            # how many “excellent” papers to keep
YEARS_BACK  = 5                             # freshness filter

# ---------- 1) SEARCH PUBMED --------------------------------------------
query = '(protein protein interaction[MeSH Terms] OR PPI[Title/Abstract]) '
query += f'AND ("{2025-YEARS_BACK}"[PDAT] : "3000"[PDAT])'

ids = Entrez.read(Entrez.esearch(db="pubmed", term=query,
                                 retmax=N_RESULTS, sort="relevance"))["IdList"]

# ---------- 2) ENRICH EACH PAPER ----------------------------------------
SEMANTIC_SCHOLAR = "https://api.semanticscholar.org/graph/v1/paper/PMID:"
CROSSREF         = "https://api.crossref.org/works/"
UNPAYWALL        = "https://api.unpaywall.org/v2/"
ALT_URL          = "https://api.altmetric.com/v1/pmid/"

papers = []
for pmid in tqdm(ids, desc="enriching"):
    paper = {"pmid": pmid}
    # a) PubMed metadata
    p = Entrez.read(Entrez.efetch(db="pubmed", id=pmid, rettype="xml"))["PubmedArticle"][0]
    art      = p["MedlineCitation"]["Article"]
    journal  = art["Journal"]["Title"]
    title    = art["ArticleTitle"]
    try:
        year = int(art["Journal"]["JournalIssue"]["PubDate"]["Year"])
    except:
        year = 0
    abstract = art.get("Abstract", {}).get("AbstractText", [""])[0]

    paper.update({"title": title, "journal": journal, "year": year, "abstract": abstract})

    # b) SemanticScholar → citation count, influential citations
    try:
        ss = requests.get(SEMANTIC_SCHOLAR + pmid +
                          "?fields=citationCount,influentialCitationCount").json()
        paper.update({"citations": ss.get("citationCount", 0),
                      "infl_citations": ss.get("influentialCitationCount", 0)})
    except Exception as e:
        paper.update({"citations": 0, "infl_citations": 0})

    # c) Crossref → journal impact factor proxy: is-referenced-by-count
    # (will not always be available)
    try:
        cr = requests.get(CROSSREF + "PMID:" + pmid).json()
        score = cr["message"].get("is-referenced-by-count", 0)
        paper.update({"journal_refs": score})
    except Exception as e:
        paper.update({"journal_refs": 0})

    # d) Unpaywall → is this Open Access? (needs email parameter at their site)
    try:
        uq = requests.get(UNPAYWALL + pmid, params={'email': Entrez.email}).json()
        paper["open_access"] = uq.get("is_oa", False)
    except Exception as e:
        paper["open_access"] = False

    # e) Altmetric (optional, may not work for all PMIDs)
    try:
        alt = requests.get(ALT_URL + pmid).json()
        paper["altmetric"] = alt.get("score", 0)
    except Exception as e:
        paper["altmetric"] = 0

    papers.append(paper)

# ------------- 3) COMPOSITE SCORING & TOP N ----------------------

import pandas as pd

df = pd.DataFrame(papers)

# Replace None with 0
for k in ["citations","infl_citations","journal_refs","altmetric","year"]:
    df[k] = df[k].fillna(0)

# Normalize quality fields (min-max normalization, avoid divide-by-zero)
def norm(col):
    if df[col].max() - df[col].min() > 0:
        return (df[col] - df[col].min())/(df[col].max() - df[col].min())
    else:
        return df[col]*0

df["cit_score"] = norm("citations")
df["inflcit_score"] = norm("infl_citations")
df["journal_score"] = norm("journal_refs")
df["alt_score"] = norm("altmetric")
df["year_score"] = (df["year"] - (2025 - YEARS_BACK))/(YEARS_BACK)  # normalized 0-1 over query window

# Final composite score (tune weights as needed)
df["quality_score"] = (
    0.4 * df["cit_score"] +
    0.2 * df["inflcit_score"] +
    0.2 * df["journal_score"] +
    0.1 * df["year_score"] +
    0.1 * df["alt_score"] +
    0.05 * df["open_access"].astype(float)
)

df.sort_values("quality_score", ascending=False, inplace=True)

# ------------ 4) PRINT TOP N -----------

print("\nTop PPI Papers by composite quality score:\n")

for i, row in df.head(TOP_N).iterrows():
    print(f"[{row['year']}] {row['journal']}\n{row['title']}")
    print(f"Citations: {row['citations']}  Influential: {row['infl_citations']}  Journal Refs: {row['journal_refs']} Altmetric: {row['altmetric']} OA: {row['open_access']}")
    print(f"Score: {row['quality_score']:.3f}   PMID: {row['pmid']}")
    print("Abstract:", row['abstract'][:400], "\n---\n")

