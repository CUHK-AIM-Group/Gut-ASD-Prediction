#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugging Face Spaces App: Microbe-Disease Relationship Mining
Supports optional full-text review: for articles initially judged as 'not_mentioned',
search local folder for full text (by DOI) and re-analyze.
"""

import os
import re
import time
import json
from typing import List, Dict, Optional, Any
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import gradio as gr
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np

# ---------- Full-text parsing dependency ----------
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: pdfplumber not installed. Cannot parse PDFs. Install with: pip install pdfplumber")

# ---------- OpenAI dependency ----------
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    raise ImportError("Please install openai: pip install openai")

# ================== PubMed utilities ==================
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def build_species_query(taxon: str) -> str:
    taxon_clean = re.sub(r'[[]\(\)_]', ' ', taxon)
    taxon_clean = re.sub(r'\s+', ' ', taxon_clean).strip()
    queries = [
        f'"{taxon_clean}"[Title/Abstract]',
        f'{taxon_clean}[Title/Abstract]',
        f'"{taxon_clean}"[All Fields]',
    ]
    return f"({' OR '.join(queries)})"

def build_species_abbreviation(full_name: str) -> Optional[str]:
    parts = re.split(r'\s+', full_name.strip())
    if len(parts) >= 2 and all(parts):
        genus = parts[0]
        rest = " ".join(parts[1:])
        if genus and genus[0].isalpha():
            return f"{genus[0]}. {rest}"
    return None

def build_species_query_with_aliases(taxon: str) -> str:
    species_qs = [build_species_query(taxon)]
    abbr = build_species_abbreviation(taxon)
    if abbr:
        abbr_clean = re.sub(r'[[]\(\)_]', ' ', abbr)
        abbr_clean = re.sub(r'\s+', ' ', abbr_clean).strip()
        abbr_queries = [
            f'"{abbr_clean}"[Title/Abstract]',
            f'{abbr_clean}[Title/Abstract]',
            f'"{abbr_clean}"[All Fields]',
        ]
        species_qs.append("(" + " OR ".join(abbr_queries) + ")")
    return "(" + " OR ".join(species_qs) + ")"

def pubmed_search(query: str, retmax: int, email: str, api_key: Optional[str] = None) -> dict:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "relevance",
        "tool": "microbe_disease_ui",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key
    r = requests.get(f"{PUBMED_BASE}/esearch.fcgi", params=params,
                     headers={"User-Agent": f"microbe_disease_ui ({email})"}, timeout=30)
    r.raise_for_status()
    return r.json()

def pubmed_fetch_xml(pmids: List[str], email: str, api_key: Optional[str] = None) -> str:
    if not pmids:
        return ""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": "microbe_disease_ui",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key
    r = requests.get(f"{PUBMED_BASE}/efetch.fcgi", params=params,
                     headers={"User-Agent": f"microbe_disease_ui ({email})"}, timeout=30)
    r.raise_for_status()
    return r.text

def parse_pubmed_xml(xml_text: str) -> List[Dict]:
    if not xml_text:
        return []
    soup = BeautifulSoup(xml_text, "lxml-xml")
    out = []
    for art in soup.find_all("PubmedArticle"):
        pmid_tag = art.find("PMID")
        pmid = pmid_tag.text.strip() if pmid_tag else None
        art_title = art.find("ArticleTitle")
        title = art_title.get_text(" ", strip=True) if art_title else ""
        abstract = ""
        abs_tags = art.find_all("AbstractText")
        if abs_tags:
            abstract = " ".join([t.get_text(" ", strip=True) for t in abs_tags])
        journal = None
        journal_tag = art.find("Journal")
        if journal_tag:
            jt = journal_tag.find("Title")
            journal = jt.get_text(" ", strip=True) if jt else None
        year = None
        pubdate = art.find("PubDate")
        if pubdate:
            y = pubdate.find("Year")
            if y and y.text.strip().isdigit():
                year = y.text.strip()
        if not year:
            dc = art.find("DateCompleted")
            if dc:
                y = dc.find("Year")
                if y:
                    year = y.text.strip()
        doi = None
        for idn in art.find_all("ArticleId"):
            if (idn.get("IdType") or "").lower() == "doi":
                doi = idn.text.strip()
        out.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "doi": doi,
        })
    return out

# ================== LLM prompts (enhanced for full text) ==================
SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert biomedical information extractor. "
    "Task: From the title/abstract/full text of a PubMed record, extract ONLY direct relationships between a microbial taxon and the disease: {disease_name}. "
    "Accept explicit statements about differences in abundance between disease vs controls, associations of abundance with disease severity/symptoms, or odds/risks relating a microbe to the disease. "
    "Carefully handle causal/mediated language around interventions/treatments: "
    "If a disease-targeted treatment/intervention improves disease outcomes while a taxon increases, then disease severity is inversely related to that taxon (direction='decrease'); "
    "if disease outcomes improve while the taxon decreases, disease severity is positively related to that taxon (direction='increase'); "
    "if disease outcomes worsen and the taxon increases, that implies direction='increase'; "
    "if disease outcomes worsen and the taxon decreases, that implies direction='decrease'. "
    "Map language carefully: 'positively associated', 'higher', 'increase', 'elevated', 'OR>1' => direction='increase'; "
    "'negatively associated', 'lower', 'decrease', 'reduced', 'inverse association', 'OR<1' => direction='decrease'. "
    "Do NOT fabricate numbers. If the text is ambiguous or does not clearly state a relationship, return 'direction = not_mentioned' or 'no_difference' accordingly. "
    "The relationship MUST be tied to the disease specifically; ignore associations to other conditions if disease linkage is not explicit."
)

USER_PROMPT_TEMPLATE = """Paper:
- pmid: {pmid}
- title: {title}
- abstract: {abstract}
- full_text: {full_text}

Target species (with common abbreviations if any):
{species_list}

Disease: {disease_name}

Extraction rules (strict):
- If the text describes an intervention/treatment for {disease_name} and reports abundance changes:
    * Intervention IMPROVES outcome AND species INCREASES → direction = "decrease"
    * Intervention IMPROVES outcome AND species DECREASES → direction = "increase"
    * Intervention WORSENS outcome AND species INCREASES → direction = "increase"
    * Intervention WORSENS outcome AND species DECREASES → direction = "decrease"
- For observational studies:
    * "positively associated", "higher", "increased", "enriched", "more abundant" → direction = "increase"
    * "negatively associated", "lower", "decreased", "depleted", "less abundant" → direction = "decrease"
- If NO clear abundance change or directional association (e.g., "was influenced by", "exerted impact", "modulates", "regulates" without comparison), set direction = "not_mentioned".
- Do NOT infer direction from vague mechanistic verbs.

Return JSON with schema:
{{
  "species_relations": [
    {{
      "species": "string",
      "direction": "increase|decrease|no_difference|not_mentioned",
      "statistical_significance": "string|null",
      "effect_size": "string|null",
      "sample_type": "human|animal|in_vitro|unspecified",
      "study_design": "observational|RCT|case_control|cohort|cross_sectional|review|meta_analysis|unknown",
      "evidence_excerpt": "string",
      "confidence": 0.0
    }}
  ]
}}"""

# ================== Full-text search & extraction ==================
def find_full_text_by_doi(doi: str, folder_path: str) -> Optional[str]:
    """
    Search for full-text file by DOI in the given folder (supports PDF and TXT).
    Returns extracted text content.
    """
    if not doi or not folder_path or not os.path.isdir(folder_path):
        return None
    doi_clean = doi.lower().replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
    doi_part = doi_clean.replace("/", "_").replace(".", "_")
    folder = Path(folder_path)
    candidates = []
    for ext in ["*.pdf", "*.txt", "*.TXT", "*.PDF"]:
        candidates.extend(folder.glob(ext))
    for file_path in candidates:
        name = file_path.stem.lower()
        if doi_clean in name or doi_part in name or name.replace("_", "") == doi_clean.replace("/", "").replace(".", ""):
            try:
                if file_path.suffix.lower() == ".pdf":
                    if not PDF_AVAILABLE:
                        print(f"PDF parsing requires pdfplumber, skipping {file_path}")
                        continue
                    with pdfplumber.open(file_path) as pdf:
                        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                    return text.strip()
                else:  # .txt
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read().strip()
            except Exception as e:
                print(f"Failed to read full text {file_path}: {e}")
                continue
    return None

# ================== LLM Extractor (with full-text support) ==================
@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4000
    api_base: Optional[str] = None
    api_key: Optional[str] = None

class LLMExtractor:
    def __init__(self, config: LLMConfig, disease_name: str):
        self.config = config
        self.disease_name = disease_name

    def call_llm(self, user_prompt: str) -> str:
        if not self.config.api_key:
            raise RuntimeError("Missing LLM API Key")
        client_params = {"api_key": self.config.api_key}
        if self.config.api_base:
            client_params["base_url"] = self.config.api_base
        client = OpenAI(**client_params)

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(disease_name=self.disease_name)

        resp = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"}
        )
        result = resp.choices[0].message.content
        return result

    def extract_species_vs_disease(self, paper: Dict, species: str, alias_map: Dict[str, List[str]], full_text: Optional[str] = None) -> Dict[str, Any]:
        pmid = paper.get("pmid") or ""
        title = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        full = full_text if full_text else ""

        list_lines = []
        for sp in [species]:
            aliases = alias_map.get(sp, [])
            if aliases:
                list_lines.append(f"- {sp} (aliases: {', '.join(aliases)})")
            else:
                list_lines.append(f"- {sp}")
        sp_block = "\n".join(list_lines)

        # Truncate full text to avoid token limits
        max_chars = self.config.max_tokens * 4
        if len(full) > max_chars:
            full = full[:max_chars] + "\n...[Full text truncated due to length]"

        up = USER_PROMPT_TEMPLATE.format(
            pmid=pmid,
            title=title,
            abstract=abstract,
            full_text=full,
            species_list=sp_block,
            disease_name=self.disease_name
        )
        try:
            raw = self.call_llm(up)
            data = json.loads(raw)
            items = data.get("species_relations", [])
            if not items:
                rel = self._empty_relation(species)
            else:
                rel = items[0]
                rel["direction"] = self._normalize_direction(rel.get("direction"))
            
            # Post-processing: intervention inference + ambiguous phrase downgrade
            text_all = (title + " " + abstract + " " + full).strip()
            rel = self._apply_postprocessing(rel, text_all, species)
            return rel
        except Exception as e:
            print(f"[LLM parsing failed] PMID={pmid}: {e}")
            return self._empty_relation(species)

    def _apply_postprocessing(self, rel: Dict, text: str, species: str) -> Dict:
        if rel["direction"] in ("not_mentioned", "no_difference"):
            return rel
        
        text_lower = text.lower()
        inc_keywords = ["increase", "increased", "higher", "elevat", "enrich", "more abundant", "positively", ">"]
        dec_keywords = ["decrease", "decreased", "lower", "reduc", "deplet", "less abundant", "negatively", "<"]
        has_inc = any(k in text_lower for k in inc_keywords)
        has_dec = any(k in text_lower for k in dec_keywords)
        if not (has_inc or has_dec):
            if rel["direction"] in ("increase", "decrease"):
                rel["direction"] = "not_mentioned"
                rel["confidence"] = 0.15
                rel["evidence_excerpt"] = (rel.get("evidence_excerpt", "") + " [Postprocessing: no clear direction keywords, changed to not_mentioned]").strip()
            return rel
        
        # Intervention scenario correction
        treat_keywords = ["treatment", "therapy", "intervention", "administer", "probiotic", "prebiotic", "synbiotic", "fmt", "fecal microbiota", "drug", "rct"]
        outcome_improve = ["improve", "ameliorat", "better", "enhance", "alleviat", "benefit", "reduce symptom", "reduce severity"]
        outcome_worsen = ["worsen", "exacerbat", "worse", "deteriorat", "adverse", "increase symptom"]
        has_treat = any(k in text_lower for k in treat_keywords)
        if has_treat:
            improve = any(k in text_lower for k in outcome_improve)
            worsen = any(k in text_lower for k in outcome_worsen)
            species_lower = species.lower()
            abbr = build_species_abbreviation(species)
            abbr_lower = abbr.lower() if abbr else ""
            idx = text_lower.find(species_lower)
            if idx == -1 and abbr_lower:
                idx = text_lower.find(abbr_lower)
            microbe_up = False
            microbe_down = False
            if idx != -1:
                context = text_lower[max(0, idx-50):min(len(text_lower), idx+100)]
                microbe_up = any(k in context for k in inc_keywords)
                microbe_down = any(k in context for k in dec_keywords)
            if improve and not worsen:
                if microbe_up:
                    correct_dir = "decrease"
                elif microbe_down:
                    correct_dir = "increase"
                else:
                    correct_dir = rel["direction"]
            elif worsen and not improve:
                if microbe_up:
                    correct_dir = "increase"
                elif microbe_down:
                    correct_dir = "decrease"
                else:
                    correct_dir = rel["direction"]
            else:
                correct_dir = rel["direction"]
            if correct_dir != rel["direction"] and correct_dir in ("increase", "decrease"):
                rel["direction"] = correct_dir
                rel["confidence"] = min(0.75, rel.get("confidence", 0.5) + 0.2)
                rel["evidence_excerpt"] = (rel.get("evidence_excerpt", "") + f" [Postprocessing: intervention inference → {correct_dir}]").strip()
        return rel

    def _empty_relation(self, species: str) -> Dict[str, Any]:
        return {
            "species": species,
            "direction": "not_mentioned",
            "statistical_significance": None,
            "effect_size": None,
            "sample_type": "unspecified",
            "study_design": "unknown",
            "evidence_excerpt": "",
            "confidence": 0.15
        }

    @staticmethod
    def _normalize_direction(d: str) -> str:
        d = (d or "").lower().strip()
        inc = {"increase", "increased", "higher", "elevated", "upregulated",
               "positively_associated", "positive", "positively associated", "positive association",
               "more abundant", "enriched"}
        dec = {"decrease", "decreased", "lower", "reduced", "downregulated",
               "negatively_associated", "negative", "negatively associated", "negative association",
               "inverse", "inversely", "inverse association", "inversely associated",
               "less abundant", "depleted"}
        nodiff = {"no_difference", "no difference", "ns", "nonsignificant", "no significant difference"}
        if d in inc:
            return "increase"
        if d in dec:
            return "decrease"
        if d in nodiff:
            return "no_difference"
        return "not_mentioned"

# ================== Helper functions ==================
def parse_sample_size_from_text(text: str) -> Optional[int]:
    if not text:
        return None
    t = re.sub(r'\s+', ' ', text)
    patterns = [
        r'\b[Nn]\s*=\s*(\d{2,5})\b',
        r'\b[nN]umber of (?:subjects|participants|patients)\s*(?:=|was|were|:)?\s*(\d{2,5})\b',
        r'\b(?:sample size|participants|subjects|patients)\s*(?:=|:)?\s*(\d{2,5})\b',
        r'\b(?:enrolled|recruited|included)\s*(\d{2,5})\s*(?:participants|subjects|patients)\b',
        r'\b(?:cohort|dataset)\s*of\s*(\d{2,5})\b',
        r'\b(\d{2,5})\s*(?:participants|subjects|patients)\b'
    ]
    cands = []
    for p in patterns:
        for m in re.finditer(p, t):
            try:
                val = int(m.group(1))
                if 10 <= val <= 100000:
                    cands.append(val)
            except Exception:
                pass
    return max(cands) if cands else None

def has_significance(sig_str: Optional[str], evidence_text: str) -> bool:
    text = (sig_str or "") + " " + (evidence_text or "")
    text = text.lower()
    if re.search(r'\bp\s*<\s*0\.05\b', text) or re.search(r'\bq\s*<\s*0\.05\b', text) or re.search(r'\bfdr\s*<\s*0\.05\b', text):
        return True
    if re.search(r'\bsignificant\b', text) or ("significant" in text):
        return True
    return False

def compute_confidence(direction: str,
                       stat_sig: Optional[str],
                       evidence_text: str,
                       study_design: str,
                       sample_type: str,
                       sample_size: Optional[int]) -> float:
    d_clear = direction in ("increase", "decrease")
    sig = has_significance(stat_sig, evidence_text)
    n = sample_size if (isinstance(sample_size, int) and sample_size > 0) else None
    if sig and d_clear:
        if n is not None and n > 200:
            base = 0.8
        elif n is not None and 100 < n <= 200:
            base = 0.7
        else:
            base = 0.6
    elif (not sig) and d_clear:
        base = 0.5
    else:
        base = 0.2
    sd = (study_design or "unknown").lower()
    if any(k in sd for k in ["longitudinal", "cohort"]):
        base += 0.1
    elif any(k in sd for k in ["cross", "cross_sectional", "cross-sectional", "unknown", "observational", "case_control"]):
        base += 0.05
    else:
        base += 0.05
    st = (sample_type or "unspecified").lower()
    if "human" in st:
        base += 0.1
    elif "animal" in st:
        base += 0.05
    return float(np.clip(base, 0.0, 1.0))

# ================== Main analysis pipeline (with full-text review) ==================
def run_analysis(species: str,
                 disease_name: str,
                 disease_query: str,
                 api_base: str,
                 api_key: str,
                 model: str,
                 email: str,
                 years: int,
                 top_k: int,
                 fulltext_folder: str,
                 progress=gr.Progress()) -> str:
    if not species or not disease_name or not disease_query or not email:
        return "❌ Please fill in all required fields (species, disease name, disease query, email)."
    if not api_key:
        return "❌ Please provide an OpenAI API Key."
    if not api_base:
        api_base = None

    # Handle folder path
    folder_path = fulltext_folder.strip() if fulltext_folder else None
    if folder_path and not os.path.isdir(folder_path):
        return f"❌ Full-text folder path is invalid or does not exist: {folder_path}"
    if folder_path and not PDF_AVAILABLE:
        progress(0, desc="Warning: pdfplumber not installed, cannot parse PDFs. Only TXT files will work.")

    progress(0, desc="Initializing...")
    alias_map = {species: []}
    abbr = build_species_abbreviation(species)
    if abbr:
        alias_map[species].append(abbr)

    species_part = build_species_query_with_aliases(species)
    disease_part = f"({disease_query})"
    query = f"{species_part} AND {disease_part}"
    if years > 0:
        import datetime
        current_year = datetime.datetime.now().year
        start_year = current_year - years
        query += f' AND ("{start_year}"[PDAT] : "{current_year}"[PDAT])'

    progress(0.1, desc=f"Searching PubMed: {query[:100]}...")
    try:
        search_res = pubmed_search(query, retmax=top_k, email=email, api_key=None)
        idlist = search_res.get("esearchresult", {}).get("idlist", [])
        total_count = search_res.get("esearchresult", {}).get("count", "0")
        progress(0.2, desc=f"Found {len(idlist)} articles (total {total_count})")
        if not idlist:
            return f"⚠️ No articles found. Try relaxing query or increasing year range.\n\nQuery: `{query}`"
    except Exception as e:
        return f"❌ PubMed search failed: {str(e)}"

    try:
        xml_text = pubmed_fetch_xml(idlist, email=email, api_key=None)
        papers = parse_pubmed_xml(xml_text)
        progress(0.3, desc=f"Parsing {len(papers)} articles...")
    except Exception as e:
        return f"❌ Failed to fetch article details: {str(e)}"

    llm_config = LLMConfig(
        provider="openai",
        model=model,
        temperature=0.1,
        max_tokens=4000,
        api_base=api_base,
        api_key=api_key
    )
    extractor = LLMExtractor(llm_config, disease_name)

    relations = []
    total = len(papers)
    for idx, paper in enumerate(papers):
        progress(0.3 + 0.5 * (idx / total), desc=f"LLM extraction ({idx+1}/{total})...")
        # Initial extraction (title/abstract only)
        rel = extractor.extract_species_vs_disease(paper, species, alias_map, full_text=None)
        text_all = (paper.get("title", "") + " " + paper.get("abstract", "")).strip()
        sample_size = parse_sample_size_from_text(text_all)
        rel["confidence"] = compute_confidence(
            direction=rel.get("direction", "not_mentioned"),
            stat_sig=rel.get("statistical_significance"),
            evidence_text=rel.get("evidence_excerpt", ""),
            study_design=rel.get("study_design", "unknown"),
            sample_type=rel.get("sample_type", "unspecified"),
            sample_size=sample_size
        )
        rel["sample_size"] = sample_size
        rel["pmid"] = paper.get("pmid")
        rel["year"] = paper.get("year")
        rel["title"] = paper.get("title")
        rel["abstract"] = paper.get("abstract")[:500]
        rel["doi"] = paper.get("doi")
        relations.append(rel)
        time.sleep(0.3)

    # ----- Full-text review for 'not_mentioned' articles -----
    if folder_path:
        progress(0.85, desc="Full-text review: searching for not_mentioned articles...")
        not_mentioned_indices = [i for i, r in enumerate(relations) if r.get("direction") == "not_mentioned"]
        if not_mentioned_indices:
            print(f"Found {len(not_mentioned_indices)} articles needing full-text review")
            for i in not_mentioned_indices:
                rel = relations[i]
                doi = rel.get("doi")
                if not doi:
                    print(f"PMID {rel['pmid']} has no DOI, skipping full-text review")
                    continue
                full_text = find_full_text_by_doi(doi, folder_path)
                if not full_text:
                    print(f"PMID {rel['pmid']} DOI {doi} full-text file not found, skipping")
                    continue
                print(f"Full text found, re-analyzing PMID {rel['pmid']}")
                paper_full = {
                    "pmid": rel["pmid"],
                    "title": rel["title"],
                    "abstract": rel["abstract"],
                    "doi": doi
                }
                new_rel = extractor.extract_species_vs_disease(paper_full, species, alias_map, full_text=full_text)
                text_all = (rel["title"] + " " + rel["abstract"] + " " + full_text[:2000]).strip()
                sample_size = parse_sample_size_from_text(text_all)
                new_rel["confidence"] = compute_confidence(
                    direction=new_rel.get("direction", "not_mentioned"),
                    stat_sig=new_rel.get("statistical_significance"),
                    evidence_text=new_rel.get("evidence_excerpt", ""),
                    study_design=new_rel.get("study_design", "unknown"),
                    sample_type=new_rel.get("sample_type", "unspecified"),
                    sample_size=sample_size
                )
                new_rel["sample_size"] = sample_size
                new_rel["pmid"] = rel["pmid"]
                new_rel["year"] = rel["year"]
                new_rel["title"] = rel["title"]
                new_rel["abstract"] = rel["abstract"][:500]
                new_rel["doi"] = doi
                relations[i] = new_rel
                time.sleep(0.5)
        else:
            print("No not_mentioned articles to review")
    else:
        print("No full-text folder provided, skipping full-text review")

    progress(0.95, desc="Generating report...")

    # Generate Markdown report
    report = f"""# Microbe-Disease Relationship Mining Report

## Query Parameters
- **Microbial species**: `{species}`
- **Disease name**: `{disease_name}`
- **PubMed disease query**: `{disease_query}`
- **Year range**: last {years} years
- **Max articles**: {top_k}
- **LLM model**: `{model}`
- **Full-text review**: {'Enabled' if folder_path else 'Disabled'}

## Search Results
- **Total hits**: {total_count}
- **Articles analyzed**: {len(relations)}

## Direction Summary
"""
    dir_counts = Counter([r.get("direction", "unknown") for r in relations])
    report += "| Direction | Count |\n|-----------|-------|\n"
    for d, cnt in dir_counts.most_common():
        report += f"| {d} | {cnt} |\n"

    confs = [r.get("confidence", 0.0) for r in relations if isinstance(r.get("confidence"), (int, float))]
    avg_conf = np.mean(confs) if confs else 0.0
    report += f"\n- **Average confidence**: {avg_conf:.2f}\n"

    report += "\n## Detailed Article List\n"
    for i, r in enumerate(relations, 1):
        pmid = r.get("pmid", "?")
        title = r.get("title", "No title")
        year = r.get("year", "Unknown")
        direction = r.get("direction", "?")
        conf = r.get("confidence", 0.0)
        excerpt = r.get("evidence_excerpt", "")[:200]
        pubmed_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid and pmid != "?" else ""
        report += f"### {i}. {title} ({year})\n"
        report += f"- **PMID**: [{pmid}]({pubmed_link})\n"
        report += f"- **Direction**: {direction}\n"
        report += f"- **Confidence**: {conf:.2f}\n"
        if excerpt:
            report += f"- **Evidence snippet**: {excerpt}\n"
        if r.get("sample_size"):
            report += f"- **Sample size**: {r.get('sample_size')}\n"
        if r.get("study_design") and r.get("study_design") != "unknown":
            report += f"- **Study design**: {r.get('study_design')}\n"
        if r.get("sample_type") and r.get("sample_type") != "unspecified":
            report += f"- **Sample type**: {r.get('sample_type')}\n"
        report += "\n"

    # Data table
    df = pd.DataFrame(relations)
    if not df.empty:
        cols = ["pmid", "title", "year", "direction", "confidence", "evidence_excerpt", "study_design", "sample_type"]
        existing_cols = [c for c in cols if c in df.columns]
        df_display = df[existing_cols].copy()
        report += "## Data Table\n"
        report += df_display.to_markdown(index=False)

    progress(1.0, desc="Done")
    return report

# ================== Gradio UI ==================
def create_ui():
    with gr.Blocks(title="Microbe-Disease Relationship Mining", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧫 Microbe-Disease Relationship Mining System
        Automatically extract associations (increase/decrease/no difference) between a specific microbe and a disease from PubMed articles using LLM.
        Optional full-text review: for articles initially classified as 'not_mentioned', the system searches a local folder for full-text files (by DOI) and re-analyzes them.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                species = gr.Textbox(label="Microbial species", placeholder="e.g., Akkermansia muciniphila", value="")
                disease_name = gr.Textbox(label="Disease display name", placeholder="e.g., ASD", value="")
                disease_query = gr.Textbox(
                    label="PubMed disease query",
                    placeholder='Example: "ASD"[Title/Abstract] OR "Autism Spectrum Disorder"[Title/Abstract]',
                    value="",
                    info="Use PubMed query syntax, supports field tags like [Title/Abstract]"
                )
                years = gr.Number(label="Year range (last N years)", value=1, precision=0)
                top_k = gr.Number(label="Max articles to fetch", value=50, precision=0)

            with gr.Column(scale=1):
                api_base = gr.Textbox(label="OpenAI API Base URL", value="")
                api_key = gr.Textbox(label="OpenAI API Key", type="password", placeholder="sk-...")
                model = gr.Textbox(label="Model name", placeholder="e.g., gpt-4o", value="")
                email = gr.Textbox(label="NCBI Email (required)", placeholder="your_email@example.com", value="user@example.com")
                fulltext_folder = gr.Textbox(label="Full-text folder path (optional)", placeholder="/path/to/pdf_txt_files", value="",
                                             info="Leave empty to skip full-text review. Files should be named with DOI (e.g., 10.1016/j.cell.2020.01.001.pdf). Supports .pdf and .txt")

        btn = gr.Button("🚀 Generate Report", variant="primary")
        output = gr.Markdown(label="Analysis Report")

        btn.click(fn=run_analysis,
                  inputs=[species, disease_name, disease_query, api_base, api_key, model, email, years, top_k, fulltext_folder],
                  outputs=output,
                  show_progress="full")

        gr.Markdown("""
        ---
        ### Instructions
        1. **Microbial species**: Enter full Latin name (e.g., *Akkermansia muciniphila*). Abbreviations are automatically added to the search.
        2. **Disease display name**: Used in LLM prompts for context.
        3. **PubMed disease query**: Directly appended to the PubMed query. Use field tags like `[Title/Abstract]`.  
           Example: `"ASD"[Title/Abstract] OR "Autism Spectrum Disorder"[Title/Abstract]`
        4. **OpenAI Configuration**: Supports any OpenAI-compatible API. Leave Base URL empty to use default OpenAI.
        5. **NCBI Email**: Required by PubMed for rate limiting and contact.
        6. **Full-text review (optional)**: Provide a folder containing PDF/TXT files. The system matches files by DOI (filename containing the DOI string). If a file is found for a 'not_mentioned' article, it re-analyzes the full text.  
           *Note:* Install `pdfplumber` (`pip install pdfplumber`) to parse PDFs.
        """)

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()