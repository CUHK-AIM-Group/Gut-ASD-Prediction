import os
import re
import time
import json
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np

# ================== LLM ==================
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: failed to import OpenAI: {e}")
    print("Please install the openai package: pip install openai")
    OPENAI_AVAILABLE = False

# ================== Configuration ==================
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = os.environ.get("NCBI_TOOL", "asd_microbiome_species_vs_asd")
EMAIL = os.environ.get("NCBI_EMAIL", "your_email@example.com")
API_KEY = os.environ.get("NCBI_API_KEY")

# LLM configuration
LLM_PROVIDER = "openai"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

INPUT_SPECIES_CSV = "./selected_species.csv"
FULL_PAPER_DIR = Path("./full_paper")
YEARS = 20
TOP_K_PER_SPECIES = 100
REQUEST_INTERVAL_SEC = 0.34
FULL_TEXT_MAX_CHARS = 45000
FULL_TEXT_WINDOW_CHARS = 1800
FULL_TEXT_MAX_PASSAGES_PER_SPECIES = 5

# Concurrency and timeout
MAX_WORKERS = 3
LLM_REQUEST_TIMEOUT = 60

# Output paths
OUT_DIR = Path("./output_llm_species_vs_asd")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_RELATIONS_CSV = OUT_DIR / "species_vs_asd_relations.csv"
OUT_RELATIONS_JSONL = OUT_DIR / "species_vs_asd_relations.jsonl"
OUT_LLM_CACHE = OUT_DIR / "llm_cache.json"
OUT_LLM_RAW = OUT_DIR / "llm_raw_results.json"

########################
# Environment and PubMed utilities
########################
def check_env():
    print("Environment check:")
    print(f"  TOOL={TOOL}")
    print(f"  EMAIL={EMAIL}")
    print(f"  API_KEY={'SET' if API_KEY else 'NOT SET'}")
    print(f"  LLM_PROVIDER={LLM_PROVIDER}")
    print(f"  OPENAI_MODEL={OPENAI_MODEL}")
    if OPENAI_API_BASE:
        print(f"  OPENAI_API_BASE={OPENAI_API_BASE[:40]}...")
    else:
        print("  OPENAI_API_BASE=DEFAULT")
    print(f"  OPENAI_API_KEY={'SET' if OPENAI_API_KEY and OPENAI_API_KEY!='YOUR_API_KEY' else 'NOT SET'}")
    issues = []
    if not EMAIL or "@" not in EMAIL:
        issues.append("NCBI_EMAIL is missing or invalid")
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY":
        issues.append("OpenAI API key is missing")
    if issues:
        print("Warnings:")
        for it in issues:
            print(" -", it)

def build_species_query(taxon: str) -> str:
    taxon_clean = re.sub(r'[[]\(\)_]', ' ', taxon)
    taxon_clean = re.sub(r'\s+', ' ', taxon_clean).strip()
    queries = [
        f'"{taxon_clean}"[Title/Abstract]',
        f'{taxon_clean}[Title/Abstract]',
        f'"{taxon_clean}"[All Fields]',
    ]
    return f"({' OR '.join(queries)})"

def build_asd_query_terms() -> str:
    ta_terms = [
        '"ASD"[Title/Abstract]',
        '"autism spectrum disorder"[Title/Abstract]',
        '"autism"[Title/Abstract]',
        '"autistic disorder"[Title/Abstract]',
    ]
    return "(" + " OR ".join(ta_terms) + ")"

def build_species_abbreviation(full_name: str) -> Optional[str]:
    # Akkermansia muciniphila -> A. muciniphila
    parts = re.split(r'\s+', full_name.strip())
    if len(parts) >= 2 and all(parts):
        genus = parts[0]
        rest = " ".join(parts[1:])
        if genus and genus[0].isalpha():
            return f"{genus[0]}. {rest}"
    return None

def build_pubmed_query_with_aliases(taxon: str, years: Optional[int] = None) -> str:
    # Query both full species names and abbreviations.
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
    species_q = "(" + " OR ".join(species_qs) + ")"
    asd_q = build_asd_query_terms()
    q = f"{species_q} AND {asd_q}"
    if years:
        import datetime
        y2 = datetime.datetime.now().year
        y1 = y2 - years
        q += f' AND ("{y1}"[PDAT] : "{y2}"[PDAT])'
    return q

def pubmed_search(query: str, retmax: int = 50) -> dict:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "relevance",
        "tool": TOOL,
        "email": EMAIL,
    }
    if API_KEY:
        params["api_key"] = API_KEY
    r = requests.get(f"{PUBMED_BASE}/esearch.fcgi", params=params,
                     headers={"User-Agent": f"{TOOL} ({EMAIL})"}, timeout=30)
    r.raise_for_status()
    return r.json()

def pubmed_fetch_xml(pmids: List[str]) -> str:
    if not pmids:
        return ""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": TOOL,
        "email": EMAIL,
    }
    if API_KEY:
        params["api_key"] = API_KEY
    r = requests.get(f"{PUBMED_BASE}/efetch.fcgi", params=params,
                     headers={"User-Agent": f"{TOOL} ({EMAIL})"}, timeout=30)
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

#########################
# Input species table
#########################
def load_selected_species(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    if not {"species", "prevalence"}.issubset(set(cols)):
        raise ValueError("selected_species.csv must contain two columns: species, prevalence")
    df["species"] = df["species"].astype(str).map(lambda s: re.sub(r'\s+', ' ', s).strip())
    df["prevalence"] = pd.to_numeric(df["prevalence"], errors="coerce")
    df = df.dropna(subset=["species"]).reset_index(drop=True)
    return df

#########################
# Strict LLM extraction prompts
#########################
SYSTEM_PROMPT = (
    "You are an expert biomedical information extractor. "
    "Task: From the title/abstract of a single PubMed record, extract ONLY direct relationships between a microbial taxon and ASD. "
    "Accept explicit statements about differences in abundance between ASD vs controls, associations of abundance with ASD severity/symptoms, or odds/risks relating a microbe to ASD. "
    "Carefully handle causal/mediated language around interventions/treatments: "
    "If an ASD-targeted treatment/intervention improves ASD outcomes while a taxon increases, then ASD severity is inversely related to that taxon (direction='decrease'); "
    "if ASD outcomes improve while the taxon decreases, ASD severity is positively related to that taxon (direction='increase'); "
    "if ASD outcomes worsen and the taxon increases, that implies direction='increase'; "
    "if ASD outcomes worsen and the taxon decreases, that implies direction='decrease'. "
    "Map language carefully: 'positively associated', 'higher', 'increase', 'elevated', 'OR>1' => direction='increase'; "
    "'negatively associated', 'lower', 'decrease', 'reduced', 'inverse association', 'OR<1' => direction='decrease'. "
    "Do NOT fabricate numbers. If the abstract is ambiguous or does not clearly state a relationship, return 'direction = not_mentioned' or 'no_difference' accordingly. "
    "The relationship MUST be tied to ASD specifically; ignore associations to other conditions if ASD linkage is not explicit."
    )

USER_PROMPT_TEMPLATE = """Paper:
- pmid: {pmid}
- title: {title}
- abstract: {abstract}

Target species (with common abbreviations if any):
{species_list}

Extraction rules (strict):
- Goal: relate each target species' abundance to ASD presence/severity (not treatment per se). Resolve mediated language:
    - If an ASD-targeted treatment/intervention improves ASD outcomes and the species increases => ASD is inversely related => direction="decrease".
    - If an ASD-targeted treatment/intervention improves ASD outcomes and the species decreases => ASD is positively related => direction="increase".
    - If ASD outcomes worsen and the species increases => direction="increase".
    - If ASD outcomes worsen and the species decreases => direction="decrease".
- For each species: direction ∈ {{"increase","decrease","no_difference","not_mentioned"}}; tie explicitly to ASD.
- If numbers present, also capture:
    - statistical_significance (e.g., "p=0.03","q<0.05","FDR<0.1", or null)
    - effect_size (e.g., "r=0.35","OR=1.8 (95% CI ...)","β=-0.12","RR=1.5", or null)
    - sample_type: "human" | "animal" | "in_vitro" | "unspecified"
    - study_design: "observational" | "RCT" | "case_control" | "cohort" | "cross_sectional" | "review" | "meta_analysis" | "unknown"
    - evidence_excerpt: short verbatim snippet.

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
}}
- IMPORTANT: Ensure all target species appear exactly once.
- If unclear, use direction='not_mentioned' with low confidence.
"""

FULL_TEXT_FALLBACK_PROMPT_TEMPLATE = """Paper:
- pmid: {pmid}
- title: {title}

Full-text fallback passages extracted from the PDF:
{full_text_passages}

Target species (with common abbreviations if any):
{species_list}

Extraction rules (strict):
- This is a fallback pass because the title/abstract analysis returned direction="not_mentioned".
- Use ONLY the supplied full-text passages. Do not infer from outside knowledge.
- Goal: relate each target species' abundance to ASD presence/severity.
- For each species: direction ∈ {{"increase","decrease","no_difference","not_mentioned"}}; tie explicitly to ASD.
- If the full text discusses the species but not in relation to ASD, use direction="not_mentioned".
- If numbers present, also capture:
    - statistical_significance (e.g., "p=0.03","q<0.05","FDR<0.1", or null)
    - effect_size (e.g., "r=0.35","OR=1.8 (95% CI ...)","β=-0.12","RR=1.5", or null)
    - sample_type: "human" | "animal" | "in_vitro" | "unspecified"
    - study_design: "observational" | "RCT" | "case_control" | "cohort" | "cross_sectional" | "review" | "meta_analysis" | "unknown"
    - evidence_excerpt: short verbatim snippet from the supplied passages.

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
}}
- IMPORTANT: Ensure all target species appear exactly once.
- If unclear, use direction='not_mentioned' with low confidence.
"""

#########################
# Full-text PDF fallback utilities
#########################
def find_full_paper_pdf(pmid: str, full_paper_dir: Path = FULL_PAPER_DIR) -> Optional[Path]:
    if not pmid or not full_paper_dir.exists():
        return None
    exact_candidates = [
        full_paper_dir / f"{pmid}.pdf",
        full_paper_dir / f"PMID{pmid}.pdf",
        full_paper_dir / f"pmid{pmid}.pdf",
    ]
    for p in exact_candidates:
        if p.exists() and p.is_file():
            return p
    matches = sorted(full_paper_dir.glob(f"*{pmid}*.pdf"))
    return matches[0] if matches else None

def extract_text_from_pdf(pdf_path: Path) -> str:
    if not pdf_path or not pdf_path.exists():
        return ""

    text_parts = []
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
    except Exception:
        text_parts = []

    if not any(t.strip() for t in text_parts):
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            text_parts = [(page.extract_text() or "") for page in reader.pages]
        except Exception:
            text_parts = []

    if not any(t.strip() for t in text_parts):
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(pdf_path))
            text_parts = [(page.extract_text() or "") for page in reader.pages]
        except Exception as e:
            print(f"[PDF parse failed] {pdf_path}: {e}")
            return ""

    text = "\n".join(text_parts)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _window_around(text: str, start: int, end: int, window_chars: int = FULL_TEXT_WINDOW_CHARS) -> str:
    left = max(0, start - window_chars)
    right = min(len(text), end + window_chars)
    return text[left:right].strip()

def build_full_text_passages(text: str,
                             species_list: List[str],
                             alias_map: Dict[str, List[str]]) -> str:
    if not text:
        return ""

    passages = []
    seen = set()
    asd_pattern = re.compile(r'\b(?:ASD|autism|autistic|autism spectrum disorder|autistic disorder)\b', re.I)

    for sp in species_list:
        names = [sp] + alias_map.get(sp, [])
        name_patterns = [re.escape(x) for x in names if x]
        if not name_patterns:
            continue
        species_pattern = re.compile(r'(' + '|'.join(name_patterns) + r')', re.I)

        matches = list(species_pattern.finditer(text))
        selected = []
        for m in matches:
            win = _window_around(text, m.start(), m.end())
            if asd_pattern.search(win):
                selected.append(win)
            if len(selected) >= FULL_TEXT_MAX_PASSAGES_PER_SPECIES:
                break

        if not selected:
            for m in matches[:FULL_TEXT_MAX_PASSAGES_PER_SPECIES]:
                selected.append(_window_around(text, m.start(), m.end()))

        for idx, win in enumerate(selected, 1):
            key = re.sub(r'\s+', ' ', win[:300]).lower()
            if key in seen:
                continue
            seen.add(key)
            passages.append(f"[Species: {sp}; passage {idx}]\n{win}")

    combined = "\n\n---\n\n".join(passages)
    if not combined:
        asd_matches = list(asd_pattern.finditer(text))
        for i, m in enumerate(asd_matches[:8], 1):
            passages.append(f"[ASD-related passage {i}]\n{_window_around(text, m.start(), m.end())}")
        combined = "\n\n---\n\n".join(passages)

    return combined[:FULL_TEXT_MAX_CHARS]

#########################
# LLM extractor
#########################
@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 1500
    api_base: Optional[str] = None
    api_key: Optional[str] = None

class LLMExtractor:
    def __init__(self, config: LLMConfig, cache_path: Path):
        self.config = config
        self.cache_path = cache_path
        self.cache: Dict[str, str] = {}
        self.load_cache()

    def load_cache(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"Loaded cache entries: {len(self.cache)}")
            except Exception:
                self.cache = {}

    def save_cache(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def call_llm(self, user_prompt: str) -> str:
        if not self.config.api_key:
            raise RuntimeError("Missing LLM API key")
        ck = str(hash(user_prompt))
        if ck in self.cache:
            return self.cache[ck]

        client_params = {"api_key": self.config.api_key}
        if self.config.api_base:
            client_params["base_url"] = self.config.api_base
        client = OpenAI(**client_params)

        resp = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"}
        )
        result = resp.choices[0].message.content
        self.cache[ck] = result
        return result

    def extract_species_vs_asd(self, paper: Dict, species_list: List[str], alias_map: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        pmid = paper.get("pmid") or ""
        title = paper.get("title") or ""
        abstract = paper.get("abstract") or ""

        list_lines = []
        for sp in sorted(set(species_list)):
            aliases = alias_map.get(sp, [])
            if aliases:
                list_lines.append(f"- {sp} (aliases: {', '.join(aliases)})")
            else:
                list_lines.append(f"- {sp}")
        sp_block = "\n".join(list_lines)

        up = USER_PROMPT_TEMPLATE.format(
            pmid=pmid,
            title=title,
            abstract=abstract,
            species_list=sp_block
        )
        try:
            raw = self.call_llm(up)
            data = json.loads(raw)
            items = data.get("species_relations", [])
            target_set = set(species_list)
            alias_to_full = {}
            for full, als in alias_map.items():
                for a in als:
                    alias_to_full[a.lower()] = full
                alias_to_full[full.lower()] = full

            seen = set()
            clean = []
            for it in items:
                sp_raw = (it.get("species") or "").strip()
                sp_key = sp_raw.lower()
                full = alias_to_full.get(sp_key, None)
                if not full and sp_raw:
                    sp_norm = re.sub(r'\s+', ' ', sp_key.replace(" .", ".")).strip()
                    full = alias_to_full.get(sp_norm, None)
                if full and full in target_set and full not in seen:
                    it["species"] = full
                    seen.add(full)
                    clean.append(it)

            missing = sorted(list(target_set - seen))
            for sp in missing:
                clean.append({
                    "species": sp,
                    "direction": "not_mentioned",
                    "statistical_significance": None,
                    "effect_size": None,
                    "sample_type": "unspecified",
                    "study_design": "unknown",
                    "evidence_excerpt": "",
                    "confidence": 0.15
                })

            text_all = (title + " " + abstract).strip()
            clean = rule_based_direction_fallback(clean, text_all)
            clean = causal_polarity_adjustment(clean, text_all)

            return clean
        except Exception as e:
            print(f"[LLM parse failed] PMID={pmid}: {e}")
            return [{
                "species": sp,
                "direction": "not_mentioned",
                "statistical_significance": None,
                "effect_size": None,
                "sample_type": "unspecified",
                "study_design": "unknown",
                "evidence_excerpt": "",
                "confidence": 0.15
            } for sp in species_list]

    def extract_species_vs_asd_from_full_text(self,
                                              paper: Dict,
                                              species_list: List[str],
                                              alias_map: Dict[str, List[str]],
                                              full_text_passages: str) -> List[Dict[str, Any]]:
        pmid = paper.get("pmid") or ""
        title = paper.get("title") or ""

        list_lines = []
        for sp in sorted(set(species_list)):
            aliases = alias_map.get(sp, [])
            if aliases:
                list_lines.append(f"- {sp} (aliases: {', '.join(aliases)})")
            else:
                list_lines.append(f"- {sp}")
        sp_block = "\n".join(list_lines)

        up = FULL_TEXT_FALLBACK_PROMPT_TEMPLATE.format(
            pmid=pmid,
            title=title,
            full_text_passages=full_text_passages,
            species_list=sp_block
        )
        try:
            raw = self.call_llm(up)
            data = json.loads(raw)
            items = data.get("species_relations", [])
            target_set = set(species_list)
            alias_to_full = {}
            for full, als in alias_map.items():
                for a in als:
                    alias_to_full[a.lower()] = full
                alias_to_full[full.lower()] = full

            seen = set()
            clean = []
            for it in items:
                sp_raw = (it.get("species") or "").strip()
                sp_key = sp_raw.lower()
                full = alias_to_full.get(sp_key, None)
                if not full and sp_raw:
                    sp_norm = re.sub(r'\s+', ' ', sp_key.replace(" .", ".")).strip()
                    full = alias_to_full.get(sp_norm, None)
                if full and full in target_set and full not in seen:
                    it["species"] = full
                    it["full_text_fallback_used"] = True
                    seen.add(full)
                    clean.append(it)

            missing = sorted(list(target_set - seen))
            for sp in missing:
                clean.append({
                    "species": sp,
                    "direction": "not_mentioned",
                    "statistical_significance": None,
                    "effect_size": None,
                    "sample_type": "unspecified",
                    "study_design": "unknown",
                    "evidence_excerpt": "",
                    "confidence": 0.15,
                    "full_text_fallback_used": True
                })

            clean = rule_based_direction_fallback(clean, full_text_passages)
            clean = causal_polarity_adjustment(clean, full_text_passages)
            for it in clean:
                it["full_text_fallback_used"] = True
            return clean
        except Exception as e:
            print(f"[Full-text LLM parse failed] PMID={pmid}: {e}")
            return [{
                "species": sp,
                "direction": "not_mentioned",
                "statistical_significance": None,
                "effect_size": None,
                "sample_type": "unspecified",
                "study_design": "unknown",
                "evidence_excerpt": "",
                "confidence": 0.15,
                "full_text_fallback_used": True
            } for sp in species_list]

#########################
# Rule-based fallback and normalization
#########################
def normalize_direction(d: str) -> str:
    d = (d or "").lower().strip()
    inc = {
        "increase", "increased", "higher", "elevated", "upregulated",
        "positively_associated", "positive", "positively associated", "positive association",
        "more abundant", "enriched"
    }
    dec = {
        "decrease", "decreased", "lower", "reduced", "downregulated",
        "negatively_associated", "negative", "negatively associated", "negative association",
        "inverse", "inversely", "inverse association", "inversely associated",
        "less abundant", "depleted"
    }
    nodiff = {"no_difference", "no difference", "ns", "nonsignificant", "no significant difference"}
    notm = {"not_mentioned", "unmentioned"}

    if d in inc:
        return "increase"
    if d in dec:
        return "decrease"
    if d in nodiff:
        return "no_difference"
    if d in notm:
        return "not_mentioned"
    return "not_mentioned"

POSITIVE_HINTS = [
    r"\bpositively (?:associated|correlated)\b",
    r"\bhigher\b", r"\bincrease[d]?\b", r"\belevat(?:ed|ion)\b",
    r"\benrich(?:ed|ment)\b", r"\bmore abundant\b",
    r"\bOR\s*>\s*1\b", r"\bRR\s*>\s*1\b", r"\bβ\s*>\s*0\b", r"\bbeta\s*>\s*0\b"
]
NEGATIVE_HINTS = [
    r"\bnegatively (?:associated|correlated)\b",
    r"\blower\b", r"\bdecrease[d]?\b", r"\breduc(?:ed|tion)\b",
    r"\bdeplet(?:ed|ion)\b", r"\binverse(?:ly)? (?:associated|correlated)?\b",
    r"\bless abundant\b",
    r"\bOR\s*<\s*1\b", r"\bRR\s*<\s*1\b", r"\bβ\s*<\s*0\b", r"\bbeta\s*<\s*0\b"
]
SIG_HINTS = [
    r"\bp\s*<\s*0\.05\b", r"\bp\s*<\s*0\.01\b", r"\bFDR\s*<\s*0\.05\b", r"\bq\s*<\s*0\.05\b",
    r"\bsignificant\b", r"\bstatistically significant\b"
]

TREATMENT_HINTS = [
    r"\btreatment\b", r"\btherapy\b", r"\bintervention\b", r"\badminister(?:ed|ing)\b",
    r"\bASD[-\s]?targeted\b", r"\bprebiotic\b", r"\bprobiotic\b", r"\bsynbiotic\b",
    r"\bfecal microbiota transplantation\b", r"\bFMT\b", r"\bdrug\b", r"\bRCT\b"
]
IMPROVE_HINTS = [
    r"\bimprov(?:e|ed|ement)\b", r"\bameliorat(?:e|ion)\b", r"\bbetter\b", r"\benhanc(?:e|ed)\b",
    r"\breduc(?:ed|tion) (?:in )?(?:ASD|autis[tm]-?(?:like )?behavio[u]?rs|symptoms|severity)\b",
    r"\balleviat(?:e|ion)\b", r"\bbenefit(?:ed|s)?\b"
]
WORSEN_HINTS = [
    r"\bworsen(?:ed|ing)?\b", r"\bexacerbat(?:e|ion)\b", r"\bworse\b",
    r"\bincreas(?:ed|e) (?:in )?(?:ASD|autis[tm]-?(?:like )?behavio[u]?rs|symptoms|severity)\b",
    r"\badvers(?:e)?\b", r"\bdeteriorat(?:e|ion)\b"
]
MICROBE_UP_HINTS = [
    r"\bincrease[d]?\b", r"\belevat(?:ed|ion)\b", r"\benhanc(?:e|ed)\b",
    r"\benhance in .*? (?:abundance|levels)\b", r"\bmore abundant\b", r"\benriched\b", r"\bhigher\b"
]
MICROBE_DOWN_HINTS = [
    r"\bdecrease[d]?\b", r"\breduc(?:ed|tion)\b", r"\bdeplet(?:ed|ion)\b",
    r"\blower\b", r"\bless abundant\b"
]

def text_has_hint(text: str, patterns: List[str]) -> bool:
    t = (text or "").lower()
    for p in patterns:
        if re.search(p, t):
            return True
    return False

def rule_based_direction_fallback(items: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    if not text:
        return items
    has_pos = text_has_hint(text, POSITIVE_HINTS)
    has_neg = text_has_hint(text, NEGATIVE_HINTS)
    sig = text_has_hint(text, SIG_HINTS)

    if not (has_pos or has_neg):
        return items

    adjusted = []
    for it in items:
        d = normalize_direction(it.get("direction", ""))
        if d in ("increase", "decrease"):
            adjusted.append(it)
            continue
        new_it = dict(it)
        if has_neg and not has_pos:
            new_it["direction"] = "decrease"
            base = 0.38
            if sig:
                base = 0.62
            new_it["confidence"] = round(float(np.clip(base, 0.0, 1.0)), 2)
            if not new_it.get("evidence_excerpt"):
                new_it["evidence_excerpt"] = "inverse/negative association hinted in abstract"
        elif has_pos and not has_neg:
            new_it["direction"] = "increase"
            base = 0.38
            if sig:
                base = 0.62
            new_it["confidence"] = round(float(np.clip(base, 0.0, 1.0)), 2)
            if not new_it.get("evidence_excerpt"):
                new_it["evidence_excerpt"] = "positive association hinted in abstract"
        adjusted.append(new_it)
    return adjusted

def causal_polarity_adjustment(items: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    if not text:
        return items
    t = text.lower()
    has_treat = text_has_hint(t, TREATMENT_HINTS)
    if not has_treat:
        return items

    improve = text_has_hint(t, IMPROVE_HINTS)
    worsen = text_has_hint(t, WORSEN_HINTS)
    microbe_up = text_has_hint(t, MICROBE_UP_HINTS)
    microbe_down = text_has_hint(t, MICROBE_DOWN_HINTS)
    if not ((improve or worsen) and (microbe_up or microbe_down)):
        return items

    desired_dir = None
    if improve and not worsen:
        if microbe_up and not microbe_down:
            desired_dir = "decrease"
        elif microbe_down and not microbe_up:
            desired_dir = "increase"
    elif worsen and not improve:
        if microbe_up and not microbe_down:
            desired_dir = "increase"
        elif microbe_down and not microbe_up:
            desired_dir = "decrease"
    else:
        return items

    if not desired_dir:
        return items

    adjusted = []
    for it in items:
        cur = normalize_direction(it.get("direction", ""))
        new_it = dict(it)
        need_flip = False
        if cur in ("not_mentioned", "no_difference"):
            need_flip = True
        elif cur in ("increase", "decrease") and cur != desired_dir:
            ev = (new_it.get("evidence_excerpt") or "").lower()
            if (not ev) or ("treatment" in ev or "intervention" in ev or "improv" in ev or "worsen" in ev):
                need_flip = True

        if need_flip:
            new_it["direction"] = desired_dir
            base = 0.48
            if text_has_hint(t, SIG_HINTS):
                base = 0.62
            try:
                old = float(new_it.get("confidence", 0.0))
            except Exception:
                old = 0.0
            new_it["confidence"] = round(float(np.clip(0.6 * base + 0.4 * old, 0.0, 1.0)), 2)
            if not new_it.get("evidence_excerpt"):
                if desired_dir == "decrease":
                    new_it["evidence_excerpt"] = "ASD improvement with treatment alongside increased microbial abundance (implies inverse relation)"
                else:
                    new_it["evidence_excerpt"] = "ASD improvement with treatment alongside decreased microbial abundance (implies positive relation)"
        adjusted.append(new_it)
    return adjusted

#########################
# Sample size and significance parsing
#########################
def parse_sample_size_from_text(text: str) -> Optional[int]:
    """
    Heuristically parse sample size from title and abstract text.
    Returns the largest plausible sample size as a proxy for total participants.
    """
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
    if not cands:
        return None
    # Prefer the largest value as a proxy for total sample size.
    return max(cands)

def has_significance(sig_str: Optional[str], evidence_text: str) -> bool:
    """
    Treat p<0.05, q<0.05, FDR<0.05, or significant language as statistically significant.
    """
    text = (sig_str or "") + " " + (evidence_text or "")
    text = text.lower()
    if re.search(r'\bp\s*<\s*0\.05\b', text) or re.search(r'\bq\s*<\s*0\.05\b', text) or re.search(r'\bfdr\s*<\s*0\.05\b', text):
        return True
    if re.search(r'\bsignificant\b', text):
        return True
    return False

def direction_is_clear(direction: str) -> bool:
    d = normalize_direction(direction)
    return d in ("increase", "decrease")

def compute_confidence_base_and_adjusted(direction: str,
                                         stat_sig: Optional[str],
                                         evidence_text: str,
                                         study_design: str,
                                         sample_type: str,
                                         sample_size: Optional[int]) -> float:
    """
    Compute confidence for a single extracted relation.
    Base score:
    - Significant and directionally clear: n>200 => 0.8; 100<n<=200 => 0.7; otherwise => 0.6.
    - Directionally clear without significance: 0.5.
    - Directionally unclear: 0.2.
    Adjustments:
    - Longitudinal/cohort design: +0.1; cross-sectional or unknown-like designs: +0.05.
    - Human sample: +0.1; animal sample: +0.05.
    """
    d_clear = direction_is_clear(direction)
    sig = has_significance(stat_sig, evidence_text)
    n = sample_size if (isinstance(sample_size, int) and sample_size > 0) else None

    # Base score.
    base = 0.0
    if sig and d_clear:
        if n is not None and n > 200:
            base = 0.8
        elif n is not None and 100 < n <= 200:
            base = 0.7
        else:
            # Includes n<=100 or unknown sample size.
            base = 0.6
    elif (not sig) and d_clear:
        base = 0.5
    else:
        if normalize_direction(direction) != "not_mentioned":
            base = 0.2
        else:
            base = 0.2

    # Study design adjustment.
    sd = (study_design or "unknown").lower()
    if any(k in sd for k in ["longitudinal", "cohort"]):
        base += 0.1
    elif any(k in sd for k in ["cross", "cross_sectional", "cross-sectional", "unknown", "observational", "case_control"]):
        base += 0.05
    else:
        # Treat unrecognized designs as unknown.
        base += 0.05

    # Sample type adjustment.
    st = (sample_type or "unspecified").lower()
    if "human" in st:
        base += 0.1
    elif "animal" in st:
        base += 0.05

    return float(np.clip(base, 0.0, 1.0))

#########################
# Pipeline
#########################
def build_species_alias_map(species_list: List[str]) -> Dict[str, List[str]]:
    alias_map: Dict[str, List[str]] = {}
    for sp in species_list:
        als = []
        abbr = build_species_abbreviation(sp)
        if abbr:
            als.append(abbr)
            als.append(abbr.replace(". ", "."))
        alias_map[sp] = sorted(list(set(als)))
    return alias_map

def search_species_asd_records(species_list: List[str], years: int, retmax: int) -> List[Dict]:
    all_records = []
    for i, sp in enumerate(species_list, 1):
        q = build_pubmed_query_with_aliases(sp, years=years)
        js = pubmed_search(q, retmax=retmax)
        res = js.get("esearchresult", {})
        idlist = res.get("idlist", [])
        print(f"[INFO] {i}/{len(species_list)} {sp}: hits={res.get('count')} used={len(idlist)}")
        if not idlist:
            continue
        xml_text = pubmed_fetch_xml(idlist)
        papers = parse_pubmed_xml(xml_text)
        for p in papers:
            p.setdefault("species", [])
            p["species"].append(sp)
        all_records.extend(papers)
        time.sleep(REQUEST_INTERVAL_SEC)
    return all_records

def build_paper_to_species(records: List[Dict], alias_map: Dict[str, List[str]]) -> Tuple[List[Dict], Dict[str, List[str]]]:
    paper_map = {}
    paper_species = defaultdict(set)
    for rec in records:
        pmid = rec.get("pmid")
        if not pmid:
            continue
        if pmid not in paper_map:
            paper_map[pmid] = {
                "pmid": pmid,
                "title": rec.get("title") or "",
                "abstract": rec.get("abstract") or "",
                "journal": rec.get("journal"),
                "year": rec.get("year"),
                "doi": rec.get("doi"),
            }
        sps = rec.get("species")
        if sps:
            for sp in sps:
                paper_species[pmid].add(sp)
    unique_papers = []
    for pmid, meta in paper_map.items():
        slist = sorted(list(paper_species[pmid]))
        meta["species"] = slist
        sp_aliases = {sp: alias_map.get(sp, []) for sp in slist}
        meta["species_aliases"] = sp_aliases
        unique_papers.append(meta)
    return unique_papers, {k: sorted(list(v)) for k, v in paper_species.items()}

def process_paper_with_llm(paper: Dict, extractor: LLMExtractor) -> Dict[str, Any]:
    pmid = paper["pmid"]
    slist = paper.get("species", [])
    alias_map = paper.get("species_aliases", {sp: [] for sp in slist})
    items = extractor.extract_species_vs_asd(paper, slist, alias_map)
    missing_species = [
        it.get("species") for it in items
        if it.get("species") and normalize_direction(it.get("direction")) == "not_mentioned"
    ]
    pdf_path = None

    if missing_species:
        pdf_path = find_full_paper_pdf(pmid)
        if pdf_path:
            print(f"[Full-text fallback] PMID {pmid}: {len(missing_species)} species marked as not_mentioned; reading {pdf_path}")
            full_text = extract_text_from_pdf(pdf_path)
            full_text_passages = build_full_text_passages(full_text, missing_species, alias_map)
            if full_text_passages:
                fallback_items = extractor.extract_species_vs_asd_from_full_text(
                    paper=paper,
                    species_list=missing_species,
                    alias_map={sp: alias_map.get(sp, []) for sp in missing_species},
                    full_text_passages=full_text_passages
                )
                fallback_by_species = {it.get("species"): it for it in fallback_items if it.get("species")}
                merged_items = []
                for it in items:
                    sp = it.get("species")
                    if sp in fallback_by_species:
                        fb = fallback_by_species[sp]
                        fb["full_text_pdf"] = str(pdf_path)
                        merged_items.append(fb)
                    else:
                        it["full_text_fallback_used"] = False
                        it["full_text_pdf"] = None
                        merged_items.append(it)
                items = merged_items
            else:
                print(f"[Full-text fallback skipped] PMID {pmid}: no usable text passages extracted from PDF")
        else:
            print(f"[Full-text fallback skipped] PMID {pmid}: no PDF found in {FULL_PAPER_DIR}")

    for it in items:
        it.setdefault("full_text_fallback_used", False)
        if it.get("full_text_fallback_used"):
            it.setdefault("full_text_pdf", str(pdf_path) if pdf_path else None)
        else:
            it["full_text_pdf"] = None
    return {"pmid": pmid, "year": paper.get("year"), "title": paper.get("title",""), "abstract": paper.get("abstract",""), "species_relations": items}

def batch_process_papers(papers: List[Dict], llm_config: LLMConfig) -> List[Dict]:
    extractor = LLMExtractor(llm_config, OUT_LLM_CACHE)
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(process_paper_with_llm, p, extractor) for p in papers]
        for i, fu in enumerate(futs, 1):
            res = fu.result(timeout=LLM_REQUEST_TIMEOUT)
            results.append(res)
            cnt = len(res.get("species_relations", []))
            print(f"[{i}/{len(futs)}] PMID {res['pmid']} completed; extracted {cnt} relations")

    extractor.save_cache()
    return results

#########################
# Relation formatting and aggregation
#########################
def build_relations_from_llm_results(llm_results: List[Dict]) -> List[Dict]:
    rows = []
    for item in llm_results:
        pmid = item.get("pmid")
        paper_year = item.get("year")
        title = item.get("title","")
        abstract = item.get("abstract","")
        text_all = (title + " " + abstract).strip()
        sample_size = parse_sample_size_from_text(text_all)

        for r in item.get("species_relations", []):
            sp = r.get("species")
            if not sp:
                continue
            direction = normalize_direction(r.get("direction"))
            sig = r.get("statistical_significance")
            eff = r.get("effect_size")
            sample_type = r.get("sample_type") or "unspecified"
            study_design = r.get("study_design") or "unknown"
            ev = (r.get("evidence_excerpt") or "")[:500]
            full_text_fallback_used = bool(r.get("full_text_fallback_used", False))
            full_text_pdf = r.get("full_text_pdf")

            # Recompute confidence with deterministic scoring.
            conf_new = compute_confidence_base_and_adjusted(
                direction=direction,
                stat_sig=sig,
                evidence_text=ev if ev else text_all,
                study_design=study_design,
                sample_type=sample_type,
                sample_size=sample_size
            )

            rows.append({
                "pmid": pmid,
                "year": paper_year,
                "species": sp,
                "direction": direction,        # increase | decrease | no_difference | not_mentioned
                "statistical_significance": sig,
                "effect_size": eff,
                "sample_type": sample_type,    # human | animal | in_vitro | unspecified
                "study_design": study_design,
                "evidence_text": ev,
                "confidence": conf_new,
                "sample_size": sample_size,
                "full_text_fallback_used": full_text_fallback_used,
                "full_text_pdf": full_text_pdf,
                "evidence_source": "full_text_pdf_fallback" if full_text_fallback_used else "title_abstract_explicit_species_vs_asd",
            })
    return rows

def aggregate_species_vs_asd(relations: List[Dict]) -> List[Dict]:
    """
    Aggregate paper-level relations into species-level evidence.
    - Encode direction as: increase=+1, decrease=-1, no_difference/not_mentioned=0.
    - Compute a confidence-weighted mean direction score and assign consensus_direction.
    - Compute base confidence as a confidence-weighted average.
    - Penalize directional conflicts using min(pos_w, neg_w) / total_w.
    - Discount sparse evidence with n / (n + k), where k=3.
    """
    bucket = defaultdict(list)
    for r in relations:
        bucket[r["species"]].append(r)

    merged = []
    k = 3.0

    for sp, lst in bucket.items():
        dir_map = {"increase": 1.0, "decrease": -1.0, "no_difference": 0.0, "not_mentioned": 0.0}
        # Use relation-level confidence as the weight.
        weights = np.array([max(1e-6, float(x.get("confidence", 0.0))) for x in lst], dtype=float)
        vals = np.array([dir_map.get(x["direction"], 0.0) for x in lst], dtype=float)

        has_inc = any(v > 0 for v in vals)
        has_dec = any(v < 0 for v in vals)
        directional_conflict = has_inc and has_dec

        # Confidence-weighted direction mean.
        wmean = float(np.average(vals, weights=weights)) if len(vals) else 0.0

        # Consensus direction.
        if wmean > 0.15:
            consensus = "increase"
        elif wmean < -0.15:
            consensus = "decrease"
        else:
            cnt_no = sum(1 for x in lst if x["direction"] == "no_difference")
            if cnt_no >= max(1, len(lst) // 3):
                consensus = "no_difference"
            else:
                consensus = "uncertain"

        # Base confidence is the confidence-weighted average.
        base_conf = float(np.clip(np.average([x.get("confidence", 0.0) for x in lst], weights=weights), 0.0, 1.0))

        # Conflict ratio.
        if directional_conflict:
            pos_w = float(np.sum(weights[vals > 0])) if np.any(vals > 0) else 0.0
            neg_w = float(np.sum(weights[vals < 0])) if np.any(vals < 0) else 0.0
            tot_w = float(np.sum(weights)) if len(weights) else 1.0
            conflict_ratio = min(1.0, min(pos_w, neg_w) / max(1e-6, tot_w))
        else:
            conflict_ratio = 0.0

        # Penalize confidence by directional conflict.
        conf_after_conflict = max(0.0, abs(base_conf) - conflict_ratio)

        # Discount sparse paper counts with n/(n+k), k=3.
        n_records = len(lst)
        factor = n_records / (n_records + k) if n_records > 0 else 0.0
        final_conf = float(np.clip(conf_after_conflict * factor, 0.0, 1.0))

        evid_samples = []
        for x in sorted(lst, key=lambda d: -d.get("confidence", 0.0)):
            if x.get("evidence_text"):
                evid_samples.append({
                    "pmid": x["pmid"],
                    "text": x["evidence_text"][:280],
                    "direction": x["direction"],
                    "year": x.get("year")
                })
            if len(evid_samples) >= 3:
                break

        merged.append({
            "species": sp,
            "consensus_direction": consensus,
            "direction_score": round(wmean, 3),
            "aggregated_confidence": round(final_conf, 3),
            "direction_conflict": directional_conflict,
            "conflict_ratio": round(conflict_ratio, 3),
            "n_records": n_records,
            "factor_n_over_n_plus_k": round(factor, 3),
            "top_evidence": evid_samples,
            "pmids": sorted(list({x["pmid"] for x in lst if x.get("pmid")})),
            "study_design_mode": Counter([x.get("study_design","unknown") for x in lst]).most_common(1)[0][0],
            "sample_type_mode": Counter([x.get("sample_type","unspecified") for x in lst]).most_common(1)[0][0],
        })
    return merged

#########################
# Main entry point
#########################
def main():
    check_env()

    llm_config = LLMConfig(
        provider=LLM_PROVIDER,
        model=OPENAI_MODEL,
        temperature=0.1,
        max_tokens=1500,
        api_base=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY
    )

    # Load species.
    species_df = load_selected_species(INPUT_SPECIES_CSV)
    species_list = species_df["species"].tolist()
    print(f"Loaded species: {len(species_list)}")

    # Build species abbreviation aliases.
    alias_map = build_species_alias_map(species_list)

    # Search PubMed.
    print("Searching PubMed records...")
    records = search_species_asd_records(species_list, years=YEARS, retmax=TOP_K_PER_SPECIES)
    print(f"Fetched records: {len(records)}")

    # Build paper-to-species mapping with aliases.
    unique_papers, paper_species_map = build_paper_to_species(records, alias_map)
    print(f"Unique papers: {len(unique_papers)}")

    # Extract species-ASD relations with the LLM.
    print("Extracting species-ASD relations with LLM...")
    llm_results = batch_process_papers(unique_papers, llm_config)

    # Flatten relations with year, sample size, and recomputed confidence.
    relations = build_relations_from_llm_results(llm_results)
    print(f"Extracted relations: {len(relations)}")

    # Aggregate to species-level evidence.
    merged = aggregate_species_vs_asd(relations)
    print(f"Aggregated species: {len(merged)}")

    # Save detailed relation outputs.
    df_rel = pd.DataFrame(relations)
    df_rel.to_csv(OUT_RELATIONS_CSV, index=False, encoding="utf-8-sig")
    with open(OUT_RELATIONS_JSONL, "w", encoding="utf-8") as f:
        for row in relations:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save raw LLM results and aggregated outputs.
    with open(OUT_LLM_RAW, "w", encoding="utf-8") as f:
        json.dump(llm_results, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "species_vs_asd_aggregated.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print("\nDone!")
    print("Output files:")
    print(f"  - Detailed relations: {OUT_RELATIONS_CSV}")
    print(f"  - Detailed JSONL: {OUT_RELATIONS_JSONL}")
    print(f"  - Raw LLM results: {OUT_LLM_RAW}")
    print(f"  - Aggregated results: {OUT_DIR / 'species_vs_asd_aggregated.json'}")

if __name__ == "__main__":
    main()
