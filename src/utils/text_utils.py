"""Utility functions for text processing and validation."""
import re
from typing import Optional

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    # Replace multiple newlines with double newline
    text = re.sub(r'\n\n+', '\n\n', text)
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()

def is_valid_url(url: str) -> bool:
    """Check if URL is valid."""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

def extract_domain(url: str) -> Optional[str]:
    """Extract domain from URL."""
    match = re.search(r'https?://([^/]+)', url)
    return match.group(1) if match else None

def clean_text(text: str) -> str:
    """Clean extracted text."""
    # Remove excessive whitespace
    text = normalize_whitespace(text)
    
    # Remove common boilerplate patterns
    boilerplate_patterns = [
        r'Cookie Policy.*?Accept',
        r'We use cookies.*?(?:\n|$)',
        r'Privacy Policy.*?(?:\n|$)',
        r'All rights reserved\.?',
        r'Copyright \d{4}.*?(?:\n|$)',
        r'Skip to (?:main )?content',
        r'Share on (?:Facebook|Twitter|LinkedIn)',
    ]
    
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return normalize_whitespace(text)

# The AIMS Cameroon WordPress theme prepends this same fixed mega-menu block
# (English or French, depending on locale) ahead of every page's real content.
# It isn't wrapped in a <nav>/<header> element, so _clean_soup() can't remove
# it, and it ends up inline in raw_text - on short pages (single
# person/researcher profiles) it can be 60-80% of the stored text. Captured
# verbatim from clean_text()+remove_duplicates() output; verified as an exact
# prefix match against ~1700 already-scraped documents.
NAV_MENU_BLOCKS = [
    "EN EN FR\n\nAIMS ECOSYSTEM\n\nAIMS ENTITIES\n\nAIMS GLOBAL SECRETARIAT\n\n"
    "AIMS SOUTH AFRICA\n\nAIMS SENEGAL\n\nAIMS GHANA\n\nAIMS CAMEROON\n\nAIMS RWANDA\n\n"
    "AIMS INITIATIVES\n\nNEXT EINSTEIN FORUM (NEF)\n\nQUANTUM LEAP AFRICA\n\n"
    "AIMS PROGRAMS\n\nAFRICAN MASTER’S IN MACHINE INTELLIGENCE (AMMI)\n\n"
    "AIMS RESEARCH\n\nMASTERCARD FOUNDATION SCHOLARS PROGRAM @ AIMS",

    "FR FR EN\n\nÉCOSYSTÈME AIMS\n\nENTITÉS AIMS\n\nSECRÉTARIAT MONDIAL AIMS\n\n"
    "AIMS AFRIQUE DU SUD\n\nAIMS SENEGAL\n\nAIMS GHANA\n\nAIMS CAMEROUN\n\nAIMS RWANDA\n\n"
    "INITIATIVES AIMS\n\nPROCHAIN FORUM EINSTEIN (NEF)\n\nQUANTUM LEAP AFRIQUE\n\n"
    "PROGRAMMES AIMS\n\nMASTER AFRICAIN EN INTELLIGENCE MACHINE (AMMI)\n\n"
    "RECHERCHE AIMS\n\nPROGRAMME DE BOURSES D'ÉTUDES DE LA FONDATION MASTERCARD @ AIMS",
]

def strip_nav_menu(text: str) -> str:
    """Strip the repeated site nav-menu block from the start of extracted text.

    No-op (returns text unchanged) if none of the known blocks match, e.g.
    for PDF-derived text, which never had a menu in the first place.
    """
    for block in NAV_MENU_BLOCKS:
        if text.startswith(block):
            return text[len(block):].lstrip('\n').lstrip()
    return text

def remove_duplicates(text: str, threshold: int = 50) -> str:
    """Remove duplicate paragraphs from text."""
    paragraphs = text.split('\n\n')
    seen = set()
    unique_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Create a signature for the paragraph (first threshold chars)
        signature = para[:threshold].lower()
        
        if signature not in seen:
            seen.add(signature)
            unique_paragraphs.append(para)
    
    return '\n\n'.join(unique_paragraphs)

def categorize_content(url: str, title: str, text: str) -> str:
    """Attempt to categorize content based on URL, title, and text."""
    content_lower = (url + " " + title + " " + text).lower()
    
    category_keywords = {
        "Admissions": ["admission", "apply", "application", "eligibility", "deadline", "entrance"],
        "Academics": ["program", "curriculum", "course", "academic", "degree", "master", "study"],
        "Scholarships": ["scholarship", "funding", "financial", "tuition", "fully funded", "stipend"],
        "Faculty": ["faculty", "professor", "lecturer", "staff", "researcher", "tutor"],
        "Research": ["research", "publication", "lab", "project", "collaboration"],
        "Student Life": ["student life", "accommodation", "campus", "housing", "facility", "dormitory"],
        "Administration": ["contact", "office", "admin", "director", "email", "phone"],
        "FAQs": ["faq", "question", "answer", "q&a"],
        "Policies": ["policy", "regulation", "rule", "guideline", "code of conduct"],
    }
    
    # Count matches for each category
    category_scores = {}
    for category, keywords in category_keywords.items():
        score = sum(content_lower.count(keyword) for keyword in keywords)
        category_scores[category] = score
    
    # Return category with highest score, default to Institutional
    if max(category_scores.values()) > 0:
        return max(category_scores, key=category_scores.get)
    
    return "Institutional"
