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
