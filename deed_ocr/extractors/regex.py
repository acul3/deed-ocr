"""Regex-based legal description extraction."""

import re
from typing import List, Tuple

# Common legal description patterns
PATTERNS = {
    "lot_block": re.compile(
        r"(Lot\s+\d+[A-Z]?,?\s*Block\s+\d+[A-Z]?,?\s*(?:of\s+)?[A-Z\s]+(?:SUBDIVISION|ADDITION))",
        re.IGNORECASE,
    ),
    "section_township_range": re.compile(
        r"((?:(?:N|S|E|W|NE|NW|SE|SW)\s*1/[24]\s+of\s+)?Section\s+\d+,?\s*Township\s+\d+\s*[NS],?\s*Range\s+\d+\s*[EW])",
        re.IGNORECASE,
    ),
    "metes_bounds_start": re.compile(
        r"(Beginning\s+at\s+(?:a\s+point\s+)?.*?(?:thence|containing).*?(?:acres?|feet))",
        re.IGNORECASE | re.DOTALL,
    ),
    "legal_desc_header": re.compile(
        r"(?:LEGAL\s+DESCRIPTION|PROPERTY\s+DESCRIPTION|DESCRIBED\s+AS\s+FOLLOWS?):?\s*",
        re.IGNORECASE,
    ),
}


def extract_with_regex(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract legal descriptions using regex patterns.
    
    Args:
        text: Input text to search
        
    Returns:
        List of (match_text, start_pos, end_pos) tuples
    """
    matches = []
    
    # Check for explicit legal description headers first
    header_match = PATTERNS["legal_desc_header"].search(text)
    if header_match:
        # Extract text after header (next 500 chars as context)
        start_pos = header_match.end()
        context_text = text[start_pos : start_pos + 500]
        # TODO: Apply more sophisticated extraction after header
    
    # Apply all patterns
    for pattern_name, pattern in PATTERNS.items():
        if pattern_name == "legal_desc_header":
            continue
        for match in pattern.finditer(text):
            matches.append((match.group(0), match.start(), match.end()))
    
    return matches 