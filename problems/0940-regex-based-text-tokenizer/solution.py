import re

def tokenize_text(text: str) -> list:
    """
    Split raw text into tokens using regex-based splitting on whitespace
    and punctuation. Returns a list of non-empty stripped tokens.
    """
    patterns = r'(\s+|--|[,.;:,?_!"()\'-])'
    parts = re.split(patterns,text)
    ans = []
    for i in range(len(parts)):
        if parts[i].strip():
            ans.append(parts[i].strip())
    return ans