from nltk import CFG
from nltk.parse import ChartParser

# Define a recursive CFG
grammar = CFG.fromstring("""
    S -> 'a' S 'b' | 'a' 'b'
""")

parser = ChartParser(grammar)

def is_recursive(text):
    words = list(text)  # Treats input as a sequence of characters
    try:
        for _ in parser.parse(words):
            return True
    except ValueError:
        pass
    return False

# Example Tests
texts = ["ab", "aabb", "aaabbb", "aabbb"]
for text in texts:
    print(f"'{text}' -> Accepted" if is_recursive(text) else f"'{text}' -> Rejected")
