from nltk.parse import RecursiveDescentParser
from nltk import CFG

# Define a Regular Grammar (which is a subset of CFG)
grammar = CFG.fromstring("""
    S -> 'a' S 'b' | 'a' 'b'
""")

parser = RecursiveDescentParser(grammar)

def is_regular(text):
    words = list(text)  # Regular grammar processes character-by-character
    try:
        for _ in parser.parse(words):
            return True
    except ValueError:
        pass
    return False

# Example Tests
texts = ["ab", "aabb", "aaabbb", "aabbb"]
for text in texts:
    print(f"'{text}' -> Accepted" if is_regular(text) else f"'{text}' -> Rejected")
