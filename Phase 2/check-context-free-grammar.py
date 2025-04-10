import nltk
from nltk import CFG

# Define a simple CFG
grammar = CFG.fromstring("""
    S -> NP VP
    NP -> 'the' 'cat' | 'the' 'dog'
    VP -> 'chases' 'the' 'mouse' | 'sleeps'
""")

# Create a parser
parser = nltk.ChartParser(grammar)

def is_accepted(text):
    words = text.split()
    try:
        for tree in parser.parse(words):
            return True  # If there is at least one parse tree, it is accepted
    except ValueError:
        pass
    return False

# Example Tests
texts = [
    "the cat chases the mouse",
    "the dog sleeps",
    "the bird sings"  # Invalid sentence (not in grammar)
]

for text in texts:
    print(f"'{text}' -> Accepted" if is_accepted(text) else f"'{text}' -> Rejected")
