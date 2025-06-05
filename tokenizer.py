import sys
import re
from file_parsing import parse_text_file

def tokenize_text(text):
    # Tokenize words and special characters, strip whitespace
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens

def main(filename):
    raw_text = parse_text_file(filename)
    if raw_text is not None:
        tokens = tokenize_text(raw_text)
        print(f"Total tokens: {len(tokens)}")
        print("Tokens:")
        print(tokens)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tokenizer.py <filename>")
    else:
        main(sys.argv[1]) 