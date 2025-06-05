import sys

def parse_text_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            raw_text = file.read()
        print(f"Total number of characters: {len(raw_text)}")
        print(f"Total number of words: {len(raw_text.split())}")
        print(f"Total number of lines: {len(raw_text.splitlines())}")
        print("Preview (first 100 characters):")
        print(raw_text[:100])
        return raw_text
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_parsing.py <filename>")
    else:
        parse_text_file(sys.argv[1])