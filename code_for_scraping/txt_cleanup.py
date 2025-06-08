import re

def fix_punctuation_spacing(input_path, output_path=None):
    """
    Fixes spaces before punctuation marks in a text file. This is needed because the html parser code usually 
    leaves spaces before punctuation marks.
    Args:
        input_path: Path to the input text file.
        output_path: Path to save the fixed text. If None, overwrites input file.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Remove space before common punctuation marks
    fixed_text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Optionally, fix multiple spaces to single space
    fixed_text = re.sub(r' {2,}', ' ', fixed_text)
    if not output_path:
        output_path = input_path
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(fixed_text)
    print(f"Fixed punctuation spacing in {output_path}")

# Example usage:
fix_punctuation_spacing('inner.txt')
