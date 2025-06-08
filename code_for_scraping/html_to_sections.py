import requests
from bs4 import BeautifulSoup, Tag
import os
import re

def sanitize_filename(text):
    # Clean and shorten for filename
    text = re.sub(r'[\\/*?:"<>|]', "", text)  # Remove illegal filename characters
    text = re.sub(r'\s+', "_", text)          # Replace spaces with underscores
    return text[:40]  # limit length

def clean_text(tag):
    # Remove hyperlinks, reference tags, figures, and figure captions
    for a in tag.find_all('a'):
        a.unwrap()
    for sup in tag.find_all('sup'):
        sup.decompose()
    # Remove all <figure> tags and their content
    for fig in tag.find_all('figure'):
        fig.decompose()
    # Remove all <figcaption> tags and their content
    for figcap in tag.find_all('figcaption'):
        figcap.decompose()
    return tag.get_text(separator=" ", strip=True)

def extract_all_headers(soup):
    # Recursively find all header tags and their content, even if nested
    headers = soup.find_all(re.compile(r'^h[1-6]$'))
    seen = set()
    sections = []

    def extract_section(header):
        header_id = id(header)
        if header_id in seen:
            return []
        seen.add(header_id)
        header_text = clean_text(header)
        content = []
        for sibling in header.next_siblings:
            if isinstance(sibling, Tag) and re.match(r'^h[1-6]$', sibling.name):
                break
            if isinstance(sibling, Tag):
                # Recursively extract headers inside this tag
                inner_headers = sibling.find_all(re.compile(r'^h[1-6]$'))
                for inner_header in inner_headers:
                    sections.extend(extract_section(inner_header))
                # Only add content if it's not a header
                if not inner_headers:
                    content.append(clean_text(sibling))
            elif isinstance(sibling, str):
                content.append(sibling.strip())
        section_text = "\n".join([t for t in content if t])
        # Only return if section_text is not empty and not just the header itself
        if section_text.strip() and section_text.strip() != header_text.strip():
            return [(header_text, section_text)]
        return []

    for header in headers:
        if id(header) not in seen:
            sections.extend(extract_section(header))
    return sections

def main(url=None):
    # Read HTML from inner.txt instead of fetching from the web
    with open("inner.txt", "r", encoding="utf-8") as file:
        html = file.read()
    soup = BeautifulSoup(html, 'html.parser')

    sections = extract_all_headers(soup)
    out_dir = "scraped_sections_md"
    os.makedirs(out_dir, exist_ok=True)

    # Group sections by header level and write to a single text file
    txt_lines = []
    for header, content in sections:
        # Determine header level (e.g., h2 -> ##)
        match = re.match(r'^(h[1-6])$', next(tag.name for tag in soup.find_all(re.compile(r'^h[1-6]$')) if clean_text(tag) == header))
        if match:
            level = int(match.group(1)[1])
        else:
            level = 2  # Default to h2 if not found
        txt_lines.append(f"{'#' * level} {header}\n")
        if content.strip():
            txt_lines.append(f"{content.strip()}\n")

    # Write all sections to a single text file
    txt_path = os.path.join(out_dir, "sections.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_lines))

    print(f"Saved text file with {len(sections)} sections to ./{out_dir}/sections.txt")

if __name__ == "__main__":
    main()
