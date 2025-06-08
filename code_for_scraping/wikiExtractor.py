
import requests

def fetch_and_save_wikitext(wiki_url: str, out_path: str = "wikitext.txt"):
    """
    Fetches plain text from a Wikipedia article using the eluni.co API and saves it to a file.
    Args:
        wiki_url: The full Wikipedia article URL (e.g., https://en.wikipedia.org/wiki/Aragorn)
        out_path: The file path to save the extracted text.
    """
    api_url = f"https://wikitext.eluni.co/api/extract?url={wiki_url}&format=text"
    response = requests.get(api_url)
    response.raise_for_status()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"Saved extracted text to {out_path}")

# Example usage:
fetch_and_save_wikitext("https://en.wikipedia.org/wiki/Samwise_Gamgee", "sam.txt")

