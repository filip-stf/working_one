import stealth_requests as requests
from bs4 import BeautifulSoup

def save_html_body(url: str, output_path: str = 'body.txt'):
    """
    Fetches the HTML from the given URL and saves the <body> content to a text file.
    Args:
        url: The URL to fetch.
        output_path: The file path to save the body HTML.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    body = soup.body
    if body:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(str(body))
        print(f"Saved <body> HTML to {output_path}")
    else:
        print("No <body> tag found in the HTML.")

def clean_html_file(input_path: str, output_path: str = 'cleaned_text.txt'):
    """
    Reads a file containing HTML, removes all tags, and writes only the plain text to output_path.
    Args:
        input_path: Path to the HTML file.
        output_path: Path to save the cleaned text.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator='\n', strip=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved cleaned text to {output_path}")

# Example usage:
save_html_body('https://tolkiengateway.net/wiki/Gandalf', 'gandalf_body.txt')
#clean_html_file('aragorn_body.txt', 'aragorn_cleaned.txt')
