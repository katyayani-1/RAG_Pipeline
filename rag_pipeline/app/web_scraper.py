import requests
from bs4 import BeautifulSoup

# Scrape content from a website
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator="\n")  # Get all text with line breaks
    return text

# Chunk scraped text for embedding
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
