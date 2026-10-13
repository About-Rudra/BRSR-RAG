import os
import requests
from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import yaml
import warnings
from src.queue.producer import submit_ingest


# Load environment variables (GOOGLE_API_KEY from .env)
load_dotenv()

# Suppress SSL warnings if using verify=False
warnings.filterwarnings('ignore', category=UserWarning)

# Load config.yaml
with open('src/config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Optional: Global session with relaxed SSL (for gov sites on Windows)
session = requests.Session()
session.verify = False  # Bypass SSL verification issues safely for scraping


def _extract_pdf_bytes(content):
    """Return the PDF bytes in a response body, or None if it is not a PDF."""
    if content.startswith(b"%PDF"):
        return content
    # Some CPCB links put a small HTML wrapper in front of the PDF bytes.
    start = content.find(b"%PDF", 0, 4096)
    if start != -1:
        return content[start:]
    return None


def _find_embedded_pdf_url(html, base_url):
    """SEBI circular pages are HTML viewers: <iframe src="../web/?file=<pdf url>">."""
    soup = BeautifulSoup(html, "html.parser")
    for iframe in soup.find_all("iframe", src=True):
        src = iframe["src"]
        file_param = parse_qs(urlparse(src).query).get("file")
        if file_param:
            return urljoin(base_url, file_param[0])
        if ".pdf" in src.lower():
            return urljoin(base_url, src)
    return None


def download_pdf(url, save_path, category):
    """
    Downloads a PDF from URL, saves it,
    and submits it for ingestion.
    If the URL is an HTML page that embeds a PDF, the embedded PDF is downloaded.
    """
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()

        pdf_bytes = _extract_pdf_bytes(response.content)
        if pdf_bytes is None:
            pdf_url = _find_embedded_pdf_url(response.text, url)
            if pdf_url:
                response = session.get(pdf_url, timeout=30)
                response.raise_for_status()
                pdf_bytes = _extract_pdf_bytes(response.content)

        if pdf_bytes is None:
            print(f"Skipped (not a PDF): {url}")
            return False

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(pdf_bytes)

        print(f"Downloaded: {save_path}")

        # Ingest now (INGEST_MODE=sync) or enqueue for the worker (INGEST_MODE=redis)
        submit_ingest(
            file_path=save_path,
            category=category,
            source="scraper"
        )

        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False


# New: Direct download for key historical/latest PDFs (reliable fallback)
def download_direct_pdfs():
    notifications = []
    direct_pdfs = config.get('direct_pdfs', {})
    for category, pdf_list in direct_pdfs.items():
        folder = f"data/raw/{category}"
        os.makedirs(folder, exist_ok=True)
        for pdf in pdf_list:
            name = pdf['name']
            url = pdf['url']
            save_path = f"{folder}/{name[:60].replace(' ', '_').replace('/', '_')}.pdf"
            if download_pdf(url, save_path, category):
                # Force relevant for key docs
                notifications.append({
                    'title': name,
                    'path': save_path,
                    'summary': 'Key regulatory document (direct download)'
                })
    return notifications

if __name__ == "__main__":
    # Test functions
    print("Testing direct download...")
    results = download_direct_pdfs()
    print(f"Downloaded {len(results)} key documents.")