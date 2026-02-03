import os
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import yaml
import warnings
from src.queue.producer import enqueue_job


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


def download_pdf(url, save_path, category):
    """
    Downloads a PDF from URL, saves it,
    and enqueues it for ingestion.
    """
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(response.content)

        print(f"Downloaded: {save_path}")

        # ✅ ENQUEUE FOR INGESTION
        enqueue_job(
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