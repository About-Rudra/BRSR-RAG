import os
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import yaml
import warnings

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

def download_pdf(url, save_path):
    """
    Downloads a PDF from URL and saves to save_path.
    Uses session with verify=False to handle CPCB SSL issues.
    """
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False

def check_relevancy_with_ai(text, keywords=config.get('relevance_keywords', [])):
    """
    Uses Gemini to check if text is relevant.
    Returns (is_relevant: bool, summary: str)
    """
    llm = ChatGoogleGenerativeAI(
        model=config['llm_model'],
        temperature=config['temperature'],
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )
    prompt = (
        f"Is this text relevant to compliance topics like {', '.join(keywords)}? "
        "Respond with 'Yes' or 'No', followed by a short summary if Yes."
    )
    try:
        response = llm.invoke(prompt + "\nText: " + text[:2000])  # Limit to avoid token overflow
        content = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        is_relevant = content.lower().startswith('yes')
        summary = content.split('\n', 1)[1].strip() if is_relevant and '\n' in content else content
        return is_relevant, summary
    except Exception as e:
        print(f"LLM error: {e}")
        return False, ""

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
            if download_pdf(url, save_path):
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