import argparse
from src.scraping.sebi_scraper import scrape_sebi
from src.scraping.cpcb_scraper import scrape_cpcb
from src.scraping.ccts_scraper import scrape_ccts
from src.scraping.utils import download_direct_pdfs

parser = argparse.ArgumentParser()
parser.add_argument('--scrape', action='store_true', help='Run all scrapers')

args = parser.parse_args()

if args.scrape:
    print("Running SEBI scraper...")
    sebi_results = scrape_sebi()
    print(f"Found {len(sebi_results)} relevant BRSR notifications.")

    print("Running CPCB scraper...")
    cpcb_results = scrape_cpcb()
    print(f"Found {len(cpcb_results)} relevant EPR notifications.")

    print("Running CCTS scraper...")
    ccts_results = scrape_ccts()
    print(f"Found {len(ccts_results)} relevant CCTS notifications.")

    print("\nDownloading key regulatory PDFs (direct - reliable)...")
    direct_results = download_direct_pdfs()
    print(f"Downloaded {len(direct_results)} essential documents for customers.")

    print("Scraping done. PDFs saved in data/raw/")