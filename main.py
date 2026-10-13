import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--scrape', action='store_true', help='Run all scrapers and download key PDFs')
parser.add_argument('--ingest', action='store_true', help='Ingest every PDF under data/raw/<regime>/ into Qdrant')

args = parser.parse_args()

if args.scrape:
    from src.scraping.sebi_scraper import scrape_sebi
    from src.scraping.cpcb_scraper import scrape_cpcb
    from src.scraping.ccts_scraper import scrape_ccts
    from src.scraping.utils import download_direct_pdfs

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

if args.ingest:
    # Always synchronous here, regardless of INGEST_MODE.
    from src.scraping.utils import REGIMES
    from src.rag.ingestion import ingest_single_file

    total, failed = 0, 0
    for regime in REGIMES:
        files = sorted(glob.glob(os.path.join("data", "raw", regime, "*.pdf")))
        print(f"\n{regime}: {len(files)} PDF(s)")
        for file_path in files:
            total += 1
            try:
                ingest_single_file(file_path, regime)
            except Exception as e:
                failed += 1
                print(f"Ingestion failed for {file_path}: {e}")

    print(f"\nIngest done: {total} file(s) attempted, {failed} raised errors (see log above).")
