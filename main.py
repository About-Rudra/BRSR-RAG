import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--scrape', action='store_true', help='Run all scrapers and download key PDFs')
parser.add_argument('--ingest', action='store_true', help='Ingest every PDF under data/raw/<regime>/ into Qdrant')
parser.add_argument('--ask', metavar='QUESTION', help='Answer a question with the research agent')
parser.add_argument('--watch', action='store_true', help='Check the regulator sites for new or changed documents')
parser.add_argument('--simulate', metavar='PDF', help='With --watch: SIMULATION - treat a local PDF as newly found')
parser.add_argument('--title', help='With --simulate: title to use for the document')
parser.add_argument('--approve', metavar='ID', help='Approve a pending regulatory update')
parser.add_argument('--reject', metavar='ID', help='Reject a pending regulatory update')
parser.add_argument('--pending', action='store_true', help='List regulatory updates waiting for approval')

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

if args.ask:
    from src.agent.graph import run_research, save_run

    result = run_research(args.ask)
    print(result["final_answer"])
    print(f"\nStats: {result['stats']}")
    print(f"Full trace saved to {save_run(result)}")

if args.watch:
    from src.watch.run import run_watch, save_watch_run

    if args.simulate:
        print("*** SIMULATION: a local file is treated as a newly found document. This is not a live scrape. ***")
    summary = run_watch(simulate_path=args.simulate, title=args.title)
    for r in summary["processed"]:
        print(f"- [{r['status']}] {r['title']} ({r['change']})" + (f" -> pending id {r['pending_id']}" if r["pending_id"] else ""))
    print(f"Processed {len(summary['processed'])}, skipped {len(summary['skipped'])}, errors {len(summary['errors'])}.")
    print(f"Trace saved to {save_watch_run(summary)}")

if args.pending:
    from src.watch.approvals import load_pending

    for e in load_pending():
        if e["status"] == "pending_approval":
            label = "[SIMULATION] " if e.get("simulated") else ""
            print(f"{e['id']}  {label}{e['title']}  ({e['regime']})")

if args.approve or args.reject:
    from src.watch.approvals import approve, reject

    entry = approve(args.approve) if args.approve else reject(args.reject)
    print(f"{entry['status']}: {entry['title']}")
