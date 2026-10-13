from urllib.parse import urljoin
from bs4 import BeautifulSoup
from .utils import download_pdf, check_relevancy_with_ai, safe_filename, session, config, REQUEST_TIMEOUT

CATEGORY = "cpcb_epr"


def list_candidates():
    """List PDF links on the configured CPCB pages as {url, title, regime}, without downloading."""
    candidates, seen = [], set()
    for entry in config.get('cpcb_urls', []):
        url = entry['url']
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except Exception as e:
            print(f"Error fetching CPCB: {e}")
            continue

        soup = BeautifulSoup(response.text, 'html.parser')
        for a in soup.find_all('a', href=True):
            link = a['href']
            if 'pdf' in link.lower() or 'pdffile' in link.lower():
                full_link = urljoin(url, link)
                if full_link not in seen:
                    seen.add(full_link)
                    candidates.append({'url': full_link, 'title': a.text.strip(), 'regime': CATEGORY})
    return candidates


def scrape_cpcb():
    notifications = []
    for c in list_candidates():
        is_relevant, summary = check_relevancy_with_ai(c['title'])
        if is_relevant:
            save_path = f"data/raw/{CATEGORY}/{safe_filename(c['title'])}.pdf"
            if download_pdf(c['url'], save_path, CATEGORY):
                notifications.append({'title': c['title'], 'summary': summary, 'path': save_path})
    return notifications


if __name__ == "__main__":
    results = scrape_cpcb()
    print(f"Found {len(results)} relevant EPR notifications.")
