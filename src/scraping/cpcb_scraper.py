import requests
from bs4 import BeautifulSoup
from .utils import download_pdf, check_relevancy_with_ai, config

def scrape_cpcb():
    notifications = []
    for entry in config.get('cpcb_urls', []):
        url = entry['url']
        try:
            response = requests.get(url)
            response.raise_for_status()
        except Exception as e:
            print(f"Error fetching CPCB: {e}")
            continue

        soup = BeautifulSoup(response.text, 'html.parser')
        for a in soup.find_all('a', href=True):
            title = a.text.strip()
            link = a['href']
            if 'pdf' in link.lower() or 'pdffile' in link.lower():
                full_link = requests.urljoin(url, link)
                is_relevant, summary = check_relevancy_with_ai(title)
                if is_relevant:
                    save_path = f"data/raw/cpcb/{title[:50].replace(' ', '_')}.pdf"
                    if download_pdf(full_link, save_path):
                        notifications.append({'title': title, 'summary': summary, 'path': save_path})
    return notifications

if __name__ == "__main__":
    results = scrape_cpcb()
    print(f"Found {len(results)} relevant EPR notifications.")