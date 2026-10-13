from urllib.parse import urljoin
from bs4 import BeautifulSoup  # For parsing HTML
from .utils import download_pdf, check_relevancy_with_ai, safe_filename, session, config, REQUEST_TIMEOUT

CATEGORY = "sebi"


def scrape_sebi():  # Main function
    """
    Scrapes the SEBI circulars list, filters relevant ones, and downloads their PDFs.
    Why: Encapsulates logic; callable from main.py.
    """
    url = config['sebi_url']  # From config
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)  # Fetch page
        response.raise_for_status()  # Check status
    except Exception as e:
        print(f"Error fetching SEBI: {e}")
        return []  # Return empty on failure

    soup = BeautifulSoup(response.text, 'html.parser')  # Parse HTML

    notifications = []  # List to store relevant ones
    # The listing is a table: each row is <td>date</td><td><a href=circular page>title</a></td>
    for row in soup.select('table tr'):
        cells = row.find_all('td')
        if len(cells) < 2:
            continue  # header row
        a = cells[1].find('a', href=True)
        if not a:
            continue
        date = cells[0].text.strip()  # e.g., "Oct 07, 2026"
        title = (a.get('title') or a.text).strip()
        full_link = urljoin(url, a['href'])  # Links may be relative

        is_relevant, summary = check_relevancy_with_ai(title)  # Quick check on title
        if is_relevant:
            # The circular page is an HTML viewer; download_pdf follows the embedded PDF.
            save_path = f"data/raw/{CATEGORY}/{safe_filename(title)}.pdf"
            if download_pdf(full_link, save_path, CATEGORY):
                notifications.append({'date': date, 'title': title, 'summary': summary, 'path': save_path})
    return notifications  # Return list of processed items


if __name__ == "__main__":  # For testing standalone
    results = scrape_sebi()
    print(f"Found {len(results)} relevant notifications.")
