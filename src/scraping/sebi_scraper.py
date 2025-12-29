import requests  # For fetching page
from bs4 import BeautifulSoup  # For parsing HTML
from .utils import download_pdf, check_relevancy_with_ai, config

def scrape_sebi():  # Main function
    """
    Scrapes SEBI circulars, filters relevant, downloads, and processes.
    Why: Encapsulates logic; callable from main.py.
    """
    url = config['sebi_url']  # From config
    try:
        response = requests.get(url)  # Fetch page
        response.raise_for_status()  # Check status
    except requests.exceptions.RequestException as e:
        print(f"Error fetching SEBI: {e}")
        return []  # Return empty on failure

    soup = BeautifulSoup(response.text, 'html.parser')  # Parse HTML

    notifications = []  # List to store relevant ones
    # Find all h3 (dates) and following content
    for date_header in soup.find_all('h3'):  # Dates are h3
        date = date_header.text.strip()  # e.g., "Dec 26, 2025"
        # Find next siblings until next h3
        current = date_header.next_sibling  # Start after header
        while current and current.name != 'h3':  # Loop until next date
            if current.name == 'h4':  # Category, e.g., "Circulars"
                category = current.text.strip()
            elif current.name == 'ul':  # List of items
                for li in current.find_all('li'):  # Each bullet
                    a = li.find('a')  # Link tag
                    if a:
                        title = a.text.strip()  # Title
                        link = a['href']  # Relative link
                        full_link = "https://www.sebi.gov.in" + link if link.startswith('/') else link
                        # Fetch linked page to check for PDF
                        is_relevant, summary = check_relevancy_with_ai(title)  # AI check on title first (quick)
                        if is_relevant:
                            # Fetch linked page for full text/PDF
                            link_resp = requests.get(full_link)
                            link_soup = BeautifulSoup(link_resp.text, 'html.parser')
                            pdf_a = link_soup.find('a', text=lambda t: 'PDF' in t if t else False)  # Find PDF link
                            if pdf_a:
                                pdf_url = pdf_a['href']
                                save_path = f"data/raw/sebi/{title[:50].replace(' ', '_')}.pdf"  # Sanitize filename
                                if download_pdf(pdf_url, save_path):  # Download if relevant
                                    notifications.append({'date': date, 'title': title, 'summary': summary, 'path': save_path})
            current = current.next_sibling  # Move to next element
    return notifications  # Return list of processed items

if __name__ == "__main__":  # For testing standalone
    results = scrape_sebi()
    print(f"Found {len(results)} relevant notifications.")