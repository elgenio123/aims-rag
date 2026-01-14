"""Web crawler for AIMS Cameroon with robots.txt respect and rate limiting."""
from typing import List, Set, Tuple, Optional
from urllib.parse import urljoin, urlparse
import time
import requests
from bs4 import BeautifulSoup
from loguru import logger
import re
import io
from pypdf import PdfReader
from config import SCRAPE_DELAY_SECONDS, MAX_DEPTH, RESPECT_ROBOTS_TXT, USER_AGENT
from src.utils import clean_text, remove_duplicates, categorize_content
from src.storage.document import Document
from src.storage.storage import DocumentStorage

try:
    from urllib.robotparser import RobotFileParser
except Exception:
    RobotFileParser = None

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

class Crawler:
    def __init__(self, start_urls: List[str], max_depth: int = MAX_DEPTH, delay: float = SCRAPE_DELAY_SECONDS):
        self.start_urls = start_urls
        self.max_depth = max_depth
        self.delay = delay
        self.visited: Set[str] = set()
        self.allowed_domains = {urlparse(u).netloc for u in start_urls}
        self.storage = DocumentStorage()
        self.robots_cache = {}

    def _get_robots(self, base_url: str):
        if not RESPECT_ROBOTS_TXT or RobotFileParser is None:
            return None
        domain = urlparse(base_url).netloc
        if domain in self.robots_cache:
            return self.robots_cache[domain]
        robots_url = f"{urlparse(base_url).scheme}://{domain}/robots.txt"
        rp = RobotFileParser()
        try:
            rp.set_url(robots_url)
            rp.read()
            self.robots_cache[domain] = rp
            return rp
        except Exception:
            logger.warning(f"Failed to read robots.txt: {robots_url}")
            self.robots_cache[domain] = None
            return None

    def _allowed(self, url: str) -> bool:
        if not RESPECT_ROBOTS_TXT or RobotFileParser is None:
            return True
        parsed = urlparse(url)
        rp = self._get_robots(url)
        if rp is None:
            return True
        return rp.can_fetch(USER_AGENT, url)

    def _is_pdf(self, url: str, headers: Optional[dict]) -> bool:
        if headers and 'content-type' in headers:
            ct = headers['content-type']
            if 'pdf' in ct:
                return True
        return url.lower().endswith('.pdf')

    def _fetch(self, url: str) -> Tuple[Optional[str], Optional[requests.Response]]:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                return resp.text, resp
            logger.warning(f"Non-200 status {resp.status_code} for {url}")
            return None, None
        except requests.RequestException as e:
            logger.error(f"Request error for {url}: {e}")
            return None, None

    def _parse_pdf(self, content: bytes) -> str:
        try:
            reader = PdfReader(io.BytesIO(content))
            texts = []
            for page in reader.pages:
                texts.append(page.extract_text() or "")
            return "\n\n".join(texts)
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            return ""

    def _extract_title(self, soup: BeautifulSoup) -> str:
        title = ""
        if soup.title and soup.title.text:
            title = soup.title.text.strip()
        h1 = soup.find('h1')
        if h1 and h1.text and not title:
            title = h1.text.strip()
        return title or "Untitled"

    def _clean_soup(self, soup: BeautifulSoup):
        # Remove scripts/styles/nav/headers/footers
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
            tag.decompose()
        # Remove cookie banners by common ids/classes
        selectors = [
            {'id': re.compile(r'cookie', re.I)},
            {'class': re.compile(r'cookie', re.I)},
            {'id': re.compile(r'consent', re.I)},
            {'class': re.compile(r'consent', re.I)},
        ]
        for sel in selectors:
            for node in soup.find_all(attrs=sel):
                node.decompose()
        return soup

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('#'):
                continue
            full = urljoin(base_url, href)
            parsed = urlparse(full)
            if parsed.netloc in self.allowed_domains:
                links.append(full)
        # Deduplicate
        return list(dict.fromkeys(links))

    def _html_to_text(self, soup: BeautifulSoup) -> str:
        # Preserve headings and paragraph structure
        texts = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
            txt = tag.get_text(separator=' ', strip=True)
            if not txt:
                continue
            # Add newlines after headings for hierarchy
            if tag.name.startswith('h'):
                texts.append(f"{txt}\n")
            else:
                texts.append(txt)
        return "\n\n".join(texts)

    def crawl(self) -> List[Document]:
        queue: List[Tuple[str, int]] = [(url, 0) for url in self.start_urls]
        documents: List[Document] = []
        seen_urls: Set[str] = set()
        logger.info(f"Starting crawl with {len(queue)} start URLs")

        while queue:
            url, depth = queue.pop(0)
            if depth > self.max_depth:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)

            if not self._allowed(url):
                logger.info(f"Skipping disallowed by robots.txt: {url}")
                continue

            # Polite delay
            time.sleep(self.delay)

            html_text, resp = self._fetch(url)
            if not resp:
                logger.warning(f"Skipping due to fetch failure: {url}")
                continue

            is_pdf = self._is_pdf(url, resp.headers)
            raw_text = ""
            title = ""

            if is_pdf:
                raw_text = self._parse_pdf(resp.content)
                title = url.split('/')[-1]
            else:
                soup = BeautifulSoup(html_text, 'lxml')

                # Extract links BEFORE cleaning so we include navigation/header/footer links
                outgoing_links = self._extract_links(soup, url)

                # Now clean soup to strip menus/headers/footers from the stored content
                soup = self._clean_soup(soup)
                title = self._extract_title(soup)
                raw_text = self._html_to_text(soup)

                # Enqueue discovered links from same domain
                for link in outgoing_links:
                    if link not in seen_urls:
                        queue.append((link, depth + 1))

            # Clean and deduplicate text
            cleaned = clean_text(raw_text)
            cleaned = remove_duplicates(cleaned)
            category = categorize_content(url, title, cleaned)

            # Skip tiny pages
            if len(cleaned) < 200:
                logger.info(f"Skipping tiny content page: {url}")
                continue

            # Build and store document
            doc_id = re.sub(r'[^a-zA-Z0-9]+', '_', url)[:100]
            document = Document(
                doc_id=doc_id,
                title=title,
                source_url=url,
                category=category,
                raw_text=cleaned,
                scrape_timestamp=Document.now_iso(),
            )
            try:
                document.validate_category()
            except Exception:
                document.category = "Institutional"
            self.storage.save_document(document)
            documents.append(document)
            logger.info(f"Stored document for {url} with category {document.category}")

        logger.info(f"Crawl complete. Stored {len(documents)} documents.")
        return documents
