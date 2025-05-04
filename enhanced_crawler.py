import requests
import re
import time
import os
import sys
import json
import hashlib
from bs4 import BeautifulSoup
from collections import deque
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

class EnhancedCrawler:
    def __init__(self, output_dir="crawled_documents", max_docs=500, max_depth=3, workers=10):
        """Initialize the crawler with configuration parameters."""
        self.output_dir = output_dir
        self.max_docs = max_docs
        self.max_depth = max_depth
        self.workers = workers
        self.visited = set()
        self.doc_count = 0
        self.crawl_state_file = os.path.join(output_dir, "crawl_state.json")
        self.metadata_file = os.path.join(output_dir, "metadata.json")
        self.domain_counts = {}
        self.metadata = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load previous state if exists
        self._load_state()
    
    def _load_state(self):
        """Load previous crawling state if available."""
        if os.path.exists(self.crawl_state_file):
            try:
                with open(self.crawl_state_file, 'r') as f:
                    state = json.load(f)
                    self.visited = set(state.get('visited', []))
                    self.doc_count = state.get('doc_count', 0)
                    self.domain_counts = state.get('domain_counts', {})
                    print(f"Resuming crawl from previous state with {self.doc_count} documents already collected")
            except Exception as e:
                print(f"Error loading previous state: {e}")
        
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")
    
    def _save_state(self):
        """Save current crawling state."""
        state = {
            'visited': list(self.visited),
            'doc_count': self.doc_count,
            'domain_counts': self.domain_counts
        }
        
        with open(self.crawl_state_file, 'w') as f:
            json.dump(state, f)
    
    def _save_metadata(self):
        """Save document metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
    
    def is_same_domain(self, base_url, target_url):
        """Check if a link belongs to the same domain."""
        return urlparse(base_url).netloc == urlparse(target_url).netloc
    
    def is_valid_link(self, link, base_url, root_domain):
        """Validate if a link is worth following."""
        if not link.startswith('http'):
            return False
            
        # Skip files that are not HTML
        if link.lower().endswith(('pdf', 'jpg', 'jpeg', 'png', 'gif', 'css', 'js', 'xml', 'ico', 'zip', 'exe')):
            return False
            
        # Stay within the same domain
        target_domain = urlparse(link).netloc
        
        # Limit per domain to ensure diversity
        if target_domain in self.domain_counts and self.domain_counts[target_domain] >= 50:
            return False
            
        return self.is_same_domain(base_url, link) or target_domain == root_domain
    
    def clean_text(self, html_content, url):
        """Extract and clean text content from HTML using BeautifulSoup."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'head', 'header', 'footer', 'nav', 'aside']):
                element.extract()
                
            # Get page title
            title = soup.title.string if soup.title else url
                
            # Extract text from paragraphs, headings, and other content elements
            content_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'article', 'section', 'div.content', 'div.main'])
            
            # If no specific content tags are found, use the body
            if not content_tags:
                content_tags = [soup.body] if soup.body else []
                
            # Extract text from each content element
            text_parts = []
            for tag in content_tags:
                if tag.text.strip():
                    text_parts.append(tag.text.strip())
                    
            # Join all text parts
            text = '\n\n'.join(text_parts)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return title, text
            
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return url, ""
    
    def fetch_page(self, url):
        """Fetch HTML content of a given URL with proper error handling."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            
            response = requests.get(url, timeout=10, headers=headers)
            
            # Check if the response is HTML
            content_type = response.headers.get('Content-Type', '').lower()
            if response.status_code == 200 and ('text/html' in content_type or 'application/xhtml+xml' in content_type):
                return response.text
                
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            
        return None
    
    def extract_links(self, html, current_url, root_domain):
        """Extract valid links from a page."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            links = []
            
            for a_tag in soup.find_all('a', href=True):
                link = a_tag['href']
                full_url = urljoin(current_url, link)
                
                if self.is_valid_link(full_url, current_url, root_domain):
                    links.append(full_url)
                    
            return links
            
        except Exception as e:
            print(f"Error extracting links from {current_url}: {e}")
            return []
    
    def process_page(self, url, depth, root_domain):
        """Process a page: fetch, extract text, and identify links."""
        if url in self.visited or self.doc_count >= self.max_docs:
            return []
            
        self.visited.add(url)
        
        # Fetch page content
        html_content = self.fetch_page(url)
        if not html_content:
            return []
            
        # Extract and clean text
        title, text = self.clean_text(html_content, url)
        
        # Only save if content is substantial
        if len(text) > 500:
            # Update domain counts
            domain = urlparse(url).netloc
            self.domain_counts[domain] = self.domain_counts.get(domain, 0) + 1
            
            # Generate unique file name based on URL
            url_hash = hashlib.md5(url.encode()).hexdigest()
            doc_id = f"doc_{self.doc_count + 1:03d}_{url_hash[:8]}"
            file_name = f"{doc_id}.txt"
            
            # Save document
            with open(os.path.join(self.output_dir, file_name), 'w', encoding='utf-8') as f:
                f.write(text)
                
            # Save metadata
            self.metadata[doc_id] = {
                'url': url,
                'title': title,
                'crawled_at': datetime.now().isoformat(),
                'length': len(text),
                'depth': depth
            }
            
            # Increment document count
            self.doc_count += 1
            
            # Save state periodically
            if self.doc_count % 10 == 0:
                self._save_state()
                self._save_metadata()
                
        # Extract links for further crawling if not at max depth
        if depth < self.max_depth:
            return self.extract_links(html_content, url, root_domain)
            
        return []
    
    def crawl(self, seed_urls):
        """Main crawling method using multi-threading."""
        start_time = time.time()
        
        # Initialize queue with seed URLs
        queue = deque([(url, 0, urlparse(url).netloc) for url in seed_urls])
        
        # Create progress bar
        pbar = tqdm(total=self.max_docs, initial=self.doc_count, desc="Crawling")
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {}
            
            while queue and self.doc_count < self.max_docs:
                # Submit jobs to the thread pool
                while len(futures) < self.workers and queue:
                    url, depth, root_domain = queue.popleft()
                    if url not in self.visited:
                        future = executor.submit(self.process_page, url, depth, root_domain)
                        futures[future] = (url, depth, root_domain)
                        
                # Process completed jobs
                for future in as_completed(list(futures.keys())):
                    url, depth, root_domain = futures[future]
                    del futures[future]
                    
                    try:
                        new_links = future.result()
                        # Add new links to the queue
                        for link in new_links:
                            if link not in self.visited:
                                queue.append((link, depth + 1, root_domain))
                                
                        # Update progress bar
                        pbar.n = self.doc_count
                        pbar.refresh()
                        
                    except Exception as e:
                        print(f"Error processing {url}: {e}")
                        
        pbar.close()
        
        # Save final state
        self._save_state()
        self._save_metadata()
        
        print(f"Crawling completed in {time.time() - start_time:.2f} seconds")
        print(f"Collected {self.doc_count} documents from {len(self.domain_counts)} domains")
        
    def reset(self):
        """Reset the crawler state to start fresh."""
        self.visited = set()
        self.doc_count = 0
        self.domain_counts = {}
        self.metadata = {}
        
        # Remove state files
        if os.path.exists(self.crawl_state_file):
            os.remove(self.crawl_state_file)
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
            
        print("Crawler state has been reset")

# Entry point of the script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Web Crawler')
    parser.add_argument('urls', nargs='+', help='Seed URLs to start crawling')
    parser.add_argument('--max-docs', type=int, default=500, help='Maximum number of documents to collect')
    parser.add_argument('--max-depth', type=int, default=3, help='Maximum crawling depth')
    parser.add_argument('--workers', type=int, default=10, help='Number of worker threads')
    parser.add_argument('--output', default='crawled_documents', help='Output directory')
    parser.add_argument('--reset', action='store_true', help='Reset crawler state before starting')
    
    args = parser.parse_args()
    
    crawler = EnhancedCrawler(
        output_dir=args.output,
        max_docs=args.max_docs,
        max_depth=args.max_depth,
        workers=args.workers
    )
    
    if args.reset:
        crawler.reset()
        
    crawler.crawl(args.urls)