#!/usr/bin/env python3
import os
import argparse
import colorama
from colorama import Fore, Style
from enhanced_crawler import EnhancedCrawler
from search_engine import EnhancedSearchEngine
from search_evaluator import SearchEvaluator

# Initialize colorama
colorama.init()

def print_banner():
    """Print a fancy banner for the application."""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║  {Fore.YELLOW}Enhanced Vector-Based Search Engine and Web Crawler{Fore.CYAN}      ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)

def crawl_data(args):
    """Run the crawler to collect data."""
    print(f"{Fore.GREEN}Starting web crawler...{Style.RESET_ALL}")
    
    # Create crawler
    crawler = EnhancedCrawler(
        output_dir=args.output_dir,
        max_docs=args.max_docs,
        max_depth=args.depth,
        workers=args.workers
    )
    
    # Reset if requested
    if args.reset:
        crawler.reset()
    
    # Start crawling
    print(f"{Fore.YELLOW}Crawling from seed URLs: {', '.join(args.urls)}{Style.RESET_ALL}")
    crawler.crawl(args.urls)
    
    print(f"{Fore.GREEN}Crawling complete! Data saved to {args.output_dir}{Style.RESET_ALL}")

def build_search_engine(args):
    """Build and run the search engine."""
    print(f"{Fore.GREEN}Building search engine...{Style.RESET_ALL}")
    
    # Create search engine
    search_engine = EnhancedSearchEngine(
        docs_dir=args.docs_dir,
        use_bm25=not args.no_bm25,
        use_positional=not args.no_positional,
        use_query_expansion=not args.no_expansion
    )
    
    # Load index if requested
    if args.load_index:
        if not search_engine.load_index():
            print("Building index from scratch instead...")
    
    # Save index if requested
    if args.save_index:
        search_engine.save_index()
    
    # Interactive search loop
    print(f"{Fore.YELLOW}Search engine ready! Enter queries below.{Style.RESET_ALL}")
    while True:
        query = input(f"\n{Fore.GREEN}Enter your search query (or 'exit' to quit): {Style.RESET_ALL}")
        
        if query.lower() == 'exit':
            break
        
        # Search for documents
        results = search_engine.search(query)
        
        # Display results
        query_terms = search_engine.preprocess_text(query, for_indexing=False)
        search_engine.display_results(results, query_terms)

def evaluate_search_engine(args):
    """Evaluate the search engine performance."""
    print(f"{Fore.GREEN}Evaluating search engine...{Style.RESET_ALL}")
    
    # Create search engine
    search_engine = EnhancedSearchEngine(
        docs_dir=args.docs_dir,
        use_bm25=not args.no_bm25,
        use_positional=not args.no_positional,
        use_query_expansion=not args.no_expansion
    )
    
    # Load index if requested
    if args.load_index:
        if not search_engine.load_index():
            print("Building index from scratch instead...")
    
    # Create evaluator
    evaluator = SearchEvaluator(search_engine)
    
    # Load or create evaluation data
    if args.eval_data and os.path.exists(args.eval_data):
        evaluator.load_evaluation_data(args.eval_data)
    else:
        print(f"{Fore.YELLOW}Creating new evaluation data...{Style.RESET_ALL}")
        evaluator.create_evaluation_data()
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Plot metrics
    evaluator.plot_metrics(save_path='evaluation_results.png')
    
    # Save index if requested
    if args.save_index:
        search_engine.save_index()
    
    print(f"{Fore.GREEN}Evaluation complete! Results saved to evaluation_results.png{Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Vector-Based Search Engine and Web Crawler')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Crawler parser
    crawler_parser = subparsers.add_parser('crawl', help='Crawl websites to collect data')
    crawler_parser.add_argument('urls', nargs='+', help='Seed URLs to start crawling')
    crawler_parser.add_argument('--output-dir', default='crawled_documents', help='Output directory for crawled data')
    crawler_parser.add_argument('--max-docs', type=int, default=500, help='Maximum number of documents to collect')
    crawler_parser.add_argument('--depth', type=int, default=3, help='Maximum crawl depth')
    crawler_parser.add_argument('--workers', type=int, default=10, help='Number of worker threads')
    crawler_parser.add_argument('--reset', action='store_true', help='Reset crawler state before starting')
    
    # Search engine parser
    search_parser = subparsers.add_parser('search', help='Run the search engine')
    search_parser.add_argument('--docs-dir', default='crawled_documents', help='Directory containing documents')
    search_parser.add_argument('--no-bm25', action='store_true', help='Disable BM25 scoring')
    search_parser.add_argument('--no-positional', action='store_true', help='Disable positional indexing')
    search_parser.add_argument('--no-expansion', action='store_true', help='Disable query expansion')
    search_parser.add_argument('--save-index', action='store_true', help='Save index to disk after building')
    search_parser.add_argument('--load-index', action='store_true', help='Load index from disk instead of building')
    
    # Evaluation parser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate search engine performance')
    eval_parser.add_argument('--docs-dir', default='crawled_documents', help='Directory containing documents')
    eval_parser.add_argument('--eval-data', help='Path to evaluation data JSON file')
    eval_parser.add_argument('--no-bm25', action='store_true', help='Disable BM25 scoring')
    eval_parser.add_argument('--no-positional', action='store_true', help='Disable positional indexing')
    eval_parser.add_argument('--no-expansion', action='store_true', help='Disable query expansion')
    eval_parser.add_argument('--save-index', action='store_true', help='Save index to disk after building')
    eval_parser.add_argument('--load-index', action='store_true', help='Load index from disk instead of building')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Execute command
    if args.command == 'crawl':
        crawl_data(args)
    elif args.command == 'search':
        build_search_engine(args)
    elif args.command == 'evaluate':
        evaluate_search_engine(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()