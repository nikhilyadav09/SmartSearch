import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from enhanced_search_engine import EnhancedSearchEngine
from tqdm import tqdm
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

class SearchEvaluator:
    def __init__(self, engine, eval_data=None):
        """Initialize the evaluator with a search engine and evaluation data."""
        self.engine = engine
        self.eval_data = eval_data or {}
        self.metrics = {}
    
    def load_evaluation_data(self, filepath):
        """Load evaluation data from a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.eval_data = json.load(f)
            print(f"Loaded evaluation data with {len(self.eval_data)} queries")
            return True
        except Exception as e:
            print(f"Error loading evaluation data: {e}")
            return False
    
    def create_evaluation_data(self, queries=None, output_file='evaluation_data.json'):
        """Create evaluation data through manual relevance judgments."""
        if queries is None:
            queries = []
            print(f"{Fore.CYAN}Enter queries for evaluation (empty line to finish):{Style.RESET_ALL}")
            while True:
                query = input("> ")
                if not query:
                    break
                queries.append(query)
        
        eval_data = {}
        
        for query in queries:
            print(f"\n{Fore.GREEN}Evaluating query: '{query}'{Style.RESET_ALL}")
            results = self.engine.search(query, top_n=20)
            
            if not results:
                print(f"{Fore.RED}No results found for this query.{Style.RESET_ALL}")
                continue
            
            relevant_docs = []
            
            # Display results and ask for relevance judgments
            query_terms = self.engine.preprocess_text(query, for_indexing=False)
            self.engine.display_results(results, query_terms)
            
            print(f"\n{Fore.YELLOW}Mark relevant documents (comma-separated doc numbers, e.g. '1,3,5'):{Style.RESET_ALL}")
            relevant_input = input("> ")
            
            if relevant_input:
                try:
                    relevant_indices = [int(idx.strip()) for idx in relevant_input.split(',')]
                    for idx in relevant_indices:
                        if 1 <= idx <= len(results):
                            doc_id = results[idx-1][0]  # Convert to 0-based index
                            relevant_docs.append(doc_id)
                except ValueError:
                    print(f"{Fore.RED}Invalid input. Using empty relevance judgments.{Style.RESET_ALL}")
            
            eval_data[query] = relevant_docs
        
        # Save evaluation data
        self.eval_data = eval_data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2)
        
        print(f"\n{Fore.GREEN}Evaluation data saved to {output_file}{Style.RESET_ALL}")
        return eval_data
    
    def evaluate(self):
        """Evaluate search engine performance using precision, recall, and MAP."""
        if not self.eval_data:
            print(f"{Fore.RED}No evaluation data available.{Style.RESET_ALL}")
            return {}
        
        precision_at_k = {1: [], 3: [], 5: [], 10: []}
        recall_at_k = {1: [], 3: [], 5: [], 10: []}
        average_precision = []
        
        for query, relevant_docs in tqdm(self.eval_data.items(), desc="Evaluating queries"):
            # Run search
            results = self.engine.search(query, top_n=20)
            retrieved_docs = [doc_id for doc_id, _ in results]
            
            # Calculate precision and recall at different K values
            for k in precision_at_k.keys():
                if len(retrieved_docs) >= k:
                    retrieved_at_k = retrieved_docs[:k]
                    relevant_at_k = set(retrieved_at_k).intersection(set(relevant_docs))
                    precision_at_k[k].append(len(relevant_at_k) / k)
                    recall_at_k[k].append(len(relevant_at_k) / len(relevant_docs) if relevant_docs else 0)
            
            # Calculate Average Precision (AP)
            ap = 0
            relevant_count = 0
            
            for i, doc_id in enumerate(retrieved_docs, 1):
                if doc_id in relevant_docs:
                    relevant_count += 1
                    precision_at_i = relevant_count / i
                    ap += precision_at_i
            
            # Normalize AP
            ap = ap / len(relevant_docs) if relevant_docs else 0
            average_precision.append(ap)
        
        # Calculate mean metrics
        mean_precision = {k: np.mean(v) for k, v in precision_at_k.items() if v}
        mean_recall = {k: np.mean(v) for k, v in recall_at_k.items() if v}
        mean_average_precision = np.mean(average_precision) if average_precision else 0
        
        # Store metrics
        self.metrics = {
            'precision_at_k': mean_precision,
            'recall_at_k': mean_recall,
            'map': mean_average_precision
        }
        
        # Print results
        print(f"\n{Fore.GREEN}Evaluation Results:{Style.RESET_ALL}")
        print(f"  Mean Average Precision (MAP): {mean_average_precision:.4f}")
        
        print("\n  Precision at K:")
        for k, value in mean_precision.items():
            print(f"    P@{k}: {value:.4f}")
        
        print("\n  Recall at K:")
        for k, value in mean_recall.items():
            print(f"    R@{k}: {value:.4f}")
        
        return self.metrics
    
    def plot_metrics(self, save_path=None):
        """Plot evaluation metrics."""
        if not self.metrics:
            print(f"{Fore.RED}No metrics available. Run evaluate() first.{Style.RESET_ALL}")
            return
        
        # Create precision-recall curve
        k_values = sorted(self.metrics['precision_at_k'].keys())
        precision_values = [self.metrics['precision_at_k'][k] for k in k_values]
        recall_values = [self.metrics['recall_at_k'][k] for k in k_values]
        
        plt.figure(figsize=(12, 5))
        
        # Precision@K plot
        plt.subplot(1, 2, 1)
        plt.plot(k_values, precision_values, 'b-o', linewidth=2)
        plt.title('Precision at K')
        plt.xlabel('K')
        plt.ylabel('Precision')
        plt.grid(True)
        
        # Precision-Recall curve
        plt.subplot(1, 2, 2)
        plt.plot(recall_values, precision_values, 'r-o', linewidth=2)
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"{Fore.GREEN}Metrics plot saved to {save_path}{Style.RESET_ALL}")
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Search Engine Evaluator')
    parser.add_argument('--docs-dir', default='crawled_documents', help='Directory containing documents')
    parser.add_argument('--eval-data', help='Path to evaluation data JSON file')
    parser.add_argument('--create-eval', action='store_true', help='Create new evaluation data')
    parser.add_argument('--output', default='evaluation_results.json', help='Output file for evaluation results')
    parser.add_argument('--plot', action='store_true', help='Plot evaluation metrics')
    parser.add_argument('--plot-file', default='evaluation_plot.png', help='File to save the plot')
    
    args = parser.parse_args()
    
    # Create search engine
    engine = EnhancedSearchEngine(
        docs_dir=args.docs_dir,
        use_bm25=True,
        use_positional=True,
        use_query_expansion=True
    )
    
    # Create evaluator
    evaluator = SearchEvaluator(engine)
    
    # Load or create evaluation data
    if args.create_eval:
        evaluator.create_evaluation_data()
    elif args.eval_data:
        evaluator.load_evaluation_data(args.eval_data)
    else:
        print(f"{Fore.YELLOW}No evaluation data specified. Creating new evaluation data...{Style.RESET_ALL}")
        evaluator.create_evaluation_data()
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n{Fore.GREEN}Evaluation results saved to {args.output}{Style.RESET_ALL}")
    
    # Plot metrics if requested
    if args.plot:
        evaluator.plot_metrics(save_path=args.plot_file)

if __name__ == "__main__":
    main()