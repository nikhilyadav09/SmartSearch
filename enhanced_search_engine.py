import os
import re
import json
import math
import string
import numpy as np
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import nltk
import heapq
from tqdm import tqdm
import argparse
import colorama
from colorama import Fore, Style
import concurrent.futures
import functools

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Define this function at the module level so it can be pickled for multiprocessing
def process_document_weights(doc_id, term_freq_doc, preprocessed_tokens, idf_values, max_freq, avgdl, k1, b, use_bm25):
    """Process a single document to calculate weights."""
    doc_vector_length = 0
    bm25_scores_doc = {}
    
    # Get length of document in tokens
    dl = len(preprocessed_tokens)
    
    for term, term_freq in term_freq_doc.items():
        # Get IDF value
        idf = idf_values.get(term, 0)
        
        # TF-IDF calculation
        tf = term_freq / max_freq
        tf_idf = tf * idf
        
        # Add squared weight to document vector length
        doc_vector_length += tf_idf ** 2
        
        # Calculate BM25 score for this term if enabled
        if use_bm25:
            numerator = term_freq * (k1 + 1)
            denominator = term_freq + k1 * (1 - b + b * dl / avgdl)
            bm25_score = idf * (numerator / denominator) if denominator > 0 else 0
            bm25_scores_doc[term] = bm25_score
    
    # Calculate vector length
    doc_length = math.sqrt(doc_vector_length)
    
    return doc_id, doc_length, bm25_scores_doc

class EnhancedSearchEngine:
    def __init__(self, docs_dir, use_bm25=True, use_positional=True, use_query_expansion=True):
        """Initialize the search engine with documents from a directory."""
        self.docs_dir = docs_dir
        self.documents = {}         # Document content keyed by doc_id
        self.doc_ids = []           # List of document IDs
        self.doc_metadata = {}      # Metadata for each document
        self.inverted_index = defaultdict(list)  # Term -> [(doc_id, term_freq, positions), ...]
        self.term_freq = defaultdict(Counter)    # doc_id -> {term: frequency, ...}
        self.doc_lengths = {}       # doc_id -> vector length (for cosine similarity)
        self.bm25_scores = {}       # doc_id -> BM25 scores for each term
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.avgdl = 0              # Average document length for BM25
        self.use_bm25 = use_bm25
        self.use_positional = use_positional
        self.use_query_expansion = use_query_expansion
        self.preprocessed_docs = {} # Preprocessed document tokens
        
        # BM25 parameters
        self.k1 = 1.2
        self.b = 0.75
        
        # Load and process all documents
        self.load_documents()
        self.build_index()
        
    def load_documents(self):
        """Load documents and their metadata from the specified directory."""
        print("Loading documents...")
        
        # Load metadata if available
        metadata_file = os.path.join(self.docs_dir, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.doc_metadata = json.load(f)
                print(f"Loaded metadata for {len(self.doc_metadata)} documents")
            except Exception as e:
                print(f"Error loading metadata: {e}")
        
        # Load document content
        for filename in tqdm(os.listdir(self.docs_dir), desc="Loading documents"):
            if filename.endswith('.txt') and filename != "metadata.json":
                doc_id = filename.split('.')[0]  # Use filename without extension as doc_id
                self.doc_ids.append(doc_id)
                
                with open(os.path.join(self.docs_dir, filename), 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.documents[doc_id] = content
        
        print(f"Loaded {len(self.documents)} documents")
    
    def preprocess_text(self, text, for_indexing=True):
        """Clean and tokenize text, removing stopwords and applying stemming."""
        # Convert to lowercase
        text = text.lower()
        
        # Use NLTK's word_tokenize for better tokenization
        tokens = word_tokenize(text)
        
        # Remove punctuation and filter tokens
        processed_tokens = []
        for token in tokens:
            # Remove punctuation
            token = re.sub(f'[{re.escape(string.punctuation)}]', '', token)
            
            # Apply filters
            if token and len(token) > 1:
                if for_indexing:
                    # For indexing, remove stopwords and apply stemming
                    if token not in self.stop_words:
                        processed_tokens.append(self.stemmer.stem(token))
                else:
                    # For query processing, keep stopwords for phrase queries
                    processed_tokens.append(self.stemmer.stem(token))
        
        return processed_tokens
    
    def build_index(self):
        """Build the inverted index with positional information and calculate term frequencies."""
        print("Building index...")
        
        total_tokens = 0
        doc_lengths_tokens = {}
        
        # First preprocess all documents once
        print("Preprocessing documents...")
        for doc_id in tqdm(self.doc_ids, desc="Preprocessing"):
            content = self.documents[doc_id]
            tokens = self.preprocess_text(content)
            self.preprocessed_docs[doc_id] = tokens
            doc_lengths_tokens[doc_id] = len(tokens)
            total_tokens += len(tokens)
        
        # Calculate average document length for BM25
        self.avgdl = total_tokens / len(self.documents) if self.documents else 0
        print(f"Average document length: {self.avgdl:.2f} tokens")
        
        # Now build the index
        for doc_id in tqdm(self.doc_ids, desc="Building index"):
            tokens = self.preprocessed_docs[doc_id]
            
            # Count term frequencies in this document
            term_counts = Counter(tokens)
            self.term_freq[doc_id] = term_counts
            
            # Update inverted index with positions
            if self.use_positional:
                positions = defaultdict(list)
                for pos, term in enumerate(tokens):
                    positions[term].append(pos)
                
                for term, positions_list in positions.items():
                    self.inverted_index[term].append((doc_id, len(positions_list), positions_list))
            else:
                # Without positional information
                for term, count in term_counts.items():
                    self.inverted_index[term].append((doc_id, count, []))
        
        print(f"Index built with {len(self.inverted_index)} unique terms")
        
        # Calculate weights for vector space model and BM25
        self.calculate_weights()
    
    def calculate_weights(self):
        """Calculate TF-IDF weights and BM25 scores more efficiently."""
        print("Calculating weights...")
        N = len(self.documents)  # Total number of documents
        
        # Precompute maximum term frequencies for each document
        print("Precomputing max frequencies...")
        max_freqs = {doc_id: max(term_freqs.values()) if term_freqs else 1 
                    for doc_id, term_freqs in self.term_freq.items()}
        
        # Precompute IDF values
        print("Calculating IDF values...")
        idf_values = {}
        for term in tqdm(self.inverted_index.keys(), desc="IDF values"):
            doc_freq = len(self.inverted_index.get(term, []))
            idf_values[term] = math.log(N / doc_freq) if doc_freq > 0 else 0
        
        # Use multi-processing for weights calculation
        print("Calculating document weights...")
        self.calculate_weights_parallel(idf_values, max_freqs)
    
    def calculate_weights_parallel(self, idf_values, max_freqs):
        """Calculate weights using parallel processing."""
        # Process documents in serial instead of parallel to avoid pickling errors
        for doc_id in tqdm(self.doc_ids, desc="Processing documents"):
            # Get document info
            term_freq_doc = self.term_freq[doc_id]
            preprocessed_tokens = self.preprocessed_docs[doc_id]
            max_freq = max_freqs[doc_id]
            
            # Process the document
            doc_id, doc_length, bm25_scores_doc = process_document_weights(
                doc_id, term_freq_doc, preprocessed_tokens, 
                idf_values, max_freq, self.avgdl, self.k1, self.b, self.use_bm25
            )
            
            # Store the results
            self.doc_lengths[doc_id] = doc_length
            if self.use_bm25:
                self.bm25_scores[doc_id] = bm25_scores_doc
    
    def expand_query(self, query_terms, top_n_docs=3, top_n_terms=5):
        """Expand the query with relevant terms from top documents."""
        if not self.use_query_expansion:
            return query_terms
            
        # First get top documents for the original query
        original_results = self._search_internal(query_terms, top_n=top_n_docs, for_expansion=True)
        
        # Collect terms from these documents
        expansion_terms = Counter()
        for doc_id, _ in original_results:
            # Get terms from this document
            for term, freq in self.term_freq[doc_id].items():
                # Skip terms already in the query
                if term not in query_terms:
                    expansion_terms[term] += freq
        
        # Select top terms to add
        top_terms = [term for term, _ in expansion_terms.most_common(top_n_terms)]
        
        # Add expansion terms to the query
        expanded_query = query_terms + top_terms
        return expanded_query
    
    def _search_internal(self, query_terms, top_n=10, for_expansion=False):
        """Internal search function used by main search and query expansion."""
        if not query_terms:
            return []
        
        N = len(self.documents)
        query_term_freq = Counter(query_terms)
        
        # For vector space model
        query_length = 0
        for term, term_freq in query_term_freq.items():
            # Calculate TF part
            tf = term_freq / max(query_term_freq.values())
            
            # Calculate IDF part
            doc_freq = len(self.inverted_index.get(term, [])) 
            idf = math.log(N / (doc_freq or 1))
            
            # TF-IDF weight
            tf_idf = tf * idf
            
            # Add squared weight to query vector length
            query_length += tf_idf ** 2
        
        query_length = math.sqrt(query_length)
        
        # Calculate query term positions for phrase matching
        query_term_positions = {}
        if self.use_positional and not for_expansion:
            for i, term in enumerate(query_terms):
                if term not in query_term_positions:
                    query_term_positions[term] = []
                query_term_positions[term].append(i)
        
        # Calculate scores
        scores = {}
        
        # Get candidate documents (docs containing at least one query term)
        candidate_docs = set()
        for term in query_terms:
            for doc_id, _, _ in self.inverted_index.get(term, []):
                candidate_docs.add(doc_id)
        
        for doc_id in candidate_docs:
            # Vector space score (cosine similarity)
            dot_product = 0
            
            # BM25 score
            bm25_score = 0
            
            # Proximity score
            proximity_boost = 0
            
            # Calculate combined score
            for term, term_freq in query_term_freq.items():
                # Skip terms not in the index
                if term not in self.inverted_index:
                    continue
                
                # Skip if term not in this document
                if term not in self.term_freq[doc_id]:
                    continue
                
                # Calculate TF-IDF for query term
                query_tf = term_freq / max(query_term_freq.values())
                doc_freq = len(self.inverted_index[term])
                idf = math.log(N / doc_freq) if doc_freq > 0 else 0
                query_tf_idf = query_tf * idf
                
                # Calculate TF-IDF for document term
                doc_tf = self.term_freq[doc_id][term] / max(self.term_freq[doc_id].values())
                doc_tf_idf = doc_tf * idf
                
                # Add to dot product (vector space model)
                dot_product += query_tf_idf * doc_tf_idf
                
                # Add BM25 score if enabled
                if self.use_bm25:
                    bm25_score += self.bm25_scores[doc_id].get(term, 0) * term_freq
            
            # Calculate cosine similarity
            if query_length > 0 and self.doc_lengths[doc_id] > 0:
                cosine_similarity = dot_product / (query_length * self.doc_lengths[doc_id])
            else:
                cosine_similarity = 0
            
            # Phrase matching boost
            if self.use_positional and len(query_terms) > 1 and not for_expansion:
                # Check for exact phrase matches
                phrase_matches = self._check_phrase_matches(query_terms, doc_id)
                if phrase_matches > 0:
                    proximity_boost = phrase_matches * 0.2  # Boost for each phrase match
            
            # Combine scores
            if self.use_bm25:
                combined_score = (0.4 * cosine_similarity) + (0.6 * bm25_score) + proximity_boost
            else:
                combined_score = cosine_similarity + proximity_boost
                
            scores[doc_id] = combined_score
        
        # Sort results by score
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N results
        return results[:top_n]
    
    def _check_phrase_matches(self, query_terms, doc_id):
        """Check for exact phrase matches in a document."""
        # Get positions for all query terms in this document
        term_positions = {}
        for term in set(query_terms):
            for t_doc_id, _, positions in self.inverted_index.get(term, []):
                if t_doc_id == doc_id:
                    term_positions[term] = positions
                    break
        
        # If any term is missing, no phrase match is possible
        if len(term_positions) < len(set(query_terms)):
            return 0
        
        # Check for consecutive positions that would indicate a phrase match
        phrase_matches = 0
        query_len = len(query_terms)
        
        # Get positions for the first term
        first_term = query_terms[0]
        if first_term not in term_positions:
            return 0
            
        for start_pos in term_positions[first_term]:
            match = True
            for i in range(1, query_len):
                term = query_terms[i]
                if term not in term_positions or start_pos + i not in term_positions[term]:
                    match = False
                    break
            
            if match:
                phrase_matches += 1
                
        return phrase_matches
    
    def search(self, query, top_n=10):
        """Search for documents matching the query."""
        print(f"\nSearching for: '{query}'")
        
        # Preprocess query
        query_terms = self.preprocess_text(query, for_indexing=False)
        
        if not query_terms:
            return []
        
        # Expand query if enabled
        if self.use_query_expansion:
            expanded_terms = self.expand_query(query_terms)
            if len(expanded_terms) > len(query_terms):
                print(f"Expanded query with terms: {', '.join(expanded_terms[len(query_terms):])}")
                query_terms = expanded_terms
        
        # Perform the search
        results = self._search_internal(query_terms, top_n=top_n)
        
        return results
    
    def generate_snippet(self, doc_id, query_terms, window_size=50):
        """Generate a snippet with highlighted query terms."""
        content = self.documents[doc_id]
        
        # If content is very short, return it all
        if len(content) < 300:
            return self._highlight_terms(content, query_terms)
        
        # Find best window containing query terms
        tokens = word_tokenize(content.lower())
        best_window = (0, 0)
        max_query_terms = 0
        
        # Find all occurrences of query terms
        term_positions = []
        for i, token in enumerate(tokens):
            stemmed = self.stemmer.stem(re.sub(f'[{re.escape(string.punctuation)}]', '', token))
            if stemmed in query_terms:
                term_positions.append(i)
        
        if not term_positions:
            # No query terms found, return beginning of document
            return content[:300] + "..."
        
        # Find best window with most query terms
        for start_pos in term_positions:
            end_pos = start_pos
            window_terms = set()
            
            # Expand window to include nearby terms
            for pos in range(start_pos, min(len(tokens), start_pos + window_size)):
                stemmed = self.stemmer.stem(re.sub(f'[{re.escape(string.punctuation)}]', '', tokens[pos].lower()))
                if stemmed in query_terms:
                    window_terms.add(stemmed)
                    end_pos = pos
            
            if len(window_terms) > max_query_terms:
                max_query_terms = len(window_terms)
                best_window = (max(0, start_pos - 5), min(len(tokens), end_pos + 5))
        
        # Extract the snippet text
        start_idx = max(0, sum(len(tokens[i]) + 1 for i in range(best_window[0])))
        try:
            end_idx = min(len(content), sum(len(tokens[i]) + 1 for i in range(best_window[1] + 1)))
        except:
            end_idx = min(300, len(content))  # Fallback if calculation fails
        
        snippet = content[start_idx:end_idx]
        
        # Add ellipsis if needed
        if start_idx > 0:
            snippet = "..." + snippet
        if end_idx < len(content):
            snippet = snippet + "..."
            
        # Highlight query terms
        return self._highlight_terms(snippet, query_terms)
    
    def _highlight_terms(self, text, query_terms):
        """Highlight query terms in the text."""
        highlighted = text
        
        # Create regex pattern to match stemmed terms
        for term in query_terms:
            # Find word stem matches
            pattern = re.compile(r'\b(\w*' + re.escape(term) + r'\w*)\b', re.IGNORECASE)
            highlighted = pattern.sub(Fore.YELLOW + r'\1' + Style.RESET_ALL, highlighted)
            
        return highlighted
    
    def display_results(self, results, query_terms):
        """Display search results with rich snippets."""
        if not results:
            print(f"{Fore.RED}No matching documents found.{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.GREEN}Found {len(results)} matching documents:{Style.RESET_ALL}")
        print("=" * 80)
        
        for i, (doc_id, score) in enumerate(results, 1):
            # Get document metadata
            metadata = self.doc_metadata.get(doc_id, {})
            title = metadata.get('title', doc_id)
            url = metadata.get('url', 'Unknown URL')
            
            # Generate snippet with highlighted terms
            snippet = self.generate_snippet(doc_id, query_terms)
            
            # Display result
            print(f"{Fore.CYAN}{i}. {title}{Style.RESET_ALL}")
            print(f"   {Fore.BLUE}Document ID: {doc_id}{Style.RESET_ALL}")
            if url != 'Unknown URL':
                print(f"   {Fore.BLUE}URL: {url}{Style.RESET_ALL}")
            print(f"   {Fore.MAGENTA}Score: {score:.4f}{Style.RESET_ALL}")
            print(f"   {Fore.WHITE}Snippet: {snippet}{Style.RESET_ALL}")
            print("-" * 80)
    
    def evaluate_search(self, query_set, relevance_judgments):
        """Evaluate search performance using precision, recall, and MAP."""
        if not relevance_judgments:
            print("No relevance judgments provided for evaluation")
            return
            
        precision_sum = 0
        recall_sum = 0
        map_sum = 0
        
        for query, relevant_docs in query_set.items():
            # Run search
            results = self.search(query, top_n=20)
            retrieved_docs = [doc_id for doc_id, _ in results]
            
            # Calculate precision and recall
            relevant_retrieved = set(retrieved_docs).intersection(set(relevant_docs))
            precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
            recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
            
            precision_sum += precision
            recall_sum += recall
            
            # Calculate Average Precision (AP)
            ap = 0
            relevant_count = 0
            
            for i, doc_id in enumerate(retrieved_docs, 1):
                if doc_id in relevant_docs:
                    relevant_count += 1
                    precision_at_k = relevant_count / i
                    ap += precision_at_k
                    
            ap = ap / len(relevant_docs) if relevant_docs else 0
            map_sum += ap
            
            print(f"Query: '{query}'")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  Average Precision: {ap:.4f}")
            
        # Calculate Mean Average Precision (MAP)
        num_queries = len(query_set)
        map_score = map_sum / num_queries if num_queries > 0 else 0
        avg_precision = precision_sum / num_queries if num_queries > 0 else 0
        avg_recall = recall_sum / num_queries if num_queries > 0 else 0
        
        print("\nOverall Evaluation:")
        print(f"  Mean Average Precision (MAP): {map_score:.4f}")
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Average Recall: {avg_recall:.4f}")
    
    def save_index(self, filepath='search_index.json'):
        """Save the index to disk."""
        print(f"Saving index to {filepath}...")
        
        # Create a serializable version of the index
        serializable_index = {
            'inverted_index': {term: [(doc_id, freq, []) for doc_id, freq, _ in postings] 
                               for term, postings in self.inverted_index.items()},
            'term_freq': {doc_id: dict(freqs) for doc_id, freqs in self.term_freq.items()},
            'doc_lengths': self.doc_lengths,
            'avgdl': self.avgdl
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_index, f)
            
        print(f"Index saved successfully")
    
    def load_index(self, filepath='search_index.json'):
        """Load the index from disk."""
        print(f"Loading index from {filepath}...")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.inverted_index = defaultdict(list)
            for term, postings in data['inverted_index'].items():
                self.inverted_index[term] = [(doc_id, freq, []) for doc_id, freq, _ in postings]
                
            self.term_freq = defaultdict(Counter)
            for doc_id, freqs in data['term_freq'].items():
                self.term_freq[doc_id] = Counter(freqs)
                
            self.doc_lengths = data['doc_lengths']
            self.avgdl = data['avgdl']
            
            print(f"Index loaded successfully with {len(self.inverted_index)} terms")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Enhanced Search Engine')
    parser.add_argument('--docs-dir', default='crawled_documents', help='Directory containing documents')
    parser.add_argument('--no-bm25', action='store_true', help='Disable BM25 scoring')
    parser.add_argument('--no-positional', action='store_true', help='Disable positional indexing')
    parser.add_argument('--no-expansion', action='store_true', help='Disable query expansion')
    parser.add_argument('--save-index', action='store_true', help='Save index to disk after building')
    parser.add_argument('--load-index', action='store_true', help='Load index from disk instead of building')
    
    args = parser.parse_args()
    
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
    while True:
        query = input(f"\n{Fore.GREEN}Enter your search query (or 'exit' to quit): {Style.RESET_ALL}")
        
        if query.lower() == 'exit':
            break
        
        # Search for documents
        results = search_engine.search(query)
        
        # Display results
        query_terms = search_engine.preprocess_text(query, for_indexing=False)
        search_engine.display_results(results, query_terms)

if __name__ == "__main__":
    main()