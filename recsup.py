import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download NLTK resources if not already available
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('wordnet')
    nltk.download('stopwords')

class SupervisorRecommender:
    def __init__(self, excel_path):
        """Initialize the recommender system with the Excel file path"""
        self.df = pd.read_excel(excel_path)
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        
        # Customize TF-IDF parameters for better topic modeling
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),     # Include both unigrams and bigrams
            min_df=2,               # Ignore terms that appear in less than 2 documents
            max_df=0.85,            # Ignore terms that appear in more than 85% of documents
            sublinear_tf=True       # Apply sublinear tf scaling (1 + log(tf))
        )
        self.process_data()
        
    def preprocess_text(self, text):
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize and lemmatize
        tokens = text.split()
        lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stopwords]
        
        return ' '.join(lemmatized)
        
    def process_data(self):
        """Process the data and create TF-IDF vectors"""
        # Clean the data
        self.df.columns = self.df.columns.str.strip().str.lower()
        required_cols = ['supervisor name', 'working title']
        
        # Check if required columns exist
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Excel file missing required columns: {', '.join(missing_cols)}")
        
        # Remove rows with missing values
        self.df = self.df.dropna(subset=required_cols)
        
        # Preprocess working titles
        self.df['processed_title'] = self.df['working title'].apply(self.preprocess_text)
        
        # Create TF-IDF matrix for the working titles
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['processed_title'])
        
        # Add a terms column for explanation
        feature_names = self.vectorizer.get_feature_names_out()
        self.df['key_terms'] = self._extract_key_terms(self.tfidf_matrix, feature_names)
        
    def _extract_key_terms(self, tfidf_matrix, feature_names, top_n=5):
        """Extract top terms for each document for explanation purposes"""
        key_terms_list = []
        
        for i in range(tfidf_matrix.shape[0]):
            # Get the TF-IDF scores for this document
            feature_index = tfidf_matrix[i,:].nonzero()[1]
            tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
            
            # Sort by score and get top terms
            top_terms = sorted([(feature_names[i], s) for (i, s) in tfidf_scores], key=lambda x: x[1], reverse=True)[:top_n]
            key_terms_list.append(', '.join([term for term, score in top_terms]))
            
        return key_terms_list
        
    def find_matching_titles(self, query, min_score=0.0, top_n=None):
        """Find all titles that match the query regardless of supervisor"""
        # Preprocess the query
        processed_query = self.preprocess_text(query)
        
        # Transform query to TF-IDF vector
        try:
            query_vector = self.vectorizer.transform([processed_query])
        except Exception as e:
            # Handle case where query terms aren't in the vocabulary
            print(f"Warning: Query processing issue - {e}")
            return pd.DataFrame(columns=['supervisor', 'working_title', 'similarity', 'key_terms'])
        
        # Calculate cosine similarity between query and all titles
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Create a dataframe with supervisors and similarity scores
        results = pd.DataFrame({
            'supervisor': self.df['supervisor name'],
            'working_title': self.df['working title'],
            'similarity': cosine_similarities,
            'key_terms': self.df['key_terms']
        })
        
        # Filter by minimum score
        results = results[results['similarity'] > min_score]
        
        # Sort by similarity score in descending order
        results = results.sort_values('similarity', ascending=False)
        
        # Limit to top_n results if specified
        if top_n is not None and top_n > 0:
            results = results.head(top_n)
            
        return results
    
    def get_recommended_supervisors(self, query, min_score=0.0, top_n=5):
        """Get the top recommended supervisors based on aggregated title matches"""
        matches = self.find_matching_titles(query, min_score)
        
        if matches.empty:
            return pd.DataFrame(columns=['supervisor', 'avg_similarity', 'matching_titles'])
            
        # Group by supervisor and aggregate scores and titles
        supervisor_scores = matches.groupby('supervisor').agg({
            'similarity': 'mean',
            'working_title': lambda x: list(x),
            'key_terms': lambda x: '; '.join(set(x))
        }).reset_index()
        
        # Rename columns
        supervisor_scores.columns = ['supervisor', 'avg_similarity', 'matching_titles', 'common_terms']
        
        # Sort by average similarity score
        supervisor_scores = supervisor_scores.sort_values('avg_similarity', ascending=False)
        
        # Limit to top_n results
        if top_n is not None and top_n > 0:
            supervisor_scores = supervisor_scores.head(top_n)
            
        return supervisor_scores

def main():
    # Create a simple interface
    try:
        recommender = SupervisorRecommender('SV_FYPtitle.xlsx')
        print("Project Title Matching System\n")
        
        while True:
            query = input("\nEnter project keywords or title (or 'q' to quit): ")
            if query.lower() == 'q':
                break
            
            # Provide default score if user enters invalid input
            try:    
                min_score = float(input("Enter minimum similarity score (0-1): "))
                if min_score < 0 or min_score > 1:
                    print("Invalid score. Using default of 0.1")
                    min_score = 0.1
            except ValueError:
                print("Invalid input. Using default score of 0.1")
                min_score = 0.1
            
            # Let user choose between individual titles or aggregated supervisors
            mode = input("Search by (1) Individual titles or (2) Recommended supervisors? [1/2]: ")
            
            if mode == '2':
                # Get recommended supervisors
                matches = recommender.get_recommended_supervisors(query, min_score=min_score)
                
                if len(matches) == 0:
                    print(f"No suitable supervisor found with similarity score above {min_score}.")
                    continue
                    
                print(f"\nFound {len(matches)} recommended supervisors:")
                print("-" * 80)
                for i, (_, row) in enumerate(matches.iterrows(), 1):
                    print(f"{i}. Supervisor: {row['supervisor']}")
                    print(f"   Average Score: {row['avg_similarity']:.4f}")
                    print(f"   Common Terms: {row['common_terms']}")
                    print(f"   Matching Titles ({len(row['matching_titles'])}):")
                    for idx, title in enumerate(row['matching_titles'][:3], 1):
                        print(f"      {idx}. {title}")
                    if len(row['matching_titles']) > 3:
                        print(f"      ... and {len(row['matching_titles'])-3} more")
                    print("-" * 80)
            else:
                # Get matching titles
                matches = recommender.find_matching_titles(query, min_score=min_score)
                
                if len(matches) == 0:
                    print(f"No suitable titles found with similarity score above {min_score}.")
                    continue
                    
                print(f"\nFound {len(matches)} matching titles:")
                print("-" * 80)
                for i, (_, row) in enumerate(matches.iterrows(), 1):
                    print(f"{i}. Supervisor: {row['supervisor']}")
                    print(f"   Title: {row['working_title']}")
                    print(f"   Score: {row['similarity']:.4f}")
                    print(f"   Key Terms: {row['key_terms']}")
                    print("-" * 80)
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()