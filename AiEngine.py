import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import mysql.connector
from flask import jsonify

# Download NLTK resources if not already available
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('wordnet')
    nltk.download('stopwords')

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  
    'database': 'project_supervisor_rec'
}

class SupervisorRecommender:
    def __init__(self):
        """Initialize the recommender system with database connection"""
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        
        # Customize TF-IDF parameters for better topic modeling
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),     # Include both unigrams and bigrams
            min_df=1,               # Lower this since we have fewer documents in the database
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
        
    def get_db_connection(self):
        """Get database connection"""
        conn = mysql.connector.connect(**db_config)
        return conn, conn.cursor(dictionary=True)
        
    def process_data(self):
        """Process the data from database and create TF-IDF vectors"""
        conn, cursor = self.get_db_connection()
        
        try:
            # Fetch supervisor data joined with expertise
            cursor.execute("""
                SELECT s.SupervisorID, s.SvName, s.SvEmail, e.Expertise 
                FROM supervisor s
                JOIN expertise e ON s.SupervisorID = e.SupervisorID
            """)
            
            results = cursor.fetchall()
            
            # Create DataFrame from results
            self.df = pd.DataFrame(results)
            
            # Group by supervisor to collect all expertise areas
            supervisor_expertise = self.df.groupby(['SupervisorID', 'SvName', 'SvEmail'])['Expertise'].apply(
                lambda x: ' '.join(x)).reset_index()
            
            # Preprocess expertise text
            supervisor_expertise['processed_expertise'] = supervisor_expertise['Expertise'].apply(self.preprocess_text)
            
            # Store the processed data
            self.supervisor_data = supervisor_expertise
            
            # Create TF-IDF matrix for the expertise
            self.tfidf_matrix = self.vectorizer.fit_transform(self.supervisor_data['processed_expertise'])
            
            # Add a terms column for explanation
            feature_names = self.vectorizer.get_feature_names_out()
            self.supervisor_data['key_terms'] = self._extract_key_terms(self.tfidf_matrix, feature_names)
            
        except Exception as e:
            print(f"Database error: {e}")
            # Create empty DataFrames in case of error
            self.supervisor_data = pd.DataFrame(columns=['SupervisorID', 'SvName', 'SvEmail', 'Expertise', 'processed_expertise'])
            self.tfidf_matrix = None
        finally:
            cursor.close()
            conn.close()
    
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
        
    def search_supervisors(self, query, min_score=0.0, top_n=5):
        """Find supervisors matching the query based on their expertise"""
        # Preprocess the query
        processed_query = self.preprocess_text(query)
        
        # Transform query to TF-IDF vector
        try:
            query_vector = self.vectorizer.transform([processed_query])
        except Exception as e:
            # Handle case where query terms aren't in the vocabulary
            print(f"Warning: Query processing issue - {e}")
            return []
        
        # Calculate cosine similarity between query and all expertise
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Create a dataframe with supervisors and similarity scores
        results = pd.DataFrame({
            'supervisor_id': self.supervisor_data['SupervisorID'],
            'supervisor_name': self.supervisor_data['SvName'],
            'supervisor_email': self.supervisor_data['SvEmail'],
            'expertise': self.supervisor_data['Expertise'],
            'similarity': cosine_similarities,
            'key_terms': self.supervisor_data['key_terms']
        })
        
        # Filter by minimum score
        results = results[results['similarity'] > min_score]
        
        # Sort by similarity score in descending order
        results = results.sort_values('similarity', ascending=False)
        
        # Limit to top_n results if specified
        if top_n is not None and top_n > 0:
            results = results.head(top_n)
            
        # Convert to list of dictionaries for JSON serialization
        return results.to_dict('records')

# Create a singleton instance
recommender = None

def get_recommender():
    global recommender
    if recommender is None:
        recommender = SupervisorRecommender()
    return recommender