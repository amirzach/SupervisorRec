import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SupervisorRecommender:
    def __init__(self, excel_path):
        """Initialize the recommender system with the Excel file path"""
        self.df = pd.read_excel(excel_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.process_data()
        
    def process_data(self):
        """Process the data and create TF-IDF vectors"""
        # Clean the data
        self.df.columns = self.df.columns.str.strip().str.lower()
        if 'supervisor name' not in self.df.columns or 'working title' not in self.df.columns:
            raise ValueError("Excel file must contain 'supervisor name' and 'working title' columns")
        
        # Remove rows with missing values
        self.df = self.df.dropna(subset=['supervisor name', 'working title'])
        
        # Create TF-IDF matrix for the working titles
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['working title'])
        
    def recommend_supervisors(self, query, top_n=5, min_score=0.0):
        """Recommend supervisors based on the input query"""
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity between query and all titles
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Create a dataframe with supervisors and similarity scores
        results = pd.DataFrame({
            'supervisor': self.df['supervisor name'],
            'working_title': self.df['working title'],
            'similarity': cosine_similarities
        })
        
        # Filter by minimum score
        results = results[results['similarity'] > min_score]
        
        # Sort by similarity score in descending order
        results = results.sort_values('similarity', ascending=False)
        
        # Get top N results
        top_results = results.head(top_n)
        return top_results
        
    def get_unique_supervisors(self):
        """Get a list of all unique supervisors in the dataset"""
        return self.df['supervisor name'].unique()
    
    def get_supervisor_titles(self, supervisor_name):
        """Get all working titles for a specific supervisor"""
        supervisor_titles = self.df[self.df['supervisor name'] == supervisor_name]['working title']
        return supervisor_titles.tolist()

def main():
    # Create a simple interface
    try:
        recommender = SupervisorRecommender('SV_FYPtitle.xlsx')
        print("Supervisor Recommendation System\n")
        
        while True:
            query = input("\nEnter project keywords or title (or 'q' to quit): ")
            if query.lower() == 'q':
                break
                
            min_score = float(input("Enter minimum similarity score (0-1): "))
            
            # Get recommendations
            recommendations = recommender.recommend_supervisors(query, min_score=min_score)
            
            if len(recommendations) == 0:
                print(f"No supervisors found with similarity score above {min_score}.")
                continue
                
            print("\nRecommended Supervisors:")
            print("-" * 80)
            for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                print(f"{i}. {row['supervisor']} (Score: {row['similarity']:.4f})")
                print(f"   Title: {row['working_title']}")
                print("-" * 80)
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()