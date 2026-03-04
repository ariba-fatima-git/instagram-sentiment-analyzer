
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class AudienceClusterer:
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        self.cluster_labels = {}
        
    def prepare_features(self, df):
        """Prepare numerical features for clustering"""
        # Select numerical features for clustering
        feature_cols = ['length', 'word_count', 'emoji_count', 'uppercase_ratio',
                       'negative_score', 'neutral_score', 'positive_score']
        
        # Ensure all columns exist
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # Extract features
        X = df[available_cols].values
        
        # Handle any missing values
        X = np.nan_to_num(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, available_cols
    
    def fit_clusters(self, df):
        """Fit KMeans clustering on the data"""
        # Prepare features
        X_scaled, feature_cols = self.prepare_features(df)
        
        # Fit KMeans
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        df['cluster'] = clusters
        
        return df
    
    def get_cluster_keywords(self, df, text_column='cleaned_text', n_keywords=5):
        """Get top keywords for each cluster using TF-IDF"""
        cluster_keywords = {}
        
        for cluster_id in range(self.n_clusters):
            # Get texts for this cluster
            cluster_texts = df[df['cluster'] == cluster_id][text_column].tolist()
            
            if len(cluster_texts) > 0:
                # Fit TF-IDF on cluster texts
                tfidf_matrix = self.tfidf.fit_transform(cluster_texts)
                
                # Get average TF-IDF scores
                avg_scores = tfidf_matrix.mean(axis=0).A1
                
                # Get top keywords
                feature_names = self.tfidf.get_feature_names_out()
                top_indices = avg_scores.argsort()[-n_keywords:][::-1]
                keywords = [feature_names[i] for i in top_indices]
                
                cluster_keywords[cluster_id] = keywords
            else:
                cluster_keywords[cluster_id] = []
        
        return cluster_keywords
    
    def label_clusters(self, df, cluster_keywords):
        """Generate human-readable labels for clusters"""
        cluster_labels = {}
        
        for cluster_id in range(self.n_clusters):
            # Get cluster statistics
            cluster_df = df[df['cluster'] == cluster_id]
            
            avg_sentiment = cluster_df['sentiment'].mode()[0] if len(cluster_df) > 0 else 'unknown'
            avg_confidence = cluster_df['confidence'].mean()
            keywords = cluster_keywords[cluster_id][:3]  # Top 3 keywords
            
            # Generate label based on sentiment and keywords
            if avg_sentiment == 'positive':
                base_label = "Enthusiastic Fans"
            elif avg_sentiment == 'negative':
                base_label = "Critical Viewers"
            else:
                base_label = "Neutral Observers"
            
            # Add specificity based on keywords
            if keywords:
                specific_label = f"{base_label} ({', '.join(keywords[:2])})"
            else:
                specific_label = base_label
            
            cluster_labels[cluster_id] = {
                'label': specific_label,
                'size': len(cluster_df),
                'percentage': (len(cluster_df) / len(df)) * 100,
                'avg_confidence': avg_confidence,
                'dominant_sentiment': avg_sentiment,
                'keywords': keywords
            }
        
        return cluster_labels
    
    def visualize_clusters(self, df, feature_cols=None):
        """Visualize clusters using PCA"""
        # Prepare features
        X_scaled, used_cols = self.prepare_features(df)
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot each cluster
        for cluster_id in range(self.n_clusters):
            mask = df['cluster'] == cluster_id
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      label=f"Cluster {cluster_id}", alpha=0.6, s=50)
        
        ax.set_title("Audience Clusters Visualization")
        ax.set_xlabel("First Principal Component")
        ax.set_ylabel("Second Principal Component")
        ax.legend()
        
        plt.tight_layout()
        return fig
