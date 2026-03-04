
import pandas as pd
import numpy as np
from collections import Counter
import re

class InsightGenerator:
    def __init__(self):
        self.insights = {}
        
    def calculate_sentiment_percentages(self, df):
        """Calculate overall sentiment distribution"""
        sentiment_counts = df['sentiment'].value_counts()
        total = len(df)
        
        percentages = {
            sentiment: (count / total) * 100 
            for sentiment, count in sentiment_counts.items()
        }
        
        return percentages
    
    def find_common_themes(self, df, sentiment=None, n_terms=10):
        """Find common words/themes in comments"""
        # Filter by sentiment if specified
        if sentiment:
            texts = df[df['sentiment'] == sentiment]['cleaned_text']
        else:
            texts = df['cleaned_text']
        
        # Combine all texts
        all_text = ' '.join(texts)
        
        # Tokenize and clean
        words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
        
        # Remove common stop words (custom list for social media)
        stop_words = set(['the', 'and', 'for', 'you', 'this', 'that', 'with', 
                         'have', 'was', 'are', 'not', 'but', 'like', 'just',
                         'get', 'can', 'your', 'its', 'amp', 'via', 'will'])
        
        words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Get most common
        common_terms = Counter(words).most_common(n_terms)
        
        return common_terms
    
    def generate_summary(self, df, cluster_labels=None):
        """Generate plain English summary of findings"""
        percentages = self.calculate_sentiment_percentages(df)
        
        # Determine overall sentiment
        if percentages.get('positive', 0) > 60:
            overall = "overwhelmingly positive"
        elif percentages.get('positive', 0) > 50:
            overall = "mostly positive"
        elif percentages.get('negative', 0) > 40:
            overall = "concerning with significant criticism"
        elif percentages.get('neutral', 0) > 50:
            overall = "neutral with room for engagement"
        else:
            overall = "mixed"
        
        # Find top positive and negative themes
        pos_themes = self.find_common_themes(df, 'positive', 5)
        neg_themes = self.find_common_themes(df, 'negative', 5)
        
        # Build summary
        summary = f"Overall audience sentiment is {overall}. "
        summary += f"Positive comments: {percentages.get('positive', 0):.1f}%, "
        summary += f"Neutral: {percentages.get('neutral', 0):.1f}%, "
        summary += f"Negative: {percentages.get('negative', 0):.1f}%. "
        
        if pos_themes:
            pos_words = ', '.join([w for w, _ in pos_themes[:3]])
            summary += f"Positive comments often mention: {pos_words}. "
        
        if neg_themes:
            neg_words = ', '.join([w for w, _ in neg_themes[:3]])
            summary += f"Criticism focuses on: {neg_words}. "
        
        # Add cluster insights if available
        if cluster_labels:
            # Find largest cluster
            largest_cluster = max(cluster_labels.values(), key=lambda x: x['size'])
            summary += f"The largest audience segment ({largest_cluster['percentage']:.1f}%) consists of {largest_cluster['label'].lower()}."
        
        return summary
    
    def generate_cluster_descriptions(self, cluster_labels):
        """Generate detailed descriptions for each cluster"""
        descriptions = {}
        
        for cluster_id, info in cluster_labels.items():
            desc = f"**{info['label']}**  \n"
            desc += f"Size: {info['size']} comments ({info['percentage']:.1f}%)  \n"
            desc += f"Dominant sentiment: {info['dominant_sentiment']}  \n"
            
            if info['keywords']:
                desc += f"Key topics: {', '.join(info['keywords'])}  \n"
            
            descriptions[cluster_id] = desc
        
        return descriptions
