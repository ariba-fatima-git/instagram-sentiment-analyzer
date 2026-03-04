# src/sentiment.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from tqdm import tqdm

class SentimentAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """Initialize the sentiment analysis model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Label mapping
        self.labels = ['negative', 'neutral', 'positive']
        
    def analyze_sentiment(self, text):
        """Analyze sentiment for a single text"""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get label and confidence
        prediction = torch.argmax(scores, dim=-1).cpu().item()
        confidence = scores[0][prediction].cpu().item()
        
        return {
            'label': self.labels[prediction],
            'confidence': confidence,
            'scores': {
                'negative': scores[0][0].item(),
                'neutral': scores[0][1].item(),
                'positive': scores[0][2].item()
            }
        }
    
    def analyze_batch(self, texts, batch_size=32):
        """Analyze sentiment for batch of texts"""
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, 
                                   max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Process results
            for j, score in enumerate(scores):
                prediction = torch.argmax(score).item()
                results.append({
                    'sentiment': self.labels[prediction],
                    'confidence': score[prediction].item(),
                    'negative_score': score[0].item(),
                    'neutral_score': score[1].item(),
                    'positive_score': score[2].item()
                })
        
        return results
    
    def add_sentiment_to_df(self, df, text_column='cleaned_text'):
        """Add sentiment columns to dataframe"""
        # Analyze all texts
        results = self.analyze_batch(df[text_column].tolist())
        
        # Add results to dataframe
        for key in results[0].keys():
            df[key] = [r[key] for r in results]
        
        return df
