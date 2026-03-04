
import pandas as pd
import re
import emoji
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import nltk
from nltk.corpus import stopwords

# Set seed for consistent language detection
DetectorFactory.seed = 42

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

class InstagramCommentCleaner:
    def __init__(self, target_language='en'):
        self.target_language = target_language
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean individual comment text"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags (keep the text after # for analysis)
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag text
        
        # Remove special characters but keep letters, numbers, and emojis
        text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_features(self, text):
        """Extract features from comment"""
        features = {}
        
        # Comment length
        features['length'] = len(text)
        
        # Word count
        words = text.split()
        features['word_count'] = len(words)
        
        # Emoji count
        features['emoji_count'] = emoji.emoji_count(text)
        
        # Hashtag count
        features['hashtag_count'] = len(re.findall(r'#\w+', text))
        
        # Uppercase ratio (for emphasis detection)
        uppercase_chars = sum(1 for c in text if c.isupper())
        features['uppercase_ratio'] = uppercase_chars / len(text) if len(text) > 0 else 0
        
        return features
    
    def detect_language(self, text):
        """Detect comment language"""
        try:
            # Only detect if text has meaningful content
            if len(text.strip()) > 10:
                return detect(text)
            return 'unknown'
        except LangDetectException:
            return 'unknown'
    
    def is_spam(self, text, min_words=2, max_repeat_ratio=0.5):
        """Detect spam comments"""
        if len(text.split()) < min_words:
            return True
        
        # Check for repeated characters (e.g., "aaaaaa")
        if len(text) > 0:
            # Count consecutive repeated characters
            max_repeat = 1
            current_repeat = 1
            for i in range(1, len(text)):
                if text[i] == text[i-1]:
                    current_repeat += 1
                    max_repeat = max(max_repeat, current_repeat)
                else:
                    current_repeat = 1
            
            # If more than 30% of text is repeated characters
            if max_repeat / len(text) > max_repeat_ratio:
                return True
        
        return False
    
    def clean_dataframe(self, df, text_column='text'):
        """Main cleaning pipeline for dataframe"""
        # Make a copy
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=[text_column])
        
        # Remove nulls
        df_clean = df_clean.dropna(subset=[text_column])
        
        # Clean text
        df_clean['cleaned_text'] = df_clean[text_column].apply(self.clean_text)
        
        # Remove empty comments after cleaning
        df_clean = df_clean[df_clean['cleaned_text'].str.len() > 0]
        
        # Extract features
        features_df = df_clean['cleaned_text'].apply(self.extract_features).apply(pd.Series)
        df_clean = pd.concat([df_clean, features_df], axis=1)
        
        # Detect language (optional - can be slow for large datasets)
        # df_clean['language'] = df_clean['cleaned_text'].apply(self.detect_language)
        # df_clean = df_clean[df_clean['language'] == self.target_language]
        
        # Remove spam
        df_clean['is_spam'] = df_clean['cleaned_text'].apply(self.is_spam)
        df_clean = df_clean[~df_clean['is_spam']]
        
        return df_clean
