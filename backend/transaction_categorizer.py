import os
import logging
import numpy as np
import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('transaction_categorizer')

# Create a file handler
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/transaction_categorizer.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Set up models directory
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.environ.get('MODELS_DIR', os.path.join(current_dir, 'models'))
os.makedirs(models_dir, exist_ok=True)

class TransactionCategorizer:
    """
    NLP-based transaction categorizer that automatically assigns categories
    based on transaction descriptions.
    """
    
    def __init__(self, user_id=None):
        """Initialize the TransactionCategorizer with user-specific models"""
        self.user_id = user_id
        self.model = None
        self.default_categories = [
            'Food & Dining', 'Shopping', 'Housing', 'Transportation', 'Healthcare', 
            'Entertainment', 'Utilities', 'Travel', 'Education', 'Income', 'Other'
        ]
        
        # Log initialization
        if user_id:
            logger.info(f"Initializing TransactionCategorizer for user: {user_id}")
        else:
            logger.info("Initializing TransactionCategorizer in demo mode (no user_id)")
        
        # Define model paths
        if user_id:
            self.model_path = os.path.join(models_dir, f'nlp_categorizer_{user_id}.joblib')
            loaded = self._load_model()
            logger.info(f"Model loaded for user {user_id}: {loaded}")
        
        # Initialize NLTK resources
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading NLTK resources...")
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def _preprocess_text(self, text):
        """Clean and normalize transaction descriptions"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers (but keep number words)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Lemmatization and stopword removal
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def _load_model(self):
        """Load the categorization model if it exists"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded NLP categorization model for user {self.user_id}")
                return True
        except Exception as e:
            logger.error(f"Error loading NLP model: {str(e)}")
        return False
    
    def _save_model(self):
        """Save the categorization model for future use"""
        if not self.user_id or not self.model:
            return False
        
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            logger.info(f"Saved NLP categorization model for user {self.user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving NLP model: {str(e)}")
            return False
    
    def train(self, transactions):
        """
        Train the NLP categorization model using transaction data
        
        Parameters:
        -----------
        transactions : list of dict or pandas DataFrame
            Transaction data containing 'description' and 'category' columns
        
        Returns:
        --------
        dict
            Training results with accuracy and classification report
        """
        try:
            # Convert to DataFrame if not already
            if not isinstance(transactions, pd.DataFrame):
                transactions = pd.DataFrame(transactions)
            
            # Check for required columns
            if 'description' not in transactions.columns or 'category' not in transactions.columns:
                logger.error("Training data must contain 'description' and 'category' columns")
                return {
                    'success': False,
                    'error': "Training data must contain 'description' and 'category' columns"
                }
            
            # Filter out rows with missing data
            transactions = transactions.dropna(subset=['description', 'category'])
            
            if len(transactions) < 20:
                logger.warning("Not enough data to train NLP model (minimum 20 transactions required)")
                return {
                    'success': False,
                    'error': "Not enough data to train NLP model (minimum 20 transactions required)"
                }
            
            # Preprocess descriptions
            transactions['processed_description'] = transactions['description'].apply(self._preprocess_text)
            
            # Create the NLP pipeline
            self.model = Pipeline([
                ('vectorizer', TfidfVectorizer(
                    min_df=2, max_df=0.8, 
                    ngram_range=(1, 2),
                    sublinear_tf=True
                )),
                ('classifier', LinearSVC(
                    C=1.0, 
                    class_weight='balanced',
                    dual=False,
                    max_iter=10000
                ))
            ])
            
            # Split data
            X = transactions['processed_description']
            y = transactions['category']
            
            # Train on all data for the production model
            self.model.fit(X, y)
            
            # Split for evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            eval_model = Pipeline([
                ('vectorizer', TfidfVectorizer(
                    min_df=2, max_df=0.8, 
                    ngram_range=(1, 2),
                    sublinear_tf=True
                )),
                ('classifier', LinearSVC(
                    C=1.0, 
                    class_weight='balanced',
                    dual=False,
                    max_iter=10000
                ))
            ])
            
            # Train for evaluation
            eval_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = eval_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Save the model
            self._save_model()
            
            return {
                'success': True,
                'accuracy': float(accuracy),
                'report': report,
                'categories': sorted(y.unique().tolist()),
                'transaction_count': len(transactions)
            }
            
        except Exception as e:
            logger.error(f"Error training NLP model: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def categorize(self, description):
        """
        Categorize a transaction based on its description
        
        Parameters:
        -----------
        description : str
            The transaction description to categorize
            
        Returns:
        --------
        dict
            The predicted category with confidence score
        """
        try:
            # Log the request
            logger.info(f"Categorizing transaction for user {self.user_id}: '{description}'")
            
            if not self.model:
                logger.warning(f"No trained model available for user {self.user_id}")
                # Return a reasonable default if model isn't trained
                return {
                    'category': 'Other',
                    'confidence': 0.0,
                    'is_trained': False
                }
            
            # Preprocess the description
            processed = self._preprocess_text(description)
            
            if not processed:
                logger.warning(f"Empty processed description for: '{description}'")
                return {
                    'category': 'Other',
                    'confidence': 0.0,
                    'is_trained': True
                }
            
            # Get prediction
            category = self.model.predict([processed])[0]
            
            # Get confidence score (distance from decision boundary)
            # Get confidence score (distance from decision boundary)
# First vectorize the text using the pipeline's vectorizer
            vectorized_features = self.model.named_steps['vectorizer'].transform([processed])
            # Then pass the numeric features to the classifier
            decision_values = self.model.named_steps['classifier'].decision_function(vectorized_features)[0]
            
            # For multiclass, get the highest decision value
            if isinstance(decision_values, np.ndarray):
                confidence = float(max(decision_values))
            else:
                confidence = float(decision_values)
            
            # Normalize confidence to 0-1 range
            confidence = min(max(0.5 + confidence/10, 0), 1)
            
            result = {
                'category': category,
                'confidence': confidence,
                'is_trained': True
            }
            
            logger.info(f"Categorization result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error categorizing transaction: {str(e)}", exc_info=True)
            return {
                'category': 'Other',
                'confidence': 0.0,
                'is_trained': True,
                'error': str(e)
            }
    
    def batch_categorize(self, transactions):
        """
        Categorize multiple transactions at once
        
        Parameters:
        -----------
        transactions : list of dict
            List of transactions with 'description' field
            
        Returns:
        --------
        list of dict
            The original transactions with updated 'category' and added 'confidence' fields
        """
        try:
            # Make a copy to avoid modifying the original
            result = []
            
            for transaction in transactions:
                # Skip if already categorized
                if 'category' in transaction and transaction['category'] and transaction['category'] != 'Uncategorized' and transaction['category'] != 'Other':
                    result.append(transaction.copy())
                    continue
                
                # Get prediction
                if 'description' in transaction and transaction['description']:
                    prediction = self.categorize(transaction['description'])
                    
                    # Make a copy and update category
                    tx_copy = transaction.copy()
                    tx_copy['category'] = prediction['category']
                    tx_copy['category_confidence'] = prediction['confidence']
                    result.append(tx_copy)
                else:
                    # No description to categorize
                    tx_copy = transaction.copy()
                    tx_copy['category'] = 'Other'
                    tx_copy['category_confidence'] = 0.0
                    result.append(tx_copy)
            
            return result
            
        except Exception as e:
            logger.error(f"Error batch categorizing transactions: {str(e)}")
            return transactions  # Return original if error occurs

# Example usage
if __name__ == "__main__":
    # Test data
    sample_data = [
        {"description": "AMAZON MKTPLACE AMZN.COM/BILL", "category": "Shopping"},
        {"description": "UBER TRIP", "category": "Transportation"},
        {"description": "NETFLIX.COM", "category": "Entertainment"},
        {"description": "TRADER JOE'S", "category": "Food & Dining"},
        {"description": "COMCAST CABLE", "category": "Utilities"},
        {"description": "CVS PHARMACY", "category": "Healthcare"},
        {"description": "SHELL OIL", "category": "Transportation"},
        {"description": "AMC THEATERS", "category": "Entertainment"},
        {"description": "AIRBNB INC", "category": "Travel"},
        {"description": "Salary Deposit", "category": "Income"}
    ]
    
    # Initialize categorizer
    categorizer = TransactionCategorizer()
    
    # Train on sample data
    train_results = categorizer.train(sample_data)
    print(f"Training results: {train_results}")
    
    # Test categorization
    test_description = "AMAZON PAYMENT"
    prediction = categorizer.categorize(test_description)
    print(f"Prediction for '{test_description}': {prediction}") 