from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
from dotenv import load_dotenv
import logging
import joblib
import numpy as np
import warnings
from firebase_admin import credentials, firestore, initialize_app, storage
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from pymongo import MongoClient
from bson import ObjectId
import base64
from io import BytesIO
from PIL import Image
from firebase_admin import auth
from lstm_predictor import LSTMPredictor
from rl_budget_optimizer import BudgetOptimizer
from typing import Dict, List, Optional, Union, Any
from functools import wraps
import scipy.stats as stats
from transaction_categorizer import TransactionCategorizer

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Allow OPTIONS requests to pass through without authentication
        if request.method == 'OPTIONS':
            return f(*args, **kwargs)
            
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Authorization header is required'}), 401

        try:
            # Remove 'Bearer ' prefix if present
            token = auth_header.replace('Bearer ', '')
            decoded_token = auth.verify_id_token(token)
            request.user_id = decoded_token['uid']
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return jsonify({'error': 'Invalid or expired token'}), 401

    return decorated_function

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize paths and directories
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
models_dir = os.environ.get('MODELS_DIR', os.path.join(current_dir, 'models'))
os.makedirs(models_dir, exist_ok=True)
os.environ['MODELS_DIR'] = models_dir

# Set up logging
log_dir = os.path.join(current_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'app.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {log_file}")
logger.info(f"Models directory set to: {models_dir}")

# Load environment variables
try:
    load_dotenv(env_path)
    logger.info(f"Successfully loaded .env file from {env_path}")
except Exception as e:
    logger.error(f"Error loading .env file: {str(e)}")
    raise

# Initialize Flask app
app = Flask(__name__)

# Configure CORS with specific settings
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

# Add after_request handler to ensure CORS headers are set
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:5173'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

# Initialize MongoDB
try:
    mongo_client = MongoClient(os.getenv('MONGODB_URI'))
    mongo_db = mongo_client[os.getenv('MONGODB_DB_NAME', 'finance_analyzer')]
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {str(e)}")
    raise

# Initialize Firebase
try:
    cred = credentials.Certificate({
        "type": "service_account",
        "project_id": os.getenv('FIREBASE_PROJECT_ID'),
        "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
        "private_key": os.getenv('FIREBASE_PRIVATE_KEY', '').replace('\\n', '\n'),
        "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
        "client_id": os.getenv('FIREBASE_CLIENT_ID'),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.getenv('FIREBASE_CLIENT_EMAIL')}"
    })
    firebase_app = initialize_app(cred, {
        'storageBucket': 'finance-analyzer-15afe.appspot.com'
    })
    firestore_db = firestore.client()
    logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Firebase: {str(e)}")
    raise

class FinanceAI:
    """
    AI class for financial data analysis, anomaly detection, and expense prediction
    """
    
    def __init__(self, user_id=None):
        """Initialize the FinanceAI with user-specific models"""
        self.user_id = user_id
        if user_id:
            logger.info(f"Initializing FinanceAI with user_id: {user_id}")
        else:
            logger.info("Initializing FinanceAI without user_id (demo mode)")
            
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.expense_predictor = None
        self.category_predictors = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.lstm_predictor = LSTMPredictor(user_id=user_id)
        self.budget_optimizer = BudgetOptimizer(user_id=user_id)
        self.transaction_categorizer = TransactionCategorizer(user_id=user_id)
        
        # Define model paths
        if user_id:
            self.model_path = os.path.join(models_dir, f'expense_predictor_{user_id}.joblib')
            self.scaler_path = os.path.join(models_dir, f'scaler_{user_id}.joblib')
            self.category_models_path = os.path.join(models_dir, f'category_models_{user_id}')
            
            # Load models if they exist
            self._load_models()

    def _load_models(self):
        """Load prediction models if they exist"""
        if not self.user_id:
            return
            
        try:
            feature_columns_path = os.path.join(current_dir, 'models', f'feature_columns_{self.user_id}.joblib')
            if os.path.exists(feature_columns_path):
                self.feature_columns = joblib.load(feature_columns_path)
                logger.info(f"Loaded feature columns for user {self.user_id}")
                
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.expense_predictor = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded prediction models for user {self.user_id}")

                
                # Load category models
                if os.path.exists(self.category_models_path):
                    for file in os.listdir(self.category_models_path):
                        if file.endswith('.joblib'):
                            category = file.replace('.joblib', '')
                            model_path = os.path.join(self.category_models_path, file)
                            self.category_predictors[category] = joblib.load(model_path)
                    logger.info(f"Loaded {len(self.category_predictors)} category models")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

    def _save_models(self):
        """Save prediction models for future use"""
        if not self.user_id or not self.expense_predictor:
            return
        
        try:
            # Save main predictor and scaler
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.expense_predictor, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
        
        # Save feature columns if available
            if self.feature_columns is not None:
                feature_columns_path = os.path.join(current_dir, 'models', f'feature_columns_{self.user_id}.joblib')
                joblib.dump(self.feature_columns, feature_columns_path)
        
        # Save category models
            os.makedirs(self.category_models_path, exist_ok=True)
            for category, model in self.category_predictors.items():
                model_path = os.path.join(self.category_models_path, f'{category}.joblib')
                joblib.dump(model, model_path)
        
            logger.info(f"Saved prediction models for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            
    def _prepare_time_features(self, df):
        """Extract time-based features from date column"""
        # Create a copy to avoid modifying the original
        df_features = df.copy()
        required_columns = ['date', 'amount', 'category']
        for col in df_features.columns:
            if col not in required_columns:
                df_features = df_features.drop(col, axis=1)
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_features['date']):
            df_features['date'] = pd.to_datetime(df_features['date'])
        
        # Extract date features
        df_features['day_of_month'] = df_features['date'].dt.day
        df_features['day_of_week'] = df_features['date'].dt.dayofweek
        df_features['month'] = df_features['date'].dt.month
        df_features['year'] = df_features['date'].dt.year
        df_features['is_weekend'] = df_features['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Create features for beginning, middle, end of month
        df_features['month_period'] = df_features['day_of_month'].apply(
            lambda x: 0 if x <= 10 else (1 if x <= 20 else 2)
        )
        
        # Drop the original date column
        df_features = df_features.drop('date', axis=1)
        
        # One-hot encode categorical features
        df_features = pd.get_dummies(df_features, columns=['category'])
        
        return df_features

    def train_prediction_models(self, df):
        """Train prediction models for future expense amounts"""
        if len(df) < 30:  # Need sufficient data for training
            logger.warning("Not enough data to train prediction models (minimum 30 records required)")
            return False
            
        try:
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
                
            # Prepare features
            df_features = self._prepare_time_features(df)
            
            # Keep track of column order for predictions
            self.feature_columns = [col for col in df_features.columns if col != 'amount']
            
            # Split data
            X = df_features[self.feature_columns]
            y = df_features['amount']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train overall expense predictor (RandomForest)
            self.expense_predictor = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            self.expense_predictor.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.expense_predictor.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            logger.info(f"Trained expense predictor with MAE: {mae:.2f}")
            
            # Train category-specific models
            self._train_category_models(df_features)
            
            # Train LSTM model
            lstm_history = self.lstm_predictor.train(df)
            logger.info(f"Trained LSTM model with final loss: {lstm_history['loss']:.4f}")
            
            # Train budget optimizer
            categories = df['category'].unique().tolist()
            income = df[df['amount'] > 0]['amount'].sum() / len(df['date'].dt.month.unique())
            savings_goal = income * 0.2  # Default 20% savings goal
            budget_training_success = self.budget_optimizer.train(
                df, categories, income, savings_goal
            )
            logger.info(f"Trained budget optimizer: {budget_training_success}")
            
            # Save models for future use
            self._save_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training prediction models: {str(e)}")
            return False
            
    def _train_category_models(self, df_features):
        """Train category-specific prediction models"""
        # Get all category columns (one-hot encoded)
        categories = [col for col in df_features.columns if col.startswith('category_')]
        
        # Train separate model for each category with sufficient data
        for cat_col in categories:
            cat_name = cat_col.replace('category_', '')
            
            # Filter rows for this category
            cat_rows = df_features[df_features[cat_col] == 1]
            
            if len(cat_rows) < 15:  # Skip if not enough data
                continue
                
            X_cat = cat_rows[self.feature_columns]
            y_cat = cat_rows['amount']
            
            # Scale features
            X_cat_scaled = self.scaler.transform(X_cat)
            
            # Train-test split if enough data
            if len(X_cat) > 20:
                X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(
                    X_cat_scaled, y_cat, test_size=0.2, random_state=42
                )
                
                # Train a linear model for this category
                cat_model = LinearRegression()
                cat_model.fit(X_cat_train, y_cat_train)
                
                # Evaluate
                y_cat_pred = cat_model.predict(X_cat_test)
                cat_mae = mean_absolute_error(y_cat_test, y_cat_pred)
                logger.info(f"Trained predictor for category {cat_name} with MAE: {cat_mae:.2f}")
            else:
                # Not enough data for test split, use all data
                cat_model = LinearRegression()
                cat_model.fit(X_cat_scaled, y_cat)
                logger.info(f"Trained predictor for category {cat_name} (limited data)")
            
            # Store model
            self.category_predictors[cat_name] = cat_model

    def predict_future_expenses(self, df, months_ahead=3):
        """Predict expenses for the next few months"""
        if not self.expense_predictor or not self.feature_columns:
            logger.warning("No prediction model available")
            return None
            
        try:
            # Get traditional predictions
            traditional_predictions = self._get_traditional_predictions(df, months_ahead)
            
            # Get LSTM predictions
            lstm_predictions = self.lstm_predictor.predict(df, days_ahead=months_ahead*30)
            
            # Combine predictions
            combined_predictions = self._combine_predictions(traditional_predictions, lstm_predictions)
            
            return combined_predictions
            
        except Exception as e:
            logger.error(f"Error predicting future expenses: {str(e)}")
            return None

    def _get_traditional_predictions(self, df, months_ahead):
        """Get predictions from traditional models"""
        try:
            # Get last date in dataset
            last_date = df['date'].max()
            
            # Generate future dates (month starts)
            future_dates = [(last_date.replace(day=1) + pd.DateOffset(months=i+1)).strftime('%Y-%m') 
                           for i in range(months_ahead)]
            
            predictions = []
            for month in future_dates:
                # Create prediction entry
                month_pred = {
                    'date': month,
                    'total_predicted': 0,
                    'category_predictions': {}
                }
                
                # Add category-specific predictions if available
                for category, model in self.category_predictors.items():
                    # Simple prediction based on historical average for this category and month
                    cat_values = df[df['category'] == category]['amount'].values
                    if len(cat_values) > 0:
                        avg_amount = np.mean(cat_values)
                        month_pred['category_predictions'][category] = float(avg_amount)
                        month_pred['total_predicted'] += float(avg_amount)
                
                predictions.append(month_pred)
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error in traditional predictions: {str(e)}")
            return []

    def _combine_predictions(self, traditional_predictions, lstm_predictions):
        """Combine predictions from different models"""
        # Convert LSTM daily predictions to monthly
        lstm_monthly = lstm_predictions.groupby(
            lstm_predictions['date'].dt.strftime('%Y-%m')
        )['predicted_amount'].sum().reset_index()
        
        # Combine predictions
        combined = []
        for trad_pred in traditional_predictions:
            month = trad_pred['date']
            lstm_pred = lstm_monthly[lstm_monthly['date'] == month]['predicted_amount'].values[0]
            
            # Weighted average (can be adjusted based on model performance)
            combined_pred = {
                'date': month,
                'total_predicted': (trad_pred['total_predicted'] * 0.6 + lstm_pred * 0.4),
                'category_predictions': trad_pred['category_predictions']
            }
            combined.append(combined_pred)
        
        return combined

    def analyze_spending_patterns(self, df):
        """Analyze spending patterns and detect anomalies"""
        if len(df) < 5:  # Need minimum data for analysis
            logger.warning("Not enough data for meaningful analysis (minimum 5 records)")
            return None

        try:
            # Data validation and preprocessing
            logger.info(f"Starting spending pattern analysis with {len(df)} records")
            
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                logger.info("Converting date column to datetime")
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                # Handle invalid dates
                invalid_dates = df['date'].isna().sum()
                if invalid_dates > 0:
                    logger.warning(f"Found {invalid_dates} invalid dates, dropping these records")
                    df = df.dropna(subset=['date'])
            
            # Create feature matrix for anomaly detection
            df_features = df.copy()
            df_features = df_features[['date', 'amount', 'category']]  # Only keep essential columns
            
            # Extract features for anomaly detection
            logger.info("Extracting features for anomaly detection")
            
            # Add date-based features
            df_features['day_of_month'] = df_features['date'].dt.day
            df_features['day_of_week'] = df_features['date'].dt.dayofweek
            df_features['month'] = df_features['date'].dt.month
            df_features['year'] = df_features['date'].dt.year
            df_features['is_weekend'] = df_features['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            
            # Add statistical features
            category_avg = df.groupby('category')['amount'].transform('mean')
            category_std = df.groupby('category')['amount'].transform('std')
            df_features['amount_vs_category_avg'] = df['amount'] / category_avg
            
            # Handle potential division by zero or NaN
            df_features.fillna(0, inplace=True)
            df_features.replace([np.inf, -np.inf], 0, inplace=True)
            
            # Get numerical columns for anomaly detection
            numeric_cols = df_features.select_dtypes(include=['number']).columns
            # Skip date-only columns that may be incorrectly detected as numeric
            numeric_cols = [col for col in numeric_cols if col not in ['date']]
            
            logger.info(f"Using {len(numeric_cols)} numeric features for anomaly detection")
            
            # One-hot encode category for anomaly detection
            df_features_encoded = pd.get_dummies(df_features, columns=['category'], prefix=['cat'])
            numeric_cols = numeric_cols.tolist() + [col for col in df_features_encoded.columns if col.startswith('cat_')]
            
            # Drop any remaining non-numeric columns
            X = df_features_encoded[numeric_cols]
            
            logger.info(f"Feature matrix shape: {X.shape}")
            
            # Check for NaN or infinite values
            if X.isna().any().any() or np.isinf(X).any().any():
                logger.warning("Found NaN or infinite values, replacing with zeros")
                X.fillna(0, inplace=True)
                X.replace([np.inf, -np.inf], 0, inplace=True)
            
            # Detect anomalies
            logger.info("Detecting anomalies with Isolation Forest")
            self.anomaly_detector.fit(X)
            anomaly_scores = self.anomaly_detector.decision_function(X)
            predictions = self.anomaly_detector.predict(X)
            
            # Get anomalous transactions (prediction == -1 means anomaly)
            anomaly_indices = np.where(predictions == -1)[0]
            
            logger.info(f"Found {len(anomaly_indices)} anomalous transactions")
            
            if len(anomaly_indices) > 0:
                anomalous_transactions = df.iloc[anomaly_indices].copy()
                # Add anomaly score for reference
                anomalous_transactions['anomaly_score'] = anomaly_scores[anomaly_indices]
            else:
                # Create empty DataFrame with same columns plus anomaly_score
                anomalous_transactions = df.head(0).copy()
                anomalous_transactions['anomaly_score'] = []
            
            # Train models if enough data and not already trained
            models_trained = False
            if not self.expense_predictor and len(df) >= 30:
                logger.info("Training prediction models")
                models_trained = self.train_prediction_models(df)
                logger.info(f"Models trained: {models_trained}")

            # Predict future expenses
            future_predictions = None
            if self.expense_predictor:
                logger.info("Generating future expense predictions")
                future_predictions = self.predict_future_expenses(df)
                if future_predictions:
                    logger.info(f"Generated predictions for {len(future_predictions)} future periods")
                else:
                    logger.warning("Failed to generate future predictions")

            # Calculate monthly trends
            monthly_trends = df.groupby(df['date'].dt.strftime('%Y-%m'))['amount'].agg(['mean', 'sum', 'count']).to_dict()
            logger.info(f"Calculated trends for {len(monthly_trends['mean'])} months")

            # Calculate spending velocity (rate of change)
            monthly_sums = df.groupby(df['date'].dt.strftime('%Y-%m'))['amount'].sum()
            spending_velocity = 0
            recent_trend_direction = "stable"
            if len(monthly_sums) >= 2:
                # Handle potential division by zero
                latest = monthly_sums.iloc[-1]
                previous = monthly_sums.iloc[-2]
                
                if previous > 0:
                    spending_velocity = ((latest - previous) / previous * 100)
                else:
                    spending_velocity = 100  # Assuming 100% increase if previous was zero
                    
                if spending_velocity > 10:
                    recent_trend_direction = "increasing_rapidly"
                elif spending_velocity > 0:
                    recent_trend_direction = "increasing_slowly"
                elif spending_velocity < -10:
                    recent_trend_direction = "decreasing_rapidly"
                else:
                    recent_trend_direction = "decreasing_slowly"
                    
                logger.info(f"Spending velocity: {spending_velocity:.2f}%, trend: {recent_trend_direction}")

            # Analyze spending by day of week
            day_of_week_spending = df.groupby(df['date'].dt.dayofweek)['amount'].sum()
            weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_of_week_analysis = {weekday_names[day]: float(amount) for day, amount in day_of_week_spending.items()}
            
            # Find top spending day
            if not day_of_week_spending.empty:
                top_spending_day = weekday_names[day_of_week_spending.idxmax()]
                logger.info(f"Top spending day: {top_spending_day}")
            else:
                top_spending_day = None
            
            # Find recurring expenses
            recurring_expenses = []
            try:
                logger.info("Analyzing for recurring expenses")
                recurring_candidates = df.groupby(['category'])['amount'].agg(['count', 'mean', 'std'])
                # Filter for categories with >3 transactions and low standard deviation
                recurring_candidates = recurring_candidates[(recurring_candidates['count'] > 3)]
                
                # Only calculate std/mean ratio where mean is not zero
                valid_means = recurring_candidates['mean'] > 0
                if valid_means.any():
                    std_mean_ratio = recurring_candidates.loc[valid_means, 'std'] / recurring_candidates.loc[valid_means, 'mean']
                    low_variance_categories = std_mean_ratio[std_mean_ratio < 0.1].index
                    
                    for cat in low_variance_categories:
                        cat_data = df[df['category'] == cat]
                        if len(cat_data) > 1:
                            # Check for consistent time intervals
                            cat_data = cat_data.sort_values('date')
                            date_diffs = cat_data['date'].diff().dropna()
                            
                            if not date_diffs.empty:
                                avg_days = date_diffs.mean().days
                                
                                if 25 <= avg_days <= 35:  # Monthly recurring
                                    recurring_expenses.append({
                                        'category': cat,
                                        'amount': float(recurring_candidates.loc[cat, 'mean']),
                                        'frequency': 'Monthly',
                                        'confidence': 'High' if recurring_candidates.loc[cat, 'count'] > 5 else 'Medium'
                                    })
                                elif 13 <= avg_days <= 17:  # Bi-weekly
                                    recurring_expenses.append({
                                        'category': cat,
                                        'amount': float(recurring_candidates.loc[cat, 'mean']),
                                        'frequency': 'Bi-weekly',
                                        'confidence': 'High' if recurring_candidates.loc[cat, 'count'] > 5 else 'Medium'
                                    })
                                elif 6 <= avg_days <= 8:  # Weekly
                                    recurring_expenses.append({
                                        'category': cat,
                                        'amount': float(recurring_candidates.loc[cat, 'mean']),
                                        'frequency': 'Weekly',
                                        'confidence': 'High' if recurring_candidates.loc[cat, 'count'] > 5 else 'Medium'
                                    })
                
                logger.info(f"Found {len(recurring_expenses)} recurring expenses")
            except Exception as e:
                logger.error(f"Error analyzing recurring expenses: {str(e)}")
                # Continue with analysis

            # Generate insights from spending data
            try:
                logger.info("Generating spending insights")
                spending_insights = self._generate_insights(df, future_predictions, anomalous_transactions)
                logger.info(f"Generated {len(spending_insights)} spending insights")
            except Exception as e:
                logger.error(f"Error generating insights: {str(e)}")
                spending_insights = []

            # Analyze seasonal patterns if enough data
            seasonal_patterns = None
            if len(df) >= 180:  # At least 6 months of data
                try:
                    logger.info("Analyzing seasonal patterns")
                    seasonal_patterns = self._analyze_seasonal_patterns(df)
                    if seasonal_patterns:
                        logger.info("Seasonal patterns analyzed successfully")
                    else:
                        logger.warning("Seasonal pattern analysis returned None")
                except Exception as e:
                    logger.error(f"Error analyzing seasonal patterns: {str(e)}")
                    # Continue with analysis
            else:
                logger.info(f"Not enough data for seasonal analysis (need 180 records, have {len(df)})")
            
            # Ensure we always have a proper object structure for seasonal_patterns
            if not seasonal_patterns:
                # Create empty structure with default values
                month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                              'July', 'August', 'September', 'October', 'November', 'December']
                
                # Default month spending data
                month_spending = {}
                for month in month_names:
                    month_spending[month] = {
                        'mean': 0.0,
                        'ci_lower': 0.0,
                        'ci_upper': 0.0,
                        'confidence': 0.0
                    }
                
                # Basic structure for seasonal patterns
                seasonal_patterns = {
                    'highest_spending_month': month_names[0],
                    'lowest_spending_month': month_names[0],
                    'month_spending': month_spending,
                    'quarter_spending': {f'Q{q}': {'mean': 0.0, 'trend': 0.0} for q in range(1, 5)},
                    'category_seasons': {},
                    'seasonality_strength': 0.0,
                    'year_over_year': {
                        'growth': {},
                        'comparison': {}
                    }
                }
                logger.info("Created default seasonal patterns structure")

            # Create summary of anomalies
            anomaly_summary = {
                'count': int(anomalous_transactions.shape[0]),
                'total_amount': float(anomalous_transactions['amount'].sum()),
                'percent_of_transactions': float(anomalous_transactions.shape[0] / len(df) * 100) if len(df) > 0 else 0,
                'percent_of_spending': float(anomalous_transactions['amount'].sum() / df['amount'].sum() * 100) if df['amount'].sum() > 0 else 0,
                'categories': anomalous_transactions['category'].value_counts().to_dict(),
                'top_anomalies': []
            }
            
            # Get top anomalies safely
            if not anomalous_transactions.empty:
                top_anomalies = anomalous_transactions.sort_values('amount', ascending=False).head(5)
                anomaly_summary['top_anomalies'] = top_anomalies.to_dict('records')
            
            # Convert datetime to string for JSON serialization
            if not anomalous_transactions.empty and 'date' in anomalous_transactions.columns:
                anomalous_transactions['date'] = anomalous_transactions['date'].dt.strftime('%Y-%m-%d')

            logger.info("Analysis completed successfully")
            return {
                'anomalies': anomalous_transactions.to_dict('records'),
                'anomaly_summary': anomaly_summary,
                'future_predictions': future_predictions,
                'monthly_trends': monthly_trends,
                'spending_velocity': float(spending_velocity),
                'recent_trend_direction': recent_trend_direction,
                'day_of_week_analysis': day_of_week_analysis,
                'top_spending_day': top_spending_day,
                'recurring_expenses': recurring_expenses,
                'seasonal_patterns': seasonal_patterns,
                'spending_insights': spending_insights,
                'models_trained': models_trained
            }
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}", exc_info=True)
            return None

    def _analyze_seasonal_patterns(self, df):
        """Analyze seasonal spending patterns with enhanced statistical analysis"""
        try:
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Add debugging
            logger.info(f"Analyzing seasonal patterns with {len(df)} records")
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
            
            # Check if we have enough unique months (at least 3 for minimal seasonality)
            unique_months = df['date'].dt.strftime('%Y-%m').nunique()
            logger.info(f"Number of unique months: {unique_months}")
            
            if unique_months < 3:
                logger.warning(f"Not enough unique months for seasonal analysis (have {unique_months}, need at least 3)")
                return None
            
            # Year-over-Year Analysis - Only if we have at least 2 years of data
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            
            # Log year distribution
            year_counts = df.groupby('year').size()
            logger.info(f"Year distribution: {year_counts.to_dict()}")
            
            # Get monthly totals before year-over-year calculation
            monthly_totals = df.groupby([df['date'].dt.year, df['date'].dt.month])['amount'].sum()
            logger.info(f"Monthly totals shape: {monthly_totals.shape}")
            
            # Handle year-over-year calculation more safely
            try:
                yearly_comparison = df.groupby(['year', 'month'])['amount'].sum().unstack()
                logger.info(f"Yearly comparison shape: {yearly_comparison.shape}")
            except Exception as e:
                logger.warning(f"Error in yearly comparison calculation: {str(e)}")
                yearly_comparison = pd.DataFrame()  # Empty DataFrame
            
            # Only calculate YoY growth if we have enough months (at least 13 months of data)
            yoy_growth = pd.DataFrame()  # Default empty DataFrame
            if not yearly_comparison.empty and len(monthly_totals) >= 13 and yearly_comparison.shape[0] >= 2:
                try:
                    logger.info("Calculating year-over-year growth")
                    yoy_growth = yearly_comparison.pct_change(periods=12) * 100
                except Exception as e:
                    logger.warning(f"Error calculating year-over-year growth: {str(e)}")
            else:
                logger.info("Not enough data for year-over-year growth calculation")
            
            # Monthly analysis with confidence intervals - with error handling
            try:
                month_totals = df.groupby(df['date'].dt.month)['amount'].agg(['mean', 'std', 'count'])
                
                # Handle cases with insufficient data per month
                for idx in month_totals.index:
                    if pd.isna(month_totals.loc[idx, 'std']) or month_totals.loc[idx, 'count'] < 2:
                        month_totals.loc[idx, 'std'] = 0
                
                month_totals['ci_lower'] = month_totals['mean'] - 1.96 * (month_totals['std'] / np.sqrt(month_totals['count']))
                month_totals['ci_upper'] = month_totals['mean'] + 1.96 * (month_totals['std'] / np.sqrt(month_totals['count']))
                
                logger.info(f"Monthly totals available: {month_totals.shape[0]} months")
            except Exception as e:
                logger.error(f"Error in monthly analysis: {str(e)}")
                # Create an empty DataFrame with the expected structure
                month_totals = pd.DataFrame(columns=['mean', 'std', 'count', 'ci_lower', 'ci_upper'])
                for month in range(1, 13):
                    month_totals.loc[month] = [0, 0, 0, 0, 0]
            
            # Convert month numbers to names
            month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            
            # Enhanced monthly spending data with safe handling
            month_spending = {}
            for m in range(1, 13):
                if m in month_totals.index:
                    month_spending[month_names[m-1]] = {
                        'mean': float(month_totals.loc[m, 'mean']),
                        'ci_lower': float(month_totals.loc[m, 'ci_lower']),
                        'ci_upper': float(month_totals.loc[m, 'ci_upper']),
                        'confidence': float(month_totals.loc[m, 'count'])
                    }
                else:
                    # Add empty data for missing months
                    month_spending[month_names[m-1]] = {
                        'mean': 0.0,
                        'ci_lower': 0.0,
                        'ci_upper': 0.0,
                        'confidence': 0.0
                    }
            
            # Quarterly analysis with trend
            try:
                df['quarter'] = df['date'].dt.quarter
                quarter_totals = df.groupby('quarter')['amount'].agg(['mean', 'std', 'count'])
                
                # Handle case where we don't have enough quarters
                if len(quarter_totals) >= 2:
                    quarter_totals['trend'] = np.polyfit(range(len(quarter_totals)), quarter_totals['mean'], 1)[0]
                else:
                    # Not enough quarters for trend analysis
                    quarter_totals['trend'] = 0.0
            except Exception as e:
                logger.warning(f"Error in quarterly analysis: {str(e)}")
                # Create default quarter data
                quarter_totals = pd.DataFrame(columns=['mean', 'std', 'count', 'trend'])
                for q in range(1, 5):
                    quarter_totals.loc[q] = [0, 0, 0, 0]
            
            # Category seasonal patterns with enhanced analysis and error handling
            category_seasons = {}
            for category in df['category'].unique():
                try:
                    cat_df = df[df['category'] == category]
                    if len(cat_df) >= 20:  # Need enough data
                        # Monthly analysis with safe handling
                        month_analysis = cat_df.groupby(cat_df['date'].dt.month)['amount'].agg(['sum', 'count', 'std'])
                        # Handle NaN values
                        month_analysis.fillna(0, inplace=True)
                        month_analysis['mean'] = month_analysis['sum'] / month_analysis['count'].clip(lower=1)  # Avoid division by zero
                        
                        # Calculate seasonality index safely
                        overall_mean = month_analysis['mean'].mean()
                        if overall_mean > 0:
                            seasonality_index = (month_analysis['mean'] / overall_mean) * 100
                        else:
                            seasonality_index = month_analysis['mean'] * 0  # All zeros
                        
                        # Identify significant patterns safely
                        significant_months = seasonality_index[abs(seasonality_index - 100) > 15].to_dict()
                        
                        # Safe trend analysis
                        if len(month_analysis) >= 2:
                            trend = np.polyfit(range(len(month_analysis)), month_analysis['mean'], 1)[0]
                        else:
                            trend = 0.0
                        
                        # Handle months with no data
                        if len(month_analysis) > 0:
                            max_idx = month_analysis['mean'].idxmax()
                            min_idx = month_analysis['mean'].idxmin()
                            peak_month = month_names[max_idx-1] if max_idx in range(1, 13) else month_names[0]
                            low_month = month_names[min_idx-1] if min_idx in range(1, 13) else month_names[0]
                            max_val = month_analysis['mean'].max()
                            min_val = month_analysis['mean'].min()
                        else:
                            peak_month = month_names[0]
                            low_month = month_names[0]
                            max_val = 0
                            min_val = 0
                        
                        # Calculate variability safely
                        if max_val > 0 and month_analysis['mean'].mean() > 0:
                            variability = (max_val - min_val) / month_analysis['mean'].mean() * 100
                        else:
                            variability = 0
                        
                        # Create seasonality index for all months
                        full_seasonality_index = {}
                        for m in range(1, 13):
                            if m in seasonality_index.index:
                                full_seasonality_index[month_names[m-1]] = float(seasonality_index[m])
                            else:
                                full_seasonality_index[month_names[m-1]] = 0.0
                        
                        category_seasons[category] = {
                            'peak_month': peak_month,
                            'low_month': low_month,
                            'peak_spending': float(max_val),
                            'low_spending': float(min_val),
                            'variability': float(variability),
                            'seasonality_index': full_seasonality_index,
                            'significant_patterns': {month_names[m-1]: float(v) for m, v in significant_months.items() if m in range(1, 13)},
                            'trend': float(trend),
                            'confidence': float(month_analysis['count'].min() if not month_analysis.empty else 0)
                        }
                except Exception as e:
                    logger.warning(f"Error processing category {category}: {str(e)}")
                    # Skip this category
                    continue
            
            # Calculate overall seasonality strength safely
            try:
                if not month_totals.empty and month_totals['mean'].mean() > 0:
                    seasonality_strength = np.std(list(month_totals['mean'])) / np.mean(month_totals['mean']) * 100
                else:
                    seasonality_strength = 0.0
                logger.info(f"Calculated seasonality strength: {seasonality_strength}")
            except Exception as e:
                logger.warning(f"Error calculating seasonality strength: {str(e)}")
                seasonality_strength = 0.0
            
            # Create the results dictionary with safer handling of year-over-year data
            result = {
                'highest_spending_month': month_names[month_totals['mean'].idxmax()-1] if not month_totals.empty and len(month_totals) > 0 else month_names[0],
                'lowest_spending_month': month_names[month_totals['mean'].idxmin()-1] if not month_totals.empty and len(month_totals) > 0 else month_names[0],
                'month_spending': month_spending,
                'quarter_spending': {f'Q{q}': {
                    'mean': float(quarter_totals.loc[q, 'mean']) if q in quarter_totals.index else 0.0,
                    'trend': float(quarter_totals.loc[q, 'trend']) if q in quarter_totals.index else 0.0
                } for q in range(1, 5)},
                'category_seasons': category_seasons,
                'seasonality_strength': float(seasonality_strength),
                'year_over_year': {
                    'growth': {} if yoy_growth.empty else {
                        str(year): {
                            str(month): float(value) 
                            for month, value in row.items()
                            if not pd.isna(value)
                        } 
                        for year, row in yoy_growth.iterrows()
                    },
                    'comparison': {} if yearly_comparison.empty else {
                        str(year): {
                            str(month): float(value) 
                            for month, value in row.items()
                            if not pd.isna(value)
                        } 
                        for year, row in yearly_comparison.iterrows()
                    }
                }
            }
            
            logger.info("Seasonal pattern analysis completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error analyzing seasonal patterns: {str(e)}", exc_info=True)
            return None

    def _generate_insights(self, df, future_predictions=None, anomalous_transactions=None):
        """Generate insights from spending data"""
        insights = []
        
        try:
            # General spending trend
            recent_df = df.sort_values('date').tail(min(len(df), 30))  # Last 30 transactions or all if fewer
            if len(recent_df) >= 5:
                # Calculate a trend line
                x = np.arange(len(recent_df))
                y = recent_df['amount'].values
                slope, _, r_value, p_value, _ = stats.linregress(x, y)
                
                trend_strength = abs(r_value)
                if p_value < 0.05 and trend_strength > 0.3:  # Statistically significant trend
                    if slope > 0:
                        trend_msg = "Your transaction amounts are trending upward"
                        if trend_strength > 0.7:
                            trend_msg += " strongly"
                        percent_increase = (slope * len(x) / y[0] * 100) if y[0] > 0 else 0
                        if percent_increase > 0:
                            trend_msg += f" (approximately {percent_increase:.1f}% increase)"
                    else:
                        trend_msg = "Your transaction amounts are trending downward"
                        if trend_strength > 0.7:
                            trend_msg += " strongly"
                        percent_decrease = (-slope * len(x) / y[0] * 100) if y[0] > 0 else 0
                        if percent_decrease > 0:
                            trend_msg += f" (approximately {percent_decrease:.1f}% decrease)"
                    
                    insights.append({
                        'type': 'spending_trend',
                        'message': trend_msg,
                        'importance': 'high' if trend_strength > 0.6 else 'medium',
                        'data': {
                            'slope': float(slope),
                            'r_value': float(r_value),
                            'p_value': float(p_value)
                        }
                    })

            # Monthly spending analysis
            monthly_spending = df.groupby(df['date'].dt.strftime('%Y-%m'))['amount'].sum()
            if len(monthly_spending) > 1:
                mom_change = ((monthly_spending.iloc[-1] - monthly_spending.iloc[-2]) / 
                            max(monthly_spending.iloc[-2], 0.01) * 100)  # Avoid division by zero
                
                month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                        'July', 'August', 'September', 'October', 'November', 'December']
                current_month_name = month_names[pd.to_datetime(monthly_spending.index[-1]).month - 1]
                previous_month_name = month_names[pd.to_datetime(monthly_spending.index[-2]).month - 1]
                
                # Format message based on magnitude of change
                if abs(mom_change) < 5:
                    change_msg = f"Your spending in {current_month_name} is similar to {previous_month_name}"
                else:
                    change_msg = f"Your spending has {'increased' if mom_change > 0 else 'decreased'} by {abs(mom_change):.1f}% in {current_month_name} compared to {previous_month_name}"
                    
                    # Add significance level
                    if abs(mom_change) > 30:
                        change_msg += " (significant change)"
                
                insights.append({
                    'type': 'monthly_change',
                    'message': change_msg,
                    'importance': 'high' if abs(mom_change) > 20 else 'medium',
                    'data': {
                        'percent_change': float(mom_change),
                        'current_month': monthly_spending.index[-1],
                        'previous_month': monthly_spending.index[-2],
                        'current_amount': float(monthly_spending.iloc[-1]),
                        'previous_amount': float(monthly_spending.iloc[-2])
                    }
                })
                
                # Identify which categories drove the change
                if len(monthly_spending) >= 2 and abs(mom_change) > 10:
                    current_month = monthly_spending.index[-1]
                    previous_month = monthly_spending.index[-2]
                    
                    # Get category totals for both months
                    current_cats = df[df['date'].dt.strftime('%Y-%m') == current_month].groupby('category')['amount'].sum()
                    previous_cats = df[df['date'].dt.strftime('%Y-%m') == previous_month].groupby('category')['amount'].sum()
                    
                    # Find categories with largest changes
                    changed_categories = []
                    for category in set(current_cats.index) | set(previous_cats.index):
                        curr = current_cats.get(category, 0)
                        prev = previous_cats.get(category, 0)
                        if prev > 0:
                            pct_change = (curr - prev) / prev * 100
                            if abs(pct_change) > 25:  # Significant category change
                                changed_categories.append({
                                    'category': category,
                                    'percent_change': float(pct_change),
                                    'current': float(curr),
                                    'previous': float(prev)
                                })
                    
                    # Sort by magnitude of change
                    changed_categories.sort(key=lambda x: abs(x['percent_change']), reverse=True)
                    
                    if changed_categories:
                        top_change = changed_categories[0]
                        change_direction = 'increase' if top_change['percent_change'] > 0 else 'decrease'
                        insights.append({
                            'type': 'category_change',
                            'message': f"The largest change was in {top_change['category']} with a {abs(top_change['percent_change']):.1f}% {change_direction}",
                            'importance': 'medium',
                            'data': changed_categories[:3]  # Top 3 changes
                        })

            # Category distribution insights
            category_totals = df.groupby('category')['amount'].sum()
            category_counts = df.groupby('category')['amount'].count()
            
            if not category_totals.empty:
                # Top spending categories
                top_cats = category_totals.nlargest(3)
                top_cat_message = f"Your top spending category is {top_cats.index[0]} (${float(top_cats.iloc[0]):.2f})"
                if len(top_cats) > 1:
                    top_cat_message += f", followed by {top_cats.index[1]} (${float(top_cats.iloc[1]):.2f})"
                
                insights.append({
                    'type': 'top_categories',
                    'message': top_cat_message,
                    'importance': 'medium',
                    'data': {cat: float(amt) for cat, amt in top_cats.items()}
                })
                
                # Category concentration
                top_percent = (category_totals.nlargest(1).sum() / category_totals.sum()) * 100
                if top_percent > 40:
                    insights.append({
                        'type': 'category_concentration',
                        'message': f"Your spending is concentrated in {top_cats.index[0]} ({top_percent:.1f}% of total)",
                        'importance': 'medium' if top_percent > 60 else 'low',
                        'data': {'top_category': top_cats.index[0], 'percentage': float(top_percent)}
                    })
                
                # Most frequent categories
                if not category_counts.empty:
                    most_frequent = category_counts.idxmax()
                    if most_frequent != category_totals.idxmax():
                        insights.append({
                            'type': 'frequency_insight',
                            'message': f"You make purchases most frequently in {most_frequent}, but spend the most in {category_totals.idxmax()}",
                            'importance': 'low'
                        })

            # Anomaly insights
            if anomalous_transactions is not None and not anomalous_transactions.empty:
                anomaly_count = len(anomalous_transactions)
                top_anomaly = anomalous_transactions.sort_values('amount', ascending=False).iloc[0]
                
                insights.append({
                    'type': 'anomalies',
                    'message': f"Found {anomaly_count} unusual transactions, with the largest being ${float(top_anomaly['amount']):.2f} for {top_anomaly['category']} on {top_anomaly['date'].strftime('%Y-%m-%d')}",
                    'importance': 'high' if anomaly_count > 2 else 'medium',
                    'data': {
                        'count': anomaly_count,
                        'total_amount': float(anomalous_transactions['amount'].sum()),
                        'top_anomaly': {
                            'amount': float(top_anomaly['amount']),
                            'category': top_anomaly['category'],
                            'date': top_anomaly['date'].strftime('%Y-%m-%d'),
                            'reason': top_anomaly.get('anomaly_reason', 'Unusual transaction')
                        }
                    }
                })
                
                # Category-specific anomalies
                cat_anomalies = anomalous_transactions.groupby('category').size()
                if not cat_anomalies.empty:
                    top_anomaly_cat = cat_anomalies.idxmax()
                    if cat_anomalies[top_anomaly_cat] > 1:
                        insights.append({
                            'type': 'category_anomalies',
                            'message': f"Found {cat_anomalies[top_anomaly_cat]} unusual transactions in {top_anomaly_cat}",
                            'importance': 'medium'
                        })
                
            # Weekend vs weekday spending
            df['is_weekend'] = df['date'].dt.dayofweek >= 5
            weekend_spending = df[df['is_weekend']]['amount'].sum()
            weekday_spending = df[~df['is_weekend']]['amount'].sum()
            
            weekday_count = (~df['is_weekend']).sum()
            weekend_count = df['is_weekend'].sum()
            
            # Calculate average daily spending for comparison
            avg_weekend_daily = weekend_spending / max(weekend_count, 1)
            avg_weekday_daily = weekday_spending / max(weekday_count, 1)
            
            if avg_weekend_daily > avg_weekday_daily * 1.5 and weekend_count > 5:  # Significant weekend spending
                insights.append({
                    'type': 'timing',
                    'message': f"Your average daily spending on weekends (${avg_weekend_daily:.2f}) is {(avg_weekend_daily/avg_weekday_daily):.1f}x higher than weekdays (${avg_weekday_daily:.2f})",
                    'importance': 'medium',
                    'data': {
                        'weekend_daily_avg': float(avg_weekend_daily),
                        'weekday_daily_avg': float(avg_weekday_daily),
                        'ratio': float(avg_weekend_daily/avg_weekday_daily)
                    }
                })
                
            # Add prediction insights if available
            if future_predictions and len(future_predictions) > 0:
                next_month = future_predictions[0]
                insights.append({
                    'type': 'prediction',
                    'message': f"Based on your patterns, we predict you'll spend ${next_month['total_predicted']:.2f} next month",
                    'importance': 'medium',
                    'data': {
                        'predicted_amount': float(next_month['total_predicted']),
                        'predicted_month': next_month['date']
                    }
                })
                
                # Growth rate prediction
                if len(monthly_spending) > 0:
                    current_month_spending = monthly_spending.iloc[-1]
                    next_month_total = next_month["total_predicted"]
                    
                    if next_month_total > current_month_spending * 1.1:  # 10% increase
                        growth_rate = ((next_month_total - current_month_spending) / 
                                      max(current_month_spending, 0.01) * 100)  # Avoid division by zero
                        insights.append({
                            'type': 'growth_prediction',
                            'message': f"Your spending is predicted to increase by {growth_rate:.1f}% next month",
                            'importance': 'high' if growth_rate > 25 else 'medium',
                            'data': {
                                'growth_rate': float(growth_rate),
                                'current': float(current_month_spending),
                                'predicted': float(next_month_total)
                            }
                        })
                    elif next_month_total < current_month_spending * 0.9:  # 10% decrease
                        decline_rate = ((current_month_spending - next_month_total) / 
                                      max(current_month_spending, 0.01) * 100)  # Avoid division by zero
                        insights.append({
                            'type': 'decline_prediction',
                            'message': f"Your spending is predicted to decrease by {decline_rate:.1f}% next month",
                            'importance': 'medium',
                            'data': {
                                'decline_rate': float(decline_rate),
                                'current': float(current_month_spending),
                                'predicted': float(next_month_total)
                            }
                        })

        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}", exc_info=True)

        return insights

    def get_budget_recommendations(self, df, income: float) -> Dict[str, Any]:
        """Get budget recommendations for the next month"""
        try:
            categories = df['category'].unique().tolist()
            recommendations = self.budget_optimizer.get_budget_recommendation(
                df, categories, income
            )
            
            if recommendations:
                # Calculate current spending
                current_spending = df.groupby('category')['amount'].sum().to_dict()
                
                # Calculate suggested changes
                changes = {}
                for cat in categories:
                    current = current_spending.get(cat, 0)
                    suggested = recommendations.get(cat, 0)
                    changes[cat] = {
                        'current': current,
                        'suggested': suggested,
                        'change': suggested - current,
                        'change_percent': ((suggested - current) / max(current, 1)) * 100
                    }
                
                return {
                    'recommendations': recommendations,
                    'changes': changes,
                    'total_income': income,
                    'total_suggested': sum(recommendations.values()),
                    'suggested_savings': income - sum(recommendations.values())
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting budget recommendations: {str(e)}")
            return None

    def train_nlp_categorizer(self, transactions):
        """Train the NLP transaction categorizer"""
        return self.transaction_categorizer.train(transactions)
    
    def categorize_transaction(self, description):
        """Categorize a single transaction description"""
        if not self.user_id:
            logger.warning("Attempting to categorize a transaction without a user_id")
            # Return a default response for users without authentication
            return {
                'category': 'Other',
                'confidence': 0.0,
                'is_trained': False,
                'demo_mode': True
            }
        
        logger.info(f"Categorizing transaction for user {self.user_id}: {description}")
        result = self.transaction_categorizer.categorize(description)
        
        # Log the result for debugging
        logger.info(f"Categorization result: {result}")
        
        return result
    
    def batch_categorize_transactions(self, transactions):
        """Categorize multiple transactions at once"""
        return self.transaction_categorizer.batch_categorize(transactions)


# Add data exploratory and cleaning utilities
class DataProcessor:
    """Helper class for data cleaning and exploratory analysis"""
    
    @staticmethod
    def clean_expense_data(expenses):
        """Clean and validate expense data"""
        if not expenses:
            return []
            
        df = pd.DataFrame(expenses)
        
        # Validate required columns
        required_columns = ['date', 'amount', 'category']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return []
        
        try:
            # Convert data types
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            
            # Drop rows with invalid date or amount
            invalid_mask = df['date'].isna() | df['amount'].isna()
            if invalid_mask.any():
                logger.warning(f"Dropping {invalid_mask.sum()} rows with invalid data")
                df = df.dropna(subset=['date', 'amount'])
            
            # Ensure category is not empty
            df['category'] = df['category'].fillna('Other')
            
            # Ensure ID exists
            if 'id' not in df.columns:
                df['id'] = [str(i) for i in range(len(df))]
                
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error cleaning expense data: {str(e)}")
            return []
    
    @staticmethod
    def perform_exploratory_analysis(df):
        """Perform exploratory analysis on expense data"""
        try:
            # Convert to DataFrame if it's a list
            if isinstance(df, list):
                df = pd.DataFrame(df)
                
            # Ensure date is datetime
            if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Basic statistics
            stats = {
                'total_count': len(df),
                'total_spent': float(df['amount'].sum()),
                'avg_transaction': float(df['amount'].mean()),
                'median_transaction': float(df['amount'].median()),
                'min_transaction': float(df['amount'].min()),
                'max_transaction': float(df['amount'].max()),
                'date_range': {
                    'start': df['date'].min().strftime('%Y-%m-%d'),
                    'end': df['date'].max().strftime('%Y-%m-%d')
                }
            }
            
            # Category analysis
            category_analysis = df.groupby('category').agg({
                'amount': ['count', 'sum', 'mean', 'median'],
                'date': ['min', 'max']
            }).reset_index()
            
            # Flatten multi-level columns
            category_analysis.columns = ['_'.join(col).strip('_') for col in category_analysis.columns.values]
            
            # Convert to dictionary
            category_stats = category_analysis.to_dict('records')
            
            # Time series analysis
            time_stats = {
                'monthly': df.groupby(df['date'].dt.strftime('%Y-%m')).agg({
                    'amount': ['count', 'sum', 'mean']
                }).to_dict(),
                'day_of_week': df.groupby(df['date'].dt.dayofweek).agg({
                    'amount': ['count', 'sum', 'mean']
                }).to_dict()
            }
            
            # Data quality stats
            data_quality = {
                'missing_values': df.isna().sum().to_dict(),
                'duplicate_records': int(df.duplicated().sum())
            }
            
            return {
                'basic_stats': stats,
                'category_stats': category_stats,
                'time_stats': time_stats,
                'data_quality': data_quality
            }
        except Exception as e:
            logger.error(f"Error in exploratory analysis: {str(e)}")
            return None


@app.route('/api/analyze', methods=['POST'])
def analyze_finances():
    try:
        user_id = request.user_id
        logger.info(f"Analysis request received for user: {user_id}")
        
        # Get user's transactions from Firestore
        try:
            user_ref = firestore_db.collection('users').document(user_id)
            user_doc = user_ref.get()
            
            if not user_doc.exists:
                logger.warning(f"User document not found for user {user_id}")
                return jsonify({
                    'error': 'User data not found',
                    'message': 'Please add some transactions before using this feature'
                }), 400
                
            # Get expenses from user document
            user_data = user_doc.to_dict()
            transactions = user_data.get('expenses', [])
            logger.info(f"Found {len(transactions)} transactions")
            
            if not transactions:
                logger.warning(f"No transaction data for user {user_id}")
                return jsonify({
                    'error': 'No transaction data available',
                    'message': 'Please add some transactions before using this feature'
                }), 400
                
            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"DataFrame created with shape: {df.shape}")
            
            # Initialize FinanceAI
            finance_ai = FinanceAI(user_id)
            
            # Analyze spending patterns
            analysis_result = finance_ai.analyze_spending_patterns(df)
            
            if not analysis_result:
                logger.warning("Analysis returned None")
                return jsonify({
                    'error': 'Analysis failed',
                    'message': 'Could not analyze spending patterns'
                }), 500
                
            # Sanitize results for Firebase
            def sanitize_for_firebase(item):
                if isinstance(item, (np.int64, np.int32, np.int16, np.int8,
                                    np.uint64, np.uint32, np.uint16, np.uint8)):
                    return int(item)
                elif isinstance(item, (np.float64, np.float32, np.float16)):
                    return float(item)
                elif isinstance(item, (np.ndarray,)):
                    return sanitize_for_firebase(item.tolist())
                elif isinstance(item, dict):
                    return {k: sanitize_for_firebase(v) for k, v in item.items()}
                elif isinstance(item, list):
                    return [sanitize_for_firebase(i) for i in item]
                else:
                    return item
            
            # Ensure all keys are strings for Firebase
            def ensure_string_keys(data):
                if isinstance(data, dict):
                    return {str(k): ensure_string_keys(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [ensure_string_keys(item) for item in data]
                else:
                    return data
            
            # Extract seasonal patterns if they exist
            seasonal_patterns = analysis_result.get('seasonal_patterns', {})
            
            # Log the seasonal patterns data for debugging
            logger.info(f"Raw seasonal patterns: {seasonal_patterns}")
            logger.info(f"Seasonal patterns type: {type(seasonal_patterns)}")
            
            # Only create default structure if we truly have no data
            if not seasonal_patterns or (isinstance(seasonal_patterns, list) and len(seasonal_patterns) == 0):
                logger.warning("No seasonal patterns data available, creating default structure")
                # Create empty structure with default values
                month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                              'July', 'August', 'September', 'October', 'November', 'December']
                
                # Default month spending data
                month_spending = {}
                for month in month_names:
                    month_spending[month] = {
                        'mean': 0.0,
                        'ci_lower': 0.0,
                        'ci_upper': 0.0,
                        'confidence': 0.0
                    }
                
                # Basic structure for seasonal patterns
                seasonal_patterns = {
                    'highest_spending_month': month_names[0],
                    'lowest_spending_month': month_names[0],
                    'month_spending': month_spending,
                    'quarter_spending': {f'Q{q}': {'mean': 0.0, 'trend': 0.0} for q in range(1, 5)},
                    'category_seasons': {},
                    'seasonality_strength': 0.0,
                    'year_over_year': {
                        'growth': {},
                        'comparison': {}
                    }
                }
            else:
                logger.info("Using actual seasonal patterns data")
            
            # Create the final response structure
            response_data = {
                'seasonal_patterns': ensure_string_keys(sanitize_for_firebase(seasonal_patterns)),
                'total_spent': float(analysis_result.get('total_spent', 0)),
                'average_expense': float(analysis_result.get('average_expense', 0)),
                'std_expense': float(analysis_result.get('std_expense', 0)),
                'category_insights': ensure_string_keys(sanitize_for_firebase(analysis_result.get('category_insights', []))),
                'monthly_trends': ensure_string_keys(sanitize_for_firebase(analysis_result.get('monthly_trends', []))),
                'anomalies': ensure_string_keys(sanitize_for_firebase(analysis_result.get('anomalies', []))),
                'transaction_count': int(analysis_result.get('transaction_count', 0))
            }
            
            # Store in Firebase
            user_ref.set({
                'last_updated': firestore.SERVER_TIMESTAMP,
                'latest_analysis': response_data
            }, merge=True)
            
            return jsonify({
                'status': 'success',
                'data': response_data
            })
            
        except Exception as e:
            logger.error(f"Error processing transactions: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': f'Error processing transactions: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in analyze_finances: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Error analyzing finances: {str(e)}'
        }), 500

@app.route('/api/exploratory', methods=['POST'])
def exploratory_analysis():
    """
    API endpoint to perform exploratory data analysis on expense data
    """
    try:
        logger.info("Received exploratory analysis request")
        data = request.json
        expenses = data.get('expenses', [])
        
        if not expenses:
            return jsonify({
                'status': 'success',
                'data': None,
                'message': 'No expense data provided'
            })

        # Convert to DataFrame
        df = pd.DataFrame(expenses)
        
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Basic statistics
        stats = {
            'total_count': len(df),
            'total_spent': float(df['amount'].sum()) if 'amount' in df.columns else 0,
            'avg_transaction': float(df['amount'].mean()) if 'amount' in df.columns else 0,
            'median_transaction': float(df['amount'].median()) if 'amount' in df.columns else 0,
            'min_transaction': float(df['amount'].min()) if 'amount' in df.columns else 0,
            'max_transaction': float(df['amount'].max()) if 'amount' in df.columns else 0
        }
        
        # Date range if available
        if 'date' in df.columns:
            valid_dates = df['date'].dropna()
            if not valid_dates.empty:
                stats['date_range'] = {
                    'start': valid_dates.min().strftime('%Y-%m-%d'),
                    'end': valid_dates.max().strftime('%Y-%m-%d')
                }
        
        # Category analysis (simplified and with error handling)
        category_stats = []
        if 'category' in df.columns and 'amount' in df.columns:
            try:
                for category, group in df.groupby('category'):
                    if not group.empty:
                        cat_stat = {
                            'category': category,
                            'amount_count': int(len(group)),
                            'amount_sum': float(group['amount'].sum()),
                            'amount_mean': float(group['amount'].mean()),
                            'amount_median': float(group['amount'].median())
                        }
                        category_stats.append(cat_stat)
            except Exception as e:
                logger.error(f"Error in category analysis: {str(e)}")
                # Continue without category stats if there's an error
        
        # Time series analysis (simplified)
        time_stats = {}
        if 'date' in df.columns and 'amount' in df.columns and not df['date'].isna().all():
            try:
                # Monthly aggregation
                df_with_date = df.dropna(subset=['date'])
                monthly_data = df_with_date.groupby(df_with_date['date'].dt.strftime('%Y-%m')).agg({
                    'amount': ['count', 'sum', 'mean']
                })
                
                # Convert to dictionary safely
                monthly_dict = {}
                for idx, row in monthly_data.iterrows():
                    monthly_dict[idx] = {
                        'count': int(row[('amount', 'count')]),
                        'sum': float(row[('amount', 'sum')]),
                        'mean': float(row[('amount', 'mean')])
                    }
                
                time_stats['monthly'] = monthly_dict
                
                # Day of week aggregation
                day_data = df_with_date.groupby(df_with_date['date'].dt.dayofweek).agg({
                    'amount': ['count', 'sum', 'mean']
                })
                
                # Convert to dictionary safely
                day_dict = {}
                for idx, row in day_data.iterrows():
                    day_dict[int(idx)] = {
                        'count': int(row[('amount', 'count')]),
                        'sum': float(row[('amount', 'sum')]),
                        'mean': float(row[('amount', 'mean')])
                    }
                
                time_stats['day_of_week'] = day_dict
                
            except Exception as e:
                logger.error(f"Error in time series analysis: {str(e)}")
                # Continue without time stats if there's an error
        
        # Return all available data
        result = {
            'basic_stats': stats,
            'category_stats': category_stats
        }
        
        if time_stats:
            result['time_stats'] = time_stats
            
        return jsonify({
            'status': 'success',
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error in exploratory analysis: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0'
    })

@app.route('/api/profile/photo', methods=['POST', 'OPTIONS'])
@require_auth
def upload_profile_photo():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        user_id = request.user_id
        if not user_id:
            return jsonify({'error': 'User ID not found'}), 400

        if 'photo' not in request.files:
            return jsonify({'error': 'No photo file provided'}), 400

        photo = request.files['photo']
        if not photo.filename:
            return jsonify({'error': 'No photo file selected'}), 400

        try:
            # Convert photo to base64
            photo_data = base64.b64encode(photo.read()).decode('utf-8')
            
            # Store in MongoDB
            profile_photos = mongo_db.profile_photos
            profile_photos.update_one(
                {'user_id': user_id},
                {
                    '$set': {
                        'photo_data': photo_data,
                        'content_type': photo.content_type,
                        'last_updated': datetime.utcnow()
                    }
                },
                upsert=True
            )
            
            # Update user document in Firestore with photo reference
            user_ref = firestore_db.collection('users').document(user_id)
            user_ref.set({
                'has_profile_photo': True,
                'last_updated': firestore.SERVER_TIMESTAMP
            }, merge=True)
            
            return jsonify({
                'success': True,
                'message': 'Profile photo uploaded successfully'
            }), 200
            
        except Exception as storage_error:
            logger.error(f"Error storing profile photo: {str(storage_error)}")
            return jsonify({'error': 'Failed to store photo'}), 500

    except Exception as e:
        logger.error(f"Error uploading profile photo: {str(e)}")
        return jsonify({'error': 'Failed to upload photo'}), 500

@app.route('/api/profile/photo/<user_id>', methods=['GET', 'OPTIONS'])
@require_auth
def get_profile_photo(user_id):
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        # Verify the requesting user has access to this photo
        if request.user_id != user_id:
            return jsonify({'error': 'Unauthorized access'}), 403

        # Get photo from MongoDB
        profile_photos = mongo_db.profile_photos
        photo_doc = profile_photos.find_one({'user_id': user_id})
        
        if not photo_doc:
            return jsonify({
                'photo_url': 'https://via.placeholder.com/150?text=No+Photo'
            }), 200
            
        # Return the base64 encoded photo
        return jsonify({
            'photo_data': photo_doc['photo_data'],
            'content_type': photo_doc['content_type']
        }), 200

    except Exception as e:
        logger.error(f"Error retrieving profile photo: {str(e)}")
        return jsonify({
            'photo_url': 'https://via.placeholder.com/150?text=No+Photo'
        }), 200

@app.route('/api/auth/forgot-password', methods=['POST'])
def forgot_password():
    try:
        data = request.get_json()
        email = data.get('email')
        
        if not email:
            return jsonify({'message': 'Email is required'}), 400
            
        # Check if user exists
        user = auth.get_user_by_email(email)
        
        # Generate password reset link
        reset_link = auth.generate_password_reset_link(email)
        
        # TODO: Send email with reset link
        # For now, we'll just return the reset link
        # In production, you should send this link via email
        
        return jsonify({
            'message': 'Password reset instructions have been sent to your email',
            'reset_link': reset_link  # Remove this in production
        }), 200
        
    except auth.UserNotFoundError:
        return jsonify({'message': 'No user found with this email address'}), 404
    except Exception as e:
        print(f"Error in forgot_password: {str(e)}")
        return jsonify({'message': 'An error occurred while processing your request'}), 500

@app.route('/api/budget/recommendations', methods=['GET'])
@require_auth
def get_budget_recommendations():
    try:
        user_id = request.user_id
        income = float(request.args.get('income', 0))
        
        if not income:
            return jsonify({'error': 'Income is required'}), 400
            
        # Get user's transactions from Firestore
        logger.info(f"Fetching transactions for user: {user_id} from Firestore")
        try:
            # Get user document from Firestore
            user_ref = firestore_db.collection('users').document(user_id)
            user_doc = user_ref.get()
            
            if not user_doc.exists:
                logger.warning(f"User document not found for user {user_id}")
                return jsonify({
                    'error': 'User data not found',
                    'total_income': income,
                    'total_suggested': 0,
                    'suggested_savings': income,
                    'recommendations': {},
                    'changes': {}
                }), 200
                
            # Get expenses from user document
            user_data = user_doc.to_dict()
            transactions = user_data.get('expenses', [])
            logger.info(f"Found {len(transactions)} transactions")
            
            if not transactions:
                logger.warning(f"No transaction data for user {user_id}")
                return jsonify({
                    'error': 'No transaction data available',
                    'message': 'Please add some transactions before using this feature',
                    'total_income': income,
                    'total_suggested': 0,
                    'suggested_savings': income,
                    'recommendations': {},
                    'changes': {}
                }), 200
        except Exception as db_err:
            logger.error(f"Firestore error fetching transactions: {str(db_err)}")
            return jsonify({'error': f'Database error: {str(db_err)}'}), 500
            
        # Convert to DataFrame
        try:
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"DataFrame created with shape: {df.shape}")
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
        except Exception as df_err:
            logger.error(f"Error creating DataFrame: {str(df_err)}")
            return jsonify({'error': f'Error processing transaction data: {str(df_err)}'}), 500
        
        # Initialize FinanceAI
        finance_ai = FinanceAI(user_id)
        
        # Get recommendations
        recommendations = finance_ai.get_budget_recommendations(df, income)
        
        if not recommendations:
            logger.warning(f"Could not generate recommendations for user {user_id}")
            # Return a valid empty response
            return jsonify({
                'total_income': income,
                'total_suggested': 0,
                'suggested_savings': income,
                'recommendations': {},
                'changes': {}
            }), 200
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8,
                              np.uint64, np.uint32, np.uint16, np.uint8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            return obj
        
        recommendations = convert_numpy_types(recommendations)
        
        # Ensure all required fields exist
        if 'total_income' not in recommendations:
            recommendations['total_income'] = income
        if 'total_suggested' not in recommendations:
            recommendations['total_suggested'] = sum(recommendations.get('recommendations', {}).values())
        if 'suggested_savings' not in recommendations:
            recommendations['suggested_savings'] = income - recommendations['total_suggested']
        if 'recommendations' not in recommendations:
            recommendations['recommendations'] = {}
        if 'changes' not in recommendations:
            recommendations['changes'] = {}
            
        logger.info(f"Generated budget recommendations for user {user_id}")
        return jsonify(recommendations)
            
    except ValueError as e:
        logger.error(f"Invalid input value: {str(e)}")
        return jsonify({
            'error': 'Invalid input value',
            'total_income': float(request.args.get('income', 0)),
            'total_suggested': 0,
            'suggested_savings': float(request.args.get('income', 0)),
            'recommendations': {},
            'changes': {}
        }), 400
    except Exception as e:
        logger.error(f"Error getting budget recommendations: {str(e)}")
        return jsonify({
            'error': str(e),
            'total_income': float(request.args.get('income', 0)),
            'total_suggested': 0,
            'suggested_savings': float(request.args.get('income', 0)),
            'recommendations': {},
            'changes': {}
        }), 500

@app.route('/api/budget/train', methods=['POST'])
@require_auth
def train_budget_model():
    try:
        user_id = request.user_id
        logger.info(f"Train budget model request received for user: {user_id}")
        
        data = request.get_json()
        logger.info(f"Request data: {data}")
        
        if not data:
            logger.warning("No data provided in request")
            return jsonify({'error': 'No data provided'}), 400
            
        income = float(data.get('income', 0))
        savings_goal = float(data.get('savings_goal', 0))
        
        logger.info(f"Parsed income: {income}, savings_goal: {savings_goal}")
        
        if not income:
            logger.warning("Income not provided or zero")
            return jsonify({'error': 'Income is required'}), 400
            
        # Get user's transactions from Firestore
        logger.info(f"Fetching transactions for user: {user_id} from Firestore")
        try:
            # Get user document from Firestore
            user_ref = firestore_db.collection('users').document(user_id)
            user_doc = user_ref.get()
            
            if not user_doc.exists:
                logger.warning(f"User document not found for user {user_id}")
                return jsonify({
                    'error': 'User data not found',
                    'message': 'User profile not found'
                }), 400
                
            # Get expenses from user document
            user_data = user_doc.to_dict()
            transactions = user_data.get('expenses', [])
            logger.info(f"Found {len(transactions)} transactions")
            
            if not transactions:
                logger.warning(f"No transaction data for user {user_id}")
                return jsonify({
                    'error': 'No transaction data available',
                    'message': 'Please add some transactions before training the model'
                }), 400
        except Exception as db_err:
            logger.error(f"Firestore error fetching transactions: {str(db_err)}")
            return jsonify({'error': f'Database error: {str(db_err)}'}), 500
            
        # Convert to DataFrame
        try:
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"DataFrame created with shape: {df.shape}")
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
        except Exception as df_err:
            logger.error(f"Error creating DataFrame: {str(df_err)}")
            return jsonify({'error': f'Error processing transaction data: {str(df_err)}'}), 500
        
        # Initialize FinanceAI
        try:
            logger.info(f"Initializing FinanceAI for user: {user_id}")
            finance_ai = FinanceAI(user_id)
        except Exception as ai_err:
            logger.error(f"Error initializing FinanceAI: {str(ai_err)}")
            return jsonify({'error': f'Error initializing AI: {str(ai_err)}'}), 500
        
        # Train budget optimizer
        try:
            categories = df['category'].unique().tolist()
            logger.info(f"Categories found: {categories}")
            logger.info(f"Training budget model for user {user_id} with {len(transactions)} transactions")
            
            # Ensure models directory exists
            models_path = os.environ.get('MODELS_DIR', os.path.join(current_dir, 'models'))
            os.makedirs(models_path, exist_ok=True)
            logger.info(f"Ensuring models directory exists at: {models_path}")
            
            # Attempt to train the model
            success = finance_ai.budget_optimizer.train(
                df, categories, income, savings_goal
            )
            logger.info(f"Training result: {success}")
        except Exception as train_err:
            error_msg = f"Error training budget model: {str(train_err)}"
            logger.error(error_msg, exc_info=True)
            return jsonify({'error': error_msg}), 500
        
        if success:
            logger.info(f"Budget model trained successfully for user {user_id}")
            
            # Get recommendations after training
            try:
                recommendations = finance_ai.budget_optimizer.get_budget_recommendation(
                    df, categories, income
                )
                logger.info(f"Recommendations generated: {recommendations is not None}")
            except Exception as rec_err:
                logger.error(f"Error getting recommendations after training: {str(rec_err)}")
                recommendations = None
            
            # If we have recommendations, add them to the response
            if recommendations:
                try:
                    # Convert numpy types to Python native types for JSON serialization
                    def convert_numpy_types(obj):
                        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8,
                                          np.uint64, np.uint32, np.uint16, np.uint8)):
                            return int(obj)
                        elif isinstance(obj, (np.float64, np.float32, np.float16)):
                            return float(obj)
                        elif isinstance(obj, dict):
                            return {k: convert_numpy_types(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_numpy_types(i) for i in obj]
                        return obj
                    
                    recommendations = convert_numpy_types(recommendations)
                    logger.info("Recommendations converted for JSON serialization")
                    
                    return jsonify({
                        'message': 'Budget model trained successfully',
                        'recommendations': recommendations
                    })
                except Exception as conv_err:
                    logger.error(f"Error converting recommendations: {str(conv_err)}", exc_info=True)
                    # Return success message without recommendations
                    return jsonify({'message': 'Budget model trained successfully'})
            else:
                logger.info("No recommendations returned after training")
                return jsonify({'message': 'Budget model trained successfully'})
        else:
            logger.error(f"Failed to train budget model for user {user_id}")
            return jsonify({
                'error': 'Failed to train budget model',
                'message': 'Please try again or use a different set of parameters'
            }), 500
            
    except ValueError as val_err:
        logger.error(f"Invalid input value: {str(val_err)}")
        return jsonify({'error': f'Invalid input value: {str(val_err)}'}), 400
    except Exception as e:
        logger.error(f"Unexpected error in train_budget_model: {str(e)}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/categorizer/train', methods=['POST'])
def train_categorizer():
    """
    Train the NLP transaction categorizer model
    """
    try:
        data = request.json
        user_id = data.get('user_id')
        transactions = data.get('transactions', [])
        
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'User ID is required'
            }), 400
            
        if not transactions or len(transactions) < 20:
            return jsonify({
                'status': 'error',
                'message': 'At least 20 labeled transactions are required for training'
            }), 400
            
        # Create FinanceAI instance
        finance_ai = FinanceAI(user_id)
        
        # Train the model
        results = finance_ai.train_nlp_categorizer(transactions)
        
        if not results['success']:
            return jsonify({
                'status': 'error',
                'message': results.get('error', 'Failed to train categorizer')
            }), 400
            
        return jsonify({
            'status': 'success',
            'data': results
        })
        
    except Exception as e:
        logger.error(f"Error training NLP categorizer: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/categorizer/categorize', methods=['POST'])
def categorize_transaction():
    """
    Categorize a single transaction
    """
    try:
        data = request.json
        user_id = data.get('user_id')
        description = data.get('description')
        
        if not description:
            return jsonify({
                'status': 'error',
                'message': 'Transaction description is required'
            }), 400
        
        # Log authentication details for debugging
        logger.info(f"Categorize transaction request: user_id={user_id}, description={description}")
        
        # Check for auth header if user_id is provided
        if user_id:
            auth_header = request.headers.get('Authorization')
            if auth_header:
                logger.info(f"Auth header present for user {user_id}")
            else:
                logger.warning(f"No auth header for user {user_id}")
        
        # Allow demo mode without user_id
        if not user_id:
            logger.info("No user_id provided, using demo mode")
            # Use a demo categorizer with default categories
            finance_ai = FinanceAI()
            demo_result = {
                'category': 'Other',  # Default category
                'confidence': 0.5,
                'is_trained': False,
                'demo_mode': True
            }
            
            # Try to make a logical guess based on keywords
            desc_lower = description.lower()
            
            # Simple keyword matching for demo
            keyword_map = {
                'amazon': 'Shopping',
                'walmart': 'Shopping',
                'target': 'Shopping',
                'ebay': 'Shopping',
                'netflix': 'Entertainment',
                'spotify': 'Entertainment',
                'hulu': 'Entertainment',
                'uber': 'Transportation',
                'lyft': 'Transportation',
                'gas': 'Transportation',
                'grocery': 'Food & Dining',
                'restaurant': 'Food & Dining',
                'coffee': 'Food & Dining',
                'starbucks': 'Food & Dining',
                'chipotle': 'Food & Dining',
                'rent': 'Housing',
                'mortgage': 'Housing',
                'electric': 'Utilities',
                'water': 'Utilities',
                'cable': 'Utilities',
                'phone': 'Utilities',
                'doctor': 'Healthcare',
                'medical': 'Healthcare',
                'pharmacy': 'Healthcare',
                'flight': 'Travel',
                'hotel': 'Travel',
                'airbnb': 'Travel',
                'tuition': 'Education',
                'school': 'Education',
                'university': 'Education',
                'salary': 'Income',
                'paycheck': 'Income',
                'deposit': 'Income'
            }
            
            for keyword, category in keyword_map.items():
                if keyword in desc_lower:
                    demo_result['category'] = category
                    demo_result['confidence'] = 0.8
                    break
            
            return jsonify({
                'status': 'success',
                'data': demo_result,
                'message': 'Demo mode: Sign in to train a personalized model'
            })
            
        # Validate user_id format if provided (to help diagnose auth issues)
        if not isinstance(user_id, str) or len(user_id) < 10:
            logger.warning(f"Invalid user_id format: {user_id}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid user ID format'
            }), 400
            
        # Create FinanceAI instance
        logger.info(f"Creating FinanceAI instance for user {user_id}")
        finance_ai = FinanceAI(user_id)
        
        # Categorize the transaction
        result = finance_ai.categorize_transaction(description)
        logger.info(f"Categorization result: {result}")
        
        return jsonify({
            'status': 'success',
            'data': result
        })
        
    except Exception as e:
        logger.error(f"Error categorizing transaction: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/categorizer/batch', methods=['POST'])
def batch_categorize():
    """
    Categorize multiple transactions at once
    """
    try:
        data = request.json
        user_id = data.get('user_id')
        transactions = data.get('transactions', [])
        
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'User ID is required'
            }), 400
            
        if not transactions:
            return jsonify({
                'status': 'error',
                'message': 'Transactions array is required'
            }), 400
            
        # Create FinanceAI instance
        finance_ai = FinanceAI(user_id)
        
        # Categorize the transactions
        results = finance_ai.batch_categorize_transactions(transactions)
        
        return jsonify({
            'status': 'success',
            'data': results
        })
        
    except Exception as e:
        logger.error(f"Error batch categorizing transactions: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    logger.info(f"Starting Flask server on port 5001")
    app.run(debug=True, port=5001, host='0.0.0.0')