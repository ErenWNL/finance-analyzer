from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
import joblib
import numpy as np
import warnings
from firebase_admin import credentials, firestore, initialize_app
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

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize paths and directories
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
models_dir = os.path.join(current_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

# Load environment variables
try:
    load_dotenv(env_path)
    logger.info(f"Successfully loaded .env file from {env_path}")
except Exception as e:
    logger.error(f"Error loading .env file: {str(e)}")
    raise

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, 
     resources={r"/api/*": {"origins": "http://localhost:5173"}},
     supports_credentials=True)

# Initialize MongoDB
try:
    mongo_client = MongoClient(os.getenv('MONGODB_URI'))
    db = mongo_client[os.getenv('MONGODB_DB_NAME', 'finance_analyzer')]
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
    initialize_app(cred)
    db = firestore.client()
    logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Firebase: {str(e)}")
    raise

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

class FinanceAI:
    """
    AI class for financial data analysis, anomaly detection, and expense prediction
    """
    
    def __init__(self, user_id=None):
        """Initialize the FinanceAI with user-specific models"""
        self.user_id = user_id
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.expense_predictor = None
        self.category_predictors = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
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
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
                
            # Get the last date in the dataset
            last_date = df['date'].max()
            
            # Prepare data for future months
            future_dates = [last_date + timedelta(days=30*i) for i in range(1, months_ahead+1)]
            future_predictions = []
            
            # Get category distribution from recent data (last 3 months)
            three_months_ago = last_date - timedelta(days=90)
            recent_df = df[df['date'] >= three_months_ago]
            
            # Calculate category proportions
            category_totals = recent_df.groupby('category')['amount'].sum()
            total_recent_spending = category_totals.sum()
            category_proportions = (category_totals / total_recent_spending).to_dict() if total_recent_spending > 0 else {}
            
            # Make predictions for each future month
            for future_date in future_dates:
                # Create a sample row with the future date
                sample = pd.DataFrame({
                    'date': [future_date],
                    'amount': [0],  # Placeholder
                    'category': ['unknown']  # Placeholder
                })
                
                # Generate features for prediction
                features_df = self._prepare_time_features(sample)
                
                # Ensure all feature columns exist
                for col in self.feature_columns:
                    if col not in features_df.columns:
                        features_df[col] = 0
                
                # Keep only the necessary columns in the right order
                features_df = features_df[self.feature_columns]
                
                # Scale features
                features_scaled = self.scaler.transform(features_df)
                
                # Predict total spending
                total_prediction = float(self.expense_predictor.predict(features_scaled)[0])
                
                # Predict category-specific amounts
                category_predictions = self._predict_category_amounts(
                    features_df, total_prediction, category_proportions
                )
                
                # Build prediction object
                prediction = {
                    'date': future_date.strftime('%Y-%m'),
                    'total_predicted': total_prediction,
                    'category_predictions': category_predictions
                }
                future_predictions.append(prediction)
            
            return future_predictions
            
        except Exception as e:
            logger.error(f"Error predicting future expenses: {str(e)}")
            return None
            
    def _predict_category_amounts(self, features_df, total_prediction, category_proportions):
        """Predict spending amounts for each category"""
        category_predictions = {}
        
        # First try to use category-specific models
        for category, model in self.category_predictors.items():
            cat_col = f'category_{category}'
            
            # Create category-specific features
            cat_features = features_df.copy()
            if cat_col in cat_features.columns:
                cat_features[cat_col] = 1
            
            # Add missing columns
            for col in self.feature_columns:
                if col not in cat_features.columns:
                    cat_features[col] = 0
            
            # Keep only necessary columns
            cat_features = cat_features[self.feature_columns]
            
            # Scale and predict
            cat_features_scaled = self.scaler.transform(cat_features)
            category_predictions[category] = float(model.predict(cat_features_scaled)[0])
        
        # If no category predictions or categories are missing, use category proportions
        if not category_predictions or len(category_predictions) < len(category_proportions):
            for cat, prop in category_proportions.items():
                if cat not in category_predictions:
                    category_predictions[cat] = total_prediction * prop
        
        # Ensure sum of category predictions roughly equals total prediction
        if category_predictions:
            sum_categories = sum(category_predictions.values())
            if sum_categories > 0:  # Avoid division by zero
                scale_factor = total_prediction / sum_categories
                category_predictions = {
                    cat: amount * scale_factor
                    for cat, amount in category_predictions.items()
                }
        
        return category_predictions

    def analyze_spending_patterns(self, df):
        """Analyze spending patterns and detect anomalies"""
        if len(df) < 5:  # Need minimum data for analysis
            logger.warning("Not enough data for meaningful analysis (minimum 5 records)")
            return None

        try:
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
                
            # Detect anomalies
            features = df[['amount']].values
            anomalies = self.anomaly_detector.fit_predict(features)
            anomalous_transactions = df[anomalies == -1]

            # Train prediction models if not already trained
            models_trained = False
            if not self.expense_predictor and len(df) >= 30:
                models_trained = self.train_prediction_models(df)

            # Predict future expenses
            future_predictions = None
            if self.expense_predictor:
                future_predictions = self.predict_future_expenses(df)

            # Calculate monthly trends
            monthly_trends = df.groupby(df['date'].dt.strftime('%Y-%m'))['amount'].agg(['mean', 'sum']).to_dict()

            # Calculate spending velocity (rate of change)
            monthly_sums = df.groupby(df['date'].dt.strftime('%Y-%m'))['amount'].sum()
            spending_velocity = 0
            if len(monthly_sums) >= 2:
                spending_velocity = ((monthly_sums.iloc[-1] - monthly_sums.iloc[-2]) / 
                                    max(monthly_sums.iloc[-2], 0.01) * 100)  # Avoid division by zero

            # Generate insights from spending data
            spending_insights = self._generate_insights(df, future_predictions)

            # Analyze seasonal patterns if enough data
            seasonal_patterns = None
            if len(df) >= 365:  # At least a year of data
                seasonal_patterns = self._analyze_seasonal_patterns(df)

            return {
                'anomalies': anomalous_transactions.to_dict('records'),
                'future_predictions': future_predictions,
                'monthly_trends': monthly_trends,
                'spending_velocity': float(spending_velocity),
                'seasonal_patterns': seasonal_patterns,
                'spending_insights': spending_insights,
                'models_trained': models_trained
            }
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            return None
            
    def _analyze_seasonal_patterns(self, df):
        """Analyze seasonal spending patterns"""
        month_totals = df.groupby(df['date'].dt.month)['amount'].mean().to_dict()
        if not month_totals:
            return None
            
        highest_month = max(month_totals, key=month_totals.get)
        lowest_month = min(month_totals, key=month_totals.get)
        
        return {
            'highest_spending_month': highest_month,
            'lowest_spending_month': lowest_month,
            'month_averages': month_totals
        }

    def _generate_insights(self, df, future_predictions=None):
        """Generate insights from spending data"""
        insights = []
        
        try:
            # Analyze recent spending trend (last 5 transactions)
            recent_trend = df.sort_values('date').tail(5)
            if len(recent_trend) >= 5:
                is_increasing = recent_trend['amount'].is_monotonic_increasing
                insights.append({
                    'type': 'trend',
                    'message': f'Your spending is {"increasing" if is_increasing else "decreasing"} in recent transactions'
                })

            # Identify top spending categories
            category_totals = df.groupby('category')['amount'].sum()
            if not category_totals.empty:
                top_category = category_totals.idxmax()
                insights.append({
                    'type': 'category',
                    'message': f'Your highest spending category is {top_category}'
                })

            # Analyze monthly patterns
            monthly_spending = df.groupby(df['date'].dt.strftime('%Y-%m'))['amount'].sum()
            if len(monthly_spending) > 1:
                mom_change = ((monthly_spending.iloc[-1] - monthly_spending.iloc[-2]) / 
                              max(monthly_spending.iloc[-2], 0.01) * 100)  # Avoid division by zero
                insights.append({
                    'type': 'monthly_change',
                    'message': f'Your spending has {"increased" if mom_change > 0 else "decreased"} by {abs(mom_change):.1f}% compared to last month'
                })
                
            # Add prediction insights if available
            if future_predictions and len(future_predictions) > 0:
                next_month = future_predictions[0]
                insights.append({
                    'type': 'prediction',
                    'message': f'Based on your spending patterns, we predict you\'ll spend ${next_month["total_predicted"]:.2f} next month'
                })
                
                # Identify predicted high-growth categories
                if len(future_predictions) >= 2 and len(monthly_spending) > 0:
                    current_month_spending = monthly_spending.iloc[-1]
                    next_month_total = next_month["total_predicted"]
                    
                    if next_month_total > current_month_spending:
                        growth_rate = ((next_month_total - current_month_spending) / 
                                      max(current_month_spending, 0.01) * 100)  # Avoid division by zero
                        insights.append({
                            'type': 'growth_prediction',
                            'message': f'Your spending is predicted to increase by {growth_rate:.1f}% next month'
                        })
                    
                    # Find fastest growing categories
                    high_growth_categories = []
                    for category, predicted_amount in next_month["category_predictions"].items():
                        if category in category_totals:
                            current_amount = category_totals[category]
                            if predicted_amount > current_amount:
                                growth = ((predicted_amount - current_amount) / 
                                         max(current_amount, 0.01) * 100)  # Avoid division by zero
                                if growth > 20:  # Significant growth threshold
                                    high_growth_categories.append((category, growth))
                    
                    if high_growth_categories:
                        highest_growth_cat = max(high_growth_categories, key=lambda x: x[1])
                        insights.append({
                            'type': 'category_growth',
                            'message': f'Your {highest_growth_cat[0]} spending is projected to increase by {highest_growth_cat[1]:.1f}% next month'
                        })

            # Spending frequency analysis
            if len(df) >= 30:
                date_diff = df['date'].sort_values().diff()
                if not date_diff.empty:
                    avg_days_between = date_diff.mean().days
                    if avg_days_between < 2:
                        insights.append({
                            'type': 'frequency',
                            'message': 'You\'re making transactions almost daily'
                        })
                    elif avg_days_between < 4:
                        insights.append({
                            'type': 'frequency',
                            'message': 'You\'re making transactions every few days'
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
            
            if avg_weekend_daily > avg_weekday_daily * 1.5:  # Significant weekend spending
                insights.append({
                    'type': 'timing',
                    'message': 'You tend to spend more on weekends'
                })

        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")

        return insights


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
    """
    Analyzes expense data and returns financial insights
    """
    try:
        logger.info("Received analyze request")
        data = request.json
        expenses = data.get('expenses', [])
        user_id = data.get('user_id')
        
        if not expenses:
            logger.warning("No expense data provided")
            return jsonify({
                'status': 'success',
                'data': {
                    'total_spent': 0,
                    'average_expense': 0,
                    'category_totals': {},
                    'monthly_totals': {},
                    'transaction_count': 0,
                    'categories': []
                }
            })

        # Convert to DataFrame
        try:
            df = pd.DataFrame(expenses)
            logger.debug(f"Created DataFrame with columns: {df.columns}")

            # Validate required columns
            required_columns = ['date', 'amount', 'category']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required columns: {missing_columns}'
                }), 400

            # Convert data types
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            
            # Remove rows with invalid data
            df = df.dropna(subset=['date', 'amount'])
            
            if len(df) == 0:
                logger.warning("All data was invalid after cleaning")
                return jsonify({
                    'status': 'success',
                    'data': {
                        'total_spent': 0,
                        'average_expense': 0,
                        'category_totals': {},
                        'monthly_totals': {},
                        'transaction_count': 0,
                        'categories': []
                    }
                })

            # Handle category if missing
            if 'category' not in df.columns:
                df['category'] = 'Other'
            else:
                df['category'] = df['category'].fillna('Other')

            # Perform basic analysis
            try:
                analysis_result = {
                    'total_spent': float(df['amount'].sum()),
                    'average_expense': float(df['amount'].mean()),
                    'category_totals': df.groupby('category')['amount'].sum().to_dict(),
                    'transaction_count': len(df),
                    'categories': df['category'].unique().tolist()
                }
            except Exception as e:
                logger.error(f"Error in basic analysis: {str(e)}")
                analysis_result = {
                    'total_spent': 0,
                    'average_expense': 0,
                    'category_totals': {},
                    'transaction_count': 0,
                    'categories': []
                }
            
            # Calculate monthly totals safely
            try:
                monthly_totals = df.groupby(df['date'].dt.strftime('%Y-%m'))['amount'].sum()
                analysis_result['monthly_totals'] = {k: float(v) for k, v in monthly_totals.to_dict().items()}
            except Exception as e:
                logger.error(f"Error calculating monthly totals: {str(e)}")
                analysis_result['monthly_totals'] = {}

            # Add AI analysis if possible
            try:
                finance_ai = FinanceAI(user_id)
                ai_analysis = finance_ai.analyze_spending_patterns(df)
                if ai_analysis:
                    # Sanitize AI analysis results for Firebase storage
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
                        def ensure_string_keys(data):
                            """Convert all dict keys to strings to ensure Firebase compatibility"""
                            if isinstance(data, dict):
                                return {str(k): ensure_string_keys(v) for k, v in data.items()}
                            elif isinstance(data, list):
                                return [ensure_string_keys(item) for item in data]
                            else:
                                return data
    
                        sanitized_analysis = {}
                        for key, value in ai_analysis.items():
                            sanitized_analysis[key] = sanitize_for_firebase(value)
            
                        # Ensure all keys are strings for Firebase
                        sanitized_analysis = ensure_string_keys(sanitized_analysis)
            
                        analysis_result['ai_insights'] = sanitized_analysis
                
            except Exception as e:
                logger.error(f"Error in AI analysis: {str(e)}")
                # Continue without AI insights if there's an error

            # Store in Firebase if user_id is provided
            try:
                if user_id:
                    doc_ref = db.collection('users').document(user_id)
                    doc_ref.set({
                        'last_updated': firestore.SERVER_TIMESTAMP,
                        'latest_analysis': analysis_result
                    }, merge=True)
            except Exception as e:
                logger.error(f"Error storing analysis in Firebase: {str(e)}")
                # Continue even if Firebase storage fails

            return jsonify({
                'status': 'success',
                'data': analysis_result
            })

        except Exception as e:
            logger.error(f"Error processing DataFrame: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Error processing expense data: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"General error in analyze_finances: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
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
def upload_profile_photo():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
        
    try:
        user_id = request.form.get('userId')
        if not user_id:
            response = jsonify({'error': 'User ID is required'})
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
            return response, 400

        if 'photo' not in request.files:
            response = jsonify({'error': 'No photo provided'})
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
            return response, 400

        photo = request.files['photo']
        if photo.filename == '':
            response = jsonify({'error': 'No selected file'})
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
            return response, 400

        # Convert image to base64
        img = Image.open(photo)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Store in MongoDB
        photo_data = {
            'userId': user_id,
            'photo': img_str,
            'uploadedAt': datetime.utcnow()
        }
        
        # Update or insert the photo
        db.profile_photos.update_one(
            {'userId': user_id},
            {'$set': photo_data},
            upsert=True
        )

        response = jsonify({'message': 'Photo uploaded successfully', 'photo': img_str})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        return response, 200
    except Exception as e:
        logger.error(f"Error uploading photo: {str(e)}")
        response = jsonify({'error': str(e)})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        return response, 500

@app.route('/api/profile/photo/<user_id>', methods=['GET', 'OPTIONS'])
def get_profile_photo(user_id):
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
        
    try:
        photo_data = db.profile_photos.find_one({'userId': user_id})
        if not photo_data:
            response = jsonify({'error': 'Photo not found'})
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
            return response, 404
        
        response = jsonify({'photo': photo_data['photo']})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        return response, 200
    except Exception as e:
        logger.error(f"Error retrieving photo: {str(e)}")
        response = jsonify({'error': str(e)})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        return response, 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)