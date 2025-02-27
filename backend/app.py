from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
from firebase_admin import credentials, firestore, initialize_app
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set up logging for better debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app and environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
models_dir = os.path.join(current_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

try:
    load_dotenv(env_path)
    logger.info(f"Successfully loaded .env file from {env_path}")
except Exception as e:
    logger.error(f"Error loading .env file: {str(e)}")
    raise

app = Flask(__name__)
CORS(app)

# Initialize Firebase
try:
    cred = credentials.Certificate({
        "type": "service_account",
        "project_id": os.getenv('FIREBASE_PROJECT_ID'),
        "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
        "private_key": os.getenv('FIREBASE_PRIVATE_KEY').replace('\\n', '\n'),
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

class FinanceAI:
    def __init__(self, user_id=None):
        self.user_id = user_id
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.expense_predictor = None
        self.category_predictors = {}
        self.scaler = StandardScaler()
        
        # Initialize model paths
        self.model_path = os.path.join(models_dir, f'expense_predictor_{user_id}.joblib') if user_id else None
        self.scaler_path = os.path.join(models_dir, f'scaler_{user_id}.joblib') if user_id else None
        
        # Load models if they exist
        self._load_models()

    def _load_models(self):
        """Load prediction models if they exist"""
        if self.user_id and os.path.exists(self.model_path):
            try:
                self.expense_predictor = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded prediction models for user {self.user_id}")
                
                # Load category models
                category_models_path = os.path.join(models_dir, f'category_models_{self.user_id}')
                if os.path.exists(category_models_path):
                    for file in os.listdir(category_models_path):
                        if file.endswith('.joblib'):
                            category = file.replace('.joblib', '')
                            model_path = os.path.join(category_models_path, file)
                            self.category_predictors[category] = joblib.load(model_path)
            except Exception as e:
                logger.error(f"Error loading models: {str(e)}")

    def _save_models(self):
        """Save prediction models for future use"""
        if self.user_id and self.expense_predictor:
            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                joblib.dump(self.expense_predictor, self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
                
                # Save category models
                category_models_path = os.path.join(models_dir, f'category_models_{self.user_id}')
                os.makedirs(category_models_path, exist_ok=True)
                
                for category, model in self.category_predictors.items():
                    model_path = os.path.join(category_models_path, f'{category}.joblib')
                    joblib.dump(model, model_path)
                
                logger.info(f"Saved prediction models for user {self.user_id}")
            except Exception as e:
                logger.error(f"Error saving models: {str(e)}")

    def _prepare_time_features(self, df):
        """Extract time-based features from date column"""
        # Create a copy of the dataframe to avoid modifying the original
        df_features = df.copy()
        
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
            logger.warning("Not enough data to train prediction models")
            return False
            
        try:
            # Prepare features
            df_features = self._prepare_time_features(df)
            
            # Keep track of column order for predictions
            self.feature_columns = df_features.columns.tolist()
            self.feature_columns.remove('amount')
            
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
                n_estimators=100, max_depth=10, random_state=42
            )
            self.expense_predictor.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.expense_predictor.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            logger.info(f"Trained expense predictor with MAE: {mae:.2f}")
            
            # Train category-specific models
            categories = [col for col in df_features.columns if col.startswith('category_')]
            category_data = {}
            
            for cat_col in categories:
                cat_name = cat_col.replace('category_', '')
                # Filter rows for this category
                cat_rows = df_features[df_features[cat_col] == 1]
                if len(cat_rows) >= 15:  # Minimum rows needed
                    category_data[cat_name] = cat_rows
            
            for cat_name, cat_df in category_data.items():
                X_cat = cat_df[self.feature_columns]
                y_cat = cat_df['amount']
                
                # Scale features
                X_cat_scaled = self.scaler.transform(X_cat)
                
                # Train-test split if enough data
                if len(X_cat) > 20:
                    X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(
                        X_cat_scaled, y_cat, test_size=0.2, random_state=42
                    )
                else:
                    X_cat_train, y_cat_train = X_cat_scaled, y_cat
                    X_cat_test, y_cat_test = X_cat_scaled, y_cat
                
                # Train a linear model for this category
                cat_model = LinearRegression()
                cat_model.fit(X_cat_train, y_cat_train)
                
                # Store model
                self.category_predictors[cat_name] = cat_model
                
                # Evaluate if possible
                if len(X_cat) > 20:
                    y_cat_pred = cat_model.predict(X_cat_test)
                    cat_mae = mean_absolute_error(y_cat_test, y_cat_pred)
                    logger.info(f"Trained predictor for category {cat_name} with MAE: {cat_mae:.2f}")
            
            # Save models for future use
            self._save_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training prediction models: {str(e)}")
            return False

    def predict_future_expenses(self, df, months_ahead=3):
        """Predict expenses for the next few months"""
        if not self.expense_predictor:
            logger.warning("No prediction model available")
            return None
            
        try:
            # Get the last date in the dataset
            last_date = df['date'].max()
            
            # Prepare data for future months
            future_dates = [last_date + timedelta(days=30*i) for i in range(1, months_ahead+1)]
            future_predictions = []
            
            # Get category distribution from recent data (last 3 months)
            three_months_ago = last_date - timedelta(days=90)
            recent_df = df[df['date'] >= three_months_ago]
            category_totals = recent_df.groupby('category')['amount'].sum()
            total_recent_spending = category_totals.sum()
            category_proportions = (category_totals / total_recent_spending).to_dict()
            
            # Make overall prediction for each month
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
                total_prediction = self.expense_predictor.predict(features_scaled)[0]
                
                # Make category-specific predictions
                category_predictions = {}
                for category, model in self.category_predictors.items():
                    # Set the category flag to 1 for this prediction
                    cat_features = features_df.copy()
                    cat_col = f'category_{category}'
                    if cat_col in cat_features.columns:
                        cat_features[cat_col] = 1
                    
                    # Ensure all feature columns exist
                    for col in self.feature_columns:
                        if col not in cat_features.columns:
                            cat_features[col] = 0
                    
                    # Keep only necessary columns
                    cat_features = cat_features[self.feature_columns]
                    
                    # Scale and predict
                    cat_features_scaled = self.scaler.transform(cat_features)
                    category_predictions[category] = model.predict(cat_features_scaled)[0]
                
                # If no category predictions, use category proportions
                if not category_predictions:
                    category_predictions = {
                        cat: total_prediction * prop
                        for cat, prop in category_proportions.items()
                    }
                
                # Build prediction object
                prediction = {
                    'date': future_date.strftime('%Y-%m'),
                    'total_predicted': float(total_prediction),
                    'category_predictions': {
                        cat: float(pred) for cat, pred in category_predictions.items()
                    }
                }
                future_predictions.append(prediction)
            
            return future_predictions
            
        except Exception as e:
            logger.error(f"Error predicting future expenses: {str(e)}")
            return None

    def analyze_spending_patterns(self, df):
        """Analyze spending patterns and detect anomalies"""
        if len(df) < 5:  # Need minimum data for analysis
            return None

        try:
            # Detect anomalies
            features = df[['amount']].values
            anomalies = self.anomaly_detector.fit_predict(features)
            anomalous_transactions = df[anomalies == -1]

            # Train prediction models if not already trained
            if not self.expense_predictor and len(df) >= 30:
                self.train_prediction_models(df)

            # Predict future expenses
            future_predictions = self.predict_future_expenses(df) if self.expense_predictor else None

            # Calculate monthly trends
            monthly_trends = df.groupby(df['date'].dt.strftime('%Y-%m'))['amount'].agg(['mean', 'sum']).to_dict()

            # Calculate spending velocity (rate of change)
            monthly_sums = df.groupby(df['date'].dt.strftime('%Y-%m'))['amount'].sum()
            if len(monthly_sums) >= 2:
                spending_velocity = ((monthly_sums.iloc[-1] - monthly_sums.iloc[-2]) / 
                                    monthly_sums.iloc[-2] * 100)
            else:
                spending_velocity = 0

            # Analyze seasonal patterns if enough data
            seasonal_patterns = None
            if len(df) >= 365:  # At least a year of data
                month_totals = df.groupby(df['date'].dt.month)['amount'].mean().to_dict()
                highest_month = max(month_totals, key=month_totals.get)
                lowest_month = min(month_totals, key=month_totals.get)
                seasonal_patterns = {
                    'highest_spending_month': highest_month,
                    'lowest_spending_month': lowest_month,
                    'month_averages': month_totals
                }

            return {
                'anomalies': anomalous_transactions.to_dict('records'),
                'future_predictions': future_predictions,
                'monthly_trends': monthly_trends,
                'spending_velocity': float(spending_velocity),
                'seasonal_patterns': seasonal_patterns,
                'spending_insights': self._generate_insights(df, future_predictions)
            }
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            return None

    def _generate_insights(self, df, future_predictions=None):
        """Generate insights from spending data"""
        insights = []
        
        try:
            # Analyze recent spending trend
            recent_trend = df.sort_values('date').tail(5)
            if len(recent_trend) >= 5:
                trend = 'increasing' if recent_trend['amount'].is_monotonic_increasing else 'decreasing'
                insights.append({
                    'type': 'trend',
                    'message': f'Your spending is {trend} in recent transactions'
                })

            # Identify top spending categories
            category_totals = df.groupby('category')['amount'].sum()
            top_category = category_totals.idxmax()
            insights.append({
                'type': 'category',
                'message': f'Your highest spending category is {top_category}'
            })

            # Analyze monthly patterns
            monthly_spending = df.groupby(df['date'].dt.strftime('%Y-%m'))['amount'].sum()
            if len(monthly_spending) > 1:
                mom_change = ((monthly_spending.iloc[-1] - monthly_spending.iloc[-2]) / 
                            monthly_spending.iloc[-2] * 100)
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
                if len(future_predictions) >= 2:
                    current_month_spending = monthly_spending.iloc[-1]
                    next_month_total = next_month["total_predicted"]
                    
                    if next_month_total > current_month_spending:
                        growth_rate = ((next_month_total - current_month_spending) / 
                                      current_month_spending * 100)
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
                                growth = ((predicted_amount - current_amount) / current_amount * 100)
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
                avg_days_between = df['date'].diff().mean().days
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
            
            if weekend_spending > weekday_spending * 0.6:  # Significant weekend spending
                insights.append({
                    'type': 'timing',
                    'message': 'You tend to spend more on weekends'
                })

        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")

        return insights

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
                'status': 'error',
                'message': 'No expense data provided'
            }), 400

        # Convert to DataFrame and process data
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
            df['date'] = pd.to_datetime(df['date'])
            df['amount'] = pd.to_numeric(df['amount'])

            # Perform basic analysis
            analysis_result = {
                'total_spent': float(df['amount'].sum()),
                'average_expense': float(df['amount'].mean()),
                'category_totals': df.groupby('category')['amount'].sum().to_dict(),
                'monthly_totals': df.groupby(df['date'].dt.strftime('%Y-%m'))['amount'].sum().to_dict(),
                'transaction_count': len(df),
                'categories': df['category'].unique().tolist()
            }

            # Add AI analysis
            finance_ai = FinanceAI(user_id)
            ai_analysis = finance_ai.analyze_spending_patterns(df)
            if ai_analysis:
                analysis_result['ai_insights'] = ai_analysis

            # Store in Firebase
            if user_id:
                doc_ref = db.collection('users').document(user_id)
                doc_ref.set({
                    'last_updated': firestore.SERVER_TIMESTAMP,
                    'latest_analysis': analysis_result
                }, merge=True)

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

if __name__ == '__main__':
    app.run(debug=True, port=5000)