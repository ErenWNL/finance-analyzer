from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import logging
from firebase_admin import credentials, firestore, initialize_app
from sklearn.ensemble import IsolationForest
import numpy as np

# Set up logging for better debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app and environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')

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
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)

    def analyze_spending_patterns(self, df):
        """Analyze spending patterns and detect anomalies"""
        if len(df) < 5:  # Need minimum data for analysis
            return None

        try:
            # Detect anomalies
            features = df[['amount']].values
            anomalies = self.anomaly_detector.fit_predict(features)
            anomalous_transactions = df[anomalies == -1]

            # Predict next month's expenses
            next_month_prediction = float(df['amount'].mean())

            # Calculate monthly trends
            monthly_trends = df.groupby(df['date'].dt.strftime('%Y-%m'))['amount'].agg(['mean', 'sum']).to_dict()

            return {
                'anomalies': anomalous_transactions.to_dict('records'),
                'next_month_prediction': next_month_prediction,
                'monthly_trends': monthly_trends,
                'spending_insights': self._generate_insights(df)
            }
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            return None

    def _generate_insights(self, df):
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
            finance_ai = FinanceAI()
            ai_analysis = finance_ai.analyze_spending_patterns(df)
            if ai_analysis:
                analysis_result['ai_insights'] = ai_analysis

            # Store in Firebase
            if 'user_id' in data:
                doc_ref = db.collection('users').document(data['user_id'])
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