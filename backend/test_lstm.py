import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from lstm_predictor import LSTMPredictor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_data(n_days=365):
    """Generate synthetic test data"""
    try:
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=n_days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate base amounts with some seasonality
        base_amounts = np.sin(np.linspace(0, 4*np.pi, len(dates))) * 100 + 200
        
        # Add some noise
        noise = np.random.normal(0, 20, len(dates))
        amounts = base_amounts + noise
        
        # Ensure positive amounts
        amounts = np.maximum(amounts, 10)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'amount': amounts,
            'category': ['test'] * len(dates)
        })
        
        logger.info(f"Generated test data with {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error generating test data: {str(e)}")
        raise

def test_lstm_predictor():
    """Test the LSTM predictor"""
    try:
        # Generate test data
        logger.info("Generating test data...")
        test_data = generate_test_data()
        
        # Initialize predictor
        predictor = LSTMPredictor(user_id='test_user')
        
        # Train model
        logger.info("Training LSTM model...")
        history = predictor.train(test_data, epochs=20)
        logger.info(f"Training completed. Final loss: {history['loss']:.4f}")
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = predictor.predict(test_data, days_ahead=30)
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = predictor.evaluate(test_data)
        logger.info(f"Model metrics: {metrics}")
        
        # Print sample predictions
        logger.info("\nSample predictions:")
        print(predictions.head())
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_lstm_predictor() 