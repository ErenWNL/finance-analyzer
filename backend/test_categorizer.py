import os
import pandas as pd
import numpy as np
from transaction_categorizer import TransactionCategorizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_categorizer')

def generate_test_data(num_examples=100):
    """Generate synthetic test data for the categorizer"""
    
    # Define sample transaction patterns for different categories
    category_patterns = {
        'Food & Dining': [
            'UBER EATS', 'DOORDASH', 'GRUBHUB', 'MCDONALD', 'BURGER KING', 
            'SUBWAY', 'CHIPOTLE', 'RESTAURANT', 'PIZZA', 'CAFE', 'STARBUCKS',
            'DUNKIN', 'BAKERY', 'COFFEE', 'DELI'
        ],
        'Shopping': [
            'AMAZON', 'WALMART', 'TARGET', 'COSTCO', 'BEST BUY', 'EBAY',
            'ETSY', 'MACY', 'NORDSTROM', 'RETAIL', 'SHOPIFY', 'STORE'
        ],
        'Housing': [
            'RENT', 'MORTGAGE', 'HOA DUES', 'PROPERTY TAX', 'HOME INSURANCE',
            'MAINTENANCE', 'CLEANING', 'FURNITURE', 'HOME DEPOT', 'LOWES'
        ],
        'Transportation': [
            'UBER', 'LYFT', 'TAXI', 'RAILWAY', 'TRANSIT', 'BUS', 'SUBWAY FARE',
            'GAS', 'SHELL', 'BP', 'EXXON', 'CHEVRON', 'AUTO INSURANCE', 'CAR PAYMENT',
            'DMV', 'PARKING', 'TOLLS'
        ],
        'Healthcare': [
            'PHARMACY', 'CVS', 'WALGREENS', 'HOSPITAL', 'DOCTOR', 'MEDICAL',
            'DENTAL', 'VISION', 'HEALTH INSURANCE', 'LABORATORY', 'CLINIC'
        ],
        'Entertainment': [
            'NETFLIX', 'HULU', 'DISNEY', 'HBO', 'SPOTIFY', 'APPLE MUSIC',
            'AMAZON PRIME', 'MOVIE', 'THEATER', 'CONCERT', 'TICKET', 'GAMING',
            'PLAYSTATION', 'XBOX', 'NINTENDO', 'STEAM'
        ],
        'Utilities': [
            'ELECTRIC', 'WATER', 'GAS', 'POWER', 'INTERNET', 'CABLE', 'PHONE',
            'MOBILE', 'T-MOBILE', 'VERIZON', 'AT&T', 'WASTE MANAGEMENT', 'SEWAGE'
        ],
        'Travel': [
            'AIRLINE', 'FLIGHT', 'HOTEL', 'MOTEL', 'AIRBNB', 'VRBO', 'BOOKING',
            'EXPEDIA', 'TRAVEL AGENCY', 'CRUISE', 'RESORT', 'VACATION'
        ],
        'Education': [
            'TUITION', 'UNIVERSITY', 'COLLEGE', 'SCHOOL', 'COURSE', 'STUDENT LOAN',
            'TEXTBOOK', 'EDUCATION', 'LEARNING', 'TRAINING', 'WORKSHOP'
        ],
        'Income': [
            'SALARY', 'PAYCHECK', 'DIRECT DEPOSIT', 'PAYMENT RECEIVED', 'VENMO',
            'ZELLE', 'CASHAPP', 'PAYPAL', 'REFUND', 'TAX REFUND', 'INTEREST', 'DIVIDEND'
        ]
    }
    
    all_transactions = []
    
    for category, patterns in category_patterns.items():
        # For each category, create a number of example transactions
        num_for_category = max(3, int(num_examples * (0.05 + 0.15 * np.random.random())))
        
        for _ in range(num_for_category):
            # Pick a random pattern for this category
            pattern = np.random.choice(patterns)
            
            # Add some variation
            suffixes = ['', ' PAYMENT', '.COM', ' INC', ' LLC', ' #1234', ' - ONLINE', ' STORE', ' 12/15', ' CORP']
            prefix = ['', 'PAYMENT TO ', 'PURCHASE ', 'POS DEBIT ', 'DEBIT CARD PURCHASE ', 'ACH DEBIT ']
            
            description = f"{np.random.choice(prefix)}{pattern}{np.random.choice(suffixes)}"
            
            # Create transaction
            transaction = {
                'description': description,
                'category': category,
                'amount': round(10 + 990 * np.random.random(), 2)  # Random amount between $10 and $1000
            }
            
            all_transactions.append(transaction)
    
    # Convert to DataFrame and shuffle
    df = pd.DataFrame(all_transactions)
    return df.sample(frac=1).reset_index(drop=True)  # Shuffle rows

def run_test():
    """Run a full test of the TransactionCategorizer"""
    
    print("\n----------------------------------------------")
    print("🧠 Testing NLP Transaction Categorizer")
    print("----------------------------------------------\n")
    
    # Generate synthetic test data
    print("Generating test data...")
    df = generate_test_data(num_examples=100)
    print(f"Generated {len(df)} test transactions across multiple categories\n")
    
    # Display sample of test data
    print("Sample of test data:")
    print(df[['description', 'category']].head())
    print("\n")
    
    # Split data into train and test sets
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    print(f"Train set: {len(train_df)} examples")
    print(f"Test set: {len(test_df)} examples\n")
    
    # Initialize the categorizer
    categorizer = TransactionCategorizer()
    
    # Train the model
    print("Training the model...")
    train_results = categorizer.train(train_df)
    
    if not train_results['success']:
        print(f"❌ Training failed: {train_results.get('error', 'Unknown error')}")
        return
    
    print(f"✅ Model trained successfully with accuracy: {train_results['accuracy']:.4f}\n")
    
    # Test individual prediction
    sample_tx = test_df.iloc[0]
    print(f"Testing prediction on: '{sample_tx['description']}'")
    prediction = categorizer.categorize(sample_tx['description'])
    print(f"Actual category: {sample_tx['category']}")
    print(f"Predicted category: {prediction['category']} (confidence: {prediction['confidence']:.2f})\n")
    
    # Test batch prediction
    print("Testing batch prediction...")
    test_descriptions = test_df['description'].tolist()
    test_categories = test_df['category'].tolist()
    
    correct = 0
    total = len(test_descriptions)
    predictions = []
    
    for idx, desc in enumerate(test_descriptions):
        prediction = categorizer.categorize(desc)
        predictions.append(prediction)
        if prediction['category'] == test_categories[idx]:
            correct += 1
    
    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.4f} ({correct}/{total} correct)")
    
    # Print confusion matrix
    print("\nTop 10 test results:")
    print("--------------------------------------------------")
    print("Description  |  Actual Category  |  Predicted Category  |  Confidence")
    print("--------------------------------------------------")
    for i in range(min(10, len(test_descriptions))):
        desc = test_descriptions[i]
        actual = test_categories[i]
        pred = predictions[i]['category']
        conf = predictions[i]['confidence']
        short_desc = (desc[:30] + '...') if len(desc) > 30 else desc
        print(f"{short_desc:35} | {actual:15} | {pred:15} | {conf:.2f}")
    
    print("\n----------------------------------------------")
    print("🎉 Test completed!")
    print("----------------------------------------------\n")

if __name__ == "__main__":
    run_test() 