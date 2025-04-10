import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import logging
import joblib

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        # Convert to numpy array first to avoid the slow tensor creation warning
        seq = np.array(self.sequences[idx])
        target = np.array([self.targets[idx]])
        return torch.FloatTensor(seq), torch.FloatTensor(target)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class LSTMPredictor:
    def __init__(self, user_id=None, sequence_length=30, n_features=1):
        self.user_id = user_id
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_path = os.path.join('models', f'lstm_model_{user_id}.pt') if user_id else None
        self.scaler_path = os.path.join('models', f'lstm_scaler_{user_id}.joblib') if user_id else None
        
    def _prepare_sequences(self, data):
        """Prepare sequences for LSTM input"""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:(i + self.sequence_length)])
            targets.append(data[i + self.sequence_length])
            
        return np.array(sequences), np.array(targets)
    
    def _prepare_data(self, df):
        """Prepare data for LSTM training"""
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        # Sort by date
        df = df.sort_values('date')
        
        # Create daily totals
        daily_totals = df.groupby('date')['amount'].sum().reset_index()
        
        # Fill missing dates with 0
        date_range = pd.date_range(start=daily_totals['date'].min(), 
                                 end=daily_totals['date'].max(), 
                                 freq='D')
        daily_totals = daily_totals.set_index('date').reindex(date_range, fill_value=0).reset_index()
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(daily_totals[['amount']])
        
        return scaled_data
    
    def train(self, df, epochs=50, batch_size=32, learning_rate=0.001):
        """Train the LSTM model"""
        try:
            # Prepare data
            scaled_data = self._prepare_data(df)
            
            # Create sequences
            X, y = self._prepare_sequences(scaled_data)
            
            # Reshape X to include the feature dimension
            X = X.reshape((X.shape[0], X.shape[1], self.n_features))
            
            # Create dataset and dataloader
            dataset = TimeSeriesDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Initialize model
            self.model = LSTMModel(input_size=self.n_features)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    # Reshape batch_y to match outputs shape
                    batch_y = batch_y.squeeze(-1)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader)
                if (epoch + 1) % 10 == 0:
                    logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            
            # Save the model and scaler
            if self.model_path:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
            
            return {'loss': avg_loss}
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            raise
    
    def predict(self, df, days_ahead=30):
        """Make predictions for future days"""
        try:
            if not self.model:
                if self.model_path and os.path.exists(self.model_path):
                    self.model = LSTMModel(input_size=self.n_features)
                    self.model.load_state_dict(torch.load(self.model_path))
                else:
                    raise ValueError("Model not trained or loaded")
            
            # Prepare data
            scaled_data = self._prepare_data(df)
            
            # Get the last sequence
            last_sequence = scaled_data[-self.sequence_length:]
            predictions = []
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                for _ in range(days_ahead):
                    # Prepare input - reshape to (batch_size, sequence_length, n_features)
                    x_input = torch.FloatTensor(last_sequence).reshape(1, -1, self.n_features)
                    
                    # Make prediction
                    y_pred = self.model(x_input)
                    
                    # Update sequence
                    last_sequence = np.append(last_sequence[1:], y_pred.numpy())
                    
                    # Inverse transform prediction
                    predictions.append(self.scaler.inverse_transform(y_pred.numpy())[0][0])
            
            # Create prediction dates (without time component)
            last_date = df['date'].max()
            prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                          periods=days_ahead,
                                          freq='D').date
            
            return pd.DataFrame({
                'date': prediction_dates,
                'predicted_amount': predictions
            })
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def evaluate(self, df):
        """Evaluate model performance"""
        try:
            if not self.model:
                if self.model_path and os.path.exists(self.model_path):
                    self.model = LSTMModel(input_size=self.n_features)
                    self.model.load_state_dict(torch.load(self.model_path))
                else:
                    raise ValueError("Model not trained or loaded")
            
            # Prepare data
            scaled_data = self._prepare_data(df)
            
            # Create sequences
            X, y = self._prepare_sequences(scaled_data)
            
            # Reshape X to include the feature dimension
            X = X.reshape((X.shape[0], X.shape[1], self.n_features))
            
            # Create dataset
            dataset = TimeSeriesDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            # Make predictions
            self.model.eval()
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch_X, batch_y in dataloader:
                    outputs = self.model(batch_X)
                    predictions.extend(outputs.numpy())
                    actuals.extend(batch_y.squeeze(-1).numpy())
            
            # Inverse transform
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            actuals = self.scaler.inverse_transform(np.array(actuals).reshape(-1, 1))
            
            # Calculate metrics
            mse = np.mean((actuals - predictions) ** 2)
            mae = np.mean(np.abs(actuals - predictions))
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(np.sqrt(mse))
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise 