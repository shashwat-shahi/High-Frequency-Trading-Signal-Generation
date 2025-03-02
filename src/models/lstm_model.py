"""
LSTM model for high-frequency trading signal generation with <100ms prediction latency.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings


class HFTDataset(Dataset):
    """Dataset for high-frequency trading time series data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 sequence_length: int, prediction_horizon: int = 1):
        """
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            targets: Target values of shape (n_samples,)
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps ahead to predict
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Ensure we have enough data
        assert len(features) >= sequence_length + prediction_horizon
        
    def __len__(self):
        return len(self.features) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx):
        # Input sequence
        x = self.features[idx:idx + self.sequence_length]
        
        # Target (prediction_horizon steps ahead)
        y = self.targets[idx + self.sequence_length + self.prediction_horizon - 1]
        
        return x, y


class FastLSTM(nn.Module):
    """
    Optimized LSTM model for high-frequency trading with low-latency inference.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, output_size: int = 1, use_attention: bool = False):
        super(FastLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Input normalization layer
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # Unidirectional for lower latency
        )
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.activation = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for faster convergence."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        
        # Normalize input
        x_norm = x.transpose(1, 2)  # (batch, features, seq_len)
        x_norm = self.input_norm(x_norm)
        x_norm = x_norm.transpose(1, 2)  # (batch, seq_len, features)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x_norm)
        
        # Apply attention if enabled
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = lstm_out + attn_out  # Residual connection
        
        # Use the last output for prediction
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Fully connected layers
        output = self.dropout(last_output)
        output = self.activation(self.fc1(output))
        output = self.fc2(output)
        
        return output
    
    def predict_fast(self, x):
        """Fast prediction method optimized for inference."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class LSTMTrainer:
    """Trainer for LSTM model with optimization for HFT."""
    
    def __init__(self, model: FastLSTM, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.scaler_features = StandardScaler()
        self.scaler_targets = StandardScaler()
        self.training_history = []
        
    def prepare_data(self, features: np.ndarray, targets: np.ndarray, 
                    sequence_length: int, prediction_horizon: int = 1,
                    train_split: float = 0.8, validation_split: float = 0.1) -> Tuple:
        """Prepare and scale data for training."""
        
        # Scale features and targets
        features_scaled = self.scaler_features.fit_transform(features)
        targets_scaled = self.scaler_targets.fit_transform(targets.reshape(-1, 1)).flatten()
        
        # Create datasets
        dataset = HFTDataset(features_scaled, targets_scaled, sequence_length, prediction_horizon)
        
        # Split data
        n_samples = len(dataset)
        train_size = int(train_split * n_samples)
        val_size = int(validation_split * n_samples)
        test_size = n_samples - train_size - val_size
        
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
        test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, n_samples))
        
        return train_dataset, val_dataset, test_dataset
    
    def train(self, train_dataset, val_dataset, epochs: int = 100, 
              batch_size: int = 32, learning_rate: float = 0.001,
              early_stopping_patience: int = 10, target_latency_ms: float = 100.0):
        """Train the LSTM model with early stopping and latency monitoring."""
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_latencies = []
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Measure inference time
                start_time = time.perf_counter()
                outputs = self.model(batch_x)
                inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
                train_latencies.append(inference_time)
                
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_latencies = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    # Measure inference time
                    start_time = time.perf_counter()
                    outputs = self.model(batch_x)
                    inference_time = (time.perf_counter() - start_time) * 1000
                    val_latencies.append(inference_time)
                    
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            # Calculate average losses and latencies
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_latency = np.mean(train_latencies)
            avg_val_latency = np.mean(val_latencies)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Store training history
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_latency_ms': avg_train_latency,
                'val_latency_ms': avg_val_latency,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_stats)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}:')
                print(f'  Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
                print(f'  Train Latency: {avg_train_latency:.2f}ms, Val Latency: {avg_val_latency:.2f}ms')
                print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
            
            # Check latency constraint
            if avg_val_latency > target_latency_ms:
                warnings.warn(f'Validation latency ({avg_val_latency:.2f}ms) exceeds target ({target_latency_ms}ms)')
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), '/tmp/best_lstm_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('/tmp/best_lstm_model.pth'))
        print(f'Training completed. Best validation loss: {best_val_loss:.6f}')
        
        return self.training_history
    
    def evaluate(self, test_dataset, batch_size: int = 32) -> Dict[str, float]:
        """Evaluate the model on test data."""
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        predictions = []
        actuals = []
        latencies = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Measure inference time
                start_time = time.perf_counter()
                outputs = self.model.predict_fast(batch_x)
                inference_time = (time.perf_counter() - start_time) * 1000
                latencies.append(inference_time)
                
                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(batch_y.cpu().numpy())
        
        # Convert back to original scale
        predictions = np.array(predictions).reshape(-1, 1)
        actuals = np.array(actuals).reshape(-1, 1)
        
        predictions_original = self.scaler_targets.inverse_transform(predictions).flatten()
        actuals_original = self.scaler_targets.inverse_transform(actuals).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(actuals_original, predictions_original)
        mae = mean_absolute_error(actuals_original, predictions_original)
        rmse = np.sqrt(mse)
        
        # Calculate direction accuracy (for price movement prediction)
        direction_actual = np.sign(actuals_original)
        direction_predicted = np.sign(predictions_original)
        direction_accuracy = np.mean(direction_actual == direction_predicted)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy,
            'avg_latency_ms': np.mean(latencies),
            'max_latency_ms': np.max(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99)
        }
        
        return metrics
    
    def predict_single(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Make a single prediction with latency measurement.
        
        Args:
            features: Feature array of shape (sequence_length, n_features)
            
        Returns:
            Tuple of (prediction, latency_ms)
        """
        # Scale features
        features_scaled = self.scaler_features.transform(features)
        
        # Convert to tensor
        x = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Measure inference time
        start_time = time.perf_counter()
        
        self.model.eval()
        with torch.no_grad():
            prediction_scaled = self.model.predict_fast(x)
        
        inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Convert back to original scale
        prediction = self.scaler_targets.inverse_transform(
            prediction_scaled.cpu().numpy().reshape(-1, 1)
        )[0, 0]
        
        return prediction, inference_time
    
    def get_feature_importance(self, test_dataset, method: str = 'permutation') -> Dict[str, float]:
        """Calculate feature importance using permutation importance."""
        if method != 'permutation':
            raise NotImplementedError("Only permutation importance is implemented")
        
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        # Get baseline performance
        baseline_metrics = self.evaluate(test_dataset)
        baseline_mse = baseline_metrics['mse']
        
        # Get a single batch with all test data
        for batch_x, batch_y in test_loader:
            original_x = batch_x.clone()
            break
        
        n_features = original_x.shape[2]
        importance_scores = {}
        
        for feature_idx in range(n_features):
            # Create permuted version
            permuted_x = original_x.clone()
            
            # Permute the feature across all samples and time steps
            perm_indices = torch.randperm(permuted_x.shape[0])
            permuted_x[:, :, feature_idx] = permuted_x[perm_indices, :, feature_idx]
            
            # Create temporary dataset with permuted features
            temp_dataset = torch.utils.data.TensorDataset(permuted_x, batch_y)
            
            # Evaluate with permuted feature
            permuted_metrics = self.evaluate(temp_dataset)
            permuted_mse = permuted_metrics['mse']
            
            # Calculate importance as increase in MSE
            importance = (permuted_mse - baseline_mse) / baseline_mse
            importance_scores[f'feature_{feature_idx}'] = importance
        
        return importance_scores