"""
Transformer model for high-frequency trading signal generation with <100ms prediction latency.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
import time
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .lstm_model import HFTDataset


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, d_model: int, max_seq_length: int = 1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :].transpose(0, 1)


class MultiHeadAttention(nn.Module):
    """Optimized multi-head attention for low latency."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear layer
        output = self.w_o(context)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU often works better than ReLU for transformers
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer encoder block optimized for HFT."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class FastTransformer(nn.Module):
    """
    Optimized Transformer model for high-frequency trading with low-latency inference.
    """
    
    def __init__(self, input_size: int, d_model: int = 128, num_heads: int = 8,
                 num_layers: int = 4, d_ff: int = 512, max_seq_length: int = 100,
                 dropout: float = 0.1, output_size: int = 1):
        super(FastTransformer, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Global average pooling for sequence aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for faster convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_length: int, device: str):
        """Create causal mask for autoregressive prediction."""
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
        mask = mask.masked_fill(mask == 1, False).masked_fill(mask == 0, True)
        return mask.to(device)
    
    def forward(self, x, use_causal_mask: bool = True):
        batch_size, seq_length, _ = x.size()
        
        # Input projection
        x = self.input_projection(x)  # (batch, seq_length, d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Create causal mask if needed
        mask = None
        if use_causal_mask:
            mask = self.create_causal_mask(seq_length, x.device)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_length, seq_length)
            mask = mask.expand(batch_size, self.num_heads, -1, -1)
        
        # Pass through transformer blocks
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, mask)
            attention_weights.append(attn_weights)
        
        # Aggregate sequence information
        # Use the last token for prediction (causal approach)
        x = x[:, -1, :]  # (batch, d_model)
        
        # Output projection
        x = self.output_norm(x)
        x = self.dropout(x)
        output = self.output_projection(x)
        
        return output
    
    def predict_fast(self, x):
        """Fast prediction method optimized for inference."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, use_causal_mask=False)  # Skip causal mask for speed


class TransformerTrainer:
    """Trainer for Transformer model with optimization for HFT."""
    
    def __init__(self, model: FastTransformer, device: str = 'cpu'):
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
              early_stopping_patience: int = 10, target_latency_ms: float = 100.0,
              warmup_steps: int = 1000):
        """Train the Transformer model with early stopping and latency monitoring."""
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer with warmup
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, 
                               weight_decay=1e-4, betas=(0.9, 0.95))
        
        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 0.95 ** ((step - warmup_steps) // 100)
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        step = 0
        
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
                scheduler.step()
                
                train_loss += loss.item()
                step += 1
            
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
                    outputs = self.model.predict_fast(batch_x)
                    inference_time = (time.perf_counter() - start_time) * 1000
                    val_latencies.append(inference_time)
                    
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            # Calculate average losses and latencies
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_latency = np.mean(train_latencies)
            avg_val_latency = np.mean(val_latencies)
            
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
                print(f'Warning: Validation latency ({avg_val_latency:.2f}ms) exceeds target ({target_latency_ms}ms)')
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), '/tmp/best_transformer_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('/tmp/best_transformer_model.pth'))
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
    
    def get_attention_weights(self, features: np.ndarray) -> List[np.ndarray]:
        """Get attention weights for interpretability."""
        features_scaled = self.scaler_features.transform(features)
        x = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            # Temporarily modify forward to return attention weights
            batch_size, seq_length, _ = x.size()
            x = self.model.input_projection(x)
            x = self.model.positional_encoding(x)
            
            attention_weights = []
            for transformer_block in self.model.transformer_blocks:
                x, attn_weights = transformer_block(x, None)
                attention_weights.append(attn_weights.cpu().numpy())
        
        return attention_weights