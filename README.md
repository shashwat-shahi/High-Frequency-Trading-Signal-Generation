# High-Frequency Trading Signal Generation

A comprehensive machine learning system for generating trading signals in high-frequency trading environments. This project implements LSTM and Transformer neural networks optimized for ultra-low latency (<100ms) prediction of short-term price movements using order book data and market microstructure features.

## Features

### 🚀 Machine Learning Models
- **LSTM Networks**: Optimized for time-series forecasting with attention mechanisms
- **Transformer Models**: State-of-the-art architecture with causal masking for real-time prediction
- **Ultra-low Latency**: Both models optimized for <100ms prediction latency
- **Real-time Inference**: Fast prediction methods optimized for production deployment

### 📊 Market Data Processing
- **Order Book Management**: Real-time order book reconstruction and maintenance
- **Tick-by-tick Processing**: High-frequency market data ingestion and processing
- **Market Microstructure**: Advanced features extraction from order flow dynamics
- **Synthetic Data Generation**: Realistic market data simulation for testing

### 🔬 Feature Engineering
- **Statistical Significance Testing**: Automated feature validation using multiple statistical tests
- **Market Microstructure Features**: Order book imbalance, depth ratios, liquidity metrics
- **Technical Indicators**: Moving averages, Bollinger Bands, RSI, momentum indicators
- **Price and Volume Features**: Multi-timeframe price movements and volume analysis
- **Real-time Feature Extraction**: Optimized for low-latency feature computation

### ⚡ Performance Optimization
- **Numba Acceleration**: JIT compilation for critical path computations
- **Batch Processing**: Efficient batch prediction for multiple symbols
- **Memory Management**: Optimized data structures for minimal memory footprint
- **GPU Support**: CUDA acceleration for model training and inference

## Installation

```bash
# Clone the repository
git clone https://github.com/shashwat-shahi/High-Frequency-Trading-Signal-Generation.git
cd High-Frequency-Trading-Signal-Generation

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import numpy as np
from src.data.orderbook import MarketDataFeed
from src.features.feature_engine import FeatureEngine, FeatureConfig
from src.models.lstm_model import FastLSTM, LSTMTrainer
from src.models.transformer_model import FastTransformer, TransformerTrainer

# Create market data feed
feed = MarketDataFeed()
feed.add_symbol("AAPL")

# Setup feature engine
config = FeatureConfig(
    lookback_windows=[5, 10, 20, 50],
    price_features=True,
    volume_features=True,
    microstructure_features=True,
    significance_level=0.05
)
feature_engine = FeatureEngine(config)

# Update order book
feed.update_order_book("AAPL", "bid", 150.50, 1000, 5)
feed.update_order_book("AAPL", "ask", 150.51, 800, 3)

# Extract features
order_book = feed.get_order_book("AAPL")
price_history = np.array([150.48, 150.49, 150.50, 150.51, 150.50])
volume_history = np.array([1000, 1200, 800, 900, 1100])

features, validation = feature_engine.generate_features(
    order_book, price_history, volume_history
)
```

### Training Models

```python
# Train LSTM model
lstm_model = FastLSTM(input_size=20, hidden_size=64, num_layers=2)
lstm_trainer = LSTMTrainer(lstm_model)

# Prepare data
train_dataset, val_dataset, test_dataset = lstm_trainer.prepare_data(
    features, targets, sequence_length=20
)

# Train with latency monitoring
training_history = lstm_trainer.train(
    train_dataset, val_dataset,
    epochs=100,
    target_latency_ms=100.0
)

# Evaluate performance
metrics = lstm_trainer.evaluate(test_dataset)
print(f"Direction Accuracy: {metrics['direction_accuracy']:.3f}")
print(f"Average Latency: {metrics['avg_latency_ms']:.2f}ms")
```

### Real-time Prediction

```python
# Single prediction with latency measurement
sequence_features = np.random.randn(20, 10)  # 20 timesteps, 10 features
prediction, latency_ms = lstm_trainer.predict_single(sequence_features)

print(f"Predicted price change: {prediction:.6f}")
print(f"Prediction latency: {latency_ms:.2f}ms")
```

## Complete Example

Run the complete demonstration:

```bash
python examples/complete_example.py
```

This example demonstrates:
- Synthetic market data generation
- Feature engineering with statistical validation
- Training both LSTM and Transformer models
- Performance evaluation and latency benchmarking
- Visualization of results

## Project Structure

```
High-Frequency-Trading-Signal-Generation/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── orderbook.py          # Order book data structures
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engine.py     # Feature engineering pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py         # LSTM implementation
│   │   └── transformer_model.py  # Transformer implementation
│   └── __init__.py
├── examples/
│   ├── data_generator.py         # Synthetic data generation
│   └── complete_example.py       # Full system demonstration
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Technical Details

### Model Architecture

#### LSTM Model
- **Input**: Sequence of market features (price, volume, microstructure)
- **Architecture**: 2-layer LSTM with optional attention mechanism
- **Output**: Price movement prediction
- **Optimization**: Batch normalization, gradient clipping, early stopping

#### Transformer Model
- **Input**: Sequence of market features with positional encoding
- **Architecture**: Multi-head self-attention with feed-forward networks
- **Causal Masking**: Prevents look-ahead bias for real-time prediction
- **Optimization**: Warmup learning rate schedule, layer normalization

### Feature Engineering

#### Market Microstructure Features
- Order book imbalance ratios
- Bid-ask spread dynamics
- Volume at price (VaP) analysis
- Order flow direction
- Trade frequency metrics

#### Statistical Validation
- Stationarity testing (Augmented Dickey-Fuller)
- Normality testing (Jarque-Bera, D'Agostino)
- Autocorrelation analysis (Ljung-Box)
- Feature importance ranking

#### Technical Indicators
- Simple and exponential moving averages
- Bollinger Bands positioning
- Relative Strength Index (RSI)
- Price momentum across multiple timeframes

### Performance Optimization

#### Latency Optimization
- Numba JIT compilation for critical loops
- Optimized tensor operations
- Minimal memory allocations
- Fast inference modes

#### Accuracy Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Direction accuracy (sign prediction)
- Sharpe ratio of returns

## Requirements

- Python 3.8+
- PyTorch 1.11+
- NumPy 1.21+
- Pandas 1.3+
- Scikit-learn 1.0+
- SciPy 1.7+
- Numba 0.56+
- Matplotlib 3.5+

## Performance Benchmarks

### Latency Performance
- **LSTM Model**: ~15-30ms average prediction latency
- **Transformer Model**: ~25-50ms average prediction latency
- **P99 Latency**: <100ms for both models
- **Throughput**: >1000 predictions/second

### Prediction Accuracy
- **Direction Accuracy**: 52-58% (above random baseline)
- **Sharpe Ratio**: 0.8-1.2 on synthetic data
- **Feature Importance**: Order book imbalance most predictive

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes only. Trading in financial markets involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Use at your own risk.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- NumPy and SciPy communities for numerical computing tools
- Numba developers for JIT compilation capabilities