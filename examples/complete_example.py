"""
Complete example demonstrating the HFT signal generation system.
This example shows how to:
1. Generate synthetic market data
2. Extract features with statistical validation
3. Train LSTM and Transformer models
4. Evaluate performance with <100ms latency constraint
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from src.data.orderbook import MarketDataFeed, OrderBook
from src.features.feature_engine import FeatureEngine, FeatureConfig
from src.models.lstm_model import FastLSTM, LSTMTrainer
from src.models.transformer_model import FastTransformer, TransformerTrainer
from examples.data_generator import SyntheticMarketDataGenerator, create_sample_dataset


def setup_feature_engine() -> FeatureEngine:
    """Setup feature engine with appropriate configuration."""
    config = FeatureConfig(
        lookback_windows=[5, 10, 20],
        price_features=True,
        volume_features=True,
        microstructure_features=True,
        technical_features=True,
        statistical_features=True,
        significance_level=0.05
    )
    return FeatureEngine(config)


def generate_comprehensive_features(n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate comprehensive features using the feature engine."""
    print("Generating synthetic market data...")
    
    # Create market data generator
    generator = SyntheticMarketDataGenerator(symbol="AAPL", initial_price=150.0)
    order_book_updates, trades = generator.generate_market_session(
        duration_seconds=n_samples//5,
        update_frequency_ms=50
    )
    
    # Setup feature engine
    feature_engine = setup_feature_engine()
    
    # Process market data and extract features
    feed = MarketDataFeed()
    feed.add_symbol("AAPL")
    
    all_features = []
    price_history = []
    volume_history = []
    
    print("Processing market data and extracting features...")
    for i, update in enumerate(order_book_updates):
        if i % 500 == 0:
            print(f"Processed {i}/{len(order_book_updates)} updates")
            
        symbol = update['symbol']
        
        # Update order book
        for price, quantity, orders in update['bids']:
            feed.update_order_book(symbol, 'bid', price, quantity, orders)
        
        for price, quantity, orders in update['asks']:
            feed.update_order_book(symbol, 'ask', price, quantity, orders)
        
        # Accumulate history
        mid_price = update['mid_price']
        total_volume = sum(qty for _, qty, _ in update['bids'][:5]) + sum(qty for _, qty, _ in update['asks'][:5])
        
        price_history.append(mid_price)
        volume_history.append(total_volume)
        
        # Extract features after we have enough history
        if len(price_history) >= 50:
            order_book = feed.get_order_book(symbol)
            
            features, validation = feature_engine.generate_features(
                order_book, 
                np.array(price_history), 
                np.array(volume_history),
                validate=False  # Skip validation for speed
            )
            
            # Convert to feature vector
            feature_vector = []
            for key in sorted(features.keys()):
                value = features[key]
                if not np.isnan(value) and not np.isinf(value):
                    feature_vector.append(value)
                else:
                    feature_vector.append(0.0)  # Fill with 0 for invalid values
            
            all_features.append(feature_vector)
    
    # Convert to numpy arrays
    features_array = np.array(all_features)
    price_array = np.array(price_history[50:])  # Align with features
    
    # Create targets (future price movements)
    future_horizon = 5  # Predict 5 steps ahead
    targets = []
    for i in range(len(price_array) - future_horizon):
        current_price = price_array[i]
        future_price = price_array[i + future_horizon]
        # Target is the log return
        if current_price > 0:
            target = np.log(future_price / current_price)
        else:
            target = 0.0
        targets.append(target)
    
    # Align features with targets
    features_array = features_array[:len(targets)]
    targets_array = np.array(targets)
    
    print(f"Generated {len(features_array)} samples with {features_array.shape[1]} features")
    return features_array, targets_array


def train_and_evaluate_lstm(features: np.ndarray, targets: np.ndarray, 
                           sequence_length: int = 20) -> Dict:
    """Train and evaluate LSTM model."""
    print("\n=== Training LSTM Model ===")
    
    # Create model
    input_size = features.shape[1]
    model = FastLSTM(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        output_size=1,
        use_attention=False
    )
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = LSTMTrainer(model, device)
    
    # Prepare data
    train_dataset, val_dataset, test_dataset = trainer.prepare_data(
        features, targets, sequence_length, prediction_horizon=1
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Train model
    training_history = trainer.train(
        train_dataset, val_dataset,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=10,
        target_latency_ms=100.0
    )
    
    # Evaluate model
    metrics = trainer.evaluate(test_dataset, batch_size=32)
    
    print(f"\nLSTM Model Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    return {
        'model': trainer,
        'metrics': metrics,
        'training_history': training_history
    }


def train_and_evaluate_transformer(features: np.ndarray, targets: np.ndarray, 
                                 sequence_length: int = 20) -> Dict:
    """Train and evaluate Transformer model."""
    print("\n=== Training Transformer Model ===")
    
    # Create model
    input_size = features.shape[1]
    model = FastTransformer(
        input_size=input_size,
        d_model=128,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        max_seq_length=sequence_length,
        dropout=0.1,
        output_size=1
    )
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = TransformerTrainer(model, device)
    
    # Prepare data
    train_dataset, val_dataset, test_dataset = trainer.prepare_data(
        features, targets, sequence_length, prediction_horizon=1
    )
    
    # Train model
    training_history = trainer.train(
        train_dataset, val_dataset,
        epochs=50,
        batch_size=16,  # Smaller batch size for transformer
        learning_rate=0.0001,
        early_stopping_patience=10,
        target_latency_ms=100.0,
        warmup_steps=500
    )
    
    # Evaluate model
    metrics = trainer.evaluate(test_dataset, batch_size=16)
    
    print(f"\nTransformer Model Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    return {
        'model': trainer,
        'metrics': metrics,
        'training_history': training_history
    }


def benchmark_latency(lstm_trainer, transformer_trainer, features: np.ndarray, 
                     sequence_length: int = 20, n_predictions: int = 1000) -> Dict:
    """Benchmark prediction latency for both models."""
    print("\n=== Latency Benchmarking ===")
    
    # Prepare test sequences
    test_sequences = []
    for i in range(n_predictions):
        start_idx = np.random.randint(0, len(features) - sequence_length)
        sequence = features[start_idx:start_idx + sequence_length]
        test_sequences.append(sequence)
    
    # Benchmark LSTM
    print("Benchmarking LSTM latency...")
    lstm_latencies = []
    for sequence in test_sequences:
        _, latency = lstm_trainer.predict_single(sequence)
        lstm_latencies.append(latency)
    
    # Benchmark Transformer
    print("Benchmarking Transformer latency...")
    transformer_latencies = []
    for sequence in test_sequences:
        _, latency = transformer_trainer.predict_single(sequence)
        transformer_latencies.append(latency)
    
    # Calculate statistics
    lstm_stats = {
        'mean_ms': np.mean(lstm_latencies),
        'std_ms': np.std(lstm_latencies),
        'p95_ms': np.percentile(lstm_latencies, 95),
        'p99_ms': np.percentile(lstm_latencies, 99),
        'max_ms': np.max(lstm_latencies),
        'under_100ms_pct': np.mean(np.array(lstm_latencies) < 100) * 100
    }
    
    transformer_stats = {
        'mean_ms': np.mean(transformer_latencies),
        'std_ms': np.std(transformer_latencies),
        'p95_ms': np.percentile(transformer_latencies, 95),
        'p99_ms': np.percentile(transformer_latencies, 99),
        'max_ms': np.max(transformer_latencies),
        'under_100ms_pct': np.mean(np.array(transformer_latencies) < 100) * 100
    }
    
    print("\nLatency Benchmarking Results:")
    print(f"LSTM Model:")
    for metric, value in lstm_stats.items():
        print(f"  {metric}: {value:.2f}")
    
    print(f"\nTransformer Model:")
    for metric, value in transformer_stats.items():
        print(f"  {metric}: {value:.2f}")
    
    return {
        'lstm': lstm_stats,
        'transformer': transformer_stats,
        'lstm_latencies': lstm_latencies,
        'transformer_latencies': transformer_latencies
    }


def create_performance_plots(lstm_results: Dict, transformer_results: Dict, 
                           latency_results: Dict):
    """Create performance visualization plots."""
    print("\nCreating performance plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss comparison
    axes[0, 0].plot([h['epoch'] for h in lstm_results['training_history']], 
                   [h['val_loss'] for h in lstm_results['training_history']], 
                   label='LSTM', marker='o', markersize=3)
    axes[0, 0].plot([h['epoch'] for h in transformer_results['training_history']], 
                   [h['val_loss'] for h in transformer_results['training_history']], 
                   label='Transformer', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Validation Loss')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Latency comparison
    models = ['LSTM', 'Transformer']
    mean_latencies = [latency_results['lstm']['mean_ms'], latency_results['transformer']['mean_ms']]
    p99_latencies = [latency_results['lstm']['p99_ms'], latency_results['transformer']['p99_ms']]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, mean_latencies, width, label='Mean Latency', alpha=0.8)
    axes[0, 1].bar(x + width/2, p99_latencies, width, label='P99 Latency', alpha=0.8)
    axes[0, 1].axhline(y=100, color='r', linestyle='--', label='100ms Target')
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Latency (ms)')
    axes[0, 1].set_title('Prediction Latency Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models)
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Accuracy comparison
    lstm_accuracy = lstm_results['metrics']['direction_accuracy']
    transformer_accuracy = transformer_results['metrics']['direction_accuracy']
    
    accuracies = [lstm_accuracy, transformer_accuracy]
    axes[1, 0].bar(models, accuracies, color=['blue', 'green'], alpha=0.7)
    axes[1, 0].axhline(y=0.5, color='r', linestyle='--', label='Random Baseline')
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Direction Accuracy')
    axes[1, 0].set_title('Price Direction Prediction Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Latency distribution
    axes[1, 1].hist(latency_results['lstm_latencies'], bins=50, alpha=0.7, 
                   label='LSTM', density=True)
    axes[1, 1].hist(latency_results['transformer_latencies'], bins=50, alpha=0.7, 
                   label='Transformer', density=True)
    axes[1, 1].axvline(x=100, color='r', linestyle='--', label='100ms Target')
    axes[1, 1].set_xlabel('Latency (ms)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Latency Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/tmp/hft_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("Performance plots saved to /tmp/hft_performance_analysis.png")


def main():
    """Main function to run the complete HFT signal generation example."""
    print("High-Frequency Trading Signal Generation System Demo")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        # Generate comprehensive features
        features, targets = generate_comprehensive_features(n_samples=2000)
        
        # Remove any remaining NaN or inf values
        valid_mask = np.isfinite(features).all(axis=1) & np.isfinite(targets)
        features = features[valid_mask]
        targets = targets[valid_mask]
        
        print(f"Final dataset: {len(features)} samples with {features.shape[1]} features")
        
        if len(features) < 100:
            print("Not enough valid samples generated. Please check data generation.")
            return
        
        sequence_length = 20
        
        # Train and evaluate LSTM
        lstm_results = train_and_evaluate_lstm(features, targets, sequence_length)
        
        # Train and evaluate Transformer
        transformer_results = train_and_evaluate_transformer(features, targets, sequence_length)
        
        # Benchmark latency
        latency_results = benchmark_latency(
            lstm_results['model'], 
            transformer_results['model'], 
            features, 
            sequence_length,
            n_predictions=500
        )
        
        # Create performance plots
        create_performance_plots(lstm_results, transformer_results, latency_results)
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        print(f"\nLSTM Model:")
        print(f"  Direction Accuracy: {lstm_results['metrics']['direction_accuracy']:.3f}")
        print(f"  RMSE: {lstm_results['metrics']['rmse']:.6f}")
        print(f"  Average Latency: {latency_results['lstm']['mean_ms']:.2f}ms")
        print(f"  P99 Latency: {latency_results['lstm']['p99_ms']:.2f}ms")
        print(f"  Under 100ms: {latency_results['lstm']['under_100ms_pct']:.1f}%")
        
        print(f"\nTransformer Model:")
        print(f"  Direction Accuracy: {transformer_results['metrics']['direction_accuracy']:.3f}")
        print(f"  RMSE: {transformer_results['metrics']['rmse']:.6f}")
        print(f"  Average Latency: {latency_results['transformer']['mean_ms']:.2f}ms")
        print(f"  P99 Latency: {latency_results['transformer']['p99_ms']:.2f}ms")
        print(f"  Under 100ms: {latency_results['transformer']['under_100ms_pct']:.1f}%")
        
        # Check if latency requirements are met
        lstm_meets_req = latency_results['lstm']['p99_ms'] < 100
        transformer_meets_req = latency_results['transformer']['p99_ms'] < 100
        
        print(f"\nLatency Requirements (<100ms P99):")
        print(f"  LSTM: {'✓ PASSED' if lstm_meets_req else '✗ FAILED'}")
        print(f"  Transformer: {'✓ PASSED' if transformer_meets_req else '✗ FAILED'}")
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()