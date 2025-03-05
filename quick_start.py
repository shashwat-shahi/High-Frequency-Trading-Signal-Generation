#!/usr/bin/env python3
"""
Quick start script for the HFT signal generation system.
This script demonstrates basic functionality without running the full training pipeline.
"""
import numpy as np
import torch
from src.data.orderbook import MarketDataFeed
from src.features.feature_engine import FeatureEngine, FeatureConfig
from src.models.lstm_model import FastLSTM, LSTMTrainer
from examples.data_generator import SyntheticMarketDataGenerator


def quick_demo():
    """Run a quick demonstration of the system."""
    print("High-Frequency Trading Signal Generation - Quick Demo")
    print("=" * 60)
    
    # 1. Create synthetic market data
    print("\n1. Generating synthetic market data...")
    generator = SyntheticMarketDataGenerator(symbol="AAPL", initial_price=150.0)
    
    # Generate price path
    n_steps = 200
    prices = generator.generate_price_path(n_steps)
    print(f"   Generated {n_steps} price points")
    print(f"   Price range: ${np.min(prices):.2f} - ${np.max(prices):.2f}")
    
    # 2. Setup order book
    print("\n2. Setting up order book...")
    feed = MarketDataFeed()
    feed.add_symbol("AAPL")
    
    # Simulate order book updates
    for i, price in enumerate(prices[:50]):  # Use first 50 prices
        bids, asks = generator.generate_order_book_levels(price)
        
        # Update order book with some levels
        for j, (bid_price, bid_qty, bid_orders) in enumerate(bids[:5]):
            feed.update_order_book("AAPL", "bid", bid_price, bid_qty, bid_orders)
        
        for j, (ask_price, ask_qty, ask_orders) in enumerate(asks[:5]):
            feed.update_order_book("AAPL", "ask", ask_price, ask_qty, ask_orders)
    
    order_book = feed.get_order_book("AAPL")
    best_bid, best_ask = order_book.get_best_bid_ask()
    spread = order_book.get_spread()
    
    print(f"   Best bid: ${best_bid:.2f}")
    print(f"   Best ask: ${best_ask:.2f}")
    print(f"   Spread: ${spread:.4f}")
    
    # 3. Feature extraction
    print("\n3. Extracting features...")
    config = FeatureConfig(
        lookback_windows=[5, 10, 20],
        price_features=True,
        volume_features=True,
        microstructure_features=True
    )
    feature_engine = FeatureEngine(config)
    
    # Create dummy volume data
    volumes = np.random.lognormal(6, 1, len(prices))
    
    features, validation = feature_engine.generate_features(
        order_book, prices, volumes[:len(prices)]
    )
    
    print(f"   Extracted {len(features)} features")
    print(f"   Sample features: {list(features.keys())[:5]}...")
    
    # 4. Create and test model
    print("\n4. Creating LSTM model...")
    
    # Create sample feature matrix
    feature_matrix = np.random.randn(100, 10)  # 100 samples, 10 features
    targets = np.random.randn(100)  # Random targets
    
    # Create model
    model = FastLSTM(input_size=10, hidden_size=32, num_layers=1, dropout=0.1)
    trainer = LSTMTrainer(model, device='cpu')
    
    print(f"   Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test single prediction
    test_sequence = np.random.randn(20, 10)  # 20 timesteps, 10 features
    
    # Scale the data first
    trainer.scaler_features.fit(feature_matrix)
    trainer.scaler_targets.fit(targets.reshape(-1, 1))
    
    prediction, latency = trainer.predict_single(test_sequence)
    
    print(f"   Sample prediction: {prediction:.6f}")
    print(f"   Prediction latency: {latency:.2f}ms")
    
    # 5. Latency test
    print("\n5. Latency benchmark...")
    latencies = []
    for _ in range(100):
        test_seq = np.random.randn(20, 10)
        _, lat = trainer.predict_single(test_seq)
        latencies.append(lat)
    
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    under_100ms = np.mean(np.array(latencies) < 100) * 100
    
    print(f"   Average latency: {avg_latency:.2f}ms")
    print(f"   P95 latency: {p95_latency:.2f}ms")
    print(f"   Under 100ms: {under_100ms:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    print("✓ Market data generation: Working")
    print("✓ Order book management: Working")
    print("✓ Feature extraction: Working")
    print("✓ LSTM model: Working")
    print(f"✓ Latency requirement (<100ms): {'PASSED' if p95_latency < 100 else 'NEEDS OPTIMIZATION'}")
    print("\nThe HFT signal generation system is ready for use!")
    print("Run 'python examples/complete_example.py' for full training demo.")


if __name__ == "__main__":
    quick_demo()