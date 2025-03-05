"""
Sample data generator for testing the HFT signal generation system.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List
import time
from src.data.orderbook import MarketDataFeed, OrderBook, Tick


class SyntheticMarketDataGenerator:
    """Generate synthetic market data for testing HFT models."""
    
    def __init__(self, symbol: str = "AAPL", initial_price: float = 150.0, 
                 volatility: float = 0.02, tick_size: float = 0.01):
        self.symbol = symbol
        self.initial_price = initial_price
        self.volatility = volatility
        self.tick_size = tick_size
        self.current_price = initial_price
        self.current_time = time.time()
        
    def generate_price_path(self, n_steps: int, dt: float = 0.001) -> np.ndarray:
        """Generate realistic price path using geometric Brownian motion with jumps."""
        prices = np.zeros(n_steps)
        prices[0] = self.initial_price
        
        for i in range(1, n_steps):
            # Geometric Brownian motion
            dW = np.random.normal(0, np.sqrt(dt))
            drift = -0.5 * self.volatility**2 * dt  # Risk-neutral drift
            diffusion = self.volatility * dW
            
            # Add occasional jumps (rare events)
            if np.random.random() < 0.001:  # 0.1% chance of jump
                jump = np.random.normal(0, 0.005)  # 0.5% jump
                diffusion += jump
            
            # Update price
            log_return = drift + diffusion
            prices[i] = prices[i-1] * np.exp(log_return)
            
            # Round to tick size
            prices[i] = round(prices[i] / self.tick_size) * self.tick_size
        
        return prices
    
    def generate_order_book_levels(self, mid_price: float, depth: int = 10) -> Tuple[List, List]:
        """Generate realistic bid and ask levels around mid price."""
        spread = max(self.tick_size, mid_price * 0.0001)  # Minimum spread
        
        bids = []
        asks = []
        
        # Generate bid levels (below mid price)
        for i in range(depth):
            price = mid_price - spread/2 - i * self.tick_size
            # Quantity decreases with distance from mid
            base_qty = 1000
            qty = base_qty * np.exp(-i * 0.2) * (0.5 + np.random.random())
            orders = max(1, int(qty / 100))
            bids.append((price, qty, orders))
        
        # Generate ask levels (above mid price)
        for i in range(depth):
            price = mid_price + spread/2 + i * self.tick_size
            base_qty = 1000
            qty = base_qty * np.exp(-i * 0.2) * (0.5 + np.random.random())
            orders = max(1, int(qty / 100))
            asks.append((price, qty, orders))
        
        return bids, asks
    
    def generate_trade_flow(self, prices: np.ndarray, intensity: float = 10.0) -> List[Tick]:
        """Generate realistic trade flow."""
        trades = []
        
        for i, price in enumerate(prices):
            timestamp = self.current_time + i * 0.001  # 1ms intervals
            
            # Generate trades based on Poisson process
            n_trades = np.random.poisson(intensity * 0.001)  # Expected trades per ms
            
            for _ in range(n_trades):
                # Trade price with some noise around mid price
                trade_price = price + np.random.normal(0, price * 0.0001)
                trade_price = round(trade_price / self.tick_size) * self.tick_size
                
                # Trade size
                trade_size = np.random.lognormal(mean=4, sigma=1)  # Log-normal distribution
                trade_size = max(1, int(trade_size))
                
                # Trade direction (buy/sell)
                side = 'buy' if np.random.random() > 0.5 else 'sell'
                
                tick = Tick(
                    timestamp=timestamp + np.random.random() * 0.001,
                    symbol=self.symbol,
                    price=trade_price,
                    quantity=trade_size,
                    side=side,
                    trade_id=f"T{len(trades)}"
                )
                trades.append(tick)
        
        # Sort by timestamp
        trades.sort(key=lambda x: x.timestamp)
        return trades
    
    def generate_market_session(self, duration_seconds: int = 3600, 
                              update_frequency_ms: int = 100) -> Tuple[List, List]:
        """Generate a complete market session with order book updates and trades."""
        n_updates = int(duration_seconds * 1000 / update_frequency_ms)
        
        # Generate price path
        prices = self.generate_price_path(n_updates, dt=update_frequency_ms/1000)
        
        # Generate order book updates
        order_book_updates = []
        for i, mid_price in enumerate(prices):
            timestamp = self.current_time + i * (update_frequency_ms / 1000)
            bids, asks = self.generate_order_book_levels(mid_price)
            
            update = {
                'timestamp': timestamp,
                'symbol': self.symbol,
                'mid_price': mid_price,
                'bids': bids,
                'asks': asks
            }
            order_book_updates.append(update)
        
        # Generate trades
        trades = self.generate_trade_flow(prices, intensity=5.0)
        
        return order_book_updates, trades


def create_sample_dataset(n_samples: int = 10000, sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Create a sample dataset for model training."""
    
    # Generate synthetic market data
    generator = SyntheticMarketDataGenerator()
    order_book_updates, trades = generator.generate_market_session(
        duration_seconds=n_samples//10,  # Adjust duration for desired samples
        update_frequency_ms=10
    )
    
    # Create market data feed
    feed = MarketDataFeed()
    feed.add_symbol("AAPL")
    
    # Process updates
    features_list = []
    prices = []
    
    for update in order_book_updates:
        # Update order book
        symbol = update['symbol']
        
        # Add bid levels
        for price, quantity, orders in update['bids']:
            feed.update_order_book(symbol, 'bid', price, quantity, orders)
        
        # Add ask levels  
        for price, quantity, orders in update['asks']:
            feed.update_order_book(symbol, 'ask', price, quantity, orders)
        
        # Extract features
        order_book = feed.get_order_book(symbol)
        if order_book:
            features = order_book.to_dict()
            
            # Convert to numeric features
            feature_vector = [
                features.get('best_bid', 0),
                features.get('best_ask', 0),
                features.get('spread', 0),
                features.get('mid_price', 0),
                features.get('weighted_mid_price', 0),
                features.get('imbalance', 0),
                # Add volume features
                sum(qty for price, qty, orders in features['bids'][:5]),
                sum(qty for price, qty, orders in features['asks'][:5]),
                # Add price level features
                len(features['bids']),
                len(features['asks'])
            ]
            
            features_list.append(feature_vector)
            prices.append(features.get('mid_price', 0))
    
    # Convert to numpy arrays
    features_array = np.array(features_list)
    prices_array = np.array(prices)
    
    # Create targets (future price changes)
    targets = np.diff(prices_array)  # Price changes
    features_array = features_array[:-1]  # Align with targets
    
    # Ensure we have enough data
    if len(features_array) < sequence_length:
        raise ValueError(f"Not enough data generated. Need at least {sequence_length} samples.")
    
    return features_array, targets


if __name__ == "__main__":
    # Generate sample data
    print("Generating sample dataset...")
    features, targets = create_sample_dataset(n_samples=5000)
    print(f"Generated {len(features)} samples with {features.shape[1]} features")
    
    # Save sample data
    np.save('/tmp/sample_features.npy', features)
    np.save('/tmp/sample_targets.npy', targets)
    print("Sample data saved to /tmp/sample_features.npy and /tmp/sample_targets.npy")