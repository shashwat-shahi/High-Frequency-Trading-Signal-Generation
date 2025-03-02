"""
Order book data structures for high-frequency trading.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class OrderBookLevel:
    """Represents a single level in the order book."""
    price: float
    quantity: float
    orders: int = 1


@dataclass
class Tick:
    """Represents a single market tick."""
    timestamp: float
    symbol: str
    price: float
    quantity: float
    side: str  # 'buy' or 'sell'
    trade_id: Optional[str] = None


class OrderBook:
    """
    Real-time order book implementation optimized for high-frequency trading.
    Maintains bid and ask levels with fast updates and feature extraction.
    """
    
    def __init__(self, symbol: str, max_levels: int = 10):
        self.symbol = symbol
        self.max_levels = max_levels
        self.bids: List[OrderBookLevel] = []  # Sorted descending by price
        self.asks: List[OrderBookLevel] = []  # Sorted ascending by price
        self.last_update: float = 0
        self.tick_history: deque = deque(maxlen=1000)  # Keep last 1000 ticks
        
    def update_level(self, side: str, price: float, quantity: float, orders: int = 1):
        """Update a single level in the order book."""
        self.last_update = time.time()
        
        levels = self.bids if side == 'bid' else self.asks
        reverse = side == 'bid'
        
        # Find insertion point
        level = OrderBookLevel(price, quantity, orders)
        
        # Remove level if quantity is 0
        if quantity == 0:
            levels[:] = [l for l in levels if l.price != price]
            return
            
        # Update existing level or insert new one
        updated = False
        for i, existing_level in enumerate(levels):
            if existing_level.price == price:
                levels[i] = level
                updated = True
                break
            elif (not reverse and price < existing_level.price) or \
                 (reverse and price > existing_level.price):
                levels.insert(i, level)
                updated = True
                break
                
        if not updated:
            levels.append(level)
            
        # Sort and limit levels
        levels.sort(key=lambda x: x.price, reverse=reverse)
        if len(levels) > self.max_levels:
            levels[:] = levels[:self.max_levels]
    
    def add_tick(self, tick: Tick):
        """Add a trade tick to the history."""
        self.tick_history.append(tick)
        
    def get_best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices."""
        best_bid = self.bids[0].price if self.bids else None
        best_ask = self.asks[0].price if self.asks else None
        return best_bid, best_ask
        
    def get_spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return best_ask - best_bid
        return None
        
    def get_mid_price(self) -> Optional[float]:
        """Calculate mid price."""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None
        
    def get_weighted_mid_price(self) -> Optional[float]:
        """Calculate quantity-weighted mid price."""
        if not self.bids or not self.asks:
            return None
            
        bid_qty = self.bids[0].quantity
        ask_qty = self.asks[0].quantity
        bid_price = self.bids[0].price
        ask_price = self.asks[0].price
        
        total_qty = bid_qty + ask_qty
        if total_qty == 0:
            return (bid_price + ask_price) / 2
            
        return (bid_price * ask_qty + ask_price * bid_qty) / total_qty
    
    def get_volume_at_level(self, level: int, side: str) -> float:
        """Get volume at specific level (0-indexed)."""
        levels = self.bids if side == 'bid' else self.asks
        if level < len(levels):
            return levels[level].quantity
        return 0.0
        
    def get_order_book_imbalance(self, levels: int = 5) -> float:
        """Calculate order book imbalance."""
        bid_volume = sum(self.get_volume_at_level(i, 'bid') for i in range(min(levels, len(self.bids))))
        ask_volume = sum(self.get_volume_at_level(i, 'ask') for i in range(min(levels, len(self.asks))))
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
            
        return (bid_volume - ask_volume) / total_volume
    
    def to_dict(self) -> Dict:
        """Convert order book state to dictionary for feature extraction."""
        return {
            'symbol': self.symbol,
            'timestamp': self.last_update,
            'bids': [(l.price, l.quantity, l.orders) for l in self.bids],
            'asks': [(l.price, l.quantity, l.orders) for l in self.asks],
            'best_bid': self.bids[0].price if self.bids else None,
            'best_ask': self.asks[0].price if self.asks else None,
            'spread': self.get_spread(),
            'mid_price': self.get_mid_price(),
            'weighted_mid_price': self.get_weighted_mid_price(),
            'imbalance': self.get_order_book_imbalance()
        }


class MarketDataFeed:
    """
    High-performance market data feed that maintains multiple order books
    and provides real-time updates.
    """
    
    def __init__(self):
        self.order_books: Dict[str, OrderBook] = {}
        self.callbacks: List = []
        
    def add_symbol(self, symbol: str, max_levels: int = 10):
        """Add a new symbol to track."""
        self.order_books[symbol] = OrderBook(symbol, max_levels)
        
    def update_order_book(self, symbol: str, side: str, price: float, 
                         quantity: float, orders: int = 1):
        """Update order book for a symbol."""
        if symbol not in self.order_books:
            self.add_symbol(symbol)
            
        self.order_books[symbol].update_level(side, price, quantity, orders)
        
        # Trigger callbacks
        for callback in self.callbacks:
            callback(symbol, self.order_books[symbol])
    
    def add_trade(self, symbol: str, price: float, quantity: float, side: str):
        """Add a trade tick."""
        if symbol not in self.order_books:
            self.add_symbol(symbol)
            
        tick = Tick(
            timestamp=time.time(),
            symbol=symbol,
            price=price,
            quantity=quantity,
            side=side
        )
        
        self.order_books[symbol].add_tick(tick)
        
    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get order book for a symbol."""
        return self.order_books.get(symbol)
        
    def register_callback(self, callback):
        """Register callback for order book updates."""
        self.callbacks.append(callback)