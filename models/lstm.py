import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def generate_random_walk(self, start_price, days, volatility=0.02):
        """Generate a random walk with momentum"""
        returns = np.random.normal(0, volatility, days) + 0.0001  # Slight upward bias
        price_path = start_price * np.exp(np.cumsum(returns))
        return price_path
    
    def add_patterns(self, prices, volatility):
        """Add cyclical patterns to the price series"""
        t = np.arange(len(prices))
        
        # Add various cycles
        weekly = 0.002 * np.sin(2 * np.pi * t / 5)   # 5-day cycle
        monthly = 0.004 * np.sin(2 * np.pi * t / 21)  # 21-day cycle
        
        # Combine patterns with random noise
        patterns = weekly + monthly
        noise = np.random.normal(0, volatility/2, len(prices))
        
        # Apply patterns multiplicatively
        return prices * (1 + patterns + noise)
    
    def predict(self, df, horizon):
        """Generate predictions with error handling"""
        try:
            # Calculate base volatility from historical data
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            current_price = df['close'].iloc[-1]
            
            # Generate base prediction using random walk
            base_predictions = self.generate_random_walk(
                start_price=current_price,
                days=horizon,
                volatility=volatility
            )
            
            # Add patterns
            predictions = self.add_patterns(base_predictions, volatility)
            
            # Ensure no negative prices
            predictions = np.maximum(predictions, current_price * 0.1)
            
            # Add momentum based on recent trend
            recent_trend = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) / 20
            trend_factor = 1 + np.linspace(0, recent_trend * horizon, horizon)
            predictions = predictions * trend_factor
            
            return predictions.tolist()
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            # Fallback to simple random walk
            try:
                base_price = df['close'].iloc[-1]
                predictions = [base_price]
                daily_change = 0.02  # 2% daily volatility
                
                for _ in range(horizon):
                    change = np.random.normal(0, daily_change)
                    next_price = predictions[-1] * (1 + change)
                    predictions.append(max(next_price, base_price * 0.1))
                
                return predictions[1:]  # Remove initial price
                
            except Exception as fallback_error:
                print(f"Fallback error: {str(fallback_error)}")
                # Ultimate fallback: constant price with small random variations
                base_price = 100  # Default if everything fails
                return [base_price * (1 + np.random.normal(0, 0.01)) for _ in range(horizon)]

def create_model():
    """Factory function to create a new predictor instance"""
    return StockPredictor() 