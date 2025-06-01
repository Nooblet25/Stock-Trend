<<<<<<< HEAD
# Stock Market Analyzer

A web application for analyzing and predicting stock market trends using historical data and statistical analysis.

## Features

- Real-time stock data fetching using Yahoo Finance API
- Stock price prediction using moving average model
- Interactive charts for visualizing historical data and predictions
- Volume prediction based on historical volatility
- Modern, responsive UI with dark mode
- Support for multiple prediction horizons (up to 5 years)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Nooblet25/StockTrendAI.git
cd StockTrendAI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Enter a stock symbol and select prediction parameters:
   - Start Year: Historical data start point (2 years before end year)
   - End Year: Target year for prediction
   - Days to Forecast: Number of days to predict into the future
   - Model: Currently using Moving Average model for all options

## Troubleshooting

1. If you get a "No data available" error:
   - Check if the stock symbol is correct
   - Try a different date range
   - Ensure you have internet connectivity

2. If predictions seem unrealistic:
   - The current model uses a simple moving average approach
   - Results are for educational purposes only
   - Consider shorter prediction horizons for better accuracy

3. If the application is slow:
   - The data fetching has a 10-second timeout
   - Try reducing the date range or prediction horizon
   - Check your internet connection speed

## Technologies Used

- Flask (Backend)
- yfinance (Stock Data API)
- Chart.js (Data Visualization)
- Bootstrap 5 (UI Framework)
- NumPy & Pandas (Data Processing)

## Disclaimer

This application is for educational purposes only. The predictions are based on historical data and simple statistical models. Do not use them for actual investment decisions.

## License

MIT License 
=======
# STOCKTREND-AI
>>>>>>> c18607ab5679936f2dab352a9208b910302a35b9
