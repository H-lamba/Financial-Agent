import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import os
import joblib
import torch.optim as optim    
SEQ_LENGTH = 90
EPOCHS = 120
BATCHSIZE = 60
LEARNING_RATE = 0.001


class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=128, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(
                            input_size,
                            hidden_size=hidden_layer_size,
                            num_layers=2,
                            dropout=0.2,
                            batch_first=True
                        )
        self.linear = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        last_time_step = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step)
        return predictions
    
def train_model(ticker):
    print(f"Training model for {ticker}")
    file_path = f"data/{ticker}_with_sentiment.csv"
    if not os.path.exists(file_path):
        print("Error, file path don't exist, perform the process_data.py")
        return
    
    df  = pd.read_csv(file_path)
    data = df[['close', 'sentiment_score', 'sma_20', 'sma_50', 'rsi']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    joblib.dump(scaler, f"models/{ticker}_scaler.pkl")
    
    X, y = [],[]
    for i in range(len(scaled_data)-SEQ_LENGTH):
        X.append(scaled_data[i:i+SEQ_LENGTH])
        y.append(scaled_data[i+SEQ_LENGTH, 0])
        
    X, y = np.array(X), np.array(y)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).view(-1,1)
    
    split_idx = int(len(X) * 0.8)
    
    X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCHSIZE, shuffle=False)
    
    model = LSTMModel()
    criterion = nn.MSELoss() # Loss function (Mean Squared Error)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("🏋️ Training the model (This might take a minute)...")
    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            
        if (epoch+1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.6f}")

    print("✅ Training Complete!")
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), f"models/{ticker}_lstm.pth")
    print(f"💾 Model saved to models/{ticker}_lstm.pth")
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test).numpy()
        
    dummy_matrix = np.zeros((len(test_predictions), 5)) 
    dummy_matrix[:, 0] = test_predictions.flatten()
    real_predictions = scaler.inverse_transform(dummy_matrix)[:, 0]
    
    dummy_matrix_y = np.zeros((len(y_test), 5))
    dummy_matrix_y[:, 0] = y_test.numpy().flatten()
    real_actuals = scaler.inverse_transform(dummy_matrix_y)[:, 0]
    rmse = np.sqrt(np.mean((real_predictions - real_actuals) ** 2))
    print(f"📉 RMSE: {rmse:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(real_actuals, label='Actual Price', color='blue')
    plt.plot(real_predictions, label='AI Predicted Price', color='orange')
    plt.title(f"{ticker} Price Prediction (LSTM)")
    plt.xlabel("Days")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.savefig(f"data/{ticker}_prediction.png")
    print(f"📊 Prediction chart saved to data/{ticker}_prediction.png")
def prepare_ticker_data(ticker, start_date="2020-01-01"):
    """
    Fetch and prepare data for a ticker if it doesn't exist.
    Returns True if data is ready, False if failed.
    """
    from process_data import get_merged_data
    import pandas as pd
    
    file_path = f"data/{ticker}_with_sentiment.csv"
    
    if os.path.exists(file_path):
        print(f"✅ Data already exists for {ticker}")
        return True
    
    print(f"📊 Fetching data for new ticker: {ticker}...")
    try:
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        df = get_merged_data(ticker, start_date, end_date)
        
        if df is None or df.empty:
            print(f"❌ Could not fetch data for {ticker}. Check if ticker is valid.")
            return False
        
        # Ensure data directory exists
        if not os.path.exists("data"):
            os.makedirs("data")
            
        df.to_csv(file_path, index=False)
        print(f"✅ Data saved to {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
        return False


def predict_next_day(ticker, auto_train=True):
    """
    Predicts tomorrow's price and calculates a Confidence Interval 
    based on the model's recent accuracy (last 30 days).
    
    Args:
        ticker (str): Stock ticker symbol
        auto_train (bool): If True, automatically fetch data and train model if not found
        
    Returns:
        float: Predicted price, or None if prediction failed
    """
    print(f"\n🔮 Predicting next day price for {ticker}...")
    
    file_path = f"data/{ticker}_with_sentiment.csv"
    model_path = f"models/{ticker}_lstm.pth"
    scaler_path = f"models/{ticker}_scaler.pkl"
    
    # --- EDGE CASE 1: Data file doesn't exist ---
    if not os.path.exists(file_path):
        if auto_train:
            print(f"⚠️ No data found for {ticker}. Fetching automatically...")
            if not prepare_ticker_data(ticker):
                return None
        else:
            print(f"❌ Data file not found: {file_path}")
            return None
    
    # --- EDGE CASE 2: Model or scaler doesn't exist ---
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        if auto_train:
            print(f"⚠️ No trained model found for {ticker}. Training now...")
            try:
                train_model(ticker)
            except Exception as e:
                print(f"❌ Training failed: {e}")
                return None
        else:
            print(f"❌ Model not found. Please train the model first.")
            return None

    # --- Load and validate data ---
    try:
        df = pd.read_csv(file_path)
        required_cols = ['close', 'sentiment_score', 'sma_20', 'sma_50', 'rsi']
        
        # EDGE CASE 3: Missing required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Data file is missing columns: {missing_cols}")
            return None
            
        # EDGE CASE 4: Not enough data for prediction
        if len(df) < SEQ_LENGTH:
            print(f"❌ Not enough data for prediction. Need at least {SEQ_LENGTH} rows, got {len(df)}")
            return None
            
        data = df[required_cols].values
        
    except Exception as e:
        print(f"❌ Error reading data file: {e}")
        return None
    
    # --- Load scaler and model ---
    try:
        scaler = joblib.load(scaler_path)
        scaled_data = scaler.transform(data)
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        return None
    
    try:
        model = LSTMModel()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    print("   Calculating confidence interval (checking last 30 days)...")
    past_errors = []
    days_to_check = 60
    if len(scaled_data) < SEQ_LENGTH + days_to_check:
        print("⚠️ Not enough data to calculate confidence interval.")
        interval = 0.0
    else:
        for i in range(days_to_check):
            current_idx = len(scaled_data) - days_to_check + i
            seq = scaled_data[current_idx - SEQ_LENGTH : current_idx]
            input_tensor = torch.FloatTensor(seq).unsqueeze(0)
            with torch.no_grad():
                pred_scaled = model(input_tensor).item()
            dummy = np.zeros((1, 5))
            dummy[0, 0] = pred_scaled
            pred_price = scaler.inverse_transform(dummy)[0, 0]
            actual_price = df['close'].iloc[current_idx]
            error = abs(pred_price - actual_price)
            past_errors.append(error)
        recent_rmse = np.sqrt(np.mean(np.square(past_errors)))      
        interval = recent_rmse

    last_sequence = scaled_data[-SEQ_LENGTH:]
    input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0)
    
    with torch.no_grad():
        predicted_scaled = model(input_tensor).item()
        
    dummy_row = np.zeros((1, 5))
    dummy_row[0, 0] = predicted_scaled
    predicted_price = scaler.inverse_transform(dummy_row)[0, 0]
    
    last_price = df['close'].iloc[-1]
    print("\n" + "="*40)
    print(f"📉 Last Closing Price:   ${last_price:.2f}")
    print(f"🚀 Model Prediction:     ${predicted_price:.2f}")
    print(f"⚖️  Confidence Interval:  ${predicted_price:.2f} ± ${interval:.2f}")
    print(f"RANGE: ${predicted_price - interval:.2f} TO ${predicted_price + interval:.2f}")
    print("="*40)
    
    if predicted_price > last_price:
        print("🟢 Signal: BULLISH (Price expected to rise)")
    else:
        print("🔴 Signal: BEARISH (Price expected to fall)")

    return predicted_price
if __name__ == "__main__":
    train_model("ITC.NS")
    predict_next_day("ITC.NS")   