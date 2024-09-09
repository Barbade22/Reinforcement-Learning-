import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the datasets
weather_df = pd.read_csv('pune.csv')
commodity_df = pd.read_csv('kalimati.csv')

# Convert the date_time to datetime format and extract date
weather_df['date_time'] = pd.to_datetime(weather_df['date_time'])
weather_df['date'] = weather_df['date_time'].dt.date

# Filter weather data within the range of commodity data
weather_df = weather_df[(weather_df['date'] >= pd.to_datetime('2013-06-16').date()) & 
                        (weather_df['date'] <= pd.to_datetime('2021-05-13').date())]

# Aggregate weather data to daily data
daily_weather = weather_df.groupby('date').agg({
    'maxtempC': 'mean',
    'mintempC': 'mean',
    'totalSnow_cm': 'sum',
    'sunHour': 'sum',
    'uvIndex': 'mean',
    'DewPointC': 'mean',
    'FeelsLikeC': 'mean',
    'HeatIndexC': 'mean',
    'WindChillC': 'mean',
    'WindGustKmph': 'mean',
    'cloudcover': 'mean',
    'humidity': 'mean',
    'precipMM': 'sum',
    'pressure': 'mean',
    'tempC': 'mean',
    'visibility': 'mean',
    'winddirDegree': 'mean',
    'windspeedKmph': 'mean'
}).reset_index()

# Convert commodity Date to datetime and then to date
commodity_df['Date'] = pd.to_datetime(commodity_df['Date']).dt.date

# Ensure the correct column names for price data
price_column = 'Average'
if price_column not in commodity_df.columns:
    raise KeyError(f"Column '{price_column}' not found in commodity data")

# Save unique commodity names to a file for later use
commodity_names = commodity_df['Commodity'].unique()
commodity_mapping = {name: idx for idx, name in enumerate(commodity_names)}
inverse_commodity_mapping = {idx: name for name, idx in commodity_mapping.items()}
joblib.dump(inverse_commodity_mapping, 'commodity_mapping.pkl')

# Map commodity names to integers
commodity_df['Commodity'] = commodity_df['Commodity'].map(commodity_mapping)

# Merge weather data with commodity data on date
merged_df = pd.merge(commodity_df, daily_weather, left_on='Date', right_on='date', how='inner')
merged_df.drop(columns=['date'], inplace=True)

# Select features and target
features = ['Commodity', 'Minimum', 'Maximum','Average',' maxtempC', 'mintempC', 'totalSnow_cm', 'sunHour', 'uvIndex', 'DewPointC',
            'FeelsLikeC', 'HeatIndexC', 'WindChillC', 'WindGustKmph', 'cloudcover', 'humidity', 'precipMM',
            'pressure', 'tempC', 'visibility', 'winddirDegree', 'windspeedKmph']
X = merged_df[features].values
y = merged_df[price_column].values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for time series prediction
def create_sequences(features, target, seq_length=5):
    X_seq, y_seq = [], []
    for i in range(len(features) - seq_length):
        X_seq.append(features[i:i+seq_length])
        y_seq.append(target[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

seq_length = 5  # Weekly data means sequence length of 5
X_seq, y_seq = create_sequences(X_scaled, y, seq_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# Define dataset and data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Transformer model with enhanced architecture
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, output_size, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.fc1 = nn.Linear(input_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc2 = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.fc1(src)  # Linear projection to d_model
        src = src.unsqueeze(0)  # Add sequence length dimension (seq_length=1)
        src = src.permute(1, 0, 2)  # Shape: (seq_length=1, batch_size, d_model)
        out = self.transformer_encoder(src)  # Shape: (seq_length=1, batch_size, d_model)
        out = out.permute(1, 0, 2)  # Reshape to (batch_size, seq_length=1, d_model)
        out = out.squeeze(1)  # Remove seq_length dimension (now shape is batch_size, d_model)
        out = self.dropout(out)
        out = self.fc2(out)  # Shape: (batch_size, output_size)
        return out

# Hyperparameters
input_size = X_train.shape[2]  # Number of features per time step
d_model = 256  # Dimensionality of the model (usually a power of 2)
nhead = 8  # Number of attention heads
num_encoder_layers = 6  # Number of Transformer encoder layers
output_size = 1  # Output size (single scalar prediction)
dropout = 0.2  # Dropout rate for regularization

# Instantiate the model, define the loss function and optimizer
model = TransformerModel(input_size, d_model, nhead, num_encoder_layers, output_size, dropout).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
eval_after = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    if epoch % eval_after == 0:
        print(f'Epoch {epoch}/{num_epochs}, Loss: {avg_loss}')

# Evaluation
model.eval()
test_loss = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f'Test Loss: {avg_test_loss}')

# Save the model state
torch.save(model.state_dict(), 'transformer_model.pth')

# Save the scaler object to a file
scaler_path = 'standard_scaler.pkl'
joblib.dump(scaler, scaler_path)
