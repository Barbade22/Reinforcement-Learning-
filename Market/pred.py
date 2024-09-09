import pandas as pd
import torch
import joblib
import torch.nn as nn

# Function to prepare input for the model
def prepare_input(commodity_name, input_date, current_price, scaler):
    try:
        # Load datasets
        market_data = pd.read_csv('kalimati.csv')  # Replace with your actual market data file path
        weather_data = pd.read_csv('pune.csv')  # Replace with your actual weather data file path

        # Convert date_time to datetime format and extract date
        weather_data['date_time'] = pd.to_datetime(weather_data['date_time'])
        weather_data['date'] = weather_data['date_time'].dt.date

        # Filter weather data within the range of commodity data
        start_date = pd.to_datetime('2013-06-16').date()  # Example start date
        end_date = pd.to_datetime('2021-05-13').date()  # Example end date
        weather_data = weather_data[(weather_data['date'] >= start_date) & (weather_data['date'] <= end_date)]

        # Filter market data for the given commodity name and date range
        market_data = market_data[(market_data['Commodity'] == commodity_name) & 
                                  (pd.to_datetime(market_data['Date']).dt.date >= start_date) & 
                                  (pd.to_datetime(market_data['Date']).dt.date <= end_date)]

        if market_data.empty:
            raise ValueError(f"No data found for commodity '{commodity_name}' between {start_date} and {end_date}")

        # Merge datasets on the date column
        historical_data = pd.merge(market_data, weather_data, left_on='Date', right_on='date')

        # Check if all required columns are present
        required_columns = ['Commodity', 'Minimum', 'Maximum', 'Average', 'maxtempC', 'mintempC', 
                            'totalSnow_cm', 'sunHour', 'uvIndex', 'DewPointC', 'FeelsLikeC', 
                            'HeatIndexC', 'WindChillC', 'WindGustKmph', 'cloudcover', 'humidity', 
                            'precipMM', 'pressure', 'tempC', 'visibility', 'winddirDegree', 'windspeedKmph']
        missing_columns = [col for col in required_columns if col not in historical_data.columns]

        if missing_columns:
            raise KeyError(f"Missing columns in historical data: {missing_columns}")

        # Sort by date if not already sorted
        historical_data = historical_data.sort_values(by='Date')

        # Create past features from historical data
        past_features = historical_data[required_columns].values

        # Ensure correct shape and type for input data
        if len(past_features) == 0:
            raise ValueError("No historical data available to create input features.")

        input_data = pd.DataFrame(past_features[-5:])  # Use last 5 weeks for prediction
        input_data.columns = required_columns

        # Update the last week's Average with current_price (assuming it's the latest data)
        input_data.iloc[-1, input_data.columns.get_loc('Average')] = current_price

        # Scale the input data
        input_scaled = scaler.transform(input_data.values)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)  # Shape: (1, sequence_length, input_size)

        return input_tensor

    except KeyError as e:
        print(f"KeyError: {e}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None

# Example usage
commodity_name = 'Garlic Green'
input_date = '2014-04-02'
current_price = 55.0

scaler = joblib.load('standard_scaler.pkl')

X_input_tensor = prepare_input(commodity_name, input_date, current_price, scaler)

if X_input_tensor is not None:
    print("Input Tensor Shape:", X_input_tensor.shape)
    print("Input Tensor:", X_input_tensor)
else:
    print("Failed to prepare input data.")
