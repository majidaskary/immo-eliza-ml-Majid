import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define the path for saving the preprocessed dataset  
advance_preprocessed_dataset = "C:/Users/becod/AI/my-projects/immo-eliza-ml-Majid/immo-eliza-ml-Majid/advance_preprocessed_dataset.csv"

dataset_for_model = pd.read_csv(advance_preprocessed_dataset)
#--------------------------------------------------------------------------
# Keep only rows with price less than or equal to 500,000 euros
#dataset_for_model = dataset_for_model[dataset_for_model['price'] <= 500_000]
#--------------------------------------------------------------------------
# Separate data for apartments and houses
apartment_data = dataset_for_model[dataset_for_model['property_type'] == 'APARTMENT']
house_data = dataset_for_model[dataset_for_model['property_type'] == 'HOUSE']

#--------------------------------------------------------------------------
apartment_features = [
    'fl_furnished', 'terrace_sqm', 'total_area_sqm', 'nbr_bedrooms',
    'total_area_per_bedroom_scaled', 'kitchen_type_encoded', 'epc_encoded',
    'Bulding_sta_encoded', 'latitude', 'longitude',  'zip_code_cut', 'construction_category'
]
house_features = [
    'surface_land_sqm', 'garden_sqm', 'nbr_frontages', 'total_area_sqm',
    'nbr_bedrooms', 'total_area_per_bedroom_scaled', 'kitchen_type_encoded',
    'epc_encoded', 'Bulding_sta_encoded', 'latitude', 'longitude', 'zip_code_cut', 'construction_category'
]



# Function to build, train, and evaluate an ANN model with hyperparameters
def train_and_evaluate_ann(data, features, target='price', model_type='Apartment'):
    if data.empty:
        print(f"No data available for {model_type.lower()}s.")
        return None
    
    # Separate features and target variable
    X = data[features]
    y = data[target]
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the feature set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Build the ANN model
    model = Sequential([
        #Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),  # Dropout for regularization
        #Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)  # Output layer with one neuron for regression
    ])
    
    # Compile the model with a specific optimizer and loss function
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    
    # Train the model
    #model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1)
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=1)

    # Evaluate the model
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Compute scores
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print model performance
    print(f"\n{model_type} Model Performance:")
    print(f"Training Mean Squared Error: {train_mse:.8f}")
    print(f"Testing Mean Squared Error: {test_mse:.8f}")
    print(f"Training R-squared Score: {train_r2:.8f}")
    print(f"Testing R-squared Score: {test_r2:.8f}")
    
    # Plot actual vs. predicted values for the test set
    plt.scatter(y_test, y_test_pred, alpha=0.7, label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Ideal Line')  # Ideal line
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f'Actual vs. Predicted Prices ({model_type} Test Set)')
    plt.grid(True)
    plt.legend()
    plt.show()

# Train and evaluate the ANN model for apartments
train_and_evaluate_ann(apartment_data, apartment_features, model_type='Apartment')

# Train and evaluate the ANN model for houses
train_and_evaluate_ann(house_data, house_features, model_type='House')