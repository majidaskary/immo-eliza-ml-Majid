import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import pickle
import os
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Canvas, Frame, Scrollbar

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor


advance_preprocessed_dataset = "C:/Users/becod/AI/my-projects/immo-eliza-ml-Majid/immo-eliza-ml-Majid/advance_preprocessed_dataset.csv"

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


class ModelTrainer:
    def __init__(self, dataset_path, apartment_features, house_features,max_price):
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(dataset_path)
        # Apply price limitation if specified
        if max_price is not None:
            self.dataset = self.dataset[self.dataset['price'] <= max_price]
        self.apartment_data = self.dataset[self.dataset['property_type'] == 'APARTMENT']
        self.house_data = self.dataset[self.dataset['property_type'] == 'HOUSE']
        self.apartment_features = apartment_features
        self.house_features = house_features
        self.models = {}
        self.trained_models = {}  # Dictionary to store trained models
        self.results = []  # List to store all results
        self.apartment_results = []  # List to store results for apartments
        self.house_results = []  # List to store results for houses
        self.predictions_data = {'Apartment': [], 'House': []}  # To store prediction data for plotting

    def add_model(self, model_name, model_instance):
        """Adds a model to the model dictionary."""
        self.models[model_name] = model_instance

    def train_and_evaluate_model(self, data, features, target='price', model_name='Model', property_type='Property'):
        """Trains and evaluates a model and collects predictions for plotting."""
        if model_name not in self.models:
            print(f"Model '{model_name}' not found. Please add it first.")
            return None

        if data.empty:
            print(f"No data available for {property_type.lower()}s.")
            return None

        # Separate features and target variable
        X = data[features]
        y = data[target]

        # Split data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Get the model
        regressor = self.models[model_name]

        # Train the model on the training set
        regressor.fit(X_train, y_train)

        # Save the trained model in the dictionary with its property type and name
        self.trained_models[(model_name, property_type)] = regressor

        # Calculate R-squared scores for training and testing sets
        train_score = regressor.score(X_train, y_train)
        test_score = regressor.score(X_test, y_test)

        # Calculate additional metrics after making predictions
        y_test_pred = regressor.predict(X_test)
        mae = np.mean(np.abs(y_test - y_test_pred))
        mse = np.mean((y_test - y_test_pred) ** 2)

        # Store results in a dictionary and append to the results list
        result = {
            'Model': model_name,
            'Property Type': property_type,
            'Training score': train_score,
            'Testing score': test_score,
            'MAE': mae,
            'MSE': mse
        }
        
        self.results.append(result)  # Append all results to results list
        if property_type == 'Apartment':
            self.apartment_results.append(result) # Append results to apartment list
        elif property_type == 'House':
            self.house_results.append(result) # Append results to house list

        # Append predictions and actuals to predictions_data for plotting later
        self.predictions_data[property_type].append({
            'model_name': model_name,
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'train_score': train_score,
            'test_score': test_score,
            'mae': mae,
            'mse': mse
        })
 
        # Print model performance
        print(f"\n{property_type} ({model_name}) Model Performance:")
        print(f"Training R-squared Score: {train_score:.8f}")
        print(f"Testing R-squared Score: {test_score:.8f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")


    def run_all_models(self):
        """Trains and evaluates all models for both apartments and houses."""
        for model_name in self.models.keys():
            print(f"Training and evaluating {model_name} for Apartments...")
            self.train_and_evaluate_model(self.apartment_data, self.apartment_features, model_name=model_name, property_type='Apartment')

            print(f"Training and evaluating {model_name} for Houses...")
            self.train_and_evaluate_model(self.house_data, self.house_features, model_name=model_name, property_type='House')

        # Print saved trained models
        print("\nTrained Models Saved:")
        for (model_name, property_type), model_instance in self.trained_models.items():
            print(f"Model: {model_name}, Property Type: {property_type} - {model_instance}")

        # Return results for external handling
        return self.results, self.apartment_results, self.house_results

# Example usage
trainer = ModelTrainer(advance_preprocessed_dataset, apartment_features, house_features, max_price=3500000)

#Add models to the trainer
trainer.add_model('Linear Regression', LinearRegression(fit_intercept=True, n_jobs=-1))
# trainer.add_model('Decision Tree', DecisionTreeRegressor(max_depth=8, min_samples_split=4, random_state=42))
# trainer.add_model('Random Forest Regressor', RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=4, random_state=42, n_jobs=-1))
# trainer.add_model('Gradient Boosting Regressor', GradientBoostingRegressor(n_estimators=150, learning_rate=0.07, max_depth=4, min_samples_split=4, random_state=42))
# trainer.add_model('XGB Regressor',XGBRegressor(n_estimators=200,learning_rate=0.05, max_depth=4, min_child_weight=6,subsample=0.8, colsample_bytree=0.8, random_state=42))
trainer.add_model('gb.LGBM Regressor', lgb.LGBMRegressor(num_leaves=31, learning_rate=0.03, n_estimators=200, max_depth=7, min_child_samples=5, subsample=0.8, colsample_bytree=0.8, random_state=42))
# trainer.add_model('CatBoost Regresso', CatBoostRegressor(iterations=200, learning_rate=0.03, depth=7, l2_leaf_reg=3, random_seed=42,verbose=0))

# Run training and evaluation for all models added to the trainer
results, apartment_results, house_results = trainer.run_all_models()

# Now the trained models are stored in `trainer.trained_models`
# Print and save the results summary outside the class
results_df = pd.DataFrame(results)
print("\nSummary of Model Performances:")
print(results_df)

# Convert the results lists for apartments and houses to DataFrames for easier sorting and display
apartment_results_df = pd.DataFrame(trainer.apartment_results)
house_results_df = pd.DataFrame(trainer.house_results)

# Sort the DataFrames based on the 'Testing score' column (or another metric if desired)
apartment_results_df = apartment_results_df.sort_values(by='Testing score', ascending=False)
house_results_df = house_results_df.sort_values(by='Testing score', ascending=False)

# Print the sorted summaries for Apartments and Houses
print("\nSummary of Model Performances for Apartments (sorted by Testing score):")
print(apartment_results_df)

print("\nSummary of Model Performances for Houses (sorted by Testing score):")
print(house_results_df)


# Function to create and display charts in a scrollable Tkinter window
def display_scrollable_charts(predictions_data):
    root = Tk()
    root.title("Scrollable Model Performance Charts")

    # Create a canvas with a scrollbar
    canvas = Canvas(root, width=1000, height=600)
    scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create charts in pairs for each model (Apartment and House)
    model_names = [model_data['model_name'] for model_data in predictions_data['Apartment']]
    for model_idx, model_name in enumerate(model_names):
        # Create a new frame for each row
        row_frame = Frame(scrollable_frame)
        row_frame.pack(fill="x", padx=10, pady=10)
        
        for idx, property_type in enumerate(['Apartment', 'House']):
            model_data = predictions_data[property_type][model_idx]
            fig, ax = plt.subplots(figsize=(5, 5))  # Fixed size for each plot
            y_test = model_data['y_test']
            y_test_pred = model_data['y_test_pred']

            # Scatter plot and regression line
            ax.scatter(y_test, y_test_pred, alpha=0.7, label="Predicted Points")
            m, b = np.polyfit(y_test, y_test_pred, 1)
            ax.plot(y_test, m * y_test + b, color='blue', linestyle='-', linewidth=2, label="Model Line")

            ax.set_xlabel('Actual Prices')
            ax.set_ylabel('Predicted Prices')
            ax.set_title(f"{model_data['model_name']} - {property_type}")
            ax.legend()

            # Integrate Matplotlib figure into Tkinter
            canvas_fig = FigureCanvasTkAgg(fig, master=row_frame)
            canvas_fig.draw()
            canvas_fig.get_tk_widget().pack(side="left", padx=10)

            plt.close(fig)  # Close the figure to avoid display outside Tkinter

            # Display model metrics below the chart using a Label
            metrics_text = (f"{model_data['model_name']} \n {property_type}\n rcd Train Score: {model_data['train_score']:.6f}\n  Test Score: {model_data['test_score']:.6f}\n"
                            f"MAE: {model_data['mae']:.2f}\n  MSE: {model_data['mse']:.2e}")
            metrics_label = Label(row_frame, text=metrics_text, justify="center", font=("Arial", 10))
            metrics_label.pack(side="left", padx=10)

    # Pack the canvas and scrollbar
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    root.mainloop()

# Call function to display scrollable charts
display_scrollable_charts(trainer.predictions_data)
