import numpy as np
import lightgbm as lgb
import pandas as pd

# Load and return your pre-trained LightGBM model
def load_pretrained_model():
    # Load the pre-trained model from a file
    print("Loading pre-trained model...")
    model = lgb.Booster(model_file="new_lgbm_model.txt")  # Loading the saved model
    return model

# Train and Return Crop recommendation model
def get_crop_model():
    path = r"data/Crop_Recommendation.csv"
    df = pd.read_csv(path)

    print("Loading crop dataset...")
    x = df.drop("label", axis=1)
    y = df["label"]

    model = lgb.LGBMClassifier()
    print("Training crop model...")
    model.fit(x, y)

    # Optional: Save the trained model if needed
    model.booster_.save_model("crop_model.txt")
    
    return model

# Define soil nutrient depletion logic for crop rotation
def update_soil_conditions(crop, conditions):
    # Example nutrient adjustments based on the crop type
    nutrient_depletion = {
        "rice": {"N": -5, "P": -3, "K": -4},
        "wheat": {"N": -4, "P": -2, "K": -3},
        "corn": {"N": -3, "P": -4, "K": -5},
        # Add more crops with their respective nutrient depletion values here
    }
    
    # Update soil conditions if crop exists in the nutrient depletion dictionary
    if crop in nutrient_depletion:
        for nutrient, change in nutrient_depletion[crop].items():
            conditions[nutrient] += change  # Adjust nutrient value
    
    return conditions

#-----------------------------------------------------------------------------------------

# Train and Return Fertilizer recommendation model
def get_fertilizer_model():
    print("Loading fertilizer dataset...")
    # Loading the downloaded dataset
    df = pd.read_csv(r"data/Fertilizer_Prediction.csv")
    df = df.rename({'Fertilizer Name': 'Fertilizer','Crop Type': 'Crop_Type','Soil Type': 'Soil_Type'}, axis=1)

    # One-hot encoding
    print("One-hot encoding...")
    categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
    categorical_features.remove('Fertilizer')

    new_encoded_columns = pd.get_dummies(df[categorical_features])
    df = pd.concat([df, new_encoded_columns], axis="columns")
    df = df.drop(categorical_features, axis="columns")

    # Prepare x and y
    print("Preparing x and y...")
    x = df.drop("Fertilizer", axis=1)
    y = df["Fertilizer"]

    # Data splitting
    print("Data splitting...")
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

    print("Training the fertilizer model...")
    model = lgb.LGBMClassifier()
    model.fit(x_train, y_train)

    # Save the model if needed
    model.booster_.save_model("fertilizer_model.txt")  # Optional: Save the trained model
    
    return model

#-----------------------------------------------------------------------------------------

def get_input(x):
    x_structure = {
        "Temparature": 0, "Humidity": 1, "Moisture": 2, "Nitrogen": 3,
        "Potassium": 4, "Phosphorous": 5, "Black": 6, "Clayey": 7, "Loamy": 8,
        "Red": 9, "Sandy": 10, "Barley": 11, "Cotton": 12, "Ground Nuts": 13, "Maize": 14,
        "Millets": 15, "Oil seeds": 16, "Paddy": 17, "Pulses": 18, "Sugarcane": 19, "Tobacco": 20,
        "Wheat": 21
    }

    output = np.zeros(len(x_structure))
    output[0] = x[0]
    output[1] = x[1]
    output[2] = x[2]
    output[3] = x[3]
    output[4] = x[4]
    output[5] = x[5]
    output[x_structure[x[6]]] = 1
    output[x_structure[x[7]]] = 1
    return output

#-----------------------------------------------------------------------------------------

# Example usage of the pre-trained model
print("Loading and using pre-trained model...")
pretrained_model = load_pretrained_model()
print("Making crop prediction...")
model = get_crop_model()
output = model.predict([[150,10,47,25.5421695,83.31883376,6.936997681,57.57343233]])
print("Predicted Crop : ",output)
