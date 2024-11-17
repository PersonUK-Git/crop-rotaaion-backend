
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware
from models import get_crop_model,get_fertilizer_model,get_input, update_soil_conditions

import google.generativeai as genai
import os

description = """
### Crop Recommendation JSON Input 
    { "array": [N,P,K,temperature,humidity,ph,rainfall] }
### Fertilizer Recommendation JSON Input 
    { "array": [Temparature,Humidity,Moisture,Nitrogen,Potassium,Phosphorous,Soil Type,Crop Type] }
"""

app = FastAPI(description=description)

# ------------------------------------------

# Enabling CORS policy

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained models
crop_model = get_crop_model()
fertilizer_model = get_fertilizer_model()

#--------------------------------------------------------------------

class InputArray(BaseModel):
    array: List[Union[float, str]] = Field(..., description="An array of floats or strings")


def get_crop_info_from_gemini(predicted_crop):
    genai.configure(api_key='AIzaSyCC4m4NoNGu3QgHv7pXjz4QMnWXzOUsr7I')
    try:
        # Make the API request to generate content about the predicted crop
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Tell me facts about {predicted_crop} and its current market rates.")
        
        # Return the generated content (crop facts and market rates)
        print(response.text)
        return response.text
    except Exception as e:
        return f"Error fetching data from Google Generative AI: {str(e)}"
    
@app.post("/crop_recommend")
def array_endpoint(request: Request, input: InputArray):
    input_values = [float(value) for value in input.array]
    print("Making Initial Crop Prediction...")
    
    # Predict the first crop
    prediction = crop_model.predict([input_values])[0]
    crop_info = get_crop_info_from_gemini(prediction)
    
    # Initialize soil conditions dictionary
    soil_conditions = {
        'N': input_values[0],
        'P': input_values[1],
        'K': input_values[2],
        'temperature': input_values[3],
        'humidity': input_values[4],
        'pH': input_values[5],
        'rainfall': input_values[6]
    }
    
    # Update soil conditions based on the first crop prediction
    updated_conditions = update_soil_conditions(prediction, soil_conditions)

    # Predict the next crop using updated soil conditions
    next_prediction = crop_model.predict([[updated_conditions['N'], updated_conditions['P'], 
                                           updated_conditions['K'], updated_conditions['temperature'], 
                                           updated_conditions['humidity'], updated_conditions['pH'], 
                                           updated_conditions['rainfall']]])[0]
    
    # Return both predictions for the initial and rotated crop
    print("Returning Crop Rotation Prediction..., ", prediction, next_prediction)
    #return {"initial_crop": prediction, "next_crop": next_prediction}
    # print(crop_info)
    return(prediction)


@app.post("/fertilizer_recommend")
def array_endpoint(request: Request, input: InputArray):
    
    # process the input array
    x = get_input(input.array)
    
    print("Making Fertilizer Prediction...")
    # Make a prediction using the input array
    prediction = fertilizer_model.predict([x])
    prediction = prediction[0]
    
    print("Returning Response...")    
    # Return the prediction in the response
    return prediction

if __name__ == '__main__':
    uvicorn.run(app, host="192.168.0.229", port=8080)
    
    # changed 127.0.0.1 to 0.0.0.0 for railway.app deployment
    # you can go to "/docs" or "/redoc" endpoint to get the API documentation
    
    # CLI command
    # uvicorn app:app --host 0.0.0.0 --port 8080

