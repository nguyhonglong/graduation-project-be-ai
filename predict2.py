import torch
import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Depends, status
from pydantic import BaseModel
from typing import List, Dict, Union
import numpy as np
from datetime import datetime, timedelta
import json
from fastapi.responses import FileResponse, JSONResponse
from sklearn.preprocessing import StandardScaler
from main1 import TimeSeriesTransformer, load_data
from contextlib import asynccontextmanager
import joblib
from pathlib import Path
from fastapi.security.api_key import APIKeyHeader, APIKey
import requests
from io import BytesIO

# Add API key configuration
API_KEY = "12345"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header or api_key_header != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Could not validate API key"
        )
    return api_key_header

# Add these URLs - replace with your actual S3 or cloud storage URLs
MODEL_URLS = {
    'transformer': "https://datn-nhl.s3.ap-northeast-1.amazonaws.com/predict/best_model.pth",
    'health': "https://datn-nhl.s3.ap-northeast-1.amazonaws.com/health_index/xgboost.joblib",
    'life_expectation': "https://datn-nhl.s3.ap-northeast-1.amazonaws.com/life_expectation/xgboost.joblib",
    'scaler1': "https://datn-nhl.s3.ap-northeast-1.amazonaws.com/health_index/scaler1.joblib",
    'scaler2': "https://datn-nhl.s3.ap-northeast-1.amazonaws.com/health_index/scaler2.joblib",
    'scaler3': "https://datn-nhl.s3.ap-northeast-1.amazonaws.com/life_expectation/scaler1.joblib",
    'scaler4': "https://datn-nhl.s3.ap-northeast-1.amazonaws.com/life_expectation/scaler2.joblib"
}

def download_model(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download model from {url}")
    return BytesIO(response.content)

def load_models():
    global model, health_model, life_expectation_model, scaler1, scaler2, scaler3, scaler4
    try:
        if model is None:
            # Load transformer model
            model_data = download_model(MODEL_URLS['transformer'])
            model = torch.load(model_data)
            model.eval()  # Set to evaluation mode

            # Load health model and its scalers
            health_data = download_model(MODEL_URLS['health'])
            health_model = joblib.load(health_data)
            
            scaler1_data = download_model(MODEL_URLS['scaler1'])
            scaler1 = joblib.load(scaler1_data)
            
            scaler2_data = download_model(MODEL_URLS['scaler2'])
            scaler2 = joblib.load(scaler2_data)

            # Load life expectation model and its scalers
            life_data = download_model(MODEL_URLS['life_expectation'])
            life_expectation_model = joblib.load(life_data)
            
            scaler3_data = download_model(MODEL_URLS['scaler3'])
            scaler3 = joblib.load(scaler3_data)
            
            scaler4_data = download_model(MODEL_URLS['scaler4'])
            scaler4 = joblib.load(scaler4_data)

    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

def predict_next_5_days(model, last_30_days):
    """Make predictions for the next 5 days without scaling"""
    with torch.no_grad():
        # Convert input directly to tensor without normalization
        input_seq = torch.FloatTensor(last_30_days).unsqueeze(0)
        
        # Make prediction
        prediction = model(input_seq, training=False)
        prediction = prediction[:, -5:, :]
        
        # Return prediction directly without denormalization
        return prediction.squeeze(0).numpy()

# Add these as global variables at the top with other globals
scaler1 = None  # for input features
scaler2 = None  # for health index
scaler3 = None  # for life expectation input
scaler4 = None  # for life expectation output

# Initialize FastAPI app
app = FastAPI(
    title="Time Series Prediction API",
    description="API for predicting heath index and life expectation"
)

# Define input/output models
class PredictionInput(BaseModel):
    historical_data: List[List[float]]

class PredictionOutput(BaseModel):
    predictions: List[Dict[str, float]]

class HealthIndexOutput(BaseModel):
    health_index: float

class LifeExpectationOutput(BaseModel):
    life_expectation: float

# Global variables for model and scaler
model = None
scaler = None
health_model = None
life_expectation_model = None
feature_names = ['Hydrogen', 'Oxygen', 'Methane', 'CO', 'CO2', 
                'Ethylene', 'Ethane', 'Acetylene', 'H2O']

# Định nghĩa model input mới
class HealthPredictionInput(BaseModel):
    Hydrogen: float
    Oxygen: float
    Methane: float
    CO: float
    CO2: float
    Ethylene: float
    Ethane: float
    Acetylene: float
    H2O: float

    # class Config:
    #     json_schema_extra = {
    #         "example": {
    #             "Hydrogen": 0.14,
    #             "Oxygen": 0.20,
    #             "Methane": 0.06,
    #             "CO": 0.03,
    #             "CO2": 0.08,
    #             "Ethylene": 0.02,
    #             "Ethane": 0.02,
    #             "Acetylene": 0.006,
    #             "H2O": 0.09
    #         }
    #     }

# Định nghĩa model input mới cho life expectation
class LifeExpectationInput(BaseModel):
    Hydrogen: float
    Oxygen: float
    Methane: float
    CO: float
    CO2: float
    Ethylene: float
    Ethane: float
    Acetylene: float
    H2O: float
    Healthy_index: float

    # class Config:
    #     json_schema_extra = {
    #         "example": {
    #             "Hydrogen": 0.14,
    #             "Oxygen": 0.20,
    #             "Methane": 0.06,
    #             "CO": 0.03,
    #             "CO2": 0.08,
    #             "Ethylene": 0.02,
    #             "Ethane": 0.02,
    #             "Acetylene": 0.006,
    #             "H2O": 0.09,
    #             "Healthy_index": 0.85
    #         }
    #     }

# Define response models for error cases
class ErrorResponse(BaseModel):
    detail: str

# Update the endpoint decorators with responses
@app.post(
    "/predict/",
    response_model=PredictionOutput,
    responses={
        200: {"description": "Successful prediction", "model": PredictionOutput},
        400: {"description": "Invalid input data", "model": ErrorResponse},
        403: {"description": "Invalid API key", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def predict(input_data: PredictionInput, api_key: APIKey = Depends(get_api_key)):
    """Endpoint to make predictions for the next 5 days"""
    try:
        # Convert input data to numpy array
        last_30_days = np.array(input_data.historical_data)
        
        if last_30_days.shape != (30, 9):
            raise HTTPException(
                status_code=400, 
                detail="Input must contain exactly 30 days of data with 9 features each"
            )

        # Make predictions
        predictions = predict_next_5_days(model, last_30_days)
        
        # # Create visualization
        create_visualization(last_30_days, predictions)
        
        # Format predictions as list of dictionaries
        predictions_list = []
        for day in range(5):
            prediction_dict = {
                feature: float(predictions[day][i])
                for i, feature in enumerate(feature_names)
            }
            predictions_list.append(prediction_dict)

        return PredictionOutput(
            predictions=predictions_list
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/predict_health/",
    response_model=HealthIndexOutput,
    responses={
        200: {"description": "Successfully predicted health index", "model": HealthIndexOutput},
        400: {"description": "Invalid input data", "model": ErrorResponse},
        403: {"description": "Invalid API key", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def predict_health(input_data: HealthPredictionInput, api_key: APIKey = Depends(get_api_key)):
    """Endpoint to predict health index from gas concentrations"""
    try:
        # Convert input data to numpy array
        gas_data = np.array([
            input_data.Hydrogen,
            input_data.Oxygen,
            input_data.Methane,
            input_data.CO,
            input_data.CO2,
            input_data.Ethylene,
            input_data.Ethane,
            input_data.Acetylene,
            input_data.H2O
        ]).reshape(1, -1)
        
        # Scale input data
        scaled_input = scaler1.transform(gas_data)
        
        # Predict health index (still scaled)
        scaled_prediction = health_model.predict(scaled_input)[0]
        
        # Inverse transform the prediction
        health_prediction = scaler2.inverse_transform([[scaled_prediction]])[0][0]
        
        return HealthIndexOutput(
            health_index=float(health_prediction)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/predict_life_expectation/",
    response_model=LifeExpectationOutput,
    responses={
        200: {"description": "Successfully predicted life expectation", "model": LifeExpectationOutput},
        400: {"description": "Invalid input data", "model": ErrorResponse},
        403: {"description": "Invalid API key", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def predict_life_expectation(input_data: LifeExpectationInput, api_key: APIKey = Depends(get_api_key)):
    """Endpoint to predict life expectation from gas concentrations and health index"""
    try:
        # Convert dictionary values to numpy array in correct order
        gas_data = np.array([
            input_data.Hydrogen,
            input_data.Oxygen,
            input_data.Methane,
            input_data.CO,
            input_data.CO2,
            input_data.Ethylene,
            input_data.Ethane,
            input_data.Acetylene,
            input_data.H2O
        ])
        
        # Combine gas concentrations with health index
        combined_input = np.append(gas_data, input_data.Healthy_index).reshape(1, -1)
        
        # Scale the input data
        scaled_input = scaler3.transform(combined_input)
        
        # Predict life expectation (still scaled)
        scaled_prediction = life_expectation_model.predict(scaled_input)[0]
        
        # Inverse transform the prediction
        life_expectation_prediction = scaler4.inverse_transform([[scaled_prediction]])[0][0]
        
        return LifeExpectationOutput(
            life_expectation=float(life_expectation_prediction)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def create_visualization(historical_data, predictions):
    """Create and save visualization plot"""
    import matplotlib.pyplot as plt
    import tempfile
    import os
    
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, 'predictions_visualization.png')
    
    # Generate dates
    end_date = datetime.now().date()
    dates = pd.date_range(end=end_date, periods=30)
    future_dates = [end_date + timedelta(days=i) for i in range(1, 6)]
    
    plt.figure(figsize=(15, 8))
    
    for i, feature in enumerate(feature_names):
        plt.subplot(3, 3, i+1)
        
        # Plot historical data
        plt.plot(dates, historical_data[:, i], 'b-', label='Historical')
        
        # Plot predictions
        all_dates = dates.tolist() + future_dates
        all_values = historical_data[:, i].tolist() + predictions[:, i].tolist()
        plt.plot(all_dates, all_values, 'r--', label='Predicted')
        
        plt.title(feature)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    return file_path

@app.get(
    "/visualization",
    responses={
        200: {"description": "Successfully retrieved visualization", "content": {"image/png": {}}},
        403: {"description": "Invalid API key", "model": ErrorResponse},
        404: {"description": "Visualization not found", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def get_visualization(api_key: APIKey = Depends(get_api_key)):
    """Endpoint to retrieve the visualization plot"""
    try:
        import tempfile
        import os
        
        file_path = os.path.join(tempfile.gettempdir(), 'predictions_visualization.png')
        if not Path(file_path).exists():
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"detail": "Visualization not found. Make a prediction first."}
            )
        return FileResponse(file_path)
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(e)}
        )

@app.get(
    "/health",
    responses={
        200: {"description": "API is healthy", "model": dict},
        403: {"description": "Invalid API key", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def health_check(api_key: APIKey = Depends(get_api_key)):
    """Check if the API is healthy"""
    try:
        return {"status": "healthy"}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "API health check failed"}
        )
