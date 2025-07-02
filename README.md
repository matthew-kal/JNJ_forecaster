# 📈 Johnson & Johnson Stock Forecaster 🚀

Welcome to the **Johnson & Johnson Stock Forecaster**\! This is a full-stack application that uses a deep learning model to predict future stock prices for JNJ and visualizes the results in a sleek web interface.

The backend, powered by **FastAPI**, serves up predictions from a powerful **PyTorch** LSTM model. The frontend, built with **React**, provides a dynamic and responsive chart to display the historical data and the model's forecast.

-----

## ✨ Full Tech Stack

This project is built with a modern, high-performance stack for both the frontend and backend.

### Backend

  * **Python**: The core language for the entire application.
  * **FastAPI**: A high-performance web framework for building APIs with Python.
  * **PyTorch**: The deep learning framework used to build and run the LSTM model.
  * **Uvicorn**: A lightning-fast ASGI server to run the FastAPI application.
  * **yfinance**: The library used to fetch historical stock data from Yahoo Finance.
  * **scikit-learn**: Used for scaling the data before feeding it to the model.

### Frontend

  * **React**: A popular JavaScript library for building user interfaces.
  * **Recharts**: A composable charting library built on top of React components.
  * **React Icons**: Provides easy access to popular icon sets.
  * **CSS**: For custom styling and layout.

-----

## API Endpoints

### POST `/predict`

This is the main endpoint for getting stock predictions.

  * **Request Body**:
    You need to send a JSON object with the stock ticker you want to predict.

    ```json
    {
        "ticker": "JNJ"
    }
    ```

  * **Response**:
    The API will respond with a JSON object containing the last 5 days of actual closing prices and the predicted closing price for the next day.

    ```json
    {
        "results": [
            145.19,
            146.52,
            147.13,
            147.31,
            148.86,
            149.32
        ]
    }
    ```

-----

## 🛠️ Setup and Installation

Ready to run the full-stack application on your local machine? Follow these simple steps.

### Backend Setup

1.  **Navigate to your backend directory.**

2.  **Create a `requirements.txt` file**:

    ```
    fastapi
    uvicorn
    torch
    joblib
    numpy
    yfinance
    scikit-learn
    pydantic
    ```

3.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    Don't forget to have your trained model `lstm_model.pth` and `scaler.pkl` files in the same directory.

4.  **Run the Backend Server**:

    ```bash
    uvicorn main:app --reload
    ```

    The API will be live at `http://localhost:8000`.

### Frontend Setup

1.  **Navigate to your frontend directory in a new terminal.**
2.  **Install Dependencies**:
    ```bash
    npm install
    ```
3.  **Run the Frontend App**:
    ```bash
    npm start
    ```
    Your React application will be live at `http://localhost:3000` and will automatically connect to the backend.

-----

## 🤖 How It Works

The magic behind this application is the seamless integration of a powerful backend and a dynamic frontend.

1.  When the React app loads, a `useEffect` hook fires a **POST** request to the FastAPI backend, specifically asking for a prediction for the 'JNJ' ticker.
2.  The backend uses `yfinance` to grab the latest historical data for Johnson & Johnson.
3.  The data is preprocessed and scaled to match the format the **PyTorch LSTM model** was trained on.
4.  The model predicts the next day's closing price.
5.  The prediction is inverse-scaled to its original price format and sent back to the React frontend.
6.  The frontend receives the data, processes it into a format suitable for **Recharts**, and displays a beautiful line chart showing the 5-day lookback period and the new prediction.



