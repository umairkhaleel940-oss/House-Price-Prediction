# House Price Prediction

## Project Overview
This project is a machine learning based web application that predicts house prices based on various housing features such as area, number of bedrooms, bathrooms, and other important factors. The model analyzes housing data and estimates the price using trained machine learning algorithms.

## Features
- Predict house prices based on user input
- Data preprocessing and feature handling
- Machine learning model trained using housing dataset
- Simple and interactive web interface
- Backend API for prediction

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- HTML
- CSS
- JavaScript

## Project Structure
house-price-prediction
│
├── app.py
├── model.pkl
├── house_price_data_100k.csv
├── requirements.txt
│
├── templates
│ └── index.html
│
├── static
│ ├── style.css
│ └── script.js
│
└── README.md

## How It Works
1. The dataset is preprocessed and used to train a machine learning model.
2. The trained model is saved as a pickle file.
3. A Flask backend loads the trained model.
4. Users enter house details through the frontend interface.
5. The model predicts the house price and returns the result.

## Installation

Clone the repository from **GitHub**:

```bash
git clone https://github.com/yourusername/house-price-prediction.git


Navigate to the project folder:
</>bash
cd house-price-prediction

Install required libraries:
</>bash
pip install -r requirements.txt

Run the application:
</>bash
python app.py

**Author**
Developed as part of a machine learning project to demonstrate house price prediction using Python
