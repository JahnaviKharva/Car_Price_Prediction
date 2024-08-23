from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved models
linear_model = pickle.load(open('linear_regression_model.pkl', 'rb'))
lasso_model = pickle.load(open('lasso_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the form data (feature values)
        year = int(request.form['Year'])
        present_price = float(request.form['Present_Price'])
        kms_driven = int(request.form['Kms_Driven'])
        fuel_type = int(request.form['Fuel_Type'])
        seller_type = int(request.form['Seller_Type'])
        transmission = int(request.form['Transmission'])
        owner = int(request.form['Owner'])
        
        # Organize input into a dataframe
        input_data = pd.DataFrame([[year, present_price, kms_driven, fuel_type, seller_type, transmission, owner]], 
                                  columns=['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'])
        
        # Choose which model to use (example: Linear Regression)
        model_choice = request.form['model_choice']  # This comes from a dropdown on the form
        if model_choice == 'Linear Regression':
            prediction = linear_model.predict(input_data)[0]
        else:
            prediction = lasso_model.predict(input_data)[0]

        # Render the result with the prediction
        return render_template('result.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
