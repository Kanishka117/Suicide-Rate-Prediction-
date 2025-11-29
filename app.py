from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the saved model and preprocessors
model = pickle.load(open('random_forest_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))

# Get all categories from the training data
type_categories = encoder.categories_[0]  # Categories for Type column
age_categories = encoder.categories_[1]   # Categories for Age_group column

# Load your CSV data to get states and years
df = pd.read_csv('data1 - Copy.csv')  # Replace with your CSV filename
state_categories = sorted(df['State'].unique())
year_categories = sorted(df['Year'].unique())
gender_categories = sorted(df['Gender'].unique())  # Add this line to get gender categories

@app.route('/')
def home():
    return render_template('index.html',
                         type_categories=type_categories,
                         age_categories=age_categories,
                         state_categories=state_categories,
                         year_categories=year_categories,
                         gender_categories=gender_categories)  # Add gender_categories here

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        input_data = {
            'Type': request.form['type'],
            'Age_group': request.form['age_group'],
            'Gender': request.form['gender'],
            'State': request.form['state'],
            'Year': int(request.form['year'])
        }

        # Create DataFrame with the input data
        input_df = pd.DataFrame([input_data])

        # Encode categorical variables
        cat_features = ['Type', 'Age_group']
        encoded_features = encoder.transform(input_df[cat_features])
        encoded_df = pd.DataFrame(
            encoded_features,
            columns=encoder.get_feature_names_out(cat_features)
        )

        # Create dummy variables for Gender and State
        gender_dummies = pd.get_dummies(input_df['Gender'], prefix='Gender')
        state_dummies = pd.get_dummies(input_df['State'], prefix='State')

        # Create a DataFrame with all necessary features
        final_df = pd.DataFrame()
        
        # Add Year
        final_df['Year'] = input_df['Year']
        
        # Add encoded features
        for col in feature_names:
            if col in encoded_df.columns:
                final_df[col] = encoded_df[col]
            elif col in gender_dummies.columns:
                final_df[col] = gender_dummies[col]
            elif col in state_dummies.columns:
                final_df[col] = state_dummies[col]
            elif col != 'Year':
                final_df[col] = 0  # Add missing columns with zero values

        # Ensure columns are in the same order as during training
        final_df = final_df.reindex(columns=feature_names, fill_value=0)

        # Scale features
        scaled_features = scaler.transform(final_df)

        # Make prediction
        prediction = model.predict(scaled_features)[0]

        # Round prediction to nearest whole number
        prediction = round(prediction)

        # Store the prediction result
        prediction_result = f'Predicted number of cases: {prediction}'

        return render_template('index.html', 
                             prediction_text=prediction_result,
                             type_categories=type_categories,
                             age_categories=age_categories,
                             state_categories=state_categories,
                             year_categories=year_categories,
                             gender_categories=gender_categories)  # Add gender_categories here

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(f"Available features: {feature_names}")
        if 'final_df' in locals():
            print(f"Input features: {list(final_df.columns)}")
        
        error_message = f'Error in prediction: {str(e)}'
        return render_template('index.html', 
                             prediction_text=error_message,
                             type_categories=type_categories,
                             age_categories=age_categories,
                             state_categories=state_categories,
                             year_categories=year_categories,
                             gender_categories=gender_categories)  # Add gender_categories here

if __name__ == '__main__':
    app.run(debug=True)