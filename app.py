from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model using pickle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Dummy scaler for illustration (optional: use your trained scaler)
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect form inputs
        geography = request.form['geography']  
        gender = request.form['gender']      
        credit_score = float(request.form['credit_score'])
        age = float(request.form['age'])
        tenure = int(request.form['tenure'])
        balance = float(request.form['balance'])
        num_of_products = int(request.form['num_of_products'])
        has_cr_card = request.form['has_cr_card']
        is_active_member = request.form['is_active_member']
        estimated_salary = float(request.form['estimated_salary'])
        if gender=='Male':
            gender=1
        else: gender=0    

        if geography=='France' or geography=='france':
            geography=0
        elif geography=='Spain' or geography=='spain':
            geography=2
        else: geography=1 


        if has_cr_card=='yes' or has_cr_card=='Yes':
            has_cr_card=1  
        else: has_cr_card=0  

        if is_active_member=='yes' or is_active_member=='Yes':
            is_active_member=1  
        else: is_active_member=0    


        # Apply same scaling as used during training

        scaled = scaler.fit_transform([[balance, estimated_salary, credit_score, age]])
        scaled_balance, scaled_salary, scaled_credit, scaled_age = scaled[0]

        # Combine all features into final input array
        input_features = np.array([[tenure, num_of_products, has_cr_card, is_active_member,
                                    geography, gender,
                                    scaled_balance, scaled_salary,
                                    scaled_credit, scaled_age]])

        prediction = model.predict(input_features)
        result = 'Customer Will Churn' if prediction[0] == 1 else 'Customer Will Stay'

        return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
