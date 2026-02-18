from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved models from your 'Models' folder
model = pickle.load(open('Models/Rainfall.pkl', 'rb'))
scale = pickle.load(open('Models/scale.pkl', 'rb'))

@app.route('/')
def home():
    # Flask looks for these files in the 'Templates' folder automatically
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Capture the 12 monthly inputs from the UI
    input_features = [float(x) for x in request.form.values()]
    
    # 2. Scale the data using your saved scale.pkl
    features_array = [np.array(input_features)]
    scaled_features = scale.transform(features_array)
    
    # 3. Model analyzes the input
    prediction = model.predict(scaled_features)
    output = round(prediction[0], 2)

    # 4. Display the result on the specific UI page
    if output > 1000:
        return render_template('chance.html', prediction_text=f'Predicted Rainfall: {output}mm. High chance of success!')
    else:
        return render_template('nochance.html', prediction_text=f'Predicted Rainfall: {output}mm. Low rainfall risk.')

if __name__ == "__main__":
    # use_reloader=False stops Spyder from crashing the Flask server
    app.run(debug=True, port=8000, use_reloader=False)