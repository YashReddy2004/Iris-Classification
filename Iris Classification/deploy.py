from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('saved_model.sav', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index.html', result=result)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve form data with error handling
            sepal_length = float(request.form.get('sepal_length', 0))
            sepal_width = float(request.form.get('sepal_width', 0))
            petal_length = float(request.form.get('petal_length', 0))
            petal_width = float(request.form.get('petal_width', 0))

            # Make prediction
            result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

            return render_template('index.html', result=result)
        except Exception as e:
            # Handle any errors that may occur during prediction
            return f"An error occurred: {str(e)}"
    else:
        return "Invalid request method"

if __name__ == '__main__':
    app.run(debug=True)
