import pickle
from flask import Flask, jsonify, request
import pandas as pd
import json
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def apicall():
    """Get json request and apply deserialized model to send prediction as response"""
    try:
        # test = request.get_json() # test response through request
        test = pd.read_json('test_json.json', orient = 'records') # testing with a stored json file
    except Exception as e:
        raise e
    model = 'model_v1.pk'
    if test.empty:
        return 'Bad request!!'
    else:
        print("Loading the model...")
        loaded_model = None
        with open('models/'+model,'rb') as f:
            loaded_model = pickle.load(f)
        print("Model loaded successfully...")
        predictions = loaded_model.predict(test)
        responses = jsonify(predictions=json.dumps(predictions.tolist()))
        responses.status_code = 200
        return responses


if __name__ == '__main__':
    app.run(debug=True, port=5000)
