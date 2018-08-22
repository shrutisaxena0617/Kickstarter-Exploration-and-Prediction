from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def apicall():
    responses = jsonify(predictions='successful')
    responses.status_code = 200

    return (responses)

if __name__ == '__main__':
    app.run(debug=True, port=5000) #run app in debug mode on port 5000