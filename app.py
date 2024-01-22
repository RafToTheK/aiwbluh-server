from flask import Flask, jsonify, request
from flask_cors import CORS

from assi import get_quantized_model

app = Flask(__name__)
CORS(app)
bot = get_quantized_model()
bot.initial_messages = [
    {'role': 'user', 'content': 'What is your name?'},
    {'role': 'assistant', 'content': 'They call me Aiwbluh'}
]

bot.reset()


# Endpoint to handle POST requests to /chat with JSON object in the request body
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        print(data['prompt'])
        answer = bot.prompt(data['prompt'])
        # perform your logic here based on the data received
        print(answer)
        response = {'answer': answer}
        return jsonify(response)
    except:
        response = {'error': 'Invalid JSON format'}
        return jsonify(response), 400


# Endpoint to handle DELETE requests to /chat with an empty request body
@app.route('/chat', methods=['DELETE'])
def reset():
    bot.reset()
    return jsonify({'message': 'Chat reset'})


# Endpoint to handle GET requests to /chat
@app.route('/chat', methods=['GET'])
def get_chat():
    return jsonify(bot.chat[2:])


if __name__ == '__main__':
    # use_reloader=True reloads the Model repeatedly, which takes too long
    app.run(debug=True, use_reloader=False)
