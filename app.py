 from flask import Flask, render_template, request, jsonify
from chatbot_model import get_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.json["message"]
    response = get_response(user_input)
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True)
