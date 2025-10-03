from flask import Flask, request, render_template_string
import joblib

# Load the trained sentiment model pipeline
model = joblib.load("sentiment_model.pkl")

app = Flask(__name__)

# HTML template with styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Sentiment Analysis Demo</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #f4f6f9;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      margin: 0;
    }
    .container {
      background: #fff;
      padding: 30px;
      border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      width: 500px;
      text-align: center;
    }
    h2 {
      color: #333;
    }
    input[type="text"] {
      width: 90%;
      padding: 10px;
      margin: 15px 0;
      border-radius: 8px;
      border: 1px solid #ccc;
      font-size: 16px;
    }
    button {
      background: #007bff;
      color: white;
      padding: 10px 20px;
      border: none;
      border-radius: 8px;
      font-size: 16px;
      cursor: pointer;
    }
    button:hover {
      background: #0056b3;
    }
    .result {
      margin-top: 20px;
      font-size: 18px;
      font-weight: bold;
    }
    .positive {
      color: green;
    }
    .negative {
      color: red;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>Sentiment Analysis Demo</h2>
    <form method="post" action="/predict">
      <input type="text" name="tweet" placeholder="Type a tweet here..." required>
      <br>
      <button type="submit">Predict Sentiment</button>
    </form>
    {% if prediction %}
      <div class="result {{ prediction|lower }}">
        Prediction: {{ prediction }}
      </div>
    {% endif %}
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        tweet = request.form.get("tweet")

        if not tweet:
            return render_template_string(HTML_TEMPLATE, prediction="No tweet provided")

        prediction = model.predict([tweet])[0]

        return render_template_string(HTML_TEMPLATE, prediction=str(prediction))

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
