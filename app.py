from flask import Flask, request, render_template_string
import joblib
import os
from openai import OpenAI

app = Flask(__name__)

# Load AI model
model = joblib.load("career_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# OpenAI client (SAFE: uses environment variable)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# HTML UI
html = """
<h2>AI Career Recommendation System</h2>

<form method="post">
Education: <input name="education"><br><br>
Skills: <input name="skills"><br><br>
Interests: <input name="interests"><br><br>
<button type="submit">Predict Career</button>
</form>

{% if result %}
<h3>Result: {{ result }}</h3>
<p><b>Confidence:</b> {{ confidence }}%</p>
<p><b>AI Explanation:</b></p>
<p>{{ explanation }}</p>
{% endif %}
"""

def get_explanation(career):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional career advisor AI. Explain careers simply, clearly, and motivationally."
                },
                {
                    "role": "user",
                    "content": f"Explain this career path in simple terms: {career}"
                }
            ]
        )
        return response.choices[0].message.content
    except Exception:
        return "This career involves specialized skills, growth opportunities, and industry demand."

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    explanation = None

    if request.method == "POST":
        edu = request.form["education"]
        skills = request.form["skills"]
        interests = request.form["interests"]

        user_input = skills + " " + interests + " " + edu
        vector = vectorizer.transform([user_input])

        prediction = model.predict(vector)[0]
        probs = model.predict_proba(vector)

        confidence = round(max(probs[0]) * 100, 2)
        explanation = get_explanation(prediction)
        result = prediction

    return render_template_string(html, result=result, confidence=confidence, explanation=explanation)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
    
