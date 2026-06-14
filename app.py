from flask import Flask, request, render_template_string
import joblib
import os
from openai import OpenAI

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("career_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# =========================
# OPENAI CLIENT (SAFE - ENV VARIABLE)
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# HTML UI
# =========================
html = """
<h2>AI Career Recommendation System</h2>

<form method="post">
Education: <input name="education"><br><br>
Skills: <input name="skills"><br><br>
Interests: <input name="interests"><br><br>
<button type="submit">Predict Career</button>
</form>

{% if result %}
<hr>
<h3>🎯 Result: {{ result }}</h3>
<p><b>Confidence:</b> {{ confidence }}%</p>
<h3>🧠 AI Explanation:</h3>
<p style="white-space: pre-line;">{{ explanation }}</p>
{% endif %}
"""

# =========================
# AI EXPLANATION FUNCTION
# =========================
def get_explanation(career):
    try:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            return "ERROR: OpenAI API key not found in environment variables."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert career advisor AI like ChatGPT. "
                        "Explain careers in a clear, structured, and motivating way. "
                        "Always include: what it is, skills needed, why it fits, and career growth."
                    )
                },
                {
                    "role": "user",
                    "content": f"Explain this career in a student-friendly way: {career}"
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI ERROR: {str(e)}"

# =========================
# ROUTE
# =========================
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

        result = prediction
        explanation = get_explanation(prediction)

    return render_template_string(html, result=result, confidence=confidence, explanation=explanation)

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
