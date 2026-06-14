from flask import Flask, request, render_template_string
import joblib
import os
from openai import OpenAI

app = Flask(__name__)

# =========================
# LOAD MODEL + VECTORIZER
# =========================
model = joblib.load("career_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# =========================
# OPENAI CLIENT (SAFE)
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
# CHATGPT-STYLE EXPLANATION
# =========================
def get_explanation(career):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert AI career advisor like ChatGPT. "
                        "You explain careers in a clear, structured, and motivational way. "
                        "Always structure your response into 4 sections:\n"
                        "1. What the career is\n"
                        "2. Key skills needed\n"
                        "3. Why this career fits the student\n"
                        "4. Career growth opportunities\n"
                        "Keep it simple, human, and student-friendly."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
The predicted career is: {career}

Give a helpful explanation that sounds like a real career counselor.
"""
                }
            ]
        )

        return response.choices[0].message.content

    except Exception:
        return "AI explanation unavailable. Please check API connection."

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
# RUN APP (RENDER READY)
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
