from flask import Flask, request, render_template_string
import joblib

app = Flask(__name__)

# Load AI model
model = joblib.load("career_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Simple career explanations (AI-style layer)
career_info = {
    "Data Scientist": "You analyze data, find patterns, and help companies make decisions using machine learning and statistics.",
    "Software Engineer": "You design and build software systems and applications used by millions of users.",
    "UX Designer": "You improve how apps and websites feel and make them easy and enjoyable to use.",
    "Digital Marketer": "You promote products online using social media, ads, and content strategies.",
    "Cybersecurity Analyst": "You protect systems and networks from cyber attacks and security threats."
}

def get_explanation(career):
    return career_info.get(career, "This career involves specialized skills and industry expertise.")

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
<p><b>AI Explanation:</b> {{ explanation }}</p>
{% endif %}
"""

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

        # Predict career
        prediction = model.predict(vector)[0]

        # Fake but realistic confidence (ML limitation workaround)
        probs = model.predict_proba(vector)
        confidence = round(max(probs[0]) * 100, 2)

        explanation = get_explanation(prediction)

        result = prediction

    return render_template_string(html, result=result, confidence=confidence, explanation=explanation)

if __name__ == "__main__":
    app.run(debug=True)