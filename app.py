from flask import Flask, request, render_template_string
import joblib
import os

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("career_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# =========================
# EXPANDED CAREER KNOWLEDGE BASE
# =========================
career_info = {
    "Data Scientist": "You analyze data using statistics, Python, and machine learning to help companies make decisions. Skills: Python, ML, SQL. Growth: AI Engineer, Lead Data Scientist.",
    
    "Data Analyst": "You interpret data and create reports that help businesses understand performance. Skills: Excel, SQL, Power BI. Growth: BI Analyst, Data Scientist.",
    
    "Software Engineer": "You build applications, websites, and systems using programming languages. Skills: Python, Java, JavaScript. Growth: Senior Engineer, Architect.",
    
    "Machine Learning Engineer": "You design AI systems that learn from data and improve automatically. Skills: Python, ML, Deep Learning. Growth: AI Researcher, Senior ML Engineer.",
    
    "AI Engineer": "You develop intelligent systems like chatbots, recommendation engines, and automation tools. Skills: AI, Python, NLP. Growth: AI Architect.",
    
    "Cybersecurity Analyst": "You protect systems from hackers and cyber threats. Skills: Networking, Security tools, Ethical hacking. Growth: Security Engineer, CISO.",
    
    "UX Designer": "You design user-friendly apps and websites that improve user experience. Skills: Figma, UI/UX, research. Growth: Senior UX Designer, Product Designer.",
    
    "UI Designer": "You design visual interfaces for apps and websites. Skills: Design tools, creativity, typography. Growth: UX/UI Lead.",
    
    "Digital Marketer": "You promote brands using social media, SEO, and ads. Skills: SEO, content, analytics. Growth: Marketing Manager.",
    
    "Content Strategist": "You plan and manage digital content for brands. Skills: Writing, SEO, storytelling. Growth: Content Manager.",
    
    "Business Analyst": "You analyze business processes to improve efficiency. Skills: SQL, Excel, communication. Growth: Product Manager.",
    
    "Product Manager": "You manage product development from idea to launch. Skills: Strategy, communication, analytics. Growth: Senior PM, Director.",
    
    "Web Developer": "You build websites and web applications. Skills: HTML, CSS, JavaScript. Growth: Full Stack Developer.",
    
    "Full Stack Developer": "You build both frontend and backend of applications. Skills: React, Node.js, databases. Growth: Tech Lead.",
    
    "Mobile App Developer": "You create mobile apps for Android and iOS. Skills: Flutter, Kotlin, Swift. Growth: Senior Mobile Engineer.",
    
    "Cloud Engineer": "You manage cloud systems like AWS and Azure. Skills: Cloud computing, DevOps. Growth: Cloud Architect.",
    
    "DevOps Engineer": "You automate software deployment and manage infrastructure. Skills: CI/CD, Docker, Linux. Growth: DevOps Lead.",
    
    "Network Engineer": "You design and maintain computer networks. Skills: Networking, routers, security. Growth: Senior Network Engineer.",
    
    "Database Administrator": "You manage and secure databases. Skills: SQL, database design. Growth: Senior DBA.",
    
    "Lecturer": "You teach students in universities and conduct research. Skills: Teaching, subject expertise. Growth: Professor.",
    
    "Research Scientist": "You conduct advanced research in science or technology fields. Skills: Research, analytics. Growth: Lead Scientist.",
    
    "Financial Analyst": "You analyze financial data to guide investment decisions. Skills: Excel, finance, modeling. Growth: Finance Manager.",
    
    "Accountant": "You manage financial records and reporting. Skills: Accounting, Excel, auditing. Growth: Senior Accountant, CFO.",
    
    "HR Manager": "You manage hiring, employee relations, and workplace policies. Skills: Communication, HR tools. Growth: HR Director.",
    
    "Graphic Designer": "You create visual content for brands and media. Skills: Photoshop, Illustrator. Growth: Senior Designer.",
    
    "Entrepreneur": "You build and manage your own business. Skills: Leadership, strategy, risk-taking. Growth: Business Owner.",
    
    "Sales Manager": "You manage sales teams and business revenue. Skills: Negotiation, communication. Growth: Head of Sales.",
    
    "AI Researcher": "You study advanced AI models and improve machine intelligence. Skills: Math, ML, deep learning. Growth: Senior Research Scientist."
}

# =========================
# SAFE EXPLANATION FUNCTION
# =========================
def get_explanation(career):
    return career_info.get(
        career,
        "This career involves specialized skills, continuous learning, and strong growth opportunities in the industry."
    )

# =========================
# UI
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

<h3>🧠 Career Explanation:</h3>
<p style="white-space: pre-line;">{{ explanation }}</p>
{% endif %}
"""

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    explanation = None

    if request.method == "POST":

        edu = request.form.get("education", "")
        skills = request.form.get("skills", "")
        interests = request.form.get("interests", "")

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
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
