from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
from disease_prediction import predict_disease
import os
from dotenv import load_dotenv
from flask_cors import CORS
from chatbot import get_response

# ---------------- Load environment variables ----------------
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")

app = Flask(__name__)
app.secret_key = SECRET_KEY  # required for sessions

# Load symptoms directly from pkl (used for dropdown values)
SYMPTOM_LIST = joblib.load("symptom_list.pkl")

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("home.html", title="Home")

@app.route("/about")
def about():
    return render_template("about.html", title="About")

@app.route("/symptoms", methods=["GET", "POST"])
def symptoms_page():
    result = None
    selected = []
    if request.method == "POST":
        selected = request.form.getlist("symptoms")  # get selected symptoms
        result = predict_disease(selected)  # call ML function
        print("DEBUG - Selected:", selected)  # <-- debug

    return render_template(
        "symptoms.html",
        symptoms=SYMPTOM_LIST,
        result=result,
        selected=selected,
        title="Symptoms"
    )

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html", title="Chatbot")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "123":
            session["user"] = username
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid credentials", title="Login")

    return render_template("login.html", title="Login")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))

@app.route("/contact")
def contact():
    return render_template("contact.html", title="Contact")

CORS(app)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = data.get("message", "")
    if not message.strip():
        return jsonify({"error": "Message is empty"}), 400
    try:
        reply = get_response(message)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(debug=True)
