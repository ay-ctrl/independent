import sys
print(sys.executable)

from flask import Flask, render_template, request
from model import predict_all_drugs_web, DRUG_NAMES

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        # Form verilerini al
        responses = {
            "age": request.form.get("age"),
            "gender": request.form.get("gender"),
            "education": request.form.get("education"),
            "country": request.form.get("country"),
            "ethnicity": request.form.get("ethnicity"),
            "nscore": float(request.form.get("nscore")),
            "escore": float(request.form.get("escore")),
            "oscore": float(request.form.get("oscore")),
            "ascore": float(request.form.get("ascore")),
            "cscore": float(request.form.get("cscore")),
            "impulsive": float(request.form.get("impulsive")),
            "ss": float(request.form.get("ss"))
        }

        predictions, responses, high_risk_drugs = predict_all_drugs_web(responses)
        return render_template("result.html", predictions=predictions, responses=responses, high_risk_drugs=high_risk_drugs, drug_names=DRUG_NAMES)
    
    return render_template("form.html")


@app.route("/blog")
def blog():
    return render_template("blog.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/hakkimizda")
def hakkimizda():
    return render_template("hakkimizda.html")

@app.route("/contact")
def contact():  
    return render_template("contact.html")

@app.route("/policy")
def policy():
    return render_template("policy.html")

if __name__ == "__main__":
    app.run(debug=True)

