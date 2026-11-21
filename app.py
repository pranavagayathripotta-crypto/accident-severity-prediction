import os
import secrets
from datetime import datetime
from zoneinfo import ZoneInfo
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, session

import predict_core
import users as users_db
import quiz as quiz_db

# ---- ADMIN LOGIN ----
# Set these as environment variables instead of hardcoding them here.
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "student")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "changeme123")
# ----------------------

PUBLIC_MODE = True

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 3600
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
DEBUG_MODE = os.environ.get("FLASK_DEBUG", "0") == "1"

# Real numbers from the training run (train_model.py output) -- static
# because they describe the model as it was trained, not per-request data.
MODEL_STATS = {
    "full_count": 74881,
    "manhattan_count": 7353,
    "midtown_count": 1540,
    "train_count": 1232,
    "test_count": 308,
    "minor_count": 1185,
    "elevated_count": 355,
    "models": [
        {"name": "Random Forest + SMOTE", "f1": 52.86, "deployed": True},
        {"name": "Random Forest", "f1": 50.31},
        {"name": "SVM (scaled)", "f1": 49.06},
        {"name": "Gradient Boosting", "f1": 47.74},
        {"name": "SVM + Scaling + SMOTE", "f1": 36.36},
        {"name": "Decision Tree", "f1": 34.72},
    ],
    "features": [
        ("Local historical risk", 17.72),
        ("Temperature", 11.02),
        ("Vehicle count", 8.50),
        ("Latitude", 8.07),
        ("Month", 7.74),
        ("Windspeed", 7.53),
        ("Longitude", 6.77),
        ("Hour of day", 5.56),
        ("Day of week", 3.65),
        ("Factor: other", 2.48),
    ],
    "accuracy": 78.57,
    "class0": {"precision": 86, "recall": 86, "f1": 86, "n": 237},
    "class1": {"precision": 54, "recall": 52, "f1": 53, "n": 71},
    "threshold": 0.36,
    "threshold_recall": 67.61,
    "threshold_precision": 40.68,
}


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if PUBLIC_MODE:
            return view(*args, **kwargs)
        if not session.get("username"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if session.get("username") != ADMIN_USERNAME:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def quiz_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        questions = quiz_db.list_questions()
        if questions and not session.get("quiz_passed"):
            return redirect(url_for("quiz"))
        return view(*args, **kwargs)
    return wrapped


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["username"] = username
            return redirect(url_for("index"))
        elif users_db.verify_user(username, password):
            session["username"] = username
            return redirect(url_for("index"))
        else:
            error = "Wrong username or password, or your access was removed."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    questions = quiz_db.list_questions()
    if not questions:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        answers = {str(q["id"]): request.form.get(str(q["id"]), "") for q in questions}
        if quiz_db.check_answers(answers):
            session["quiz_passed"] = True
            return redirect(url_for("index"))
        else:
            error = "One or more answers were incorrect. Please try again."
    return render_template("quiz.html", questions=questions, error=error)


@app.route("/admin", methods=["GET", "POST"])
@admin_required
def admin():
    message = None
    if request.method == "POST":
        action = request.form.get("action")

        if action == "add_user":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            if username and password:
                users_db.add_user(username, password)
                message = f"Added user {username}."
        elif action == "remove_user":
            username = request.form.get("username", "").strip()
            users_db.remove_user(username)
            message = f"Removed user {username}."
        elif action == "toggle_user":
            username = request.form.get("username", "").strip()
            current_users = users_db.list_users()
            current = current_users.get(username, {}).get("active", True)
            users_db.set_active(username, not current)
            message = f"Updated {username}."

        elif action == "add_question":
            question_text = request.form.get("question_text", "").strip()
            answer_text = request.form.get("answer_text", "").strip()
            if question_text and answer_text:
                quiz_db.add_question(question_text, answer_text)
                message = "Question added."
        elif action == "remove_question":
            question_id = request.form.get("question_id", "")
            quiz_db.remove_question(question_id)
            message = "Question removed."

    return render_template(
        "admin.html",
        users=users_db.list_users(),
        questions=quiz_db.list_questions(),
        message=message,
        admin_username=ADMIN_USERNAME,
    )


@app.route("/model-data")
def model_data():
    return render_template(
        "model_data.html",
        stats=MODEL_STATS,
        current_user=session.get("username"),
        is_admin=(session.get("username") == ADMIN_USERNAME),
    )


@app.route("/", methods=["GET", "POST"])
@login_required
@quiz_required
def index():
    model, feature_columns, category_maps, threshold, risk_map, weather_medians, load_error = (
        predict_core.get_resources()
    )
    result = None
    error = None
    form_values = request.form if request.method == "POST" else {}

    if request.method == "POST":
        if load_error:
            error = load_error
        else:
            try:
                address = request.form.get("address", "").strip()
                latitude = longitude = None
                location_label = None

                if address:
                    geocoded = predict_core.geocode_address(address)
                    if geocoded:
                        latitude, longitude, location_label = geocoded

                if latitude is None:
                    lat_raw = request.form.get("latitude", "").strip()
                    lon_raw = request.form.get("longitude", "").strip()
                    if lat_raw and lon_raw:
                        latitude, longitude = float(lat_raw), float(lon_raw)

                if latitude is None:
                    raise ValueError("Couldn't find that address, and no latitude/longitude was given.")

                use_now = request.form.get("use_now") == "on"
                if use_now:
                    now = datetime.now(ZoneInfo('America/New_York'))
                    hour, day_of_week, month = now.hour, now.weekday(), now.month
                    crash_date = now.date()
                else:
                    custom = request.form.get("custom_datetime", "")
                    dt = datetime.strptime(custom, "%Y-%m-%dT%H:%M")
                    hour, day_of_week, month = dt.hour, dt.weekday(), dt.month
                    crash_date = dt.date()

                contributing_factor_raw = request.form.get("contributing_factor", "").strip() or "Unspecified"
                vehicle_type_raw = request.form.get("vehicle_type", "").strip() or "Unspecified"
                vehicle_count_raw = request.form.get("vehicle_count", "").strip()
                vehicle_count = int(vehicle_count_raw) if vehicle_count_raw else 2

                input_df, meta = predict_core.build_prediction_input(
                    latitude, longitude, hour, day_of_week, month, crash_date,
                    contributing_factor_raw, vehicle_type_raw, vehicle_count,
                    feature_columns, category_maps, risk_map, weather_medians,
                )
                prediction, prob_by_class = predict_core.predict(input_df, model, threshold)

                result = {
                    "label": predict_core.RISK_LABELS.get(prediction, "Unknown"),
                    "is_elevated": prediction == 1,
                    "minor_pct": round(prob_by_class.get(0, 0) * 100, 1),
                    "elevated_pct": round(prob_by_class.get(1, 0) * 100, 1),
                    "threshold_pct": round(threshold * 100, 1),
                    "location_label": location_label,
                    "latitude": round(latitude, 5),
                    "longitude": round(longitude, 5),
                    "local_risk_pct": meta["local_risk_pct"],
                    "weather": meta["weather"],
                    "weather_source": meta["weather_source"],
                    "factor_was_other": meta["factor_was_other"],
                    "vehicle_was_other": meta["vehicle_was_other"],
                    # display-only fields, not used by the model itself
                    "hour": hour,
                    "day_name": crash_date.strftime("%A"),
                    "contributing_factor_display": contributing_factor_raw,
                    "vehicle_type_display": vehicle_type_raw,
                }
            except Exception as exc:
                error = f"Check your inputs -- {exc}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        load_error=load_error,
        form_values=form_values,
        current_user=session.get("username"),
        is_admin=(session.get("username") == ADMIN_USERNAME),
    )


if __name__ == "__main__":
    app.run(debug=DEBUG_MODE)