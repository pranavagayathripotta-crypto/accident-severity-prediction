import json
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

USERS_PATH = Path(__file__).resolve().parent / "users.json"


def _load():
    if not USERS_PATH.exists():
        return {}
    with open(USERS_PATH, "r") as f:
        return json.load(f)


def _save(users):
    with open(USERS_PATH, "w") as f:
        json.dump(users, f, indent=2)


def list_users():
    return _load()


def add_user(username, password):
    users = _load()
    users[username] = {
        "password_hash": generate_password_hash(password),
        "active": True,
    }
    _save(users)


def remove_user(username):
    users = _load()
    if username in users:
        del users[username]
        _save(users)


def set_active(username, active):
    users = _load()
    if username in users:
        users[username]["active"] = active
        _save(users)


def verify_user(username, password):
    users = _load()
    user = users.get(username)
    if not user:
        return False
    if not user.get("active", True):
        return False
    return check_password_hash(user["password_hash"], password)