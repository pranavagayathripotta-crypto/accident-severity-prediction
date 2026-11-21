import json
from pathlib import Path

QUIZ_PATH = Path(__file__).resolve().parent / "quiz_questions.json"


def _load():
    if not QUIZ_PATH.exists():
        return []
    with open(QUIZ_PATH, "r") as f:
        return json.load(f)


def _save(questions):
    with open(QUIZ_PATH, "w") as f:
        json.dump(questions, f, indent=2)


def list_questions():
    return _load()


def add_question(question_text, correct_answer):
    questions = _load()
    new_id = max([q["id"] for q in questions], default=0) + 1
    questions.append({"id": new_id, "question": question_text, "answer": correct_answer})
    _save(questions)


def remove_question(question_id):
    questions = _load()
    questions = [q for q in questions if str(q["id"]) != str(question_id)]
    _save(questions)


def check_answers(submitted: dict) -> bool:
    """submitted: {question_id_str: answer_text}. Passes automatically if no questions exist."""
    questions = _load()
    if not questions:
        return True
    for q in questions:
        given = submitted.get(str(q["id"]), "").strip().lower()
        expected = q["answer"].strip().lower()
        if given != expected:
            return False
    return True