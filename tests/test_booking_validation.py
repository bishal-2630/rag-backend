from datetime import date

from app.services.booking import parse_json_object, validate_field

TODAY = date(2026, 11, 3)


def test_valid_email_is_accepted() -> None:
    value, error = validate_field("email", "bishal@example.com", TODAY)
    assert value == "bishal@example.com"
    assert error is None


def test_invalid_email_is_rejected() -> None:
    value, error = validate_field("email", "bishal@", TODAY)
    assert value is None
    assert error is not None


def test_past_date_is_rejected() -> None:
    value, error = validate_field("interview_date", "2026-01-01", TODAY)
    assert value is None
    assert error == "That date is in the past."


def test_future_date_is_accepted() -> None:
    value, _ = validate_field("interview_date", "2026-11-09", TODAY)
    assert value == "2026-11-09"


def test_time_is_normalised() -> None:
    value, _ = validate_field("interview_time", "14:00", TODAY)
    assert value == "14:00"


def test_json_with_code_fences_is_parsed() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_invalid_json_returns_empty_dict() -> None:
    assert parse_json_object("not json") == {}