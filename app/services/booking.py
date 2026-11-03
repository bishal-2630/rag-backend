import json
import re
from datetime import date, time
from typing import Any

from pydantic import BaseModel, EmailStr, TypeAdapter, ValidationError
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.booking import Booking
from app.services.llm import LLMService

BOOKING_HINTS = re.compile(
    r"\b(book|booking|schedule|reserve|appointment|interview|slot)\b", re.IGNORECASE
)

INTENT_SYSTEM_PROMPT = (
    "Decide whether the user's message asks to book, schedule or arrange an "
    "interview. Questions ABOUT the interview process (stages, duration, what to "
    "expect) are NOT booking requests. "
    'Reply with JSON only: {"wants_booking": true} or {"wants_booking": false}.'
)

EXTRACT_SYSTEM_PROMPT = (
    "Extract interview booking details from the user's message. Reply with JSON "
    'only, with exactly these keys: "name" (full name or null), "email" (or null), '
    '"interview_date" (YYYY-MM-DD or null), "interview_time" (24-hour HH:MM or '
    'null), "cancel" (true only if the user wants to stop booking). '
    "Resolve relative dates such as 'tomorrow' or 'next Monday' using today's "
    "date. Only use details the user actually stated; use null for anything "
    "missing. Never guess."
)

FIELD_LABELS: dict[str, str] = {
    "name": "your full name",
    "email": "your email address",
    "interview_date": "the interview date",
    "interview_time": "the interview time",
}

_email_adapter: TypeAdapter[EmailStr] = TypeAdapter(EmailStr)


class BookingSlots(BaseModel):
    """Booking details collected so far. None means 'not provided yet'."""

    name: str | None = None
    email: str | None = None
    interview_date: str | None = None
    interview_time: str | None = None

    def missing(self) -> list[str]:
        return [field for field in FIELD_LABELS if getattr(self, field) is None]


def parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def validate_field(
    field: str, value: str, today: date
) -> tuple[str | None, str | None]:
    """Return (clean_value, None) if valid, or (None, error_message) if not."""
    value = value.strip()
    if field == "name":
        if len(value) < 2:
            return None, "That name looks too short."
        return value, None
    if field == "email":
        try:
            return str(_email_adapter.validate_python(value)), None
        except ValidationError:
            return None, f"'{value}' doesn't look like a valid email address."
    if field == "interview_date":
        try:
            parsed_date = date.fromisoformat(value)
        except ValueError:
            return None, "I couldn't understand that date."
        if parsed_date < today:
            return None, "That date is in the past."
        return parsed_date.isoformat(), None
    if field == "interview_time":
        try:
            parsed_time = time.fromisoformat(value)
        except ValueError:
            return None, "I couldn't understand that time."
        return parsed_time.strftime("%H:%M"), None
    return None, None


class BookingService:
    def __init__(self, llm: LLMService, ttl_seconds: int = 1800) -> None:
        self._llm = llm
        self._redis: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
        self._ttl = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"booking:{session_id}"

    async def get_state(self, session_id: str) -> BookingSlots | None:
        raw = await self._redis.get(self._key(session_id))
        return BookingSlots.model_validate_json(raw) if raw else None

    async def _save_state(self, session_id: str, slots: BookingSlots) -> None:
        await self._redis.set(
            self._key(session_id), slots.model_dump_json(), ex=self._ttl
        )

    async def _clear_state(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))

    async def detect_intent(self, message: str) -> bool:
        if not BOOKING_HINTS.search(message):
            return False
        raw = await self._llm.generate(
            message, system=INTENT_SYSTEM_PROMPT, json_mode=True
        )
        return bool(parse_json_object(raw).get("wants_booking"))

    async def _extract(
        self, message: str, slots: BookingSlots, today: date
    ) -> dict[str, Any]:
        prompt = (
            f"Today is {today:%A, %Y-%m-%d}.\n"
            f"Already collected: {slots.model_dump_json()}\n"
            f"User message: {message}"
        )
        raw = await self._llm.generate(
            prompt, system=EXTRACT_SYSTEM_PROMPT, json_mode=True
        )
        return parse_json_object(raw)

    async def handle_turn(
        self, db: AsyncSession, session_id: str, message: str, slots: BookingSlots
    ) -> str:
        today = date.today()
        extracted = await self._extract(message, slots, today)

        if extracted.get("cancel") is True:
            await self._clear_state(session_id)
            return "No problem, I've cancelled the booking request."

        errors: list[str] = []
        for field in FIELD_LABELS:
            value = extracted.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            valid, error = validate_field(field, value, today)
            if valid is not None:
                setattr(slots, field, valid)
            elif error:
                errors.append(error)

        missing = slots.missing()
        if missing:
            await self._save_state(session_id, slots)
            needed = ", ".join(FIELD_LABELS[f] for f in missing)
            return " ".join([*errors, f"To book your interview, I still need: {needed}."])

        assert slots.name and slots.email
        assert slots.interview_date and slots.interview_time
        booking = Booking(
            name=slots.name,
            email=slots.email,
            interview_date=date.fromisoformat(slots.interview_date),
            interview_time=time.fromisoformat(slots.interview_time),
        )
        db.add(booking)
        await db.commit()
        await self._clear_state(session_id)
        return (
            f"Your interview is booked, {booking.name}! "
            f"Date: {booking.interview_date:%A, %d %B %Y} at "
            f"{booking.interview_time:%H:%M}. I've saved it under {booking.email}."
        )