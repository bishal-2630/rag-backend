from io import BytesIO

from pypdf import PdfReader


class UnsupportedFileTypeError(ValueError):
    """Raised when the uploaded file is not a .pdf or .txt."""


def extract_text(filename: str, content: bytes) -> str:
    """Return the plain text of a PDF or TXT file."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    if lower.endswith(".txt"):
        return content.decode("utf-8", errors="ignore").strip()
    raise UnsupportedFileTypeError(f"Unsupported file type: {filename}")