import re


def cleanText(text: str) -> str:
    """
    This function cleans the text by removing all the unnecessary characters.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
