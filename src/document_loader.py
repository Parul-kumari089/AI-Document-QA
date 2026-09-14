from pypdf import PdfReader


def clean_text(text):
    """
    Remove unwanted control characters
    from extracted PDF text.
    """

    cleaned = ""

    for char in text:

        if char.isprintable() or char in "\n\t":

            cleaned += char

    return cleaned


def extract_pages(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            text = clean_text(text)

            pages.append({
                "text": text,
                "page": page_number
            })

    return pages