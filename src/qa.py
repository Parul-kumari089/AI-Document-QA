import ollama


def generate_answer(question, context):

    prompt = f"""
You are an AI assistant that answers questions
using information from an uploaded document.

Use ONLY the provided context to answer the question.

Do not use outside knowledge.

If the answer is not present in the context,
say:

"I could not find the answer in the uploaded document."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]