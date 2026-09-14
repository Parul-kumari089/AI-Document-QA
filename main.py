import json

from src.document_loader import extract_pages
from src.text_splitter import split_pages
from src.embeddings import create_embeddings

from src.vector_store import (
    create_vector_store,
    search_vector_store,
    save_vector_store
)

from src.qa import generate_answer


# --------------------------------
# File paths
# --------------------------------

PDF_PATH = "data/documents/employee_handbook.pdf"

VECTORSTORE_PATH = "vectorstore/document.index"

CHUNKS_PATH = "vectorstore/chunks.json"


def main():

    # --------------------------------
    # 1. Load PDF
    # --------------------------------

    print("\nLoading PDF...")

    pages = extract_pages(PDF_PATH)

    print("PDF loaded successfully.")
    print("Number of pages:", len(pages))


    # --------------------------------
    # 2. Split document into chunks
    # --------------------------------

    print("\nSplitting document...")

    chunks = split_pages(pages)

    print("Number of chunks:", len(chunks))


    # --------------------------------
    # 3. Save chunks with page numbers
    # --------------------------------

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("Chunks saved successfully.")


    # --------------------------------
    # 4. Get only the text
    # --------------------------------

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]


    # --------------------------------
    # 5. Create embeddings
    # --------------------------------

    print("\nCreating embeddings...")

    embeddings = create_embeddings(
        chunk_texts
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )


    # --------------------------------
    # 6. Create FAISS vector store
    # --------------------------------

    print("\nCreating FAISS vector store...")

    index = create_vector_store(
        embeddings
    )

    print("FAISS vector store created.")


    # --------------------------------
    # 7. Save FAISS vector store
    # --------------------------------

    save_vector_store(
        index,
        VECTORSTORE_PATH
    )

    print("FAISS vector store saved.")


    # --------------------------------
    # 8. Ask question
    # --------------------------------

    question = input(
        "\nAsk a question about the document: "
    )


    # --------------------------------
    # 9. Create question embedding
    # --------------------------------

    query_embedding = create_embeddings(
        [question]
    )

    query_embedding = query_embedding.astype(
        "float32"
    )


    # --------------------------------
    # 10. Search FAISS
    # --------------------------------

    distances, indices = search_vector_store(
        index,
        query_embedding,
        k=3
    )


    # --------------------------------
    # 11. Retrieve relevant chunks
    # --------------------------------

    retrieved_chunks = []

    for index_id in indices[0]:

        retrieved_chunks.append(
            chunks[index_id]["text"]
        )


    # --------------------------------
    # 12. Create context
    # --------------------------------

    context = "\n\n".join(
        retrieved_chunks
    )


    # --------------------------------
    # 13. Generate answer
    # --------------------------------

    print("\nGenerating answer...")

    answer = generate_answer(
        question,
        context
    )


    # --------------------------------
    # 14. Display answer
    # --------------------------------

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(answer)


    # --------------------------------
    # 15. Display sources
    # --------------------------------

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)


    for rank, (distance, index_id) in enumerate(
        zip(distances[0], indices[0]),
        start=1
    ):

        chunk = chunks[index_id]

        print(f"\nSource {rank}")

        print(f"Page: {chunk['page']}")

        print(f"Distance: {distance:.4f}")

        print("-" * 60)

        print(chunk["text"])


if __name__ == "__main__":
    main()