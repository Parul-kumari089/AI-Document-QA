import json
import os

import streamlit as st

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

VECTORSTORE_PATH = "vectorstore/document.index"
CHUNKS_PATH = "vectorstore/chunks.json"


# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="📄"
)


# --------------------------------
# Title
# --------------------------------

st.title("📄 AI Document Q&A")

st.write(
    "Upload a PDF and ask questions about it."
)


# --------------------------------
# Upload PDF
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# --------------------------------
# Process PDF
# --------------------------------

if uploaded_file is not None:

    if st.button("Process Document"):

        os.makedirs(
            "vectorstore",
            exist_ok=True
        )

        # Save uploaded PDF
        pdf_path = os.path.join(
            "data",
            "documents",
            uploaded_file.name
        )

        os.makedirs(
            "data/documents",
            exist_ok=True
        )

        with open(
            pdf_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )


        # Extract pages
        with st.spinner(
            "Extracting text..."
        ):

            pages = extract_pages(
                pdf_path
            )


        # Split into chunks
        with st.spinner(
            "Splitting document..."
        ):

            chunks = split_pages(
                pages
            )


        # Get chunk text
        chunk_texts = [
            chunk["text"]
            for chunk in chunks
        ]


        # Create embeddings
        with st.spinner(
            "Creating embeddings..."
        ):

            embeddings = create_embeddings(
                chunk_texts
            )


        # Create FAISS
        with st.spinner(
            "Creating vector store..."
        ):

            index = create_vector_store(
                embeddings
            )


        # Save FAISS
        save_vector_store(
            index,
            VECTORSTORE_PATH
        )


        # Save chunks
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


        # Store index and chunks in session
        st.session_state["index"] = index
        st.session_state["chunks"] = chunks


        st.success(
            "Document processed successfully!"
        )


# --------------------------------
# Question section
# --------------------------------

if "index" in st.session_state:

    st.subheader("Ask a Question")

    question = st.text_input(
        "Enter your question:"
    )


    if st.button("Ask Question"):

        if question.strip() == "":

            st.warning(
                "Please enter a question."
            )

        else:

            index = st.session_state[
                "index"
            ]

            chunks = st.session_state[
                "chunks"
            ]


            # Create question embedding
            with st.spinner(
                "Searching document..."
            ):

                query_embedding = (
                    create_embeddings(
                        [question]
                    )
                )

                query_embedding = (
                    query_embedding.astype(
                        "float32"
                    )
                )


                # Search FAISS
                distances, indices = (
                    search_vector_store(
                        index,
                        query_embedding,
                        k=3
                    )
                )


            # Get relevant chunks
            retrieved_chunks = []

            for index_id in indices[0]:

                retrieved_chunks.append(
                    chunks[index_id]["text"]
                )


            # Create context
            context = "\n\n".join(
                retrieved_chunks
            )


            # Generate answer
            with st.spinner(
                "Generating answer..."
            ):

                answer = generate_answer(
                    question,
                    context
                )


            # Display answer
            st.subheader("Answer")

            st.write(answer)


            # Display sources
            st.subheader("Sources")

            for rank, (
                distance,
                index_id
            ) in enumerate(
                zip(
                    distances[0],
                    indices[0]
                ),
                start=1
            ):

                chunk = chunks[index_id]

                st.write(
                    f"**Source {rank} — "
                    f"Page {chunk['page']}**"
                )

                st.write(
                    chunk["text"]
                )

                st.divider()