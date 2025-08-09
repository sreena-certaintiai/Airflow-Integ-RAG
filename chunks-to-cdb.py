import os

# --- Import our custom functions and new DB manager ---
from sen_based_chunk import extract_text_from_pdfs, split_text_into_chunks
from cdb_manager import get_collection, get_embedding_model

# --- Main Execution ---
if __name__ == "__main__":
    pdf_folder = "data"

    if not os.path.exists(pdf_folder) or not any(f.endswith('.pdf') for f in os.listdir(pdf_folder)):
        print(f"The '{pdf_folder}' folder does not exist or is empty.")
        print("Please add your PDF files to this folder and run step1_chunker.py first.")
    else:
        # --- 1. Get DB Collection and Embedding Model from our manager ---
        collection = get_collection()
        embedding_model = get_embedding_model()

        # --- 2. Embed and Store Data (if not already done) ---
        if collection.count() > 0:
            print(
                f"Collection '{collection.get().get('name')}' already contains {collection.count()} documents. Skipping ingestion.")
        else:
            print("Loading and chunking documents...")
            raw_text = extract_text_from_pdfs(pdf_folder)
            text_chunks = split_text_into_chunks(raw_text)
            print("Data preparation complete.")

            print("Generating embeddings and storing in ChromaDB. This may take a moment...")
            embeddings = embedding_model.encode(text_chunks, show_progress_bar=True)
            ids = [f"chunk_{i}" for i in range(len(text_chunks))]

            collection.add(documents=text_chunks, embeddings=embeddings, ids=ids)
            print("Successfully stored documents in ChromaDB.")

        # --- 3. Quick Test (Optional) ---
        # print("\n--- Running a quick test query ---")
        # results = collection.query(
        #     query_texts=["What is the main topic of the document?"],
        #     n_results=2
        # )
        # print("Top 2 relevant document chunks:")
        # for doc in results['documents'][0]:
        #     print("  - " + doc.replace("\n", " ")[:150] + "...")