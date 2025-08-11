# query_data.py

import textwrap
# Notice we only need to import our manager now
from Core.cdb_manager import get_collection, get_embedding_model


def search_db(query: str, num_results: int = 3):
    """
    Takes a user query, embeds it, and searches the ChromaDB collection.

    Args:
        query (str): The user's question.
        num_results (int): The number of results to return.

    Returns:
        A dictionary containing the search results.
    """
    # Get the singleton instances from our manager
    collection = get_collection()
    embedding_model = get_embedding_model()

    # 1. Create the embedding for the user's query
    query_embedding = embedding_model.encode(query).tolist()

    # 2. Query the collection with the embedding
    results = collection.query(
        query_embeddings=[query_embedding],  # Note: it's a list of embeddings
        n_results=num_results
    )

    return results['documents'][0]


if __name__ == "__main__":
    print("--- 📚 Interactive Document Query Session ---")
    print("Ask a question about your documents. Type 'exit' to quit.")

    while True:
        # Added an extra newline for better spacing before the prompt
        user_question = input("\n\nAsk a question: ")

        if user_question.lower() == 'exit':
            print("Exiting session. Goodbye!")
            break

        if not user_question.strip():
            continue

        relevant_chunks = search_db(user_question, num_results=3)

        print("\n✅ --- Here are the most relevant results ---")
        if relevant_chunks:
            for i, chunk in enumerate(relevant_chunks):
                # This block is the main change for better formatting
                print(f"\n--- Result {i + 1} ---")
                # We use textwrap to indent the whole block of text
                formatted_text = textwrap.indent(text=chunk, prefix="    ")
                print(formatted_text)
        else:
            print("\n❌ No relevant documents found for your query.")