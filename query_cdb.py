# query_data.py

import pprint

# Notice we only need to import our manager now
from cdb_manager import get_collection, get_embedding_model


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

    # A loop to keep the session going
    while True:
        user_question = input("\nAsk a question: ")

        if user_question.lower() == 'exit':
            print("Exiting session. Goodbye!")
            break

        if not user_question.strip():
            continue

        # 3. Perform the search and get the results
        relevant_chunks = search_db(user_question)

        # 4. Print the results nicely
        print("\n--- Most Relevant Chunks ---")
        if relevant_chunks:
            for i, chunk in enumerate(relevant_chunks):
                print(f"Result {i + 1}:")
                # Using pprint to handle multiline strings nicely
                pprint.pprint(chunk)
                print("-" * 20)
        else:
            print("No relevant documents found for your query.")