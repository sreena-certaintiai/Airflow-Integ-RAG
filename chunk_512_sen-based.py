import fitz  # PyMuPDF
import os
import tiktoken
import nltk

# Download the sentence tokenizer model (only needs to be done once)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading 'punkt' model for NLTK...")
    nltk.download('punkt')


def extract_text_from_pdfs(folder_path: str) -> str:
    """
    Extracts all text from every PDF file in a given folder.

    Args:
        folder_path: The path to the folder containing PDF files.

    Returns:
        A single string concatenating the text from all PDFs.
    """
    print(f"Reading PDFs from: {folder_path}")
    full_text = ""
    # Ensure the folder exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder '{folder_path}' does not exist.")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                with fitz.open(file_path) as doc:
                    # Extract text from each page and join
                    text = "".join(page.get_text() for page in doc)
                    full_text += text + "\n"  # Add a newline between docs
                    print(f"  - Extracted text from {filename}")
            except Exception as e:
                print(f"Could not read {filename}: {e}")
    return full_text


def split_text_into_chunks(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """
    Splits a long text into smaller chunks based on token count,
    while trying to respect sentence boundaries and adding overlap.

    Args:
        text: The input text to be split.
        chunk_size: The target maximum size of each chunk in tokens.
        overlap: The number of tokens to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    print("Splitting text into chunks...")
    # Get the tokenizer for a modern OpenAI model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Split the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Get the token count for each sentence
    sentence_tokens = [len(tokenizer.encode(s)) for s in sentences]

    chunks = []
    current_chunk_tokens = 0
    current_chunk_sentences = []

    for i, sentence in enumerate(sentences):
        # If adding the next sentence doesn't exceed the chunk size, add it
        if current_chunk_tokens + sentence_tokens[i] <= chunk_size:
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens[i]
        else:
            # Chunk is full, finalize it
            chunks.append(" ".join(current_chunk_sentences))

            # --- Start a new chunk with overlap ---
            # We'll work backwards from the current sentence to create the overlap
            new_chunk_sentences = []
            overlap_tokens = 0

            # Iterate backwards from the end of the last chunk
            for j in range(len(current_chunk_sentences) - 1, -1, -1):
                sentence_to_add = current_chunk_sentences[j]
                tokens_to_add = len(tokenizer.encode(sentence_to_add))

                if overlap_tokens + tokens_to_add <= overlap:
                    # Add sentence to the start of the new chunk
                    new_chunk_sentences.insert(0, sentence_to_add)
                    overlap_tokens += tokens_to_add
                else:
                    break  # Stop when overlap is filled

            # The new chunk starts with the overlap and the current sentence
            current_chunk_sentences = new_chunk_sentences + [sentence]
            current_chunk_tokens = overlap_tokens + sentence_tokens[i]

    # Add the last remaining chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    print(f"Successfully created {len(chunks)} chunks.")
    return chunks


# --- Main Execution ---
if __name__ == "__main__":
    # Create a folder named 'data' and put your PDFs inside it
    pdf_folder = "data"

    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        print(f"Created a folder named '{pdf_folder}'.")
        print("Please add your PDF files to this folder and run the script again.")
    else:
        # Step 1: Extract text from all PDFs in the folder
        raw_text = extract_text_from_pdfs(pdf_folder)

        if raw_text:
            # Step 2: Split the extracted text into chunks
            text_chunks = split_text_into_chunks(raw_text, chunk_size=512, overlap=50)

            # Let's inspect the first chunk to see the result
            if text_chunks:
                print("\n--- Example Chunk (First Chunk) ---")
                print(text_chunks[0])
                print("\n------------------------------------")
                print(
                    f"Total tokens in first chunk: {len(tiktoken.get_encoding('cl100k_base').encode(text_chunks[0]))}")
        else:
            print("No text was extracted. Ensure your 'data' folder contains valid PDF files.")
