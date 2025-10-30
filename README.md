
# Semantic Book Recommender  
**Natural-Language Book Discovery Powered by Embeddings & Large Language Models**

![Semantic Book Recommender Dashboard](https://github.com/yashkhan42/LLM-semantic-book-recommender/blob/main/dashboard-selfhelp-query.png)  
*Live dashboard in action: Query "a self help book" → nonfiction category → surprising emotional tone → semantic matches with visual book covers*

## Project Overview

This project builds an **intelligent, natural-language book recommender** that understands user intent beyond keyword matching. Users describe a book in free-form text (e.g., "a dark fantasy story about betrayal and revenge with morally grey characters" or "a story about galaxies"), optionally filter by fiction/non-fiction, and sort by emotional tone (e.g., most "joyful" or "suspenseful" first).

The system uses **semantic vector search** on book descriptions + lightweight emotion classification to deliver highly relevant recommendations — a modern Retrieval-Augmented Generation (RAG)-style approach without requiring a massive collaborative filtering dataset.


## Key Features & Technologies

- **Semantic Retrieval** — Dense vector embeddings of book blurbs for meaning-based matching (OpenAI embeddings or free local alternatives like BAAI/bge-small-en-v1.5)
- **Fiction / Non-Fiction Filtering** — Zero-shot classification with transformer pipelines (`facebook/bart-large-mnli`)
- **Emotional Tone Enrichment** — Multi-label emotion scoring on descriptions (go_emotions model) enabling mood-based sorting
- **Interactive Web Interface** — Clean, dark-mode Gradio dashboard with query input, category dropdown, emotion selector, and thumbnail grid display
- **Efficient Local Vector Store** — ChromaDB for fast similarity search on ~7,000 books
- **Data Pipeline** — Kaggle 7k Books dataset → cleaning → tagging → embedding → classification → emotion extraction

### Tech Stack
- Python 3.12+
- LangChain + LangChain-Community / LangChain-Chroma / LangChain-OpenAI (or HuggingFace)
- Transformers (Hugging Face) for zero-shot & emotion models
- Gradio for the responsive UI
- Pandas for data wrangling
- Chroma for vector database
- python-dotenv for secure API key handling

## Results & Demonstration

For the query **"a story about galaxies"** (fiction, any emotional tone), the system returns highly relevant science-fiction classics and modern titles with strong thematic similarity:

- Isaac Asimov's *Gold*
- Terry Brooks / George Lucas-inspired space opera entries
- John Brunner's *The Crucible of Time*
- Douglas Adams' *The Ultimate Hitchhiker's Guide*
- Various Star Wars / space adventure novels

The interface displays cover thumbnails (when available), titles, authors, and short snippets — making discovery intuitive and visually engaging.

![Dashboard Screenshot - Galaxies Query](https://github.com/yashkhan42/LLM-semantic-book-recommender/blob/main/dashboard-galaxies-query.png)  
*(Click to enlarge — full grid of semantic matches with dark theme UI)*

## File Structure

| File / Folder                  | Description |
|-------------------------------|-------------|
| `data-exploration.ipynb`      | Dataset download (Kaggle 7k books), EDA, cleaning, tagged description generation |
| `vector-search.ipynb`         | Embedding computation + Chroma vector database creation |
| `text-classification.ipynb`   | Zero-shot fiction/non-fiction labeling |
| `sentiment-analysis.ipynb`    | Emotion score extraction (anger, joy, suspense, etc.) |
| `gradio-dashboard.py`         | Main interactive application (runs the live recommender) |
| `books_with_emotions.csv`     | Final enriched dataset (~7k books with embeddings metadata, fiction flag, emotion scores) |
| `chroma_db/`                  | Persistent vector store (do not commit to git — large) |
| `requirements.txt`            | All dependencies (pip install -r requirements.txt) |
| `.env`                        | API keys (not committed — use `.env.example`) |
| `screenshots.png`                | Dashboard images for documentation |

## Local Setup

### Prerequisites
- Python 3.10+
- Git
- (Optional) OpenAI API key for high-quality embeddings (or use free local models)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yashkhan42/LLM-semantic-book-recommender.git
   cd LLM-semantic-book-recommender