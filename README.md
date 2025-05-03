# RAG for Biomedical Question Answering at BioASQ 13b

This repository provides the source code and a small batch of dataset samples used to build a Retrieval-Augmented Generation (RAG) pipeline for biomedical question answering based on the BioASQ 13b challenge.


## Table of Content
- [Introduction](#intro)
- [Target] (#target)
- [Main Components](#main)

- [Usage](#usage)

## Introduction 
<a name = 'intro'></a>

This project implements a RAG pipeline that combines dense retrieval and generative models to answer biomedical questions. The system is designed to be compatible with the BioASQ 13b benchmark.


## Target
<a name = 'target'> </a>

The main objective is to build a robust and flexible Retrieval-Augmented Generation (RAG) system tailored to biomedical question answering in the BioASQ 13b challenge. This system integrates dense retrieval using vector databases with large language models to generate accurate and explainable answers across different question types (yes/no, factoid, list, and summary).

## Main Components
<a name='main-components'></a>

- **Retriever**: Uses SentenceTransformers (`avsolatorio/GIST-large-Embedding-v0`) to generate embeddings. Retrieval is done via Weaviate (supports BM25, dense, and hybrid).
- **Generator**: Utilizes LLMs to generate answers from retrieved documents (gemini-2.0-flash, gpt-4.1-mini)
- **Indexing**: PubMed abstracts are split into sentence-level chunks using a sliding window, then indexed into Weaviate.
- **Evaluation**: Custom evaluation scripts that compare system outputs with BioASQ ground truth (F1, Accuracy, ROUGE, BLEU).


## Usage
<a name = 'usage'></a>

### 1. Clone the repository and navigate to the working directory
```
git clone https://github.com/hddat2k4/BioASQ_UIT.git
cd /d BioASQ
``` 
### 2. Create a virtual environment
```
python -m venv venv
.\venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/MacOS
```

### 3. Install Weaviate via Docker using the docker-compose.yml file in repo
```
docker-compose up -d
```

### 4. Create a schema for the Weaviate collection

```
python create_weaviate.py
```

### 5. Index vectors into the collection 

⚠️ Note: Ensure you have already embedded your data and verified the correct directory path before indexing.

```
python we_index.py
```

### 6. Edit train.py to point to your question file
Ensure your question file follows the format below:


```
{
    "question_id": "ID of the question",
    "question": "Your question will be here",
    "type": "Question type"

}
```