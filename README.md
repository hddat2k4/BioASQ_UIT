# RAG for Biomedical Question Answering at BioASQ 13b

This repository provides the source code and a small batch of dataset samples used to build a Retrieval-Augmented Generation (RAG) pipeline for biomedical question answering based on the BioASQ 13b challenge.


## 📜 Table of Content
- [Introduction](#intro)
- [Usage](#usage)

## Introduction 
<a name = 'intro'></a>

This project implements a RAG pipeline that combines dense retrieval and generative models to answer biomedical questions. The system is designed to be compatible with the BioASQ 13b benchmark.



## Usage
<a name = 'usage'></a>

### 1. Clone the repository and navigate to the working directory
```
git init
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