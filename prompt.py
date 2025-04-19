from langchain.prompts import ChatPromptTemplate


## 01 - yesno, 02 - factoid, 03 - list, 04 - summary
## Create an LLM prompt for each type of question


prompt_01 = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a biomedical expert specializing in question answering and scientific reasoning. "
        "You answer biomedical yes/no questions using short, academic-style sentences grounded in scientific context."
    ),
    (
        "user",
        '''
Given the following yes/no biomedical question:

"{question}"

And the following context:

{context}

Follow this format for your answer:
Example 1:
Question: Is RANKL secreted from the cells?  
Ideal answer: Receptor activator of nuclear factor κB ligand (RANKL) is a cytokine predominantly secreted by osteoblasts.
Exact answer: Yes

Example 2:
Question: Is the protein Papilin secreted?  
Ideal answer: Yes, papilin is a secreted protein.
Exact Answer: Yes

If the context does not provide enough information, still provide your best yes/no guess based on scientific reasoning.

Your output must be strictly in this **valid JSON format**:

{{
  "exact_answer": "Just answer "yes" or "no", without any extra informations"
  "ideal_answer": "Your academic-style explanation here"
}}

Only return **valid JSON**. Do **not** include any other explanation, markdown, or text outside the JSON.
        '''
    )
])


prompt_02 = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are BioASQ-GPT, a biomedical expert specializing in question answering and scientific reasoning. "
        "You answer questions with short, academic-style sentences using biomedical context."
    ),
    (
        "user",
        """
Given the following biomedical Factoid question:

"{question}"

And the following context:

{context}

Return your result strictly in this JSON format:

{{
  "ideal_answer": "A complete, academic-style sentence that fully answers the question.",
  "exact_answer": "A short phrase with the core information, no abbreviation behind."
}}

Only return valid JSON. Do not include any explanations, introductions, or additional text outside the JSON.
"""
    )

])


prompt_03 = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are BioASQ-GPT, a biomedical expert specializing in question answering and scientific reasoning. "
        "You answer questions with short, academic-style sentences using biomedical context. "
        "Be precise and extract only biomedical entities or observations relevant to the question."
    ),
    ("user", """
Given the following biomedical list-type question:

"{question}"

And the relevant context:

{context}

Please return two types of answers:

1. "ideal_answer": A concise, natural academic-style summary as if written by an expert who understands the biomedical facts. 
The answer should read fluently without referring to the context or source.

2. "exact_answer": A list of specific biomedical entities, diagnoses, or clinical findings directly extracted from the context. 
Each entry must be a list containing exactly one string. For example:
[
  ["hallucination"],
  ["orthostatic hypotension"],
  ["dizziness"]
]

Respond strictly in the following valid JSON format (without extra explanation):

{{
  "ideal_answer": "...",
  "exact_answer": [
    ["..."],
    ["..."]
  ]
}}
""")
])



prompt_04 = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are BioASQ-GPT, a biomedical expert specializing in question answering and scientific reasoning. "
        "You answer questions with short, academic-style sentences using biomedical context."
    ),
    ("user", """
Given the biomedical summary-type question:

"{question}"

And the relevant context:

{context}

Return a single "ideal_answer": a well-written, academic-style paragraph that directly and completely answers the question.

Respond strictly in this JSON format:

{{
    "ideal_answer": "...",

}}

Only return valid JSON. Do not add any other text outside the JSON object.
""")
])

## function for getting prompt
def get_prompt(text: str) -> ChatPromptTemplate:
    prompt_map = {
        "yesno": prompt_01,
        "factoid": prompt_02,
        "list": prompt_03,
        "summary": prompt_04
    }
    return prompt_map.get(text, None)