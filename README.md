# EX 02 – Design and Implementation of LangChain Expression Language (LCEL) Expressions

## AIM

To design and implement a LangChain Expression Language (LCEL) pipeline using Prompt Templates, Google Gemini Language Model, and StrOutputParser to generate personalized career guidance. The project demonstrates both simple and context-aware LCEL workflows.

---

## PROBLEM STATEMENT

Develop an AI Career Guidance Assistant using LangChain Expression Language (LCEL). The system accepts user inputs such as technical skills and career interests, processes them using the Google Gemini Language Model, and generates structured career recommendations. It also includes a context-based LCEL workflow to answer career-related questions.

---

## DESIGN STEPS

### STEP 1
Install the required Python libraries (`langchain` and `langchain-google-genai`) and configure the Google Gemini API key.

### STEP 2
Initialize the Google Gemini model, create prompt templates with multiple input parameters, and configure the output parser.

### STEP 3
Build a simple LCEL pipeline by connecting the Prompt Template, Gemini Model, and Output Parser using the `|` operator.

### STEP 4
Create a context-aware LCEL workflow using `RunnableMap` to provide additional context.

### STEP 5
Execute both the simple and context-based LCEL pipelines and display the generated outputs.

---

## PROGRAM

```python
# Install Required Libraries
!pip install -U langchain-google-genai langchain

import os

# Configure Gemini API Key
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap

# Initialize Gemini Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# Prompt Template
career_prompt = ChatPromptTemplate.from_template(
"""
You are a professional career counselor.

Student Skills:
{skills}

Career Interest:
{interest}

Suggest:
1. Best Career Path
2. Required Skills
3. Learning Resources
4. Career Tips

Keep the explanation simple.
"""
)

# Output Parser
parser = StrOutputParser()

# Simple LCEL Chain
career_chain = career_prompt | model | parser

response = career_chain.invoke(
    {
        "skills": "Python, Machine Learning, SQL",
        "interest": "Artificial Intelligence Engineer"
    }
)

print(response)

# Context-Based Prompt
context_prompt = ChatPromptTemplate.from_template(
"""
Context:
{context}

Question:
{question}

Answer:
"""
)

inputs = RunnableMap({
    "context": lambda x:
        "Artificial Intelligence Engineers develop intelligent systems using Machine Learning, Deep Learning, Python, TensorFlow, and Data Science.",
    "question": lambda x: x["question"]
})

context_chain = inputs | context_prompt | model | parser

response = context_chain.invoke(
    {
        "question": "Which programming languages should I learn for becoming an AI Engineer?"
    }
)

print(response)
```

---

## OUTPUT

### Simple LCEL Output

```text
Best Career Path:
Artificial Intelligence Engineer

Required Skills:
- Python
- Machine Learning
- Deep Learning
- TensorFlow
- SQL

Learning Resources:
- Coursera
- Kaggle
- Fast.ai

Career Tips:
Build AI projects, contribute to GitHub, and improve your problem-solving skills.
```

### Context-Based LCEL Output

```text
Python is the primary programming language for AI development.

You should also learn:
- SQL
- Java or C++

Popular AI frameworks include TensorFlow, PyTorch, and Scikit-learn.
```

---

## RESULT

The AI Career Guidance Assistant was successfully developed using LangChain Expression Language (LCEL). The project integrates Prompt Templates, Google Gemini Language Model, RunnableMap, and StrOutputParser to implement both simple and context-aware AI workflows. The generated outputs demonstrate the effectiveness of LCEL in building modular and efficient AI applications.
