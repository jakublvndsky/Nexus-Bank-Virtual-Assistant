import os
import pandas as pd
import asyncio
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from dotenv import load_dotenv
from app import build_agent, initialize_vector_db
from ragas import EvaluationDataset
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    FactualCorrectness,
    ContextRecall,
    ContextPrecision,
)
from ragas.run_config import RunConfig

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
vector_store = initialize_vector_db()
agent = build_agent()
llm = ChatOpenAI(api_key=OPENAI_API_KEY)


def get_answer(question: str):
    docs = vector_store.similarity_search(query=question, k=4)

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f""" You are a chatbot for a Nexus Bank, your job is mainly customer service. 
    You need to answer the questions base on the given context

    Context:
    {context}

    Question:
    {question}

    Answer:
     """

    response = llm.invoke(prompt)

    return response.content


user_question = "What type of accounts do you have?"
get_answer(user_question)

queries = [
    {
        "question": "What are the account types offered by Nexus Bank, Summarize in 1 line",
        "answer": "Nexus Bank offers Standard, Premium, and Savings Plus accounts for its retail clients.",
        "reference": "Always search your knowledge base before answering questions related to account types (Standard, Premium, Savings Plus), fees, commissions, interest rates, or banking procedures.",
    },
    {
        "question": "What is the mandatory commission fee for foreign exchange transactions, Summarize in 1 line",
        "answer": "Every foreign exchange transaction incurs a flat operational commission of 2.50 PLN.",
        "reference": "Mandatory Commission Calculation: According to the Terms and Conditions, every foreign exchange transaction incurs a flat operational commission of 2.50 PLN.",
    },
    {
        "question": "What sensitive information is the Nexus Assistant strictly prohibited from asking the user, Summarize in 1 line",
        "answer": "The assistant must never ask users for their passwords, PIN codes, full debit card numbers, or SMS OTPs.",
        "reference": "No Authentication Processing: Do not ask users for their passwords, PIN codes, full debit card numbers, or SMS OTPs.",
    },
    {
        "question": "Can the Nexus Assistant provide financial advice on when to buy currency, Summarize in 1 line",
        "answer": "No, the assistant is strictly an informational tool and cannot advise clients on whether it is a good time to buy currency.",
        "reference": "No Financial Advice: You are an informational assistant, not a financial advisor. Never advise clients on whether it is a 'good time' to buy a currency or open a deposit.",
    },
    {
        "question": "What is the fee for withdrawing cash from a foreign ATM using a Standard Account, Summarize in 1 line",
        "answer": "A foreign ATM withdrawal with a Standard Account costs a 15.00 PLN fee, plus an additional 2.50 PLN FX commission for the currency conversion.",
        "reference": "Retrieve Standard Account foreign ATM withdrawal fees from knowledge base (Result: 15.00 PLN). Note that foreign ATMs require currency conversion (EUR). Retrieve NBP FX Commission from knowledge base (Result: 2.50 PLN).",
    },
]

df = pd.DataFrame(queries)

retriver = vector_store.as_retriever()


def process_query(query):
    result = get_answer(query)
    relevant_docs = retriver.invoke(query)
    return result, relevant_docs


process_query("What document says about fees?")

### RAG Evaluation
llm_2 = AsyncOpenAI(api_key=OPENAI_API_KEY, max_retries=2)
results = []

for _, row in df.iterrows():
    question = row["question"]
    ground_truth = row["answer"]

    answer, relevant_docs = process_query(question)

    results.append(
        {
            "user_input": question,
            "reference": ground_truth,
            "response": answer,
            "retrived_contexts": [relevant_docs[0].page_content],
        }
    )

evaluation_dataset = EvaluationDataset.from_list(results)
evaluator_llm = llm_factory(model="gpt-5-mini", client=llm_2, max_tokens=8192)

run_config = RunConfig(timeout=120, max_retries=3, max_workers=3, log_tenacity=True)


faithfulness_score = Faithfulness(llm=evaluator_llm)
factual_correctness = FactualCorrectness(llm=evaluator_llm)
context_recall = ContextRecall(llm=evaluator_llm)
context_precision = ContextPrecision(llm=evaluator_llm)


async def evaluation():
    final_score = []
    for i, result in enumerate(results):
        print(f"=== Starting faithfulness eval for {result['user_input']} ===")
        user_input = result["user_input"]
        response = result["response"]
        retrieved_contexts = result["retrived_contexts"]
        score = await faithfulness_score.ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        final_score.append({f"faithfulness_{i}": score.value})
    for i, result in enumerate(results):
        print(f"=== Starting factual correctness eval for {result['user_input']} ===")
        response = result["response"]
        reference = result["reference"]
        score = await factual_correctness.ascore(response=response, reference=reference)
        final_score.append({f"factual_correctness_{i}": score.value})
    for i, result in enumerate(results):
        print(f"=== Starting context recall eval for {result['user_input']} ===")
        user_input = result["user_input"]
        reference = result["reference"]
        retrieved_contexts = result["retrived_contexts"]
        score = await context_recall.ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
        final_score.append({f"context_recall_{i}": score.value})
    for i, result in enumerate(results):
        print(f"=== Starting context precision eval for {result['user_input']} ===")
        user_input = result["user_input"]
        reference = result["reference"]
        retrieved_contexts = result["retrived_contexts"]
        score = await context_precision.ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
        final_score.append({f"context_precision_{i}": score.value})
    return final_score


result = asyncio.run(evaluation())
print(result)
