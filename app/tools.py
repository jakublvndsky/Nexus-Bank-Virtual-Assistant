import requests
from langchain.tools import tool
from app.vector_db import initialize_vector_db


@tool(
    "currency_exchange",
    description=("Used to search for a currency exchange rate by its ISO 4217 code."),
    parse_docstring=True,
)
def check_currency_rate(currency_code: str) -> dict:
    """Checks exchange rates from Polish Zloty (PLN) to foreign currencies.

    Args:
        currency_code: The currency code according to the ISO 4217 standard.
    """

    r = requests.get(
        f"http://api.nbp.pl/api/exchangerates/rates/A/{currency_code}/?format=json",
        timeout=120,
    )
    if r.status_code == 200:
        response = r.json()
        return response
    else:
        raise Exception(f"Invalid service status, specifically: {r.status_code}")


vector_store = initialize_vector_db()


@tool
def retrieve_from_vector_db(query: str):
    """
    Searches a vector database containing the file @data/Nexus Bank Terms and Conditions.pdf,
    which is a PDF document about the fictional Nexus Bank.
    Use this tool when you want to retrieve information, extracts, rules, policies, or details
    about the services and terms offered by Nexus Bank, based on text queries.
    """
    search_sim = vector_store.similarity_search(query=query, k=3)
    return search_sim
