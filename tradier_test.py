# Tradier-test
import requests

API_TOKEN = "SbABem4wpijXKWdRzAQfrQwVVsXA"
BASE_URL = "https://sandbox.tradier.com/v1/markets"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Accept": "application/json"
}

def get_expirations(symbol):
    url = f"{BASE_URL}/options/expirations?symbol={symbol}"
    resp = requests.get(url, headers=headers)
    return resp.json()

def get_option_chain(symbol, expiration):
    url = f"{BASE_URL}/options/chains?symbol={symbol}&expiration={expiration}"
    resp = requests.get(url, headers=headers)
    return resp.json()

# Example usage:
symbol = "AAPL"

expirations = get_expirations(symbol)
print("Expirations:", expirations)

# Pick first expiration date
first_exp = expirations["expirations"]["date"][0]
chain = get_option_chain(symbol, first_exp)
print("Option Chain:", chain)
