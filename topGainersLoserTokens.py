import requests

url = "https://pro-api.coingecko.com/api/v3/coins/top_gainers_losers?duration=24h&top_coins=1000"

headers = {"accept": "application/json"}

response = requests.get(url, headers=headers)

print(response.text)
