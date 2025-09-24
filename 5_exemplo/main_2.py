from langchain_community.tools import DuckDuckGoSearchResults
def search_yf(query: str) -> str:
  engine = DuckDuckGoSearchResults(backend="news")
  return engine.run(f"site:finance.yahoo.com {query}")

tool_search_yf = {'type':'function', 'function':{
  'name': 'search_yf',
  'description': 'Search for specific financial news',
  'parameters': {'type': 'object',
                'required': ['query'],
                'properties': {
                    'query': {'type':'str', 'description':'the financial topic or subject to search'},
}}}}

## test
search_yf(query="nvidia")