from langchain_community.tools import DuckDuckGoSearchResults
def search_web(query: str) -> str:
  return DuckDuckGoSearchResults(backend="news").run(query)

tool_search_web = {'type':'function', 'function':{
  'name': 'search_web',
  'description': 'Search the web',
  'parameters': {'type': 'object',
                'required': ['query'],
                'properties': {
                    'query': {'type':'str', 'description':'the topic or subject to search on the web'},
}}}}
## test
search_web(query="nvidia")