from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()

tavily = TavilySearch(max_results=3)

result = tavily.invoke({
    "query": "latest machine learning trends"
})

print(result)