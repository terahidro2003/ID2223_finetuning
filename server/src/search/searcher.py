"""DuckDuckGo search implementation"""
from typing import List, Dict
from duckduckgo_search import DDGS


class DuckDuckGoSearcher:
    """DuckDuckGo web search"""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        region: str = "wt-wt",
        safesearch: str = "moderate"
    ) -> Dict:
        """
        Args:
            query: Search query
            max_results: Maximum number of results
            region: Region code (wt-wt, us-en, etc.)
            safesearch: Safety level (on, moderate, off)
            
        Returns:
            Dict with 'results' list containing search results
        """
        try:
            results = DDGS().text(
                query,
                region=region,
                safesearch=safesearch,
                max_results=max_results
            )
            
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "content": r.get("body", "")
                })
            
            return {"results": formatted_results}
            
        except Exception as e:
            return {"results": [], "error": str(e)}
    
    def multi_query_search(
        self,
        queries: List[str],
        max_results_per_query: int = 3
    ) -> Dict:
        """
        Returns:
            Dict with deduplicated results and queries used
        """
        all_results = []
        seen_urls = set()
        
        for query in queries:
            search_result = self.search(query, max_results_per_query)
            
            for result in search_result.get("results", []):
                url = result.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    result["query_used"] = query
                    all_results.append(result)
        
        return {
            "results": all_results,
            "queries_used": queries
        }
