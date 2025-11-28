"""DuckDuckGo search implementation"""
from typing import List, Dict
import requests, os
from ddgs import DDGS


class DuckDuckGoSearcher:
    """DuckDuckGo web search"""

    SITE_EXCLUSIONS =  ' '.join([f'-site:{site}' for site in
        ['reddit.com', 'twitter.com', 'facebook.com', 'instagram.com']
    ])
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
    
    def search(
        self,
        query: str,
        max_results: int = 5,
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
                query + " " + self.SITE_EXCLUSIONS,
                safesearch=safesearch,
                max_results=max_results,
                backend='wikipedia,google,duckduckgo',
            )
            
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                })

            print('results before content fetch:', formatted_results)
            for idx, result in enumerate(formatted_results):
                formatted_results[idx]['content'] = requests.get("https://r.jina.ai/" + result['url'], headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {os.getenv('JINA_API_KEY')}",
                    "X-Retain-Images": "none",
                    "X-Return-Format": "text",
                }).text
            
            print("SEARCH RESULTS:", formatted_results)
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
            
            if "error" in search_result:
                print("Some error happened:", search_result["error"])

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
