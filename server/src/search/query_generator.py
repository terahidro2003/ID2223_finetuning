"""Smart search query generation using LLM"""
from typing import List
from ..llm.base import BaseLLMProvider


class QueryGenerator:
    """Generate optimized search queries from user prompts"""
    
    def __init__(self, llm_provider: BaseLLMProvider, model: str):
        self.llm = llm_provider
        self.model = model
    
    def generate_queries(
        self,
        user_prompt: str,
        num_queries: int = 3
    ) -> List[str]:
        """
        Generate multiple optimized search queries
        
        Args:
            user_prompt: User's question/prompt
            num_queries: Number of queries to generate
            
        Returns:
            List of optimized search queries
        """
        system_prompt = f"""You are a search query optimization expert. Given a user's question, 
generate {num_queries} different search queries that will retrieve the most relevant information.

Guidelines:
- Extract key entities, facts, and concepts
- Remove conversational words ("please", "can you", "I want to know")
- Use specific, searchable terms
- Cover different angles of the question
- Keep queries concise (3-8 words)
- Use keywords, not full sentences

Return ONLY the queries, one per line, without numbering, enumaration at the start or explanations.
Each line MUST begin with a - followed by a space and then the query, just like a markdown list."""
        
        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=200
            )
            
            # Parse response
            queries = response.strip().split('\n')
            queries = [q.strip('- ').strip() for q in queries if q.strip()]
            
            return queries[:num_queries] if queries else [user_prompt]
            
        except Exception as e:
            print(f"Query generation error: {e}")
            return [user_prompt]
    
    def should_search(self, user_prompt: str) -> bool:
        """
        Determine if search is needed for this prompt
        
        Args:
            user_prompt: User's question
            
        Returns:
            True if search should be performed
        """
        system_prompt = """You are a routing assistant that determines if a web search is required. 
Respond with ONLY 'YES' or 'NO'.

Answer 'YES' if ANY of these conditions apply:
- User explicitly asks to search, look up, find, or check something online
- User uses phrases like "search for", "look up", "find me", "check online"
- Question involves current events, news, or recent happenings
- Question asks about live data (weather, stock prices, sports scores, traffic)
- Question involves time-sensitive information (today, this week, latest, recent)
- Question asks about facts that change frequently (prices, rankings, statistics)
- Question requires verification of up-to-date information
- Question asks "what is the current/latest" for any metric or status
- Question involves information that may have been recently updated

Answer 'NO' only if:
- Question is about general knowledge unlikely to change (historical facts, basic definitions)
- Question is conversational or opinion-based without factual lookup needs
- Question involves math, logic, or reasoning without external data needs

When uncertain, default to 'YES'.
"""
        
        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0,
                max_tokens=10
            )
            
            return response.strip().upper() == "YES"
            
        except:
            return False
