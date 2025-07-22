from typing import Dict, List

class QueryProcessor:
    def __init__(self):
        self.financial_terms = {
            'banking': ['account', 'deposit', 'withdrawal', 'transfer', 'balance'],
            'investment': ['stocks', 'bonds', 'mutual funds', 'portfolio', 'returns'],
            'loans': ['interest', 'mortgage', 'credit', 'EMI', 'collateral'],
            'taxation': ['tax', 'returns', 'deduction', 'exemption', 'filing']
        }
        self.last_query = None
        self.last_category = None

    def process_query(self, query: str) -> Dict:
        category = self._detect_category(query)
        is_followup = self._is_followup_query(query)
        
        result = {
            'processed_query': query,
            'category': category,
            'is_followup': is_followup
        }
        
        self.last_query = query
        self.last_category = category
        return result

    def _detect_category(self, query: str) -> str:
        query_lower = query.lower()
        max_matches = 0
        detected_category = 'general'

        for category, terms in self.financial_terms.items():
            matches = sum(1 for term in terms if term in query_lower)
            if matches > max_matches:
                max_matches = matches
                detected_category = category

        return detected_category

    def _is_followup_query(self, query: str) -> bool:
        if not self.last_query:
            return False

        followup_indicators = ['also', 'additionally', 'moreover', 'what about', 'and']
        query_lower = query.lower()
        
        # Check for pronouns referring to previous context
        has_pronouns = any(word in query_lower for word in ['it', 'this', 'that', 'these', 'those'])
        
        # Check for followup phrases
        has_followup = any(indicator in query_lower for indicator in followup_indicators)
        
        return has_pronouns or has_followup