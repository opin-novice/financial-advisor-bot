import os
import re
import logging
import json
from typing import Dict, List, Tuple, Optional
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from functools import lru_cache
import numpy as np
from collections import Counter
import asyncio

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Config ---
GROQ_MODEL = "llama3-8b-8192"
GROQ_API_KEY = "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"

class EnhancedRAGUtils:
    def __init__(self):
        """Initialize enhanced RAG utilities with optimized Groq LLM"""
        try:
            self.llm = ChatGroq(
                model=GROQ_MODEL,
                groq_api_key=GROQ_API_KEY,
                temperature=0.2,  # Lower temperature for more consistent results
                max_tokens=800,   # Optimized for M1 memory
                model_kwargs={
                    "top_p": 0.8,
                    "frequency_penalty": 0.1
                }
            )
            
            # Financial domain knowledge
            self.financial_keywords = {
                "banking": ["account", "deposit", "withdrawal", "transfer", "atm", "branch", "balance"],
                "loans": ["loan", "credit", "mortgage", "interest", "installment", "emi", "collateral"],
                "investment": ["investment", "bond", "savings", "certificate", "profit", "dividend"],
                "taxation": ["tax", "vat", "income tax", "return", "nbr", "tin", "withholding"],
                "insurance": ["insurance", "policy", "premium", "claim", "coverage", "life insurance"],
                "business": ["business", "startup", "license", "registration", "trade", "sme"],
                "forex": ["foreign exchange", "remittance", "dollar", "currency", "export", "import"]
            }
            
            logger.info("✅ Enhanced RAG utilities initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced RAG utilities: {e}")
            raise

    def multi_stage_query_refinement(self, query: str, category: str = "general") -> List[str]:
        """
        Multi-stage query refinement with domain-specific enhancement
        
        Args:
            query: Original user query
            category: Query category (banking, loans, etc.)
            
        Returns:
            List of refined queries ordered by expected relevance
        """
        refined_queries = [query]  # Always include original
        
        try:
            # Stage 1: Domain-specific expansion
            domain_expanded = self._expand_with_domain_knowledge(query, category)
            if domain_expanded != query:
                refined_queries.append(domain_expanded)
            
            # Stage 2: LLM-based refinement
            llm_refined = self._llm_query_refinement(query, category)
            if llm_refined and llm_refined not in refined_queries:
                refined_queries.append(llm_refined)
            
            # Stage 3: Synonym expansion
            synonym_expanded = self._expand_with_synonyms(query)
            if synonym_expanded not in refined_queries:
                refined_queries.append(synonym_expanded)
            
            logger.info(f"Generated {len(refined_queries)} refined queries for: {query}")
            return refined_queries
            
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}. Using original query.")
            return [query]

    def _expand_with_domain_knowledge(self, query: str, category: str) -> str:
        """Expand query with domain-specific financial terms"""
        query_lower = query.lower()
        
        # Get relevant keywords for the category
        relevant_keywords = self.financial_keywords.get(category, [])
        
        # Add Bangladesh-specific context
        bangladesh_terms = []
        if "bank" in query_lower:
            bangladesh_terms.extend(["bangladesh bank", "bb", "central bank"])
        if "tax" in query_lower:
            bangladesh_terms.extend(["nbr", "national board of revenue", "bangladesh"])
        if "loan" in query_lower:
            bangladesh_terms.extend(["bangladesh", "taka", "interest rate"])
        
        # Combine original query with relevant terms
        expansion_terms = []
        for keyword in relevant_keywords[:3]:  # Limit to top 3
            if keyword not in query_lower:
                expansion_terms.append(keyword)
        
        expansion_terms.extend(bangladesh_terms[:2])  # Add max 2 Bangladesh terms
        
        if expansion_terms:
            expanded = f"{query} {' '.join(expansion_terms)}"
            return expanded
        
        return query

    def _llm_query_refinement(self, query: str, category: str) -> Optional[str]:
        """LLM-based intelligent query refinement"""
        refinement_prompt = PromptTemplate.from_template("""
        You are a query refinement expert for Bangladesh financial services.
        
        Original Query: {query}
        Category: {category}
        
        Instructions:
        1. Improve the query for better document retrieval
        2. Add relevant Bangladesh financial context
        3. Expand abbreviations and technical terms
        4. Keep the original intent intact
        5. Make it more specific and searchable
        6. Limit to 20 words maximum
        
        Return ONLY the refined query, nothing else:
        """)
        
        try:
            prompt = refinement_prompt.format(query=query, category=category)
            response = self.llm.invoke(prompt)
            refined = response.content.strip()
            
            # Validate refinement quality
            if (len(refined) > 10 and 
                len(refined.split()) <= 20 and 
                refined.lower() != query.lower()):
                return refined
            
        except Exception as e:
            logger.warning(f"LLM query refinement failed: {e}")
        
        return None

    def _expand_with_synonyms(self, query: str) -> str:
        """Expand query with financial synonyms"""
        synonym_map = {
            "loan": "credit financing",
            "bank": "financial institution",
            "account": "banking account",
            "interest": "profit rate",
            "tax": "taxation levy",
            "investment": "financial investment",
            "business": "enterprise company",
            "money": "funds capital",
            "payment": "transaction settlement"
        }
        
        words = query.lower().split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            if word in synonym_map:
                expanded_words.extend(synonym_map[word].split())
        
        return " ".join(expanded_words)

    def enhanced_relevance_check(self, query: str, retrieved_docs: List[Document], 
                                category: str = "general") -> Tuple[bool, float]:
        """
        Enhanced multi-factor relevance checking
        
        Args:
            query: User's input query
            retrieved_docs: List of documents retrieved from vector store
            category: Query category for domain-specific checking
            
        Returns:
            Tuple of (is_relevant: bool, confidence_score: float)
        """
        if not retrieved_docs:
            return False, 0.0

        try:
            # Multi-factor relevance assessment
            factors = {}
            
            # Factor 1: LLM-based semantic relevance
            llm_relevance = self._llm_relevance_check(query, retrieved_docs[:3])
            factors['llm_semantic'] = llm_relevance
            
            # Factor 2: Keyword overlap analysis
            keyword_relevance = self._keyword_overlap_analysis(query, retrieved_docs, category)
            factors['keyword_overlap'] = keyword_relevance
            
            # Factor 3: Document quality assessment
            quality_score = self._assess_document_quality(retrieved_docs)
            factors['document_quality'] = quality_score
            
            # Factor 4: Category-specific relevance
            category_relevance = self._category_specific_relevance(query, retrieved_docs, category)
            factors['category_specific'] = category_relevance
            
            # Weighted combination of factors
            weights = {
                'llm_semantic': 0.4,
                'keyword_overlap': 0.3,
                'document_quality': 0.15,
                'category_specific': 0.15
            }
            
            final_score = sum(factors[factor] * weights[factor] for factor in factors)
            is_relevant = final_score > 0.2  # Threshold for relevance
            
            logger.info(f"Relevance factors: {factors}, Final score: {final_score:.3f}")
            return is_relevant, final_score
            
        except Exception as e:
            logger.warning(f"Enhanced relevance check failed: {e}. Using fallback.")
            return self._fallback_relevance_check(query, retrieved_docs)

    def _llm_relevance_check(self, query: str, docs: List[Document]) -> float:
        """LLM-based semantic relevance assessment"""
        relevance_prompt = PromptTemplate.from_template("""
        Analyze the relevance between the query and document contexts for Bangladesh financial services.
        
        Query: {query}
        
        Document Contexts:
        {contexts}
        
        Rate relevance from 0.0 (completely irrelevant) to 1.0 (highly relevant).
        Consider:
        - Semantic similarity
        - Topic alignment
        - Information usefulness
        - Bangladesh financial context
        
        Respond with ONLY a number between 0.0 and 1.0:
        """)
        
        try:
            contexts = "\n\n".join([f"Doc {i+1}: {doc.page_content[:300]}..." 
                                   for i, doc in enumerate(docs)])
            
            prompt = relevance_prompt.format(query=query, contexts=contexts)
            response = self.llm.invoke(prompt)
            
            # Extract numeric score
            score_text = response.content.strip()
            score = float(re.findall(r'0\.\d+|1\.0|0|1', score_text)[0])
            return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
            
        except Exception as e:
            logger.warning(f"LLM relevance check failed: {e}")
            return 0.5  # Neutral score

    def _keyword_overlap_analysis(self, query: str, docs: List[Document], category: str) -> float:
        """Advanced keyword overlap analysis with TF-IDF-like scoring"""
        query_terms = set(query.lower().split())
        category_terms = set(self.financial_keywords.get(category, []))
        all_query_terms = query_terms.union(category_terms)
        
        if not all_query_terms:
            return 0.0
        
        doc_scores = []
        for doc in docs[:5]:  # Analyze top 5 documents
            doc_text = doc.page_content.lower()
            doc_words = doc_text.split()
            
            # Calculate term frequency
            term_freq = Counter(doc_words)
            
            # Score based on query term presence and frequency
            score = 0
            for term in all_query_terms:
                if term in doc_text:
                    # TF-IDF-like scoring
                    tf = term_freq.get(term, 0) / len(doc_words) if doc_words else 0
                    score += tf * (2 if term in query_terms else 1)  # Higher weight for original query terms
            
            doc_scores.append(score)
        
        return np.mean(doc_scores) if doc_scores else 0.0

    def _assess_document_quality(self, docs: List[Document]) -> float:
        """Assess the quality of retrieved documents"""
        if not docs:
            return 0.0
        
        quality_scores = []
        for doc in docs[:3]:
            content = doc.page_content
            
            # Quality indicators
            word_count = len(content.split())
            sentence_count = len([s for s in content.split('.') if s.strip()])
            
            # Penalize very short or very long documents
            length_score = 1.0
            if word_count < 20:
                length_score = 0.3
            elif word_count > 1000:
                length_score = 0.8
            
            # Check for informative content vs form fields
            informative_score = 1.0
            if content.count(':') > word_count * 0.2:  # Too many colons (form-like)
                informative_score = 0.4
            
            # Check for complete sentences
            sentence_score = min(1.0, sentence_count / max(1, word_count / 15))
            
            quality = (length_score + informative_score + sentence_score) / 3
            quality_scores.append(quality)
        
        return np.mean(quality_scores)

    def _category_specific_relevance(self, query: str, docs: List[Document], category: str) -> float:
        """Category-specific relevance assessment"""
        if category == "general":
            return 0.5  # Neutral for general queries
        
        category_keywords = self.financial_keywords.get(category, [])
        if not category_keywords:
            return 0.5
        
        relevance_scores = []
        for doc in docs[:3]:
            content = doc.page_content.lower()
            matches = sum(1 for keyword in category_keywords if keyword in content)
            score = matches / len(category_keywords)
            relevance_scores.append(score)
        
        return np.mean(relevance_scores) if relevance_scores else 0.0

    def _fallback_relevance_check(self, query: str, retrieved_docs: List[Document]) -> Tuple[bool, float]:
        """Enhanced fallback relevance check"""
        query_terms = set(query.lower().split())
        relevance_scores = []
        
        for doc in retrieved_docs[:5]:
            doc_content = doc.page_content.lower()
            doc_words = set(doc_content.split())
            
            # Jaccard similarity
            intersection = query_terms.intersection(doc_words)
            union = query_terms.union(doc_words)
            jaccard = len(intersection) / len(union) if union else 0
            
            # Exact phrase matching bonus
            phrase_bonus = 0
            if len(query.split()) > 1:
                if query.lower() in doc_content:
                    phrase_bonus = 0.3
            
            score = jaccard + phrase_bonus
            relevance_scores.append(score)
        
        avg_score = np.mean(relevance_scores) if relevance_scores else 0
        is_relevant = avg_score > 0.15
        return is_relevant, avg_score

    def comprehensive_answer_validation(self, query: str, answer: str, 
                                      contexts: List[str], category: str = "general") -> Dict:
        """
        Comprehensive answer validation with multiple checks
        
        Args:
            query: Original user query
            answer: Generated answer from RAG system
            contexts: Retrieved document contexts
            category: Query category
            
        Returns:
            Validation result with detailed feedback
        """
        try:
            validation_factors = {}
            
            # Factor 1: LLM-based validation
            llm_validation = self._llm_answer_validation(query, answer, contexts)
            validation_factors['llm_validation'] = llm_validation
            
            # Factor 2: Factual consistency check
            consistency_score = self._check_factual_consistency(answer, contexts)
            validation_factors['factual_consistency'] = consistency_score
            
            # Factor 3: Query-answer alignment
            alignment_score = self._check_query_answer_alignment(query, answer)
            validation_factors['query_alignment'] = alignment_score
            
            # Factor 4: Answer completeness
            completeness_score = self._assess_answer_completeness(query, answer, category)
            validation_factors['completeness'] = completeness_score
            
            # Weighted combination
            weights = {
                'llm_validation': 0.35,
                'factual_consistency': 0.25,
                'query_alignment': 0.25,
                'completeness': 0.15
            }
            
            final_confidence = sum(validation_factors[factor] * weights[factor] 
                                 for factor in validation_factors)
            
            is_valid = final_confidence > 0.3
            
            return {
                "valid": is_valid,
                "confidence": final_confidence,
                "factors": validation_factors,
                "feedback": self._generate_validation_feedback(validation_factors)
            }
            
        except Exception as e:
            logger.warning(f"Comprehensive answer validation failed: {e}")
            return {
                "valid": True,
                "confidence": 0.5,
                "factors": {},
                "feedback": f"Validation failed: {str(e)}"
            }

    def _llm_answer_validation(self, query: str, answer: str, contexts: List[str]) -> float:
        """LLM-based answer validation"""
        validation_prompt = PromptTemplate.from_template("""
        Validate if the answer correctly addresses the query based on the provided contexts.
        
        Query: {query}
        Answer: {answer}
        Contexts: {contexts}
        
        Rate the answer quality from 0.0 (poor) to 1.0 (excellent) considering:
        - Accuracy based on contexts
        - Completeness of response
        - Relevance to query
        - Clarity and usefulness
        
        Respond with ONLY a number between 0.0 and 1.0:
        """)
        
        try:
            context_text = "\n\n".join([f"Context {i+1}: {ctx[:200]}..." 
                                       for i, ctx in enumerate(contexts[:2])])
            
            prompt = validation_prompt.format(
                query=query, answer=answer, contexts=context_text
            )
            response = self.llm.invoke(prompt)
            
            score_text = response.content.strip()
            score = float(re.findall(r'0\.\d+|1\.0|0|1', score_text)[0])
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"LLM answer validation failed: {e}")
            return 0.5

    def _check_factual_consistency(self, answer: str, contexts: List[str]) -> float:
        """Check if answer facts are consistent with contexts"""
        if not contexts:
            return 0.0
        
        # Extract key facts from answer (simplified approach)
        answer_sentences = [s.strip() for s in answer.split('.') if s.strip()]
        
        consistency_scores = []
        for sentence in answer_sentences[:5]:  # Check first 5 sentences
            # Check if sentence content appears in any context
            sentence_lower = sentence.lower()
            found_support = False
            
            for context in contexts:
                context_lower = context.lower()
                # Simple overlap check
                sentence_words = set(sentence_lower.split())
                context_words = set(context_lower.split())
                overlap = len(sentence_words.intersection(context_words))
                
                if overlap >= len(sentence_words) * 0.3:  # 30% word overlap
                    found_support = True
                    break
            
            consistency_scores.append(1.0 if found_support else 0.0)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5

    def _check_query_answer_alignment(self, query: str, answer: str) -> float:
        """Check if answer addresses the query intent"""
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())
        
        # Check term overlap
        overlap = len(query_terms.intersection(answer_terms))
        overlap_score = overlap / len(query_terms) if query_terms else 0
        
        # Check if answer contains question-specific terms
        question_indicators = ['what', 'how', 'when', 'where', 'why', 'which']
        query_type = None
        for indicator in question_indicators:
            if indicator in query.lower():
                query_type = indicator
                break
        
        # Simple heuristic for answer appropriateness
        appropriateness_score = 0.5  # Default
        if query_type:
            if query_type in ['what', 'which'] and any(word in answer.lower() 
                                                      for word in ['is', 'are', 'means', 'refers']):
                appropriateness_score = 0.8
            elif query_type == 'how' and any(word in answer.lower() 
                                           for word in ['steps', 'process', 'way', 'method']):
                appropriateness_score = 0.8
        
        return (overlap_score + appropriateness_score) / 2

    def _assess_answer_completeness(self, query: str, answer: str, category: str) -> float:
        """Assess if answer is complete for the query type"""
        # Basic completeness indicators
        answer_length = len(answer.split())
        
        # Length-based scoring
        if answer_length < 10:
            length_score = 0.2
        elif answer_length < 30:
            length_score = 0.6
        elif answer_length < 100:
            length_score = 1.0
        else:
            length_score = 0.8  # Very long answers might be less focused
        
        # Category-specific completeness
        category_score = 0.5
        if category != "general":
            category_keywords = self.financial_keywords.get(category, [])
            answer_lower = answer.lower()
            keyword_presence = sum(1 for keyword in category_keywords 
                                 if keyword in answer_lower)
            category_score = min(1.0, keyword_presence / max(1, len(category_keywords) * 0.3))
        
        return (length_score + category_score) / 2

    def _generate_validation_feedback(self, factors: Dict) -> str:
        """Generate human-readable validation feedback"""
        feedback_parts = []
        
        for factor, score in factors.items():
            if score < 0.3:
                feedback_parts.append(f"Low {factor.replace('_', ' ')}")
            elif score > 0.7:
                feedback_parts.append(f"Good {factor.replace('_', ' ')}")
        
        if not feedback_parts:
            return "Answer quality is moderate"
        
        return "; ".join(feedback_parts)

    def enhance_answer(self, answer: str, contexts: List[str], query: str) -> str:
        """Enhance answer quality when validation confidence is low"""
        enhancement_prompt = PromptTemplate.from_template("""
        Improve the following answer to better address the user's query using the provided contexts.
        
        Original Query: {query}
        Current Answer: {answer}
        Additional Context: {contexts}
        
        Instructions:
        1. Make the answer more specific and accurate
        2. Add relevant details from contexts
        3. Ensure clarity and completeness
        4. Keep Bangladesh financial context
        5. Maintain conversational tone
        
        Enhanced Answer:
        """)
        
        try:
            context_text = "\n".join(contexts[:2])  # Use top 2 contexts
            prompt = enhancement_prompt.format(
                query=query, answer=answer, contexts=context_text
            )
            
            response = self.llm.invoke(prompt)
            enhanced = response.content.strip()
            
            # Return enhanced answer if it's substantially different and better
            if len(enhanced) > len(answer) * 0.8 and enhanced != answer:
                logger.info("Answer enhanced successfully")
                return enhanced
            
        except Exception as e:
            logger.warning(f"Answer enhancement failed: {e}")
        
        return answer  # Return original if enhancement fails

    @lru_cache(maxsize=64)
    def get_domain_context(self, category: str) -> str:
        """Get domain-specific context for better query understanding"""
        domain_contexts = {
            "banking": "Bangladesh banking services, account opening, deposits, transfers",
            "loans": "Bangladesh loan services, credit facilities, interest rates, EMI",
            "investment": "Bangladesh investment opportunities, savings certificates, bonds",
            "taxation": "Bangladesh tax system, VAT, income tax, NBR regulations",
            "insurance": "Bangladesh insurance policies, life insurance, claims",
            "business": "Bangladesh business registration, startup procedures, SME support",
            "forex": "Bangladesh foreign exchange, remittance, export-import regulations"
        }
        return domain_contexts.get(category, "Bangladesh financial services")
