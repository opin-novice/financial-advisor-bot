#!/usr/bin/env python3
"""
Language Detection and Response Utilities for Bangla-English Telegram Bot
"""

import re
import logging
from typing import Dict, Tuple, Optional
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

def normalize_language_code(detected_lang: str) -> str:
    """Normalize language codes for consistency"""
    lang_mapping = {
        'bangla': 'bengali',
        'bn': 'bengali',
        'en': 'english'
    }
    return lang_mapping.get(detected_lang.lower(), detected_lang.lower())

class LanguageDetector:
    """
    Detects language (Bangla vs English) and provides language-specific prompts
    """
    
    def __init__(self):
        # Bangla Unicode ranges
        self.bangla_range = r'[\u0980-\u09FF]'  # Bengali/Bangla Unicode block
        
        # Common Bangla words for additional detection
        self.common_bangla_words = {
            'কি', 'কী', 'কে', 'কোথায়', 'কখন', 'কেন', 'কিভাবে', 'কত', 'কোন',
            'আমি', 'তুমি', 'তিনি', 'আমার', 'তোমার', 'তার', 'আমাদের', 'তোমাদের',
            'এই', 'সেই', 'ওই', 'এটা', 'সেটা', 'ওটা', 'এখানে', 'সেখানে', 'ওখানে',
            'হ্যাঁ', 'না', 'নাই', 'আছে', 'নেই', 'হবে', 'হয়েছে', 'করেছে',
            'ব্যাংক', 'টাকা', 'পয়সা', 'লোন', 'ঋণ', 'সুদ', 'হিসাব', 'একাউন্ট',
            'বিনিয়োগ', 'সঞ্চয়', 'জমা', 'উত্তোলন', 'লেনদেন', 'কর', 'ট্যাক্স'
        }
        
        # Common English financial terms for context
        self.common_english_words = {
            'bank', 'loan', 'account', 'money', 'investment', 'savings', 'deposit',
            'withdrawal', 'transaction', 'tax', 'interest', 'credit', 'debit',
            'what', 'how', 'when', 'where', 'why', 'who', 'which', 'can', 'will'
        }
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect if text is primarily in Bangla or English
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (language, confidence_score)
            language: 'bangla' or 'english'
            confidence_score: float between 0 and 1
        """
        if not text or not text.strip():
            return 'english', 0.5  # Default to English for empty text
        
        text = text.strip().lower()
        
        # Count Bangla characters
        bangla_chars = len(re.findall(self.bangla_range, text))
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return 'english', 0.5  # Default for non-alphabetic text
        
        bangla_char_ratio = bangla_chars / total_chars if total_chars > 0 else 0
        
        # Count Bangla words
        words = text.split()
        bangla_word_count = sum(1 for word in words if word in self.common_bangla_words)
        english_word_count = sum(1 for word in words if word in self.common_english_words)
        total_words = len(words)
        
        # Calculate scores
        char_score = bangla_char_ratio
        word_score = 0
        
        if total_words > 0:
            if bangla_word_count > 0 or english_word_count > 0:
                word_score = bangla_word_count / (bangla_word_count + english_word_count)
        
        # Combined score (weighted average)
        if bangla_chars > 0:
            # If there are Bangla characters, heavily weight character-based detection
            combined_score = 0.8 * char_score + 0.2 * word_score
        else:
            # If no Bangla characters, rely more on word detection
            combined_score = 0.3 * char_score + 0.7 * word_score
        
        # Determine language and confidence
        if combined_score > 0.3:  # Threshold for Bangla detection
            language = 'bengali'
            confidence = min(0.95, 0.5 + combined_score)  # Scale confidence
        else:
            language = 'english'
            confidence = min(0.95, 0.5 + (1 - combined_score))
        
        # Special handling for mixed language queries
        # If there are Bengali characters but also many English words, consider it mixed
        if bangla_chars > 0 and english_word_count > bangla_word_count and total_words > 2:
            # Mixed query with English dominance - treat as English but lower confidence
            if combined_score < 0.7:  # Not heavily Bengali
                language = 'english'
                confidence = max(0.6, confidence * 0.8)  # Reduce confidence for mixed queries
        
        # Normalize language code for consistency
        language = normalize_language_code(language)
        
        logger.info(f"Language detection: '{text[:50]}...' -> {language} (confidence: {confidence:.2f})")
        return language, confidence
    
    def get_language_specific_prompt(self, language: str) -> PromptTemplate:
        """
        Get language-specific prompt template for RAG responses
        
        Args:
            language: 'bengali' or 'english'
            
        Returns:
            PromptTemplate configured for the specified language
        """
        if language == 'bengali':
            return self._get_bangla_prompt()
        else:
            return self._get_english_prompt()
    
    def _get_bangla_prompt(self) -> PromptTemplate:
        """Get Bangla-specific prompt template"""
        template = """আপনি একজন বাংলাদেশী আর্থিক পরামর্শদাতা। নিচের প্রসঙ্গ ব্যবহার করে প্রশ্নের উত্তর দিন।

প্রসঙ্গ:
{context}

প্রশ্ন: {input}

নির্দেশনা:
- বাংলায় উত্তর দিন
- সহজ ও বোধগম্য ভাষা ব্যবহার করুন
- বাংলাদেশের আর্থিক নিয়মকানুন অনুযায়ী তথ্য দিন
- যদি নিশ্চিত না হন, তাহলে বলুন যে আরও তথ্যের প্রয়োজন
- প্রয়োজনে উদাহরণ দিন

উত্তর:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "input"]
        )
    
    def _get_english_prompt(self) -> PromptTemplate:
        """Get English-specific prompt template"""
        template = """You are a Bangladeshi financial advisor. Answer the question using the provided context.

Context:
{context}

Question: {input}

Instructions:
- Answer in English
- Use clear and understandable language
- Provide information according to Bangladesh's financial regulations
- If you're not certain, say that more information is needed
- Provide examples when necessary
- Be helpful and professional

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "input"]
        )
    
    def translate_system_messages(self, message: str, target_language: str) -> str:
        """
        Translate common system messages to the target language
        
        Args:
            message: System message in English
            target_language: 'bengali' or 'english'
            
        Returns:
            Translated message
        """
        if target_language != 'bengali':
            return message
        
        # Translation dictionary for common system messages
        translations = {
            "Processing your question...": "আপনার প্রশ্ন প্রক্রিয়া করা হচ্ছে...",
            "Please enter a valid question.": "অনুগ্রহ করে একটি বৈধ প্রশ্ন লিখুন।",
            "I could not find relevant information in my database for your query.": "আপনার প্রশ্নের জন্য আমার ডাটাবেসে প্রাসঙ্গিক তথ্য খুঁজে পাইনি।",
            "I could not find sufficiently relevant information in my database.": "আমার ডাটাবেসে পর্যাপ্ত প্রাসঙ্গিক তথ্য খুঁজে পাইনি।",
            "I'm not confident in my answer based on the available information. Please rephrase your question or ask about a different topic.": "উপলব্ধ তথ্যের ভিত্তিতে আমি আমার উত্তরে নিশ্চিত নই। অনুগ্রহ করে আপনার প্রশ্নটি অন্যভাবে লিখুন বা অন্য বিষয়ে প্রশ্ন করুন।",
            "📄 Retrieved Documents:": "📄 প্রাপ্ত নথিসমূহ:",
            "Hi! Ask me any financial question.": "হ্যালো! আমাকে যেকোনো আর্থিক প্রশ্ন করুন।"
        }
        
        return translations.get(message, message)
    
    def determine_response_language(self, query_language: str, user_preference: Optional[str] = None) -> str:
        """Determine appropriate response language based on query and preferences"""
        if user_preference:
            return user_preference
        elif query_language == 'bengali':
            return 'bengali'  # Bengali queries get Bengali responses
        else:
            return 'english'  # English queries get English responses
    
    def format_confidence_message(self, language: str) -> str:
        """
        Get confidence disclaimer message in the appropriate language
        
        Args:
            language: 'bengali' or 'english'
            
        Returns:
            Formatted confidence message
        """
        if language == 'bengali':
            return "\n\n⚠️ *দ্রষ্টব্য: এই উত্তরে আমার মাঝারি আস্থা রয়েছে। অনুগ্রহ করে সরকারি সূত্র থেকে তথ্য যাচাই করুন বা নির্দিষ্ট পরামর্শের জন্য একজন আর্থিক পরামর্শদাতার সাথে পরামর্শ করুন।*"
        else:
            return "\n\n⚠️ *Note: I have moderate confidence in this answer. Please verify the information with official sources or consult a financial advisor for specific advice.*"
    
    def translate_bangla_to_english(self, bangla_text: str) -> str:
        """
        Translate Bangla query to English using LLM
        
        Args:
            bangla_text: Text in Bangla to translate
            
        Returns:
            English translation
        """
        # from langchain_groq import ChatGroq
        from langchain_ollama import ChatOllama
        from langchain.prompts import PromptTemplate
        import os
        
        # Initialize LLM for translation (reuse existing config)
        try:
            # llm = ChatGroq(
            #     model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
            #     groq_api_key=os.getenv("GROQ_API_KEY"),
            #     temperature=0.2,  # Low temperature for consistent translation
            #     max_tokens=300,
            # )
            llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0.2,  # Low temperature for consistent translation
                num_predict=300,
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM for translation: {e}")
            return bangla_text
        
        translation_prompt = PromptTemplate.from_template("""
        Translate the following Bangla text to English. Focus on preserving the meaning, especially financial and banking terms.
        
        Bangla text: {bangla_text}
        
        Instructions:
        - Translate accurately to English
        - Keep financial terms clear (e.g., ব্যাংক = bank, লোন = loan, একাউন্ট = account)
        - Maintain the question structure and intent
        - Return ONLY the English translation, no explanations
        
        English translation:
        """)
        
        try:
            prompt = translation_prompt.format(bangla_text=bangla_text)
            response = llm.invoke(prompt)
            english_translation = response.content.strip()
            
            # Basic validation - ensure we got a reasonable translation
            if english_translation and len(english_translation) > 5:
                logger.info(f"Translated: '{bangla_text}' → '{english_translation}'")
                return english_translation
            else:
                logger.warning(f"Translation failed, using original: {bangla_text}")
                return bangla_text
                
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return bangla_text

    def translate_english_to_bangla(self, english_text: str) -> str:
        """
        Translate English answer back to Bangla using LLM
        
        Args:
            english_text: English answer to translate
            
        Returns:
            Bangla translation
        """
        # from langchain_groq import ChatGroq
        from langchain_ollama import ChatOllama
        from langchain.prompts import PromptTemplate
        import os
        
        # Get token limit from config
        from config import config
        max_tokens = config.get_feedback_loop_config().get("translation_max_tokens", 1500)
        
        try:
            # llm = ChatGroq(
            #     model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
            #     groq_api_key=os.getenv("GROQ_API_KEY"),
            #     temperature=0.3,  # Slightly higher for natural language
            #     max_tokens=max_tokens,
            # )
            llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0.3,  # Slightly higher for natural language
                num_predict=max_tokens,
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM for translation: {e}")
            return english_text
        
        translation_prompt = PromptTemplate.from_template("""
        Translate the following English financial advice to natural Bangla. This is a response to a user's question about Bangladesh banking/finance.
        
        English text: {english_text}
        
        Instructions:
        - Translate the COMPLETE text to natural, conversational Bangla
        - Use appropriate Bengali financial terms (ব্যাংক, হিসাব, লোন, সুদ, etc.)
        - Keep the helpful, advisory tone
        - Maintain formatting (bullet points, numbers, etc.)
        - IMPORTANT: Translate the entire response completely - do not cut off mid-sentence
        - If the text is long, ensure you translate all of it
        - Return ONLY the complete Bangla translation, no explanations
        
        Bangla translation:
        """)
        
        try:
            prompt = translation_prompt.format(english_text=english_text)
            response = llm.invoke(prompt)
            bangla_translation = response.content.strip()
            
            # Check if translation is complete and reasonable
            if bangla_translation and len(bangla_translation) > 20:
                # Check if translation looks complete (doesn't end abruptly)
                if not bangla_translation.endswith(('...', 'অসম্পূর্ণ', 'প')):
                    logger.info(f"Answer translated to Bangla: {len(bangla_translation)} chars")
                    return bangla_translation
                else:
                    logger.warning("Translation appears incomplete, trying again with higher token limit")
                    # Try again with higher token limit
                    retry_tokens = min(max_tokens + 500, 2500)  # Add 500 tokens or cap at 2500
                    llm_extended = ChatGroq(
                        model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
                        groq_api_key=os.getenv("GROQ_API_KEY"),
                        temperature=0.3,
                        max_tokens=retry_tokens,
                    )
                    response_retry = llm_extended.invoke(prompt)
                    bangla_retry = response_retry.content.strip()
                    if bangla_retry and len(bangla_retry) > len(bangla_translation):
                        logger.info(f"Retry successful: {len(bangla_retry)} chars")
                        return bangla_retry
                    else:
                        return bangla_translation  # Return original if retry didn't help
            else:
                logger.warning("Bangla translation failed, returning English")
                return english_text
                
        except Exception as e:
            logger.error(f"Bangla translation failed: {e}")
            return english_text


class BilingualResponseFormatter:
    """
    Formats responses appropriately for bilingual context
    """
    
    def __init__(self, language_detector: LanguageDetector):
        self.language_detector = language_detector
    
    def format_sources_section(self, language: str) -> str:
        """Get sources section header in appropriate language"""
        if language == 'bengali':
            return "📄 প্রাপ্ত নথিসমূহ:"
        else:
            return "📄 Retrieved Documents:"
    
    def format_document_header(self, doc_idx: int, filename: str, language: str) -> str:
        """Format document header in appropriate language"""
        if language == 'bengali':
            return f"\n📂 **নথি {doc_idx}: {filename}**\n"
        else:
            return f"\n📂 **Document {doc_idx}: {filename}**\n"
    
    def format_chunk_header(self, chunk_idx: int, language: str) -> str:
        """Format chunk header in appropriate language"""
        if language == 'bengali':
            return f"\n🔹 অংশ {chunk_idx}:\n"
        else:
            return f"\n🔹 Chunk {chunk_idx}:\n"
    
    def enhance_response_with_language_context(self, response: str, detected_language: str, confidence: float) -> str:
        """
        Enhance response with language-specific formatting and context
        
        Args:
            response: Original response text
            detected_language: Detected language of the query
            confidence: Confidence score of language detection
            
        Returns:
            Enhanced response with appropriate language context
        """
        # Add language detection info for debugging (only in development)
        debug_info = ""
        if confidence < 0.8:  # Low confidence detection
            if detected_language == 'bengali':
                debug_info = f"\n\n🔍 *ভাষা সনাক্তকরণ: বাংলা (আস্থা: {confidence:.1%})*"
            else:
                debug_info = f"\n\n🔍 *Language detection: English (confidence: {confidence:.1%})*"
        
        return response + debug_info
