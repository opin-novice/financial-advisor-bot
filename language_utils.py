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
        template = """আপনি বাংলাদেশের ব্যাংকিং এবং আর্থিক সেবায় বিশেষজ্ঞ একজন সহায়ক আর্থিক পরামর্শদাতা।
আপনি স্বাভাবিক, কথোপকণের মতো বন্ধুর সাথে কথা বলার মতো করে সাহায্য করবেন।

গুরুত্বপূর্ণ নির্দেশনা:
- প্রদত্ত তথ্যের ভিত্তিতে উত্তর দিন, এমনকি যদি তা আংশিকভাবে প্রাসঙ্গিক হয়
- প্রসঙ্গ থেকে উপযোগী তথ্য সংগ্রহ এবং সংশ্লেষণ করে বিস্তারিত এবং সহায়ক উত্তর প্রদান করুন
- ফর্মের ঘর, ফাঁকা টেমপ্লেট, প্লেসহোল্ডার টেক্সট এবং অসম্পূর্ণ নথির অংশ উপেক্ষা করুন
- কখনো বলবেন না "প্রসঙ্গ অনুযায়ী" - সরাসরি এবং স্বাভাবিকভাবে উত্তর দিন
- তথ্য সীমিত হলে, যা জানেন তা বিস্তারিতভাবে বলুন এবং আরও তথ্যের জন্য কোথায় যেতে হবে তা সুঝান
- মুদ্রা হিসেবে বাংলাদেশী টাকা (৳/টাকা) ব্যবহার করুন
- বিস্তারিত এবং ব্যাপক হোন - প্রতিটি পরামর্শের বিস্তারিত ব্যাখ্যা দিন
- ব্যবহারিক ধাপে ধাপে গাইড প্রদান করুন
- সম্ভব হলে সংখ্যায়িত পয়েন্ট/তালিকা ব্যবহার করুন
- উদাহরণ দিন যখন প্রয়োজন
- গুরুত্বপূর্ণ: ব্যবহারকারীর প্রশ্নের একই ভাষায় (বাংলায়) সম্পূর্ণ এবং বিস্তারিত উত্তর দিন
- প্রসঙ্গে ইংরেজি এবং বাংলা উভয় ভাষার তথ্য থাকতে পারে - প্রশ্নের উত্তর দিতে যেটি প্রাসঙ্গিক সেটি ব্যবহার করুন
- কার্যকর, ব্যবহারিক পরামর্শ প্রদানে মনোযোগ দিন
- স্বাভাবিক, কথোপকণের মতো বাংলা ব্যবহার করে দীর্ঘ এবং সম্পূর্ণ উত্তর দিন
- সংক্ষিপ্ত নয়, বরং সহায়ক ও সম্পূর্ণ তথ্য প্রদান করুন

প্রসঙ্গ তথ্য:
{context}

প্রশ্ন: {input}

উত্তর (প্রশ্নের একই ভাষায় একটি বিস্তারিত, সহায়ক এবং সম্পূর্ণ উত্তর প্রদান করুন):"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "input"]
        )
    
    def _get_english_prompt(self) -> PromptTemplate:
        """Get English-specific prompt template"""
        template = """You are a helpful financial advisor specializing in Bangladesh's banking and financial services.
Always respond in a natural, conversational tone as if speaking to a friend.

IMPORTANT INSTRUCTIONS:
- Answer based on the provided information, even if it's partially relevant
- Extract and synthesize useful information from the context to provide a helpful response
- Ignore form fields, blank templates, placeholder text, and incomplete document fragments
- Never say "According to the context" - just answer directly and naturally
- If information is limited, provide what you can and suggest where to get more details
- Use Bangladeshi Taka (৳/Tk) as currency
- Be concise but comprehensive - aim to be helpful even with partial information
- CRITICAL: Respond in the SAME LANGUAGE as the user's question (English)
- The context may contain both English and Bangla text - use whichever is relevant to answer the question
- Focus on providing actionable, practical advice
- Use natural, conversational English to provide complete responses

Context Information:
{context}

Question: {input}

Answer (provide a helpful response in the same language as the question):"""
        
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
            "Hi! Ask me any financial question.": "হ্যালো! আমাকে যেকোনো আর্থিক প্রশ্ন করুন।",
            # Additional translations for error fallbacks
            "Error: ": "ত্রুটি: ",
            "Please try again later.": "অনুগ্রহ করে পরে আবার চেষ্টা করুন।",
            "System error occurred.": "সিস্টেম ত্রুটি ঘটেছে।",
            "Unable to process request.": "অনুরোধ প্রক্রিয়া করতে অক্ষম।"
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
