#!/usr/bin/env python3
"""
Spanish Translation Module for Financial Advisor Bot
Handles Spanish query translation and response translation
"""

import re
import logging
from typing import Dict, Tuple, Optional
from langdetect import detect
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logger = logging.getLogger(__name__)

class SpanishTranslator:
    """
    Handles Spanish-English translation for the financial advisor bot
    Uses Ollama for translation to maintain consistency with the bot
    """
    
    def __init__(self, ollama_model: str = "gemma3n:e2b"):
        self.ollama_model = ollama_model
        self.spanish_keywords = {
            'banco', 'cuenta', 'préstamo', 'crédito', 'dinero', 'inversión',
            'impuesto', 'ahorro', 'tarjeta', 'financiero', 'económico',
            'solicitud', 'documento', 'formulario', 'interés', 'tasa'
        }
        
        # Translation prompts
        self.spanish_to_english_prompt = """
You are a professional translator specializing in financial and banking terminology.
Translate the following Spanish text to English. Focus on accuracy for financial terms.
Keep the meaning precise and use appropriate banking/financial vocabulary.

Spanish text: {spanish_text}

English translation:"""

        self.english_to_spanish_prompt = """
You are a professional translator specializing in financial and banking terminology.
Translate the following English text to Spanish. Focus on accuracy for financial terms.
Use formal, professional Spanish appropriate for banking and financial contexts.
Use Latin American Spanish conventions.

English text: {english_text}

Spanish translation:"""

    def detect_language(self, text: str) -> str:
        """
        Enhanced language detection for Spanish, English, and Bangla
        """
        try:
            # First try langdetect
            detected = detect(text.lower())
            if detected == 'es':
                return 'spanish'
            elif detected == 'bn':
                return 'bangla'
            elif detected == 'en':
                return 'english'
        except:
            pass
        
        # Fallback to keyword-based detection
        text_lower = text.lower()
        
        # Check for Spanish keywords
        spanish_count = sum(1 for word in self.spanish_keywords if word in text_lower)
        
        # Check for Bangla Unicode characters
        bangla_chars = len([c for c in text if '\u0980' <= c <= '\u09FF'])
        
        if bangla_chars > 0:
            return 'bangla'
        elif spanish_count > 0:
            return 'spanish'
        else:
            return 'english'

    def translate_spanish_to_english(self, spanish_text: str) -> str:
        """
        Translate Spanish text to English using Ollama
        """
        try:
            from langchain_ollama import OllamaLLM
            
            llm = OllamaLLM(
                model=self.ollama_model,
                temperature=0.1,  # Low temperature for consistent translation
                max_tokens=800
            )
            
            prompt = self.spanish_to_english_prompt.format(spanish_text=spanish_text)
            translation = llm.invoke(prompt)
            
            # Clean up the translation
            translation = translation.strip()
            
            # Remove any extra explanations that might be added
            if '\n' in translation:
                translation = translation.split('\n')[0]
            
            logger.info(f"Spanish to English translation completed")
            logger.debug(f"Original: {spanish_text[:100]}...")
            logger.debug(f"Translation: {translation[:100]}...")
            
            return translation
            
        except Exception as e:
            logger.error(f"Error in Spanish to English translation: {e}")
            # Fallback: return original text
            return spanish_text

    def translate_english_to_spanish(self, english_text: str) -> str:
        """
        Translate English text to Spanish using Ollama
        """
        try:
            from langchain_ollama import OllamaLLM
            
            llm = OllamaLLM(
                model=self.ollama_model,
                temperature=0.1,  # Low temperature for consistent translation
                max_tokens=1000
            )
            
            prompt = self.english_to_spanish_prompt.format(english_text=english_text)
            translation = llm.invoke(prompt)
            
            # Clean up the translation
            translation = translation.strip()
            
            # Remove any extra explanations that might be added
            lines = translation.split('\n')
            # Take the first substantial line
            for line in lines:
                if len(line.strip()) > 10:
                    translation = line.strip()
                    break
            
            logger.info(f"English to Spanish translation completed")
            logger.debug(f"Original: {english_text[:100]}...")
            logger.debug(f"Translation: {translation[:100]}...")
            
            return translation
            
        except Exception as e:
            logger.error(f"Error in English to Spanish translation: {e}")
            # Fallback: return original text
            return english_text

    def process_spanish_query(self, spanish_query: str) -> Tuple[str, str]:
        """
        Process a Spanish query: translate to English and return both versions
        Returns: (english_query, original_spanish_query)
        """
        logger.info(f"Processing Spanish query: {spanish_query[:50]}...")
        
        english_query = self.translate_spanish_to_english(spanish_query)
        
        return english_query, spanish_query

    def process_english_response(self, english_response: str, original_spanish_query: str) -> str:
        """
        Translate English response back to Spanish
        """
        logger.info("Translating English response to Spanish...")
        
        spanish_response = self.translate_english_to_spanish(english_response)
        
        return spanish_response

# Test functions
def test_spanish_translator():
    """Test the Spanish translator functionality"""
    print("🧪 Testing Spanish Translator...")
    
    translator = SpanishTranslator()
    
    # Test language detection
    test_cases = [
        ("¿Cómo abrir una cuenta bancaria?", "spanish"),
        ("How to open a bank account?", "english"),
        ("ব্যাংক অ্যাকাউন্ট কীভাবে খুলবো?", "bangla"),
        ("¿Cuáles son los requisitos para un préstamo?", "spanish"),
    ]
    
    print("\n📝 Language Detection Test:")
    for text, expected in test_cases:
        detected = translator.detect_language(text)
        status = "✅" if detected == expected else "❌"
        print(f"{status} '{text}' -> {detected} (expected: {expected})")
    
    # Test translation
    print("\n🔄 Translation Test:")
    spanish_query = "¿Qué documentos necesito para abrir una cuenta bancaria?"
    print(f"Original Spanish: {spanish_query}")
    
    english_query, _ = translator.process_spanish_query(spanish_query)
    print(f"Translated to English: {english_query}")
    
    # Simulate English response
    english_response = "To open a bank account, you typically need: national ID, photographs, initial deposit, and proof of address."
    print(f"English Response: {english_response}")
    
    spanish_response = translator.process_english_response(english_response, spanish_query)
    print(f"Translated to Spanish: {spanish_response}")

if __name__ == "__main__":
    test_spanish_translator()
