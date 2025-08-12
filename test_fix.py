#!/usr/bin/env python3
"""
Test script to compare Bengali and English responses after the fix
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import bot components
from main import FinancialAdvisorTelegramBot

def test_both_languages():
    """Test the same question in both languages to compare response quality"""
    
    print("🧪 Testing Response Quality After Fix")
    print("=" * 60)
    
    # Initialize bot
    bot = FinancialAdvisorTelegramBot()
    
    # Test queries
    queries = [
        {
            "english": "How do I pay my income tax?",
            "bengali": "আমি আমার আয়কর কিভাবে দিব?"
        }
    ]
    
    for i, query_pair in enumerate(queries, 1):
        print(f"\n📝 Test {i}: Income Tax Payment")
        print("-" * 50)
        
        # Test English
        print("\n🇺🇸 ENGLISH QUERY:")
        print(f"Query: {query_pair['english']}")
        print("\n🤖 Response:")
        
        try:
            english_response = bot.process_query(query_pair['english'])
            english_answer = english_response.get("response", "No response")
            print(english_answer)
            print(f"\n📊 Response Length: {len(english_answer)} characters")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "="*50)
        
        # Test Bengali
        print("\n🇧🇩 BENGALI QUERY:")
        print(f"Query: {query_pair['bengali']}")
        print("\n🤖 Response:")
        
        try:
            bengali_response = bot.process_query(query_pair['bengali'])
            bengali_answer = bengali_response.get("response", "No response")
            print(bengali_answer)
            print(f"\n📊 Response Length: {len(bengali_answer)} characters")
        except Exception as e:
            print(f"Error: {e}")
        
        # Compare lengths
        try:
            en_len = len(english_answer)
            bn_len = len(bengali_answer)
            ratio = bn_len / en_len if en_len > 0 else 0
            
            print(f"\n🔍 COMPARISON:")
            print(f"English length: {en_len} characters")
            print(f"Bengali length: {bn_len} characters")
            print(f"Bengali/English ratio: {ratio:.2f}")
            
            if ratio >= 0.8:
                print("✅ Response quality: GOOD - Bengali response is comprehensive")
            elif ratio >= 0.5:
                print("⚠️ Response quality: MODERATE - Bengali response needs improvement") 
            else:
                print("❌ Response quality: POOR - Bengali response too short")
                
        except:
            print("Unable to compare response lengths")

if __name__ == "__main__":
    test_both_languages()
