#!/usr/bin/env python3
"""
Comprehensive Multilingual Feature Accuracy Test
"""
import os
import sys
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import json
import time

# Load environment variables
load_dotenv()

class MultilingualAccuracyTester:
    def __init__(self):
        self.results = {
            'language_detection': {},
            'cross_language_similarity': {},
            'retrieval_quality': {},
            'response_quality': {},
            'overall_metrics': {}
        }
        
    def test_language_detection_accuracy(self):
        """Test language detection accuracy"""
        print("🌐 Testing Language Detection Accuracy...")
        print("-" * 60)
        
        try:
            from language_utils import LanguageDetector
            detector = LanguageDetector()
            
            # Test cases with expected languages
            test_cases = [
                # English queries
                ("How to open a bank account?", "english"),
                ("What are the loan eligibility criteria?", "english"),
                ("Tax calculation methods for individuals", "english"),
                ("Investment opportunities in Bangladesh", "english"),
                ("Mobile banking services available", "english"),
                
                # Bengali queries
                ("ব্যাংক অ্যাকাউন্ট কিভাবে খুলবো?", "bengali"),
                ("ঋণের যোগ্যতার মাপদণ্ড কি?", "bengali"),
                ("ব্যক্তিগত কর গণনার পদ্ধতি", "bengali"),
                ("বাংলাদেশে বিনিয়োগের সুযোগ", "bengali"),
                ("মোবাইল ব্যাংকিং সেবা", "bengali"),
                
                # Mixed/Edge cases
                ("How to ব্যাংক অ্যাকাউন্ট খুলবো?", "english"),  # Mixed
                ("NID card requirements", "english"),  # Technical terms
                ("জাতীয় পরিচয়পত্র", "bengali"),  # Bengali technical
            ]
            
            correct_detections = 0
            total_tests = len(test_cases)
            detailed_results = []
            
            for query, expected_lang in test_cases:
                detected_lang, confidence = detector.detect_language(query)
                is_correct = detected_lang.lower() == expected_lang.lower()
                
                if is_correct:
                    correct_detections += 1
                
                result = {
                    'query': query,
                    'expected': expected_lang,
                    'detected': detected_lang,
                    'confidence': confidence,
                    'correct': is_correct
                }
                detailed_results.append(result)
                
                status = "✅" if is_correct else "❌"
                print(f"{status} Query: {query[:40]}...")
                print(f"   Expected: {expected_lang} | Detected: {detected_lang} | Confidence: {confidence:.2f}")
            
            accuracy = correct_detections / total_tests
            
            self.results['language_detection'] = {
                'accuracy': accuracy,
                'correct_detections': correct_detections,
                'total_tests': total_tests,
                'detailed_results': detailed_results
            }
            
            print(f"\n📊 Language Detection Accuracy: {accuracy:.2%} ({correct_detections}/{total_tests})")
            return accuracy >= 0.80  # 80% threshold
            
        except Exception as e:
            print(f"❌ Language detection test failed: {e}")
            return False
    
    def test_cross_language_similarity(self):
        """Test cross-language semantic similarity"""
        print("\n🔗 Testing Cross-Language Semantic Similarity...")
        print("-" * 60)
        
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from config import config
            
            embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"}
            )
            
            # Equivalent queries in English and Bengali
            query_pairs = [
                ("How to open a bank account", "ব্যাংক অ্যাকাউন্ট খোলার নিয়ম"),
                ("Loan eligibility requirements", "ঋণের যোগ্যতার শর্ত"),
                ("Tax calculation methods", "কর গণনার পদ্ধতি"),
                ("Investment opportunities", "বিনিয়োগের সুযোগ"),
                ("Mobile banking services", "মোবাইল ব্যাংকিং সেবা"),
            ]
            
            similarities = []
            detailed_results = []
            
            for en_query, bn_query in query_pairs:
                en_emb = np.array(embeddings.embed_query(en_query)).reshape(1, -1)
                bn_emb = np.array(embeddings.embed_query(bn_query)).reshape(1, -1)
                similarity = cosine_similarity(en_emb, bn_emb)[0][0]
                
                similarities.append(similarity)
                
                result = {
                    'english_query': en_query,
                    'bengali_query': bn_query,
                    'similarity': similarity
                }
                detailed_results.append(result)
                
                quality = "🟢 Good" if similarity > 0.4 else "🟡 Fair" if similarity > 0.2 else "🔴 Poor"
                print(f"EN: {en_query}")
                print(f"BN: {bn_query}")
                print(f"Similarity: {similarity:.4f} ({quality})")
                print()
            
            avg_similarity = np.mean(similarities)
            
            self.results['cross_language_similarity'] = {
                'average_similarity': avg_similarity,
                'similarities': similarities,
                'detailed_results': detailed_results
            }
            
            print(f"📊 Average Cross-Language Similarity: {avg_similarity:.4f}")
            return avg_similarity >= 0.25  # 25% threshold for cross-language
            
        except Exception as e:
            print(f"❌ Cross-language similarity test failed: {e}")
            return False
    
    def test_retrieval_quality(self):
        """Test retrieval quality for different languages"""
        print("\n📄 Testing Retrieval Quality...")
        print("-" * 60)
        
        try:
            from main import FinancialAdvisorTelegramBot
            
            bot = FinancialAdvisorTelegramBot()
            
            # Test queries in both languages
            test_queries = [
                ("english", "How to open a bank account in Bangladesh?"),
                ("bengali", "ব্যাংক অ্যাকাউন্ট খোলার নিয়ম কি?"),
                ("english", "What are the loan eligibility criteria?"),
                ("bengali", "ঋণের যোগ্যতার মাপদণ্ড কি?"),
                ("english", "Tax calculation process"),
                ("bengali", "কর গণনার প্রক্রিয়া"),
            ]
            
            retrieval_results = []
            successful_retrievals = 0
            
            for lang, query in test_queries:
                print(f"🔍 Testing {lang.title()}: {query}")
                
                try:
                    result = bot.process_query(query)
                    
                    sources_found = len(result.get('sources', []))
                    response_generated = bool(result.get('response'))
                    
                    if sources_found > 0 and response_generated:
                        successful_retrievals += 1
                        status = "✅"
                    else:
                        status = "❌"
                    
                    retrieval_result = {
                        'language': lang,
                        'query': query,
                        'sources_found': sources_found,
                        'response_generated': response_generated,
                        'successful': sources_found > 0 and response_generated
                    }
                    retrieval_results.append(retrieval_result)
                    
                    print(f"   {status} Sources: {sources_found} | Response: {'Yes' if response_generated else 'No'}")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    retrieval_results.append({
                        'language': lang,
                        'query': query,
                        'sources_found': 0,
                        'response_generated': False,
                        'successful': False,
                        'error': str(e)
                    })
            
            retrieval_success_rate = successful_retrievals / len(test_queries)
            
            self.results['retrieval_quality'] = {
                'success_rate': retrieval_success_rate,
                'successful_retrievals': successful_retrievals,
                'total_queries': len(test_queries),
                'detailed_results': retrieval_results
            }
            
            print(f"\n📊 Retrieval Success Rate: {retrieval_success_rate:.2%} ({successful_retrievals}/{len(test_queries)})")
            return retrieval_success_rate >= 0.70  # 70% threshold
            
        except Exception as e:
            print(f"❌ Retrieval quality test failed: {e}")
            return False
    
    def test_response_language_consistency(self):
        """Test if responses are in appropriate languages"""
        print("\n💬 Testing Response Language Consistency...")
        print("-" * 60)
        
        try:
            from main import FinancialAdvisorTelegramBot
            from language_utils import LanguageDetector
            
            bot = FinancialAdvisorTelegramBot()
            detector = LanguageDetector()
            
            # Test cases with expected response languages
            test_cases = [
                ("english", "How to open a bank account?", ["english", "bengali"]),  # Can be bilingual
                ("bengali", "ব্যাংক অ্যাকাউন্ট কিভাবে খুলবো?", ["bengali"]),  # Should be Bengali
                ("english", "What are loan requirements?", ["english", "bengali"]),
                ("bengali", "ঋণের শর্তাবলী কি?", ["bengali"]),
            ]
            
            consistent_responses = 0
            response_results = []
            
            for query_lang, query, expected_response_langs in test_cases:
                print(f"🔍 Query ({query_lang}): {query}")
                
                try:
                    result = bot.process_query(query)
                    response = result.get('response', '')
                    
                    if response:
                        # Detect response language
                        response_lang, confidence = detector.detect_language(response)
                        
                        is_consistent = response_lang.lower() in [lang.lower() for lang in expected_response_langs]
                        
                        if is_consistent:
                            consistent_responses += 1
                        
                        response_result = {
                            'query_language': query_lang,
                            'query': query,
                            'expected_response_languages': expected_response_langs,
                            'actual_response_language': response_lang,
                            'confidence': confidence,
                            'consistent': is_consistent,
                            'response_preview': response[:100] + "..." if len(response) > 100 else response
                        }
                        response_results.append(response_result)
                        
                        status = "✅" if is_consistent else "❌"
                        print(f"   {status} Response Language: {response_lang} (confidence: {confidence:.2f})")
                        print(f"   Preview: {response[:80]}...")
                    else:
                        print("   ❌ No response generated")
                        response_results.append({
                            'query_language': query_lang,
                            'query': query,
                            'consistent': False,
                            'error': 'No response generated'
                        })
                
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    response_results.append({
                        'query_language': query_lang,
                        'query': query,
                        'consistent': False,
                        'error': str(e)
                    })
                
                print()
            
            consistency_rate = consistent_responses / len(test_cases)
            
            self.results['response_quality'] = {
                'consistency_rate': consistency_rate,
                'consistent_responses': consistent_responses,
                'total_tests': len(test_cases),
                'detailed_results': response_results
            }
            
            print(f"📊 Response Language Consistency: {consistency_rate:.2%} ({consistent_responses}/{len(test_cases)})")
            return consistency_rate >= 0.75  # 75% threshold
            
        except Exception as e:
            print(f"❌ Response language consistency test failed: {e}")
            return False
    
    def calculate_overall_metrics(self, test_results):
        """Calculate overall multilingual performance metrics"""
        
        # Weight different aspects
        weights = {
            'language_detection': 0.25,
            'cross_language_similarity': 0.20,
            'retrieval_quality': 0.30,
            'response_consistency': 0.25
        }
        
        scores = {
            'language_detection': self.results['language_detection'].get('accuracy', 0),
            'cross_language_similarity': min(self.results['cross_language_similarity'].get('average_similarity', 0) * 4, 1.0),  # Scale to 0-1
            'retrieval_quality': self.results['retrieval_quality'].get('success_rate', 0),
            'response_consistency': self.results['response_quality'].get('consistency_rate', 0)
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        
        self.results['overall_metrics'] = {
            'overall_score': overall_score,
            'individual_scores': scores,
            'weights': weights,
            'grade': self._get_grade(overall_score)
        }
        
        return overall_score
    
    def _get_grade(self, score):
        """Convert score to letter grade"""
        if score >= 0.90:
            return "A+ (Excellent)"
        elif score >= 0.80:
            return "A (Very Good)"
        elif score >= 0.70:
            return "B (Good)"
        elif score >= 0.60:
            return "C (Fair)"
        elif score >= 0.50:
            return "D (Poor)"
        else:
            return "F (Failing)"
    
    def run_all_tests(self):
        """Run all multilingual accuracy tests"""
        print("🚀 Comprehensive Multilingual Feature Accuracy Test")
        print("=" * 80)
        
        test_results = {}
        
        # Run all tests
        test_results['language_detection'] = self.test_language_detection_accuracy()
        test_results['cross_language_similarity'] = self.test_cross_language_similarity()
        test_results['retrieval_quality'] = self.test_retrieval_quality()
        test_results['response_consistency'] = self.test_response_language_consistency()
        
        # Calculate overall metrics
        overall_score = self.calculate_overall_metrics(test_results)
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 MULTILINGUAL ACCURACY TEST RESULTS")
        print("=" * 80)
        
        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            test_display = test_name.replace('_', ' ').title()
            print(f"{test_display:.<50} {status}")
        
        print(f"\nOverall Multilingual Score: {overall_score:.2%}")
        print(f"Grade: {self.results['overall_metrics']['grade']}")
        
        return self.results

def main():
    """Run multilingual accuracy tests"""
    tester = MultilingualAccuracyTester()
    results = tester.run_all_tests()
    
    # Save results to JSON for report generation
    with open('multilingual_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: multilingual_test_results.json")
    return results

if __name__ == "__main__":
    main()
