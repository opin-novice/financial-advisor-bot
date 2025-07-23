import subprocess
import sys
import logging
from datetime import datetime
import json
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
log_dir = "logs/baseline_tests"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
json_output_file = os.path.join(log_dir, f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json") # Define JSON output file

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_test(test_script):
    """Run a test script and return the result"""
    logger.info(f"Running test: {test_script}")
    print(f"Running {test_script}...")
    
    # Get the current environment variables
    env = os.environ.copy()
    # Add the project root to the PYTHONPATH for the subprocess
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = project_root

    try:
        result = subprocess.run(
            [sys.executable, test_script],
            capture_output=True,
            text=True,
            check=True,
            env=env # Pass the modified environment variables
        )
        logger.info(f"Test {test_script} completed successfully")
        print(f"✅ {test_script} completed successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Test {test_script} failed with error: {e.stderr}")
        print(f"❌ {test_script} failed")
        return False, e.stderr

def main():
    """Run all test scripts in the tests directory"""
    logger.info("Starting test run")
    print("Starting Financial Advisor Bot test run...\n")
    
    # Get all test scripts
    test_dir = "scripts/tests"
    test_scripts = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.startswith("test_") and f.endswith(".py")]
    
    if not test_scripts:
        logger.warning("No test scripts found")
        print("No test scripts found in the tests directory")
        return
    
    # Run each test script
    results = []
    rag_test_metrics = {
        "quality_metrics": {
            "overall_confidence": "N/A (requires implementation)",
            "answer_quality": "N/A (requires implementation)",
            "source_quality": "N/A (requires implementation)",
            "context_match": "N/A (requires implementation)"
        },
        "average_response_time_seconds": "N/A (requires implementation)"
    }

    for script in test_scripts:
        success, output = run_test(script)
        
        if "test_rag_system.py" in script and success:
            # Attempt to extract JSON output from test_rag_system.py
            start_delimiter = "====TEST_RAG_SYSTEM_RESULTS_START===="
            end_delimiter = "====TEST_RAG_SYSTEM_RESULTS_END===="
            
            if start_delimiter in output and end_delimiter in output:
                json_start = output.find(start_delimiter) + len(start_delimiter)
                json_end = output.find(end_delimiter)
                json_str = output[json_start:json_end].strip()
                try:
                    rag_results = json.loads(json_str)
                    # Calculate average response time
                    total_response_time = sum(item.get('response_time_seconds', 0) for item in rag_results)
                    average_response_time = total_response_time / len(rag_results) if len(rag_results) > 0 else 0
                    rag_test_metrics["average_response_time_seconds"] = average_response_time

                    # Calculate average quality metrics
                    total_context_match = 0
                    total_answer_quality = 0
                    for item in rag_results:
                        if 'quality_metrics' in item:
                            total_context_match += item['quality_metrics'].get('context_match', 0) if isinstance(item['quality_metrics'].get('context_match'), (int, float)) else 0
                            total_answer_quality += item['quality_metrics'].get('answer_quality', 0) if isinstance(item['quality_metrics'].get('answer_quality'), (int, float)) else 0
                    
                    num_rag_results = len(rag_results)
                    if num_rag_results > 0:
                        rag_test_metrics["quality_metrics"]["context_match"] = total_context_match / num_rag_results
                        rag_test_metrics["quality_metrics"]["answer_quality"] = total_answer_quality / num_rag_results
                        rag_test_metrics["quality_metrics"]["overall_confidence"] = "N/A"
                        rag_test_metrics["quality_metrics"]["source_quality"] = "N/A"
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from test_rag_system.py output: {e}")
                    print(f"Warning: Could not parse detailed metrics from test_rag_system.py output. Error: {e}")
            else:
                logger.warning("JSON delimiters not found in test_rag_system.py output.")
                print("Warning: Detailed JSON output not found in test_rag_system.py output.")

        results.append({
            "script": script,
            "success": success,
            "output": output
        }) # Store results as dictionaries
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Run Summary:")
    print("=" * 50)
    
    passed = sum(1 for r in results if r["success"]) # Adjust for dictionary access
    failed = len(results) - passed
    
    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 50)
    
    # Log detailed results
    logger.info(f"Test run completed. Passed: {passed}, Failed: {failed}")
    for r in results:
        status = "PASSED" if r["success"] else "FAILED"
        logger.info(f"Test {r['script']}: {status}")
        logger.debug(f"Output for {r['script']}:\n{r['output']}")

    # Prepare and save JSON output
    json_summary = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total_tests": len(results),
        "passed_tests": passed,
        "failed_tests": failed,
        "test_details": [
            {"script": r["script"], "status": "PASSED" if r["success"] else "FAILED"}
            for r in results
        ],
        "quality_metrics": rag_test_metrics["quality_metrics"],
        "average_response_time_seconds": rag_test_metrics["average_response_time_seconds"]
    }

    try:
        with open(json_output_file, 'w') as f:
            json.dump(json_summary, f, indent=4)
        logger.info(f"Detailed test results saved to {json_output_file}")
        print(f"Detailed test results saved to {json_output_file}")
    except IOError as e:
        logger.error(f"Error saving JSON results to {json_output_file}: {e}")
        print(f"Error saving JSON results: {e}")

    # Return exit code based on test results
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())