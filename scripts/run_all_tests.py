import os
import subprocess
import sys
import logging
from datetime import datetime

# Setup logging
log_dir = "logs/baseline_tests"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

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
    
    try:
        result = subprocess.run(
            [sys.executable, test_script],
            capture_output=True,
            text=True,
            check=True
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
    for script in test_scripts:
        success, output = run_test(script)
        results.append((script, success, output))
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Run Summary:")
    print("=" * 50)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 50)
    
    # Log detailed results
    logger.info(f"Test run completed. Passed: {passed}, Failed: {failed}")
    for script, success, output in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"Test {script}: {status}")
        logger.debug(f"Output for {script}:\n{output}")
    
    # Return exit code based on test results
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())