import os
import shutil
import logging

# Setup logging
logging.basicConfig(
    filename='logs/cleaning.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_processed_folder():
    processed_dir = "data/processed"
    categories = ['banking', 'investment', 'loans', 'regulations', 'sme', 'taxation']
    
    logger.info("Starting cleanup of processed folder")
    
    for category in categories:
        category_path = os.path.join(processed_dir, category)
        if os.path.exists(category_path):
            try:
                # Remove all files in the category directory
                shutil.rmtree(category_path)
                # Recreate the empty directory
                os.makedirs(category_path)
                logger.info(f"Cleaned and recreated {category} directory")
            except Exception as e:
                logger.error(f"Error cleaning {category} directory: {str(e)}")

if __name__ == "__main__":
    clean_processed_folder()
    print("✅ Processed folder cleaned successfully")