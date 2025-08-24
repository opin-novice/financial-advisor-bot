#!/usr/bin/env python3
"""
Convert existing RAGAS JSON results to CSV format
"""
import json
import pandas as pd
import os

def convert_json_to_csv():
    """Convert the existing JSON evaluation results to CSV"""
    
    # Input and output files
    json_file = "logs/ragas_evaluation_fixed.json"
    csv_file = "logs/ragas_evaluation_results_from_json.csv"
    
    if not os.path.exists(json_file):
        print(f"❌ JSON file not found: {json_file}")
        return
    
    try:
        # Load JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📄 Loaded data from: {json_file}")
        print(f"📊 Data keys: {list(data.keys())}")
        
        # Create DataFrame from the metrics
        metrics_data = {
            'Metric': [],
            'Score': [],
            'Status': []
        }
        
        # Extract metrics
        metric_names = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
        
        for metric in metric_names:
            if metric in data:
                score = data[metric]
                status = "Good" if score > 0.5 else "Poor" if score > 0 else "Failed"
                
                metrics_data['Metric'].append(metric.replace('_', ' ').title())
                metrics_data['Score'].append(score if score is not None else 0.0)
                metrics_data['Status'].append(status)
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data)
        
        # Add metadata as additional rows if available
        if 'metadata' in data:
            metadata = data['metadata']
            
            # Add separator row
            df = pd.concat([df, pd.DataFrame({
                'Metric': ['--- Metadata ---'],
                'Score': [''],
                'Status': ['']
            })], ignore_index=True)
            
            # Add metadata rows
            for key, value in metadata.items():
                df = pd.concat([df, pd.DataFrame({
                    'Metric': [key.replace('_', ' ').title()],
                    'Score': [str(value)],
                    'Status': ['Info']
                })], ignore_index=True)
        
        # Save to CSV
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"✅ CSV saved successfully: {csv_file}")
        print(f"📊 DataFrame shape: {df.shape}")
        print("\n📄 Preview:")
        print(df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error converting JSON to CSV: {e}")

if __name__ == "__main__":
    convert_json_to_csv()
