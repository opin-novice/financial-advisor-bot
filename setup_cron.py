#!/usr/bin/env python3
"""
Setup cron job for automated multilingual delta index updates
"""

import os
import sys
from pathlib import Path
import subprocess

def get_project_path():
    """Get the absolute path of the project directory"""
    return Path(__file__).parent.absolute()

def create_cron_script():
    """Create a shell script for cron to run"""
    project_path = get_project_path()
    python_path = sys.executable
    
    script_content = f"""#!/bin/bash
# Multilingual Delta Index Update Script
# Generated automatically - do not edit manually

# Set environment variables
export PATH="{os.environ.get('PATH', '')}"
export PYTHONPATH="{project_path}:$PYTHONPATH"

# Change to project directory
cd "{project_path}"

# Log file with timestamp
LOG_FILE="logs/cron_delta_update_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the delta update with logging
echo "Starting multilingual delta update at $(date)" >> "$LOG_FILE"
{python_path} multilingual_delta_index.py --verify >> "$LOG_FILE" 2>&1

# Check exit status
if [ $? -eq 0 ]; then
    echo "Delta update completed successfully at $(date)" >> "$LOG_FILE"
else
    echo "Delta update failed at $(date)" >> "$LOG_FILE"
    # Optionally send notification (uncomment and configure)
    # echo "Multilingual delta index update failed. Check $LOG_FILE" | mail -s "Index Update Failed" admin@example.com
fi

# Clean up old log files (keep last 30 days)
find logs/ -name "cron_delta_update_*.log" -mtime +30 -delete 2>/dev/null

echo "Cron job completed at $(date)" >> "$LOG_FILE"
"""
    
    script_path = project_path / "cron_delta_update.sh"
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    return script_path

def suggest_cron_entries():
    """Suggest cron entries for different update frequencies"""
    script_path = get_project_path() / "cron_delta_update.sh"
    
    cron_suggestions = {
        "Every hour": f"0 * * * * {script_path}",
        "Every 6 hours": f"0 */6 * * * {script_path}",
        "Daily at 2 AM": f"0 2 * * * {script_path}",
        "Daily at midnight": f"0 0 * * * {script_path}",
        "Weekly (Sunday 2 AM)": f"0 2 * * 0 {script_path}",
        "Twice daily (6 AM & 6 PM)": f"0 6,18 * * * {script_path}",
    }
    
    return cron_suggestions

def check_cron_availability():
    """Check if cron is available on the system"""
    try:
        result = subprocess.run(['which', 'crontab'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def install_cron_job(cron_expression):
    """Install the cron job"""
    try:
        # Get current crontab
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        current_cron = result.stdout if result.returncode == 0 else ""
        
        # Check if our job already exists
        job_marker = "# Multilingual Financial Advisor Delta Update"
        if job_marker in current_cron:
            print("⚠️ Cron job already exists. Please remove it manually first.")
            print("Run: crontab -e")
            return False
        
        # Add our job
        new_cron = current_cron + f"\n{job_marker}\n{cron_expression}\n"
        
        # Install new crontab
        process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
        process.communicate(input=new_cron)
        
        if process.returncode == 0:
            print("✅ Cron job installed successfully!")
            return True
        else:
            print("❌ Failed to install cron job")
            return False
            
    except Exception as e:
        print(f"❌ Error installing cron job: {e}")
        return False

def main():
    """Main setup function"""
    print("🕐 Multilingual Delta Index - Cron Job Setup")
    print("=" * 50)
    
    # Check if cron is available
    if not check_cron_availability():
        print("❌ Cron is not available on this system")
        print("You'll need to set up automated updates manually")
        return
    
    print("✅ Cron is available")
    
    # Create the cron script
    print("\n📝 Creating cron script...")
    script_path = create_cron_script()
    print(f"✅ Cron script created: {script_path}")
    
    # Show suggestions
    print("\n⏰ Suggested cron schedules:")
    suggestions = suggest_cron_entries()
    
    for i, (description, cron_expr) in enumerate(suggestions.items(), 1):
        print(f"{i}. {description}")
        print(f"   {cron_expr}")
        print()
    
    # Interactive setup
    try:
        print("Options:")
        print("1-6: Use one of the suggested schedules")
        print("c: Enter custom cron expression")
        print("m: Manual setup instructions only")
        print("q: Quit without installing")
        
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == 'q':
            print("👋 Setup cancelled")
            return
        
        elif choice == 'm':
            print("\n📋 Manual Setup Instructions:")
            print("1. Run: crontab -e")
            print("2. Add one of the following lines:")
            for desc, expr in suggestions.items():
                print(f"   # {desc}")
                print(f"   {expr}")
                print()
            print("3. Save and exit")
            return
        
        elif choice == 'c':
            custom_expr = input("Enter custom cron expression: ").strip()
            if custom_expr:
                cron_expression = custom_expr
            else:
                print("❌ Invalid expression")
                return
        
        elif choice.isdigit() and 1 <= int(choice) <= len(suggestions):
            cron_expression = list(suggestions.values())[int(choice) - 1]
        
        else:
            print("❌ Invalid choice")
            return
        
        # Confirm installation
        print(f"\n📅 Selected cron expression: {cron_expression}")
        confirm = input("Install this cron job? (y/N): ").strip().lower()
        
        if confirm in ['y', 'yes']:
            if install_cron_job(cron_expression):
                print("\n🎉 Cron job setup completed!")
                print("\nThe system will now automatically update the multilingual index")
                print("according to your selected schedule.")
                print(f"\nLogs will be saved in: {get_project_path()}/logs/")
                print("\nTo view current cron jobs: crontab -l")
                print("To remove the cron job: crontab -e")
            else:
                print("\n❌ Failed to install cron job")
        else:
            print("👋 Installation cancelled")
    
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted")
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")

if __name__ == "__main__":
    main()
