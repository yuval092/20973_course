import re
from collections import defaultdict

def analyze_logs(log_file):
    scenarios = ['transit', 'descend', 'ascend']
    success_counts = {s: 0 for s in scenarios}
    total_counts = {s: 0 for s in scenarios}
    
    # Track the last scenario
    current_scenario = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Detect scenario start
            m = re.search(r'Scenario: (\w+)', line)
            if m:
                current_scenario = m.group(1).lower()
                total_counts[current_scenario] += 1
            
            # Detect success
            if "SUCCESS" in line and current_scenario:
                success_counts[current_scenario] += 1
                
    print("Scenario Analysis:")
    for s in scenarios:
        rate = (success_counts[s] / total_counts[s] * 100) if total_counts[s] > 0 else 0
        print(f"  {s.upper()}: {success_counts[s]}/{total_counts[s]} successes ({rate:.1f}%)")

analyze_logs('logs/training_progress_detailed.log')
