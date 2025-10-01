import re
import matplotlib.pyplot as plt
import numpy as np
# Added modules for command-line interface and file handling
import argparse
import os

def parse_log_file(file_path):
    """
    Parse the log file and extract the values for "step" and
    "curr/step_reward_mean_for_curriculum".
    """
    steps = []
    rewards = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Remove ANSI color codes
            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            
            # Extract step value
            step_match = re.search(r'step:(\d+(?:\.\d+)?)', clean_line)
            # Extract curr/step_reward_mean_for_curriculum value
            reward_match = re.search(r'curr/step_reward_mean_for_curriculum:([+-]?\d+(?:\.\d+)?)', clean_line)
            
            if step_match and reward_match:
                step = float(step_match.group(1))
                reward = float(reward_match.group(1))
                steps.append(step)
                rewards.append(reward)
    
    return steps, rewards

def plot_training_progress(steps, rewards, save_path=None):
    """
    Plot the training progress curve.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot the primary line chart
    plt.plot(steps, rewards, 'b-', linewidth=2, alpha=0.8, label='Step Reward Mean')
    plt.scatter(steps, rewards, c='red', s=30, alpha=0.6, zorder=5)
    
    # Add a trend line
    if len(steps) > 1:
        z = np.polyfit(steps, rewards, 1)
        p = np.poly1d(z)
        plt.plot(steps, p(steps), "r--", alpha=0.7, linewidth=1.5, label=f'Trend Line (slope: {z[0]:.4f})')
    
    # Set chart properties
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Curriculum Step Reward Mean', fontsize=14)
    plt.title('Training Reward Progress', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some statistics to the plot
    if rewards:
        plt.text(0.02, 0.98, f'Max Reward: {max(rewards):.3f}\nMin Reward: {min(rewards):.3f}\nMean Reward: {np.mean(rewards):.3f}', 
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()

def main():
    """
    Main function: Example usage
    """
    # Specify your log file path
    log_file_path = "your_log_file.txt"  # Please replace with your actual file path
    
    try:
        # Parse the log file
        steps, rewards = parse_log_file(log_file_path)
        
        if not steps:
            print("No valid training data found. Please check the log file format.")
            return
        
        print(f"Successfully parsed {len(steps)} data points")
        print(f"Step range: {min(steps)} - {max(steps)}")
        print(f"Reward range: {min(rewards):.3f} - {max(rewards):.3f}")
        
        # Plot the chart
        plot_training_progress(steps, rewards, save_path="vis_training_progress.png")
        
    except FileNotFoundError:
        print(f"Error: File not found {log_file_path}")
        print("Please ensure the file path is correct.")
    except Exception as e:
        print(f"Error processing file: {e}")

# If you want to parse data directly from a string (for testing)
def parse_log_string(log_string):
    """
    Parse data from a string (for testing single lines or small amounts of data)
    """
    steps = []
    rewards = []
    
    lines = log_string.strip().split('\n')
    for line in lines:
        # Remove ANSI color codes
        clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
        
        step_match = re.search(r'step:(\d+(?:\.\d+)?)', clean_line)
        reward_match = re.search(r'curr/step_reward_mean_for_curriculum:([+-]?\d+(?:\.\d+)?)', clean_line)
        
        if step_match and reward_match:
            step = float(step_match.group(1))
            reward = float(reward_match.group(1))
            steps.append(step)
            rewards.append(reward)
    
    return steps, rewards

# -------------------------
# CLI entry point
# -------------------------

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Parse training log file and visualize reward curve.")
    parser.add_argument(
        "--log_file",
        default='lic_mixed_Qwen3_1B_cl_1to5_win5_factor08_abs000.log',
        help="Path to the log file to be parsed.")
    parser.add_argument(
        "--save",
        "-s",
        default='vis_lic_mixed_Qwen3_1B_cl_1to5_win5_factor08_abs000.png',
        help="Optional path to save the generated figure (e.g., output.png). If not provided, the figure is only shown on screen.")
    args = parser.parse_args()

    # Check if the file exists
    if not os.path.isfile(args.log_file):
        print(f"Error: File not found {args.log_file}")
        print("Please ensure the file path is correct.")
        exit(1)

    # Parse the log file
    steps, rewards = parse_log_file(args.log_file)

    if not steps:
        print("No valid training data found. Please check the log file format.")
        exit(1)

    print(f"Successfully parsed {len(steps)} data points")
    print(f"Step range: {min(steps)} - {max(steps)}")
    print(f"Reward range: {min(rewards):.3f} - {max(rewards):.3f}")

    # Plot and save/display the chart
    plot_training_progress(steps, rewards, save_path=args.save)