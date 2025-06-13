import os
import subprocess
import time
from datetime import datetime

def run_model_test(model_name, output_dir):
    """Run test for a specific model and save results in its directory"""
    print(f"\n{'='*50}")
    print(f"Testing {model_name.upper()} model...")
    print(f"{'='*50}\n")
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    cmd = [
        "python", "main.py",
        "--model", model_name,
        "--num-samples", "500",
        "--subgraph-size", "100",
        "--num-epochs", "100",
        "--batch-size", "32",
        "--hidden-dim", "64",
        "--learning-rate", "0.01",
        "--patience", "10",
        "--output-dir", model_dir
    ]
    if model_name in ['gat', 'transformer']:
        cmd.extend(["--num-heads", "4"])
    start_time = time.time()
    subprocess.run(cmd)
    end_time = time.time()
    with open(os.path.join(model_dir, "execution_time.txt"), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Execution time: {end_time - start_time:.2f} seconds\n")
        f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    models = ['gcn', 'gin', 'gat', 'sage', 'transformer']
    for model in models:
        run_model_test(model, output_dir)
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Model Testing Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for model in models:
            model_dir = os.path.join(output_dir, model)
            results_file = os.path.join(model_dir, "results.json")
            time_file = os.path.join(model_dir, "execution_time.txt")
            
            f.write(f"\n{model.upper()} Model:\n")
            f.write("-" * 30 + "\n")
            if os.path.exists(time_file):
                with open(time_file, "r") as tf:
                    f.write(tf.read())
            if os.path.exists(results_file):
                f.write("\nResults:\n")
                with open(results_file, "r") as rf:
                    f.write(rf.read())
            
            f.write("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    main() 