import subprocess
import sys
from pathlib import Path
from datetime import datetime


# ============================================================
# FINAL FORECASTING PIPELINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIVE_DIR = PROJECT_ROOT / "ML" / "Predictive"


# ============================================================
# Helper function
# ============================================================

def run_script(script_name):
    """
    Run one forecasting component and stop the pipeline
    if the component fails.
    """

    script_path = PREDICTIVE_DIR / script_name

    print("\n" + "=" * 70)
    print(f"RUNNING: {script_name}")
    print("=" * 70)

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print("\nPipeline stopped.")
        print(f"Failed script: {script_name}")
        sys.exit(result.returncode)

    print(f"\nCompleted successfully: {script_name}")


# ============================================================
# Pipeline start
# ============================================================

start_time = datetime.now()

print("\n")
print("=" * 70)
print("US REAL ESTATE ANALYTICS & AI")
print("FINAL FORECASTING PIPELINE")
print("=" * 70)

print(f"Project root: {PROJECT_ROOT}")
print(f"Started: {start_time}")


# ============================================================
# 1. Model comparison
# ============================================================

run_script("model_comparison.py")


# ============================================================
# 2. Walk-forward validation
# ============================================================

run_script("walk_forward_validation_v1.py")


# ============================================================
# 3. Residual analysis
# ============================================================

run_script("residual_analysis.py")


# ============================================================
# 4. Model explainability
# ============================================================

run_script("model_explainability.py")


# ============================================================
# 5. Future forecasting
# ============================================================

run_script("future_forecast.py")


# ============================================================
# 6. Future forecast visualization
# ============================================================

run_script("future_forecast_visualization.py")


# ============================================================
# Pipeline completed
# ============================================================

end_time = datetime.now()

print("\n")
print("=" * 70)
print("FINAL FORECASTING PIPELINE COMPLETED")
print("=" * 70)

print(f"Started : {start_time}")
print(f"Finished: {end_time}")
print(f"Duration: {end_time - start_time}")

print("\nForecast outputs are available in:")
print(PROJECT_ROOT / "data" / "forecasts")

print("\nNext step:")
print("Review final forecast, uncertainty results, and visualizations.")