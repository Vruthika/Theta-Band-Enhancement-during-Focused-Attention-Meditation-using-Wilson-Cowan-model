import os, subprocess, sys
os.environ["MPLBACKEND"] = "Agg"                  # save figures, do not open windows
steps = ["step1_simulate.py", "step2_fixed_points.py", "step3_phase_plane.py",
         "step4_bifurcation.py", "step5_frequency_tuning.py",
         "step6_rest_vs_meditation.py", "step7_coupled_regions.py", "step8_robustness.py"]
for s in steps:
    print(f"\n===== {s} =====", flush=True)
    r = subprocess.run([sys.executable, s])
    if r.returncode != 0:
        sys.exit(f"{s} failed")
print("\nAll done. Figures: fig1 ... fig8 (.png) in this folder.")
