"""Backward-compatible alias for generate_all.py (numerical computation only)."""
from generate_all import run1, run2, run3

if __name__ == "__main__":
    run1()
    run2()
    run3()
    print("All CSV summaries saved. Run plot_all.py to draw figures.")
