"""Run all three numerical experiments and save CSV; plot separately."""
from experiment1_mismatch import main as run1
from experiment2_tradeoff import main as run2
from experiment3_comparative import main as run3

if __name__ == "__main__":
    run1()
    run2()
    run3()
    print("All CSV summaries saved. Run plot_all.py to generate figures.")
