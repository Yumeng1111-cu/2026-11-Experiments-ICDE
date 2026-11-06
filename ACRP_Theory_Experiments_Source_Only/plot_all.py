"""Only replot saved experiment 1/2/3 CSV results. No simulations."""
import plot_settings as S
from plot_saved import draw

if __name__ == "__main__":
    print("绘图模式:", S.PLOT_DATA_MODE)
    for e in ("1", "2", "3"):
        draw(e)
    print("三张理论实验图已完成；没有重新计算实验。")
