# 初次测试建议 True。确认正常后改 False，执行完整 5000+5000 正式模拟。
QUICK_MODE = False
QUICK_REPS = 5
TEST_REPS = 5000             # 每市场独立测试历史数
PILOT_REPS = 5000            # 每种 Oracle 各自独立训练历史数
SEED = 20261019
FORCE_RECOMPUTE = False     # 默认复用附带的正式CSV；设 True 可强制重算

# 数学模型：ln(X)~N(0,0.5²)，Theta~Gamma(mean=mu, CV=cv)
LOG_X_STD = 0.5
PRICE_MIN = 0.01
PRICE_MAX = 5.0
FV_PERIODS = 20
LAMBDA_STEP = 0.005
LAMBDA_MAX = 0.4

# 实验 A: 时间敏感度均值 × 异质性, n=80,t=80, 12 scenarios
A_MU = (0.01, 0.02, 0.03)
A_CV = (0.2, 0.3, 0.4, 0.5)
A_N = 80
A_T = 80

# 实验 B: 每期买家数量 × 异质性, mu=.02,t=80, 18 scenarios
B_MU = 0.02
B_CV = (0.3, 0.4, 0.5)
B_N = (20, 40, 60, 80, 100, 120)
B_T = 80

# 实验 C: 目标时期 × 异质性, mu=.02,n=80, 21 scenarios
C_MU = 0.02
C_CV = (0.3, 0.4, 0.5)
C_N = 80
C_T = (40, 50, 60, 70, 80, 90, 100)
