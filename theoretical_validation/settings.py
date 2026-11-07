QUICK_MODE = False
QUICK_REPS = 20
TEST_REPS = 5000
PILOT_REPS = 5000

SEED = 20261019
MU = 0.02              # 市场平均时间敏感度
T = 80                 # 目标时期
N = 80                 # 基准期内买家数
SIGMAS = (0.0, 0.005, 0.010, 0.015, 0.020)
SAMPLE_SIZES = (10, 20, 30, 50, 100)
PRICE_MIN, PRICE_MAX = 0.01, 5.0
LOG_X_STD = 0.5
LAMBDA_STEP = 0.005
LAMBDA_MAX = 0.4
BOUND_CONFIDENCE_DELTA = 0.05

# True 会重新运行所有 Monte Carlo；False 会读取“同参数同种子”的本地缓存。
# 注意：缓存也必须由本项目代码先生成，压缩包不含任何实验结果。
FORCE_RECOMPUTE = True
