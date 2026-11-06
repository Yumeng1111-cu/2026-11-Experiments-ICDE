from __future__ import annotations
from pathlib import Path
from math import pi, sqrt, log
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr, roots_genlaguerre, gammaln
from settings import (MU, T, N, SIGMAS, SAMPLE_SIZES, PRICE_MIN, PRICE_MAX,
                      LOG_X_STD, SEED, LAMBDA_MAX, LAMBDA_STEP,
                      BOUND_CONFIDENCE_DELTA, QUICK_MODE, QUICK_REPS,
                      PILOT_REPS, TEST_REPS, FORCE_RECOMPUTE)

from numba import njit


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
DATA = OUT / 'data'
FIG = OUT / 'figures'
DATA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
LAMBDAS = np.arange(int(round(LAMBDA_MAX/LAMBDA_STEP)) + 1) * LAMBDA_STEP
KAPPA_X = 1 / (LOG_X_STD * sqrt(2 * pi))
C_DELTA = 4 + sqrt(0.5 * log(1/BOUND_CONFIDENCE_DELTA))
COLORS = ['#2475AB','#db8b39','#2e9d83','#9867ae','#d14756']


def reps():
    if QUICK_MODE:
        return QUICK_REPS, QUICK_REPS
    return TEST_REPS, PILOT_REPS


def distribution_quadrature(sigma: float):
    """Gamma 时间敏感度：均值 MU，标准差 sigma；sigma=0 是退化常数。"""
    if sigma == 0.0:
        return np.array([MU]), np.array([1.0])
    shape = (MU/sigma)**2
    scale = sigma**2 / MU
    nodes, raww = roots_genlaguerre(96, shape-1)
    weights = raww / np.exp(gammaln(shape))
    return scale*nodes, weights/weights.sum()


def demand(p, age: int, sigma: float, calibration_distance: int = 0):
    """P(X exp(-theta*age) exp(-MU*d) >= p)，p 可为数组。"""
    prices = np.atleast_1d(p).astype(float)
    theta, w = distribution_quadrature(sigma)
    z = (-np.log(prices[:,None]) - age*theta[None,:] - MU*calibration_distance) / LOG_X_STD
    return ndtr(z) @ w


def truth_grid(sigma: float):
    """从理论分布重新计算目标期收益函数和最大收益，绝不读取既存结果。"""
    from scipy.optimize import minimize_scalar
    grid = np.linspace(PRICE_MIN, PRICE_MAX, 30001)
    theta, w = distribution_quadrature(sigma)
    y = np.empty_like(grid)
    for l in range(0, len(grid), 1000):
        r = min(len(grid), l+1000)
        z = (-np.log(grid[l:r,None]) - (T-1)*theta[None,:]) / LOG_X_STD
        y[l:r] = grid[l:r] * (ndtr(z) @ w)
    k = int(np.argmax(y))
    left, right = grid[max(0,k-2)], grid[min(len(grid)-1,k+2)]
    def revenue(p):
        return float(p * (ndtr((-np.log(p) - (T-1)*theta)/LOG_X_STD) @ w))
    opt = minimize_scalar(lambda p: -revenue(p), bounds=(left,right),method='bounded',options={'xatol':1e-12})
    best = max(float(y.max()), float(-opt.fun))
    return grid, y, best


@njit(cache=True)
def _weighted_exact_price(sorted_q, sorted_d, weights, lower, upper):
    """报价断点+边界精确经验定价（返回缩放后的报价 q）；相同报价合并。"""
    length = len(sorted_q)
    hi_tail = 0.0
    for j in range(length-1,-1,-1):
        if sorted_q[j] < upper:
            break
        hi_tail += weights[sorted_d[j]]
    bestq = upper
    bestval = upper*hi_tail
    tail = 0.0
    for j in range(length-1,-1,-1):
        q = sorted_q[j]
        if q < lower:
            break
        tail += weights[sorted_d[j]]
        if q <= upper and (j == 0 or q > sorted_q[j-1]):
            candidate = q*tail
            if candidate > bestval:
                bestval = candidate
                bestq = q
    lowval = lower*tail
    if lowval > bestval:
        bestq = lower
    return bestq


@njit(cache=True)
def _paths(sorted_q, sorted_d, weights_ew, low, high, mode):
    """mode=1 SW；mode=2 EW；mode=3 Both，返回 SW 与 EW 所有候选价格。"""
    hmax = weights_ew.shape[1]-1
    nlam = weights_ew.shape[0]
    sw_prices = np.empty(hmax)
    ew_prices = np.empty(nlam)
    ww = np.zeros(hmax+1)
    if mode == 1 or mode == 3:
        for h in range(1,hmax+1):
            ww[h] = 1.0
            sw_prices[h-1] = _weighted_exact_price(sorted_q, sorted_d, ww, low, high)
    if mode == 2 or mode == 3:
        for i in range(nlam):
            ew_prices[i] = _weighted_exact_price(sorted_q, sorted_d, weights_ew[i], low, high)
    return sw_prices, ew_prices


def make_history(sigma: float, n: int, stream: int, rep: int):
    """每期 n 名新买家；所有 (X,theta) 独立抽样。不同 stream 保持训练/测试独立。"""
    sigma_seed = int(round(sigma * 1_000_000))
    rng = np.random.default_rng(np.random.SeedSequence([SEED, sigma_seed, n, stream, rep]))
    H = T-1
    ages = np.arange(H)[:,None]
    logx = rng.normal(0, LOG_X_STD, size=(H,n))
    if sigma == 0:
        theta = MU
    else:
        shape, scale = (MU/sigma)**2, (sigma**2/MU)
        theta = rng.gamma(shape, scale, size=(H,n))
    # 先把历史样本去除共同年龄变化，再统一缩放到当前期（只涉及公共均值 MU）
    normalized = np.exp(logx + (MU-theta)*ages).reshape(-1)
    distances = np.repeat(np.arange(H,0,-1), n)
    perm = np.argsort(normalized, kind='mergesort')
    return np.ascontiguousarray(normalized[perm]), np.ascontiguousarray(distances[perm])


def simulate_averages(sigma: float, n: int, count: int, stream: str):
    """从头生成 count 条历史，返回每个 SW/EW 参数对应的平均总体期望收益。

    stream=test：共同测试历史计算两种曲线；pilot_sw 与 pilot_ew：
    分别用完全独立的训练历史选收益最优固定参数。
    """
    mapping = {'test': (303,3), 'pilot_sw': (101,1), 'pilot_ew': (202,2)}
    stream_id, mode = mapping[stream]
    pg, rev, _ = truth_grid(sigma)
    H = T-1
    weight_ew = np.zeros((len(LAMBDAS),H+1))
    for i, lam in enumerate(LAMBDAS):
        weight_ew[i,1:] = np.exp(-lam*np.arange(1,H+1))
    scale = np.exp(-MU*H)
    low = PRICE_MIN/scale
    high = PRICE_MAX/scale
    total_sw = np.zeros(H)
    total_ew = np.zeros(len(LAMBDAS))
    for rep in range(count):
        q, dist = make_history(sigma,n,stream_id,rep)
        sw, ew = _paths(q,dist,weight_ew,low,high,mode)
        if mode in (1,3):
            total_sw += np.interp(sw*scale,pg,rev)
        if mode in (2,3):
            total_ew += np.interp(ew*scale,pg,rev)
        if (rep+1) % max(10, count//10) == 0:
            print(f'  {stream}: {rep+1}/{count} histories finished', flush=True)
    return total_sw/count, total_ew/count


def run_scenario(sigma: float, n: int, stream: str, count: int):
    """生成或复用本项目已经计算过的同参数 CSV；正式复现用 FORCE_RECOMPUTE=True。"""
    # Cache must depend on the complete model/price grid, not only sigma/n/seed.
    # This prevents accidental reuse after changing the experimental settings.
    relevant_settings = {
        'MU': MU, 'T': T, 'LOG_X_STD': LOG_X_STD,
        'PRICE_MIN': PRICE_MIN, 'PRICE_MAX': PRICE_MAX,
        'LAMBDA_MAX': LAMBDA_MAX, 'LAMBDA_STEP': LAMBDA_STEP,
    }
    config_hash = hashlib.sha256(
        json.dumps(relevant_settings, sort_keys=True).encode('utf-8')
    ).hexdigest()[:10]
    key = f'sigma{sigma:.3f}_n{n}_{stream}_reps{count}_seed{SEED}_cfg{config_hash}'
    path = DATA/f'{key}.csv'
    if path.exists() and not FORCE_RECOMPUTE:
        d = pd.read_csv(path)
        # The rectangular CSV pads the shorter SW/EW array with NaN.
        # Strip these before argmax to avoid selecting a padded element.
        sw = d['sw_revenue'].dropna().to_numpy(dtype=float)
        ew = d['ew_revenue'].dropna().to_numpy(dtype=float)
        # All streams serialize both arrays. The inactive stream is all zeros.
        expected_sw = T - 1
        expected_ew = len(LAMBDAS)
        if len(sw) == expected_sw and len(ew) == expected_ew:
            return sw, ew
        print(f'Cache shape mismatch, recalculating: {path.name}', flush=True)
    print(f'RUN sigma={sigma:.3f}, n={n}, {stream}={count}', flush=True)
    sw, ew = simulate_averages(sigma,n,count,stream)
    # SW 有 H 个候选, EW 有 81 个; 允许 NaN 补齐，保持单文件结构。
    size = max(len(sw),len(ew))
    a = np.full(size,np.nan);b = np.full(size,np.nan)
    a[:len(sw)] = sw; b[:len(ew)] = ew
    pd.DataFrame({'sw_revenue':a,'ew_revenue':b}).to_csv(path,index=False)
    return sw,ew


def calc_bound(sigma: float,n: int):
    h=np.arange(1,T)
    bound_sw = KAPPA_X*sigma*(h+1)/2 + C_DELTA/np.sqrt(n*h)
    d=np.arange(1,T,dtype=float)
    weights=np.exp(-LAMBDAS[:,None]*d[None,:])
    mean_distance=(weights*d).sum(axis=1)/weights.sum(axis=1)
    neff=n*(weights.sum(axis=1)**2)/(weights**2).sum(axis=1)
    bound_ew=KAPPA_X*sigma*mean_distance + C_DELTA/np.sqrt(neff)
    return int(h[np.argmin(bound_sw)]), float(LAMBDAS[np.argmin(bound_ew)])


def plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.labelsize':11,'axes.titlesize':12,'pdf.fonttype':42,
                         'ps.fonttype':42,'savefig.dpi':250})


def savefig(fig, name):
    for ext in ('png','pdf','svg'):
        fig.savefig(FIG/f'{name}.{ext}', dpi=600, bbox_inches='tight', facecolor='white')
