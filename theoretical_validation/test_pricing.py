"""右键 Run：验证 NumPy/Numba 经验最优报价与穷举断点法一致。"""
import numpy as np
from core import _weighted_exact_price


def main():
    rng = np.random.default_rng(2197)
    for k in range(50):
        n = 3 + k
        prices = rng.choice(np.array([0.05,.2,.3,.4,.7,.9,1.2,1.5,2.,3.]), size=n).astype(np.float64)
        ages = rng.integers(1,6,size=n,dtype=np.int64)
        order = np.argsort(prices)
        q, d = prices[order], ages[order]
        weight = np.r_[0.,rng.uniform(.05,2,size=5)]
        lo,hi = .1,2.5
        estimated=_weighted_exact_price(q,d,weight,lo,hi)
        candidate_prices = np.unique(np.r_[lo, hi, q[(q>=lo)&(q<=hi)]])
        values = np.array([p*np.sum(weight[d[q>=p]]) for p in candidate_prices])
        optimum = values.max()
        achieved = estimated*np.sum(weight[d[q>=estimated]])
        assert np.isclose(optimum,achieved,atol=1e-11),f'{k}: {optimum} vs {achieved}'
    print('验证成功：50组含重复报价的随机样本全部通过精确穷举检验。')

if __name__=='__main__':
    main()
