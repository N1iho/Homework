import numpy as np 
import matplotlib.pyplot  as plt 
import pandas as pd 
from scipy.optimize  import minimize_scalar 
import time 
 
# 定义目标函数和一阶导数 
def phi(alpha):
    return 3*alpha**4 - 16*alpha**3 + 30*alpha**2 - 24*alpha + 8 
 
def dphi(alpha):
    return 12*alpha**3 - 48*alpha**2 + 60*alpha - 24 
 
# 二分法实现 
def binary_search(func_deriv, a_init, b_init, tol=1e-6, max_iter=100):
    a, b = a_init, b_init 
    history = []
    for _ in range(max_iter):
        mid = (a + b) / 2 
        f_mid = func_deriv(mid)
        f_a = func_deriv(a)
        
        history.append({ 
            'iteration': len(history)+1,
            'a': a,
            'b': b,
            'midpoint': mid,
            'interval': b - a,
            'f_value': phi(mid)
        })
        
        if f_mid == 0:
            break 
        if f_mid * f_a < 0:
            b = mid 
        else:
            a = mid 
            
        if (b - a) < tol:
            break 
    return (a + b)/2, pd.DataFrame(history)
 
# 三分法实现 
def ternary_search(func, a_init, b_init, tol=1e-6, max_iter=100):
    a, b = a_init, b_init 
    history = []
    for _ in range(max_iter):
        u = a + (b - a)/3 
        v = b - (b - a)/3 
        fu, fv = func(u), func(v)
        
        optimal = u if fu < fv else v 
        history.append({ 
            'iteration': len(history)+1,
            'a': a,
            'b': b,
            'u': u,
            'v': v,
            'interval': b - a,
            'f_value': min(fu, fv)
        })
        
        if fu < fv:
            b = v 
        else:
            a = u 
            
        if (b - a) < tol:
            break 
    return (a + b)/2, pd.DataFrame(history)
 
# 黄金分割法实现 
def golden_section(func, a_init, b_init, tol=1e-6, max_iter=100):
    a, b = a_init, b_init 
    lambda_ = (np.sqrt(5)-1)/2  
    history = []
    
    u = a + (1 - lambda_)*(b - a)
    v = a + lambda_*(b - a)
    fu, fv = func(u), func(v)
    
    for _ in range(max_iter):
        current_optimal = u if fu < fv else v 
        history.append({ 
            'iteration': len(history)+1,
            'a': a,
            'b': b,
            'u': u,
            'v': v,
            'interval': b - a,
            'f_value': min(fu, fv)
        })
        
        if fu < fv:
            b = v 
            v, fv = u, fu 
            u = a + (1 - lambda_)*(b - a)
            fu = func(u)
        else:
            a = u 
            u, fu = v, fv 
            v = a + lambda_*(b - a)
            fv = func(v)
            
        if (b - a) < tol:
            break 
    return (a + b)/2, pd.DataFrame(history)
 
# 主程序 
if __name__ == "__main__":
    # 执行所有算法 
    results = {}
    
    # 二分法 
    start = time.time() 
    opt_bin, df_bin = binary_search(dphi, 0, 3)
    results['Binary'] = {
        'time': time.time()  - start,
        'df': df_bin,
        'optimal': opt_bin 
    }
    
    # 三分法 
    start = time.time() 
    opt_ter, df_ter = ternary_search(phi, 0, 3)
    results['Ternary'] = {
        'time': time.time()  - start,
        'df': df_ter,
        'optimal': opt_ter 
    }
    
    # 黄金分割法 
    start = time.time() 
    opt_gold, df_gold = golden_section(phi, 0, 3)
    results['Golden'] = {
        'time': time.time()  - start,
        'df': df_gold,
        'optimal': opt_gold 
    }
    
    # SciPy对比 
    res = minimize_scalar(phi, bounds=(0,3), method='bounded')
    results['SciPy'] = {
        'optimal': res.x,
        'fval': res.fun  
    }
    
    # 结果展示 
    print(f"{'Method':<10} | {'Optimal (α)':<12} | {'Time (s)':<8} | {'Iterations':<10}")
    for method in ['Binary', 'Ternary', 'Golden']:
        data = results[method]
        print(f"{method:<10} | {data['optimal']:12.8f} | {data['time']:8.6f} | {len(data['df']):10}")
    print(f"\nSciPy result: α = {results['SciPy']['optimal']:.8f}, f(α) = {results['SciPy']['fval']:.8f}")
 
    # 绘制收敛曲线 
    plt.figure(figsize=(12,  6))
    
    # 区间长度收敛 
    plt.subplot(1,  2, 1)
    for method in ['Binary', 'Ternary', 'Golden']:
        plt.semilogy(results[method]['df']['interval'],  label=method)
    plt.title('Interval  Length Convergence')
    plt.xlabel('Iteration') 
    plt.ylabel('Log(Interval  Length)')
    plt.legend() 
    
    # 函数值收敛 
    plt.subplot(1,  2, 2)
    for method in ['Binary', 'Ternary', 'Golden']:
        plt.semilogy(results[method]['df']['f_value'],  label=method)
    plt.title('Function  Value Convergence')
    plt.xlabel('Iteration') 
    plt.ylabel('Log(Function  Value)')
    plt.legend() 
    
    plt.tight_layout() 
    plt.savefig('convergence_curves.png') 
    plt.show() 
 
    # 导出迭代数据到Excel 
    with pd.ExcelWriter('optimization_process.xlsx')  as writer:
        for method in ['Binary', 'Ternary', 'Golden']:
            results[method]['df'].to_excel(writer, sheet_name=method, index=False)
    
    # 绘制函数图像 
    x = np.linspace(0,  3, 400)
    y = phi(x)
    plt.figure(figsize=(8,  6))
    plt.plot(x,  y, label='φ(α)')
    plt.scatter(opt_bin,  phi(opt_bin), c='red', label='Optimal')
    plt.title('Function  Visualization')
    plt.xlabel('α') 
    plt.ylabel('φ(α)') 
    plt.legend() 
    plt.grid(True) 
    plt.savefig('function_plot.png') 
    plt.show()