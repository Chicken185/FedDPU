import numpy as np
from scipy.optimize import fsolve

# 定义需要求解的非线性三角方程
def equation(theta):
    return 1500 * np.cos(theta) + 1740 * np.sin(theta) - 2191.6

# 给定一个初始猜测值：0.5 弧度
initial_guess = 0.5
theta_solution = fsolve(equation, initial_guess)[0]

# 将弧度转换为角度
theta_degrees = np.degrees(theta_solution)

# 计算总耗时 
T_seconds = 1160 / (1.5 * np.cos(theta_solution))
T_minutes = T_seconds / 60

# 输出结果
print(f"求解得到的游向角(角度):{theta_degrees:.2f}度")
print(f"预估抢渡总成绩(秒):{T_seconds:.2f}秒")