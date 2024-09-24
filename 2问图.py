import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

cases = [
    {'零配件1次品率': 0.1, '零配件2次品率': 0.1, '成品次品率': 0.1, '零配件1检测成本': 2, '零配件2检测成本': 3,
     '成品检测成本': 3, '装配成本': 6, '市场售价': 56, '调换损失': 6, '拆解费用': 5},
    {'零配件1次品率': 0.2, '零配件2次品率': 0.2, '成品次品率': 0.2, '零配件1检测成本': 2, '零配件2检测成本': 3,
     '成品检测成本': 3, '装配成本': 6, '市场售价': 56, '调换损失': 6, '拆解费用': 5},
    {'零配件1次品率': 0.1, '零配件2次品率': 0.1, '成品次品率': 0.1, '零配件1检测成本': 2, '零配件2检测成本': 3,
     '成品检测成本': 3, '装配成本': 6, '市场售价': 56, '调换损失': 30, '拆解费用': 5},
    {'零配件1次品率': 0.2, '零配件2次品率': 0.2, '成品次品率': 0.2, '零配件1检测成本': 1, '零配件2检测成本': 1,
     '成品检测成本': 2, '装配成本': 6, '市场售价': 56, '调换损失': 30, '拆解费用': 5},
    {'零配件1次品率': 0.1, '零配件2次品率': 0.2, '成品次品率': 0.1, '零配件1检测成本': 8, '零配件2检测成本': 1,
     '成品检测成本': 2, '装配成本': 6, '市场售价': 56, '调换损失': 10, '拆解费用': 5},
    {'零配件1次品率': 0.05, '零配件2次品率': 0.05, '成品次品率': 0.05, '零配件1检测成本': 2, '零配件2检测成本': 3,
     '成品检测成本': 3, '装配成本': 6, '市场售价': 56, '调换损失': 10, '拆解费用': 40}
]

def calc_total_cost(case, detect_parts1=True, detect_parts2=True, detect_final=True, dismantle=True):
    n_parts1 = 100
    n_parts2 = 100
    cost_parts1 = n_parts1 * case['零配件1检测成本'] if detect_parts1 else 0
    cost_parts2 = n_parts2 * case['零配件2检测成本'] if detect_parts2 else 0

    loss_parts1 = (n_parts1 * case['零配件1次品率']) * (case['装配成本']) if not detect_parts1 else 0
    loss_parts2 = (n_parts2 * case['零配件2次品率']) * (case['装配成本']) if not detect_parts2 else 0

    n_final_products = 100
    cost_final = n_final_products * case['成品检测成本'] if detect_final else 0
    loss_final = (n_final_products * case['成品次品率']) * case['调换损失'] if not detect_final else 0

    dismantle_cost = n_final_products * case['拆解费用'] if dismantle else 0
    dismantle_revenue = (n_final_products * case['成品次品率']) * case['市场售价'] if dismantle else 0

    total_cost = (cost_parts1 + cost_parts2 + loss_parts1 + loss_parts2 +
                  cost_final + loss_final + dismantle_cost - dismantle_revenue)

    defective_rate = case['成品次品率'] * (1 if detect_final else 0.5)

    return total_cost, defective_rate

strategies = []
for detect_parts1 in [True, False]:
    for detect_parts2 in [True, False]:
        for detect_final in [True, False]:
            for dismantle in [True, False]:
                strategies.append({
                    "detect_parts1": detect_parts1,
                    "detect_parts2": detect_parts2,
                    "detect_final": detect_final,
                    "dismantle": dismantle
                })

strategy_explanations = []
for i, strategy in enumerate(strategies):
    strategy_explanations.append(
        f"策略 {i + 1}: 检测零配件 1 {'是' if strategy['detect_parts1'] else '否'}，检测零配件 2 {'是' if strategy['detect_parts2'] else '否'}，"
        f"检测成品 {'是' if strategy['detect_final'] else '否'}，拆解不合格成品 {'是' if strategy['dismantle'] else '否'}"
    )

costs = []
defective_rates = []

for i, case in enumerate(cases):
    print(f"\n情况 {i + 1}:")
    case_costs = []
    case_defective_rates = []
    for j, strategy in enumerate(strategies):
        total_cost, defective_rate = calc_total_cost(case, **strategy)
        case_costs.append(total_cost)
        case_defective_rates.append(defective_rate)
        print(f"{strategy_explanations[j]}: 总成本 = {total_cost}")
    costs.append(case_costs)
    defective_rates.append(case_defective_rates)

costs = np.array(costs)
defective_rates = np.array(defective_rates)

plt.figure(figsize=(10, 6))
for i in range(costs.shape[1]):
    plt.plot(range(1, costs.shape[0] + 1), costs[:, i], marker='o', label=f"策略 {i + 1} - 总成本")
plt.xlabel("情况")
plt.ylabel("总成本")
plt.title("不同情况和策略下的总成本比较")
plt.legend()
plt.grid(True)
plt.show()
