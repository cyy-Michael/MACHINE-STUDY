print("测试开始")
import sys
print("=== 当前Python搜索路径 ===")
print(sys.path)

try:
    from backend.algorithms.registry import list_algorithms, create_algorithm
    print("\n✅ 导入registry成功")

    # 获取全部算法
    alg_list = list_algorithms()
    print("\n可用算法列表：")
    for item in alg_list:
        print(item)

    # 实例化线性回归作为测试
    lr = create_algorithm("linear_regression")
    print("\n✅ 创建线性回归实例成功：", lr)

except Exception as e:
    print(f"\n❌ 出错了：{type(e)}")
    print(e)
