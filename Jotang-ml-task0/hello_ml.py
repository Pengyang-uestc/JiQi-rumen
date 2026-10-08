"""Python 热身：成绩统计与 NumPy 矩阵乘法。"""

import numpy as np


scores = {
    "张三": [88, 92, 90],
    "李四": [76, 85, 80],
    "王五": [95, 91, 97],
}


def average_score(student_scores: list[int]) -> float:
    """计算一组成绩的平均分。"""
    return sum(student_scores) / len(student_scores)


def highest_score(all_scores: dict[str, list[int]]) -> tuple[str, int]:
    """找出最高分及其对应的姓名。"""
    name, scores_for_student = max(
        all_scores.items(), key=lambda item: max(item[1])
    ) #key=lambda item是给max()函数指定比较规则，item是这个小函数的返回值，item又来自all_scores里面的每一项，0是字符串，1是成绩列表。
    return name, max(scores_for_student)


print("各人成绩平均分：")
for name, student_scores in scores.items():
    print(f"{name}: {average_score(student_scores):.2f}")

top_name, top_score = highest_score(scores)
print(f"最高分：{top_name}，{top_score} 分")


matrix_a = np.array([[1, 2, 3], [4, 5, 6]])
matrix_b = np.array([[7, 8], [9, 10], [11, 12]])
matrix_result = matrix_a @ matrix_b  #@代表矩阵乘法

print("\n矩阵乘法结果：")
print(matrix_result)
print(f"输入矩阵 A 形状：{matrix_a.shape}") #{}里面可以写变量/表达式，Python会自动替换
print(f"输入矩阵 B 形状：{matrix_b.shape}")
print(f"输出矩阵形状：{matrix_result.shape}") #f直接把变量放进字符串里面输出，不用拼接

