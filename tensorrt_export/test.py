import os

# 获取CPU的逻辑核心数
cpu_count = os.cpu_count()

print("CPU的线程数:", cpu_count)