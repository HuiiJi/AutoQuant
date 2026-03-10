import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath('.'))

try:
    from autoquant import ObserverBase, MinMaxObserver, HistogramObserver, MovingAverageMinMaxObserver, PercentileObserver, MSEObserver
    print('All observers imported successfully!')
except Exception as e:
    print(f'Error importing observers: {e}')
    import traceback
    traceback.print_exc()
