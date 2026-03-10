import sys
print(f"Python version: {sys.version}")
print(f"sys.path: {sys.path}")

# 尝试导入 autoquant 模块
try:
    import autoquant
    print("✓ autoquant 模块导入成功")
    print(f"AutoQuant version: {autoquant.__version__}")
except ImportError as e:
    print(f"✗ autoquant 模块导入失败: {e}")
    import traceback
    traceback.print_exc()
