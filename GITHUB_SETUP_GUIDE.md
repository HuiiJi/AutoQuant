# GitHub 上传指南

## 第一步：在GitHub上创建新仓库

1. 访问 https://github.com 并登录
2. 点击右上角 "+" → "New repository"
3. 填写：
   - Repository name: `autoquant`
   - Description: `专业的AI模型量化工具链 - 支持PTQ/QAT、混合精度、ONNX导出`
   - 选择 Public
   - **不要**勾选 "Initialize this repository with a README"
4. 点击 "Create repository"

---

## 第二步：本地初始化Git（在Git Bash中）

打开Git Bash，进入项目目录：

```bash
cd /c/Users/75241/Desktop/trae_code/autoquant
```

### 1. 检查是否已有Git仓库
```bash
ls -la
# 看有没有 .git 文件夹，如果没有，继续
```

### 2. 初始化Git仓库
```bash
git init
```

### 3. 配置用户信息（如果还没配置）
```bash
git config user.name "你的GitHub用户名"
git config user.email "你的GitHub邮箱"
```

### 4. 创建 .gitignore 文件（重要！）
我们不希望把临时文件、模型文件等上传到GitHub

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# PyTorch/ONNX models
*.pth
*.pt
*.onnx
*.pkl

# Data
*.jpg
*.png
*.jpeg
data/
datasets/

# Logs
*.log
logs/

# Jupyter Notebook
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
EOF
```

### 5. 添加文件到Git
```bash
git add .
```

### 6. 提交
```bash
git commit -m "Initial commit: AutoQuant - Professional AI quantization toolkit"
```

### 7. 关联GitHub仓库
把下面命令中的 `你的用户名` 换成你真实的GitHub用户名：
```bash
git remote add origin https://github.com/你的用户名/autoquant.git
```

### 8. 推送到GitHub
```bash
git branch -M main
git push -u origin main
```

如果提示输入用户名和密码，按要求输入。

---

## 第三步：刷新GitHub页面

回到你的GitHub仓库页面，刷新一下，应该能看到代码了！

---

## （可选）添加README和完善

### 如果想要更好的README，可以更新一下

你的项目已有README.md，可以根据需要修改。

### 添加标签（Topics）
在GitHub仓库页面，点击右侧的 "About" 旁边的齿轮图标，添加标签：
- `quantization`
- `pytorch`
- `onnx`
- `tensorrt`
- `ptq`
- `qat`
- `deep-learning`

---

## 常见问题

### Q: 提示 "fatal: remote origin already exists"
A: 先删除旧的，再添加：
```bash
git remote remove origin
git remote add origin https://github.com/你的用户名/autoquant.git
```

### Q: 提示认证失败
A: 现在GitHub需要用Personal Access Token，不是密码。
1. 去 https://github.com/settings/tokens
2. 生成新token，勾选 repo 权限
3. 用token当密码输入

---

## 上传完成后

恭喜！你的项目现在是开源的了！可以把链接分享给别人：
`https://github.com/你的用户名/autoquant`
