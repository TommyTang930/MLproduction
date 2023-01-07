# backend/Dockerfile
FROM python:3.9
# 设置目录
WORKDIR /app
# 由于H2O基于Java，需要安装Java(JRE)
RUN apt-get update && apt-get install -y default-jre
# 拷贝requirements.txt
COPY requirements-backend.txt requirements.txt
# 安装依赖包
RUN pip install -r requirements.txt
# 拷贝项目文件到工作区
COPY . /app
# 对外暴露的端口
EXPOSE 8080
# 执行命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]