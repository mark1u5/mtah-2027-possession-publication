FROM python:3.12-slim
WORKDIR /app
COPY . /app/demo
EXPOSE 4174
CMD ["python", "demo/server.py", "--host", "0.0.0.0", "--port", "4174"]
