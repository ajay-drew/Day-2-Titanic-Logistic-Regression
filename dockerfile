FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY logistic_regression_model.py .
COPY train.csv .
VOLUME /app/
ENV PYTHONUNBUFFERED=1
CMD ["python", "logistic_regression_model.py"]