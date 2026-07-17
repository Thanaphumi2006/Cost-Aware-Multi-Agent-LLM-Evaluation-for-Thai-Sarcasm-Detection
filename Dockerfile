# Light web-only image for the /app demo (no WangchanBERTa / torch).
# Three of the four models work (the LLM ones); the free offline model shows as unavailable.
FROM python:3.12-slim
WORKDIR /app
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt
COPY Gold/ /app/Gold/
# Hugging Face Spaces serves on 7860; bind all interfaces so the host can reach it.
ENV HOST=0.0.0.0 PORT=7860
EXPOSE 7860
CMD ["python", "Gold/app.py"]
