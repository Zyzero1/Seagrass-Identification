FROM python:3.10-slim

# Library sistem yang dibutuhkan opencv & sejenisnya
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user app/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user app/ /app

USER user
ENV PATH="/home/user/.local/bin:$PATH"

EXPOSE 7860

CMD ["gunicorn", "--worker-class", "gthread", "--threads", "4", "-w", "1", "-b", "0.0.0.0:7860", "app_backup:app"]