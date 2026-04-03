FROM python:3.14-slim

ENV OPENAI_MODEL=gpt-4.1
ENV DB_PATH=/data/bot.db

WORKDIR /app

COPY IPA_Discbot/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY IPA_Discbot /app/IPA_Discbot

CMD ["python3", "-m", "IPA_Discbot.bot"]
