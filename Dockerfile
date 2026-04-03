FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DB_PATH=/data/bot.db
ENV PAAS_MCP_URL=https://solver.planning.domains/mcp
ENV L2P_MCP_URL=http://host.docker.internal:8002/mcp

WORKDIR /app

COPY IPA_Discbot/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY IPA_Discbot /app/IPA_Discbot

CMD ["python3", "-m", "IPA_Discbot.bot"]
