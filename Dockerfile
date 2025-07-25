FROM ubuntu:22.04

# Install wine and Python
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        wget software-properties-common git \
        python3 python3-pip \
        wine64 xvfb && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/bot
COPY . /opt/bot
RUN pip3 install --no-cache-dir -r requirements.txt

# Optional MetaTrader 5 install under Wine
RUN mkdir -p /opt/mt5 && \
    wget -O /tmp/mt5setup.exe https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe && \
    wine /tmp/mt5setup.exe /silent /dir=/opt/mt5 || true && \
    rm /tmp/mt5setup.exe && \
    python3 scripts/setup_terminal.py /opt/mt5

ENV DISPLAY=:0
HEALTHCHECK CMD ["python3", "scripts/healthcheck.py"]
CMD ["bash", "scripts/run_bot.sh"]
