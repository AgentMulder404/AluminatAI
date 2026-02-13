#!/bin/bash
set -e

echo "🚀 AluminatAI GPU Agent Installer"
echo "=================================="
echo

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ Error: nvidia-smi not found.${NC}"
    echo "Please install NVIDIA drivers first:"
    echo "  https://www.nvidia.com/download/index.aspx"
    exit 1
fi

echo -e "${GREEN}✅ NVIDIA drivers detected${NC}"
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "   GPU: $GPU_INFO"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: Python 3 not found.${NC}"
    echo "Please install Python 3.8 or higher:"
    echo "  https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✅ Python $PYTHON_VERSION detected${NC}"

# Check Python version is 3.8+
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}❌ Error: Python 3.8+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

# Install directory
INSTALL_DIR="/opt/aluminatai-agent"
echo
echo "📁 Install directory: $INSTALL_DIR"

# Check if already installed
if [ -d "$INSTALL_DIR" ]; then
    echo -e "${YELLOW}⚠️  Previous installation found${NC}"
    read -p "   Remove and reinstall? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Removing old installation..."
        sudo systemctl stop aluminatai-agent 2>/dev/null || true
        sudo systemctl disable aluminatai-agent 2>/dev/null || true
        sudo rm -rf "$INSTALL_DIR"
    else
        echo "Installation cancelled."
        exit 0
    fi
fi

# Create install directory
echo "📦 Installing agent..."
sudo mkdir -p "$INSTALL_DIR"

# Copy agent files
if [ -f "main.py" ]; then
    # Installing from source directory
    sudo cp -r main.py collector.py uploader.py config.py "$INSTALL_DIR/"
    sudo cp requirements.txt "$INSTALL_DIR/" 2>/dev/null || true
else
    echo -e "${RED}❌ Error: Agent files not found${NC}"
    echo "Please run this script from the agent directory."
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
cd "$INSTALL_DIR"
sudo python3 -m pip install -r requirements.txt --quiet

# Prompt for API key
echo
echo -e "${GREEN}🔑 API Key Setup${NC}"
echo "   Find your API key in your AluminatAI dashboard:"
echo "   https://aluminatai.com/dashboard/setup"
echo
read -p "   Enter API Key: " API_KEY

# Validate API key format
if [[ ! $API_KEY =~ ^alum_ ]]; then
    echo -e "${YELLOW}⚠️  Warning: API key should start with 'alum_'${NC}"
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
fi

# Create environment configuration
echo "⚙️  Creating configuration..."
sudo tee "$INSTALL_DIR/.env" > /dev/null <<EOF
# AluminatAI Agent Configuration
ALUMINATAI_API_KEY=$API_KEY
ALUMINATAI_API_ENDPOINT=https://aluminatai.com/api/metrics/ingest
SAMPLE_INTERVAL=5.0
UPLOAD_INTERVAL=60
LOG_LEVEL=INFO
ENABLE_LOCAL_BACKUP=true
EOF

sudo chmod 600 "$INSTALL_DIR/.env"

# Create log directory
sudo mkdir -p /var/log/aluminatai
sudo chmod 755 /var/log/aluminatai

# Create systemd service
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/aluminatai-agent.service > /dev/null <<EOF
[Unit]
Description=AluminatAI GPU Energy Monitoring Agent
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=/usr/bin/python3 $INSTALL_DIR/main.py --quiet --output /var/log/aluminatai/metrics.csv
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Enable service
echo "▶️  Enabling service..."
sudo systemctl enable aluminatai-agent

# Start service
echo "▶️  Starting service..."
sudo systemctl start aluminatai-agent

# Wait and check status
sleep 3

if sudo systemctl is-active --quiet aluminatai-agent; then
    echo
    echo -e "${GREEN}✅ Installation complete!${NC}"
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}🎉 Agent is running and sending metrics to AluminatAI${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    echo "Useful commands:"
    echo "  Status:   sudo systemctl status aluminatai-agent"
    echo "  Logs:     sudo journalctl -u aluminatai-agent -f"
    echo "  Stop:     sudo systemctl stop aluminatai-agent"
    echo "  Restart:  sudo systemctl restart aluminatai-agent"
    echo
    echo "Dashboard: https://aluminatai.com/dashboard"
    echo
    echo "Metrics will appear in your dashboard within 60 seconds."
else
    echo
    echo -e "${RED}❌ Service failed to start${NC}"
    echo
    echo "Check logs for errors:"
    echo "  sudo journalctl -u aluminatai-agent -n 50"
    echo
    exit 1
fi
