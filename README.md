# 🏙️ SORA-ATMAS: Adaptive Trust Management & Multi-LLM Aligned Governance for Smart Cities

[![python](https://img.shields.io/badge/python-3.9%252B-blue)](https://www.python.org/)
[![MultiChain](https://img.shields.io/badge/MultiChain-2.3.3-orange)](https://www.multichain.com/)
[![LLMs](https://img.shields.io/badge/LLMs-GPT4%7CGrok%7CDeepSeek-green)](https://platform.openai.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2501.XXXXX-blue)](https://doi.org/10.48550/arXiv.2501.XXXXX)

## 📋 Table of Contents
- [System Overview](#-system-overview)
- [Architecture](#️-dual-chain-system-architecture)
- [Prerequisites](#-prerequisites)
- [Complete Installation Guide](#-complete-installation-guide)
- [Agentic & SORA Blockchain Setup](#️-agentic--sora-blockchain-setup)
- [Blockchain Streams Configuration](#-blockchain-streams-configuration)
- [Core Modules Detailed](#-core-modules-detailed)
- [Trust & Risk Model Mathematics](#-trust--risk-model-mathematics)
- [Running the System](#-running-the-system)
- [Performance Benchmarks](#-performance-benchmarks)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)
- [System Monitoring](#-system-monitoring)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🎯 System Overview

SORA-ATMAS (Security & Operational Response Agent - Adaptive Trust Management System) is a cutting-edge governance framework for smart cities that integrates:

- **Dual-Chain Blockchain Architecture**: Agentic Blockchain for edge-level provenance + SORA Blockchain for centralized governance
- **Multi-LLM Reasoning Ensemble**: GPT-4, Grok, and DeepSeek for policy-aligned semantic reasoning
- **Three Intelligent Agents**: Weather, Traffic, and Safety agents with specialized perception capabilities
- **Adaptive Trust Regulation**: Context-aware risk assessment with dynamic threshold enforcement
- **Real-time Governance**: MAE-based LLM selection with error-directed feedback loops

**Key Innovation**: Reduces mean absolute error by 35% through governance-guided multi-LLM convergence while maintaining throughput of 13.8–17.2 requests/second with <100ms governance delay.

## 🏗️ Dual-Chain System Architecture

```ascii
+---------------------+     +-----------------------+     +-----------------------+
|    Perception Layer |     |   Agentic Layer       |     |  Blockchain Layer     |
|                     |     |                       |     |                       |
|  ┌──────────────┐   |     |  ┌─────────────────┐  |     |  ┌─────────────────┐  |
|  │ Weather      │━━━┿━━━━━┿━▶│ Security        │  |     |  │ Agentic         │  |
|  │ Sensors/API  │   |     |  │ Compliance      │━━┿━━━━━┿━▶│ Blockchain      │  |
|  └──────────────┘   |     |  └─────────────────┘  |     |  │ (Edge)          │  |
|                     |     |                       |     |  └─────────────────┘  |
|  ┌──────────────┐   |     |  ┌─────────────────┐  |     |                       |
|  │ Traffic      │━━━┿━━━━━┿━▶│ Domain-Specific │  |     |  ┌─────────────────┐  |
|  │ CCTV Streams │   |     |  │ Compliance      │━━┿━━━━━┿━▶│ SORA Blockchain │  |
|  └──────────────┘   |     |  └─────────────────┘  |     |  │ (Governance)    │  |
|                     |     |                       |     |  └─────────────────┘  |
|  ┌──────────────┐   |     |  ┌─────────────────┐  |     |                       |
|  │ Safety       │━━━┿━━━━━┿━▶│ Context &       │  |     |  ┌─────────────────┐  |
|  │ Cameras      │   |     |  │ Policy Adapter  │━━┿━━━━━┿━▶│ Global          │  |
|  └──────────────┘   |     |  └─────────────────┘  |     |  │ Repository      │  |
+---------------------+     +-----------------------+     +-----------------------+
                                |                                       |
                                v                                       v
+---------------------+     +-----------------------+     +-----------------------+
|  Multi-LLM Layer    |     |  SORA Governance      |     |  Policy Enforcement   |
|                     |     |                       |     |                       |
|  ┌─────────────────┐|     |  ┌─────────────────┐  |     |  ┌─────────────────┐  |
|  │ GPT-4          │┿━━━━━┿━▶│ Security Policy │  |     |  │ Cross-Domain    │  |
|  │ Reasoning      │|     |  │ Engine          │━━┿━━━━━┿━▶│ Operational      │  |
|  └─────────────────┘|     |  └─────────────────┘  |     |  │ Policy Engine   │  |
|                     |     |                       |     |  └─────────────────┘  |
|  ┌─────────────────┐|     |  ┌─────────────────┐  |     |                       |
|  │ Grok           │┿━━━━━┿━▶│ Adaptive Trust  │  |     |  ┌─────────────────┐  |
|  │ Reasoning      │|     |  │ & Risk Engine   │━━┿━━━━━┿━▶│ Ecosystem       │  |
|  └─────────────────┘|     |  └─────────────────┘  |     |  │ Metrics         │  |
|                     |     |                       |     |  └─────────────────┘  |
|  ┌─────────────────┐|     |  ┌─────────────────┐  |     |                       |
|  │ DeepSeek       │┿━━━━━┿━▶│ MAE-Based       │  |     |  ┌─────────────────┐  |
|  │ Reasoning      │|     |  │ Selection       │━━┿━━━━━┿━▶│ Hysteresis &    │  |
|  └─────────────────┘|     |  └─────────────────┘  |     |  │ Cooldown        │  |
+---------------------+     +-----------------------+     +-----------------------+
```
## 🏗️ Architecture Components

- **Perception Layer**: IoT sensors, CCTV cameras, and API data sources
- **Agentic Layer**: Domain-specific agents with security and compliance modules
- **Multi-LLM Layer**: Parallel LLM reasoning for contextual interpretation
- **Blockchain Layer**: Dual-chain immutable logging (Agentic + SORA)
- **Governance Layer**: Policy enforcement with adaptive trust regulation
- **Policy Layer**: Cross-domain coordination and ecosystem management

## 📋 Prerequisites

### System Requirements

- **Operating System**: Ubuntu 20.04/22.04 LTS (recommended) or Windows 10/11 with WSL2
- **Memory**: 8GB minimum, 16GB recommended
- **Storage**: 20GB free space
- **GPU**: Optional (NVIDIA with CUDA recommended for YOLO)
- **Network**: Stable internet connection

### Software Requirements

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git wget curl build-essential libssl-dev libffi-dev python3-dev
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
```
### API Keys Required

| Service              | Purpose                          | Access Link                                   |
|----------------------|----------------------------------|-----------------------------------------------|
| OpenMeteo API        | Weather data (free)              | https://open-meteo.com/                       |
| OpenAI GPT-4         | Primary LLM reasoning            | https://platform.openai.com/                  |
| xAI Grok API         | Alternative LLM reasoning        | https://x.ai/                                 |
| DeepSeek API         | Third LLM reasoning              | https://platform.deepseek.com/                |
| Google Sheets API    | Structured logging (optional)    | https://console.cloud.google.com/             |

### Python Packages

```bash
pip install requests numpy pandas matplotlib seaborn scikit-learn cryptography
pip install openai xai-api deepseek-sdk google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
pip install ultralytics opencv-python pillow torch torchvision
pip install xgboost streamlit plotly
pip install multichain-python
pip install python-dotenv tqdm
```
## 🔧 Complete Installation Guide

### Step 1: Clone Repository


#### Clone the SORA-ATMAS repository
```bash
git clone https://github.com/Usama-Antuley/SORA-ATMAS-Adaptive-Trust-Management-and-Multi-LLM-Aligned-Governance-for-Future-Smart-Cities.git
cd SORA-ATMAS-Adaptive-Trust-Management-and-Multi-LLM-Aligned-Governance-for-Future-Smart-Cities
```
#### Navigate to main directory
```bash
cd Multichain
```
#### Create virtual environment
```bash
python3 -m venv sora_env
source sora_env/bin/activate  # Linux/Mac
sora_env\Scripts\activate  # Windows
```
### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Environment
```bash
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/
wget https://github.com/sayedgamal99/Real-Time-Smoke-Fire-Detection-YOLO11/releases/download/v1.0/flare_guard.pt -P models/
```

### Step 4: Download Pre-trained Models
```bash
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/
wget https://github.com/sayedgamal99/Real-Time-Smoke-Fire-Detection-YOLO11/releases/download/v1.0/flare_guard.pt -P models/
```
### Step 5: Directory Structure Verification
```bash
# Verify the directory structure
tree -L 3 -a

# Expected output:
# .
# ├── agent_weather.py
# ├── agent_traffic.py
# ├── agent_safety.py
# ├── sora_governance.py
# ├── monitor_dashboard.py
# ├── benchmark_sora.py
# ├── requirements.txt
# ├── .env
# ├── .env.example
# ├── logs/
# ├── keys/
# │   ├── agentic_rpc.conf
# │   └── sora_rpc.conf
# ├── blockchain/
# ├── streams/
# ├── models/
# │   ├── yolov8n.pt
# │   └── flare_guard.pt
# ├── datasets/
# │   ├── weather_samples.csv
# │   ├── traffic_samples/
# │   └── safety_samples/
# └── config/
#     ├── policy_config.json
#     └── threshold_config.yaml
```

## ⛓️ Agentic & SORA Blockchain Setup

### Step 1: Install MultiChain

```bash
# Download and install MultiChain 2.3.3
cd /tmp
wget https://www.multichain.com/download/multichain-2.3.3.tar.gz
tar -xvzf multichain-2.3.3.tar.gz
cd multichain-2.3.3
```
#### Move binaries to system path
```bash
sudo mv multichaind multichain-cli multichain-util /usr/local/bin/
```
#### Verify installation
```bash
multichaind --version
```
#### Expected: MultiChain 2.3.3 Daemon (community edition)

### Step 2 Create two separate blockchains: Agentic (for edge agents) and SORA (for governance):
#### Create blockchain directories
```bash
mkdir -p ~/.multichain/{agentic,sora}
mkdir -p ~/blockchains/{agentic,sora}
```
#### Create Agentic Blockchain (for agents)
```bash
echo "Creating Agentic Blockchain..."
multichain-util create agentic
```
#### Create SORA Blockchain (for governance)
```bash
echo "Creating SORA Blockchain..."
multichain-util create sora
```
#### Configure custom ports to avoid conflicts
```bash
sed -i 's/default-network-port = [0-9]*/default-network-port = 9740/' ~/.multichain/agentic/params.dat
sed -i 's/default-rpc-port = [0-9]*/default-rpc-port = 9741/' ~/.multichain/agentic/params.dat
sed -i 's/default-network-port = [0-9]*/default-network-port = 9742/' ~/.multichain/sora/params.dat
sed -i 's/default-rpc-port = [0-9]*/default-rpc-port = 9743/' ~/.multichain/sora/params.dat
```
#### Enable mining for both chains

```bash
sed -i 's/mine-empty-rounds = 0/mine-empty-rounds = 1/' ~/.multichain/agentic/params.dat
sed -i 's/mine-empty-rounds = 0/mine-empty-rounds = 1/' ~/.multichain/sora/params.dat
```
### Step 3: Start Blockchain Daemons
```bash
# Start Agentic Blockchain daemon
multichaind agentic -daemon -rpcallowip=127.0.0.1

# Start SORA Blockchain daemon
multichaind sora -daemon -rpcallowip=127.0.0.1

# Check if both are running
multichain-cli agentic getinfo
multichain-cli sora getinfo

# Expected output should show chain information and block height
```
### Step 4: Configure RPC Access
```bash
# Generate secure RPC passwords
AGENTIC_PASSWORD=$(openssl rand -base64 32)
SORA_PASSWORD=$(openssl rand -base64 32)
```
#### Configure Agentic Blockchain RPC
```bash
cat > ~/.multichain/agentic/multichain.conf << EOF
rpcuser=multichainrpc
rpcpassword=$AGENTIC_PASSWORD
rpcallowip=127.0.0.1
rpcconnect=127.0.0.1
rpcport=9741
EOF
```
#### Configure SORA Blockchain RPC
```bash
cat > ~/.multichain/sora/multichain.conf << EOF
rpcuser=multichainrpc
rpcpassword=$SORA_PASSWORD
rpcallowip=127.0.0.1
rpcconnect=127.0.0.1
rpcport=9743
EOF
```
#### Save passwords to keys directory
```bash
mkdir -p keys
echo "$AGENTIC_PASSWORD" > keys/agentic_rpc.conf
echo "$SORA_PASSWORD" > keys/sora_rpc.conf
chmod 600 keys/*.conf
```
#### Restart daemons with new configuration
```bash
pkill multichaind
sleep 5

multichaind agentic -daemon
multichaind sora -daemon
```
### Step 5: Test Blockchain Connections
**test_blockchain.py**
```bash
import multichain
import json

def test_blockchain_connection(chain_name, port, password):
    try:
        mc = multichain.MultiChainClient("127.0.0.1", port, "multichainrpc", password)
        info = mc.getinfo()
        print(f"✓ {chain_name} Blockchain: Connected")
        print(f"  Chain: {info['chainname']}")
        print(f"  Blocks: {info['blocks']}")
        print(f"  Version: {info['version']}")
        return True
    except Exception as e:
        print(f"✗ {chain_name} Blockchain: Connection failed - {e}")
        return False

# Read passwords from keys
with open('keys/agentic_rpc.conf', 'r') as f:
    agentic_pass = f.read().strip()
    
with open('keys/sora_rpc.conf', 'r') as f:
    sora_pass = f.read().strip()

# Test connections
print("=== Testing Blockchain Connections ===")
test_blockchain_connection("Agentic", 9741, agentic_pass)
test_blockchain_connection("SORA", 9743, sora_pass)
```
#Run the test:
```bash
python test_blockchain.py
```
## 📊 Blockchain Streams Configuration

#### Use `create_sora_streams.py` to initialize all required streams on both blockchains.
#### Create a comprehensive Python script to initialize all streams for both blockchains:
 **create_sora_streams.py**

```bash
import multichain
import json
import time
from pathlib import Path

def load_password(chain_name):
    """Load RPC password from keys directory"""
    key_file = Path(f"keys/{chain_name}_rpc.conf")
    if key_file.exists():
        return key_file.read_text().strip()
    else:
        raise FileNotFoundError(f"Password file for {chain_name} not found")

def create_streams():
    """Create all required streams on both blockchains"""
    
    # Blockchain configurations
    chains = {
        "agentic": {
            "port": 9741,
            "password": load_password("agentic"),
            "streams": [
                "WeatherAgentLogs",
                "TrafficAgentLogs", 
                "SafetyAgentLogs",
                "AgentDecisions",
                "LocalPolicies",
                "TrustMetrics",
                "RiskMetrics",
                "PerformanceLogs"
            ]
        },
        "sora": {
            "port": 9743,
            "password": load_password("sora"),
            "streams": [
                "GovDecisions",
                "PolicyEnforcement",
                "CrossDomainActions",
                "EcosystemMetrics",
                "AuditTrail",
                "LLMSelectionLog",
                "FeedbackLogs",
                "EscalationRecords",
                "ComplianceChecks"
            ]
        }
    }
    
    print("=== Creating SORA-ATMAS Blockchain Streams ===\n")
    
    for chain_name, config in chains.items():
        print(f"Configuring {chain_name.upper()} Blockchain...")
        
        try:
            # Connect to blockchain
            mc = multichain.MultiChainClient(
                "127.0.0.1", 
                config["port"], 
                "multichainrpc", 
                config["password"]
            )
            
            created_count = 0
            existing_count = 0
            
            # Create each stream
            for stream in config["streams"]:
                try:
                    # Check if stream already exists
                    existing_streams = mc.liststreams()
                    stream_exists = any(s['name'] == stream for s in existing_streams)
                    
                    if not stream_exists:
                        result = mc.create('stream', stream, True)
                        print(f"  ✅ Created: {stream}")
                        created_count += 1
                        time.sleep(0.5)  # Small delay to avoid rate limiting
                    else:
                        print(f"  ℹ️  Exists: {stream}")
                        existing_count += 1
                        
                except Exception as e:
                    print(f"  ❌ Error creating {stream}: {str(e)[:100]}")
                    
            print(f"  Summary: {created_count} created, {existing_count} already exist\n")
            
        except Exception as e:
            print(f"  ❌ Connection failed: {e}\n")
    
    print("=== Stream Creation Complete ===")
    print("\nNext steps:")
    print("1. Verify streams: python verify_streams.py")
    print("2. Test publishing: python test_stream_publish.py")

if __name__ == "__main__":
    create_streams()
```
#### Run the stream creation script:
```bash
python create_sora_streams.py
```
## Step 2: Verify Stream Creation
```bash
# Create verification script
cat > verify_streams.py << 'EOF'
import multichain
import json

def verify_streams():
    chains = {
        "agentic": {"port": 9741, "password": open('keys/agentic_rpc.conf').read().strip()},
        "sora": {"port": 9743, "password": open('keys/sora_rpc.conf').read().strip()}
    }
    
    for chain_name, config in chains.items():
        print(f"\n{'='*50}")
        print(f"{chain_name.upper()} BLOCKCHAIN STREAMS")
        print('='*50)
        
        try:
            mc = multichain.MultiChainClient("127.0.0.1", config["port"], "multichainrpc", config["password"])
            streams = mc.liststreams()
            
            if streams:
                for stream in streams:
                    print(f"  • {stream['name']}")
                    print(f"    Items: {stream.get('items', 0)}, "
                          f"Confirmed: {stream.get('confirmed', 0)}, "
                          f"Subscribed: {stream.get('subscribed', 'No')}")
            else:
                print("  No streams found")
                
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    verify_streams()
EOF
```

#### Run verification
```bash
python verify_streams.py
```

### Step 3: Test Stream Publishing
```bash
# test_stream_publish.py
import multichain
import json
from datetime import datetime

def test_publishing():
    """Test publishing data to all streams"""
    
    # Connect to blockchains
    agentic_mc = multichain.MultiChainClient("127.0.0.1", 9741, "multichainrpc", 
                                             open('keys/agentic_rpc.conf').read().strip())
    sora_mc = multichain.MultiChainClient("127.0.0.1", 9743, "multichainrpc", 
                                          open('keys/sora_rpc.conf').read().strip())
    
    print("=== Testing Stream Publishing ===\n")
    
    # Test data
    test_data = {
        "timestamp": datetime.now().isoformat(),
        "test_id": "initial_setup",
        "status": "operational",
        "version": "SORA-ATMAS-v1.0"
    }
    
    # Test Agentic Blockchain streams
    print("Testing Agentic Blockchain...")
    agentic_streams = ["WeatherAgentLogs", "TrustMetrics"]
    for stream in agentic_streams:
        try:
            txid = agentic_mc.publish(stream, "test_key", {'json': test_data})
            print(f"  ✅ {stream}: Published (TX: {txid[:16]}...)")
        except Exception as e:
            print(f"  ❌ {stream}: Failed - {str(e)[:80]}")
    
    # Test SORA Blockchain streams
    print("\nTesting SORA Blockchain...")
    sora_streams = ["GovDecisions", "AuditTrail"]
    for stream in sora_streams:
        try:
            txid = sora_mc.publish(stream, "test_key", {'json': test_data})
            print(f"  ✅ {stream}: Published (TX: {txid[:16]}...)")
        except Exception as e:
            print(f"  ❌ {stream}: Failed - {str(e)[:80]}")
    
    print("\n=== Publishing Test Complete ===")

if __name__ == "__main__":
    test_publishing()
```

## 🔧 Core Modules Detailed

1. **Weather Agent** (`agent_weather.py`)
#### Processes meteorological data using XGBoost and multi-LLM reasoning
#### Key Components:
```bash
class WeatherAgent:
    """Smart-city weather monitoring and regime classification agent"""
    
    def __init__(self, city="Islamabad", debug=False):
        self.city = city
        self.api_endpoint = "https://api.open-meteo.com/v1/forecast"
        self.model = self.load_xgboost_model()
        self.llm_clients = self.initialize_llm_clients()
        
    def fetch_weather_data(self):
        """Fetch real-time weather data from OpenMeteo API"""
        params = {
            "latitude": self.get_coordinates()[0],
            "longitude": self.get_coordinates()[1],
            "current": ["temperature_2m", "precipitation", "relative_humidity_2m", 
                       "wind_speed_10m", "uv_index"],
            "forecast_days": 1
        }
        response = requests.get(self.api_endpoint, params=params)
        return self.parse_weather_data(response.json())
    
    def classify_regime(self, weather_data):
        """Classify weather regime using XGBoost model"""
        features = self.extract_features(weather_data)
        prediction = self.model.predict(features)
        
        # Regimes: Normal, Rain, Heavy Rain, Heatwave
        regimes = ["Normal", "Rain", "Heavy Rain", "Heatwave"]
        return regimes[prediction[0]]
    
    def compute_environmental_risk(self, regime, weather_data):
        """Compute environmental risk based on weather regime"""
        risk_weights = {
            "Normal": 0.1,
            "Rain": 0.3,
            "Heavy Rain": 0.7,
            "Heatwave": 0.8
        }
        
        base_risk = risk_weights.get(regime, 0.5)
        
        # Adjust based on actual values
        if weather_data["precipitation"] > 40:  # Heavy rain threshold
            base_risk = min(0.9, base_risk + 0.2)
        if weather_data["temperature"] > 40:  # Heatwave threshold
            base_risk = min(0.9, base_risk + 0.15)
            
        return round(base_risk, 3)
    
    def invoke_llm_reasoning(self, weather_data, regime, risk_score):
        """Invoke GPT-4, Grok, and DeepSeek for trust-risk assessment"""
        prompt = self.build_llm_prompt(weather_data, regime, risk_score)
        llm_outputs = {}
        
        for llm_name, client in self.llm_clients.items():
            try:
                response = client.complete(prompt)
                parsed = self.parse_llm_response(response)
                llm_outputs[llm_name] = {
                    "R": parsed["risk"],
                    "T": parsed["trust"],
                    "explanation": parsed["explanation"],
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                print(f"Error invoking {llm_name}: {e}")
                llm_outputs[llm_name] = {"error": str(e)}
        
        return llm_outputs
    
    def log_to_blockchain(self, agent_data, llm_outputs):
        """Log agent decisions to Agentic Blockchain"""
        log_entry = {
            "agent": "weather",
            "timestamp": datetime.now().isoformat(),
            "weather_data": agent_data,
            "regime": agent_data["regime"],
            "risk_score": agent_data["risk_score"],
            "llm_outputs": llm_outputs,
            "metadata": {
                "city": self.city,
                "api_version": "open-meteo-v1"
            }
        }
        
        mc = self.get_blockchain_client()
        txid = mc.publish("WeatherAgentLogs", f"weather_{datetime.now().timestamp()}", 
                         {'json': log_entry})
        return txid
```
#### Configuration File (config/weather_config.yaml):
```bash
weather_agent:
  cities:
    islamabad:
      latitude: 33.6844
      longitude: 73.0479
      timezone: "Asia/Karachi"
    karachi:
      latitude: 24.8607
      longitude: 67.0011
      timezone: "Asia/Karachi"
  
  thresholds:
    heavy_rain: 40  # mm/day
    heatwave_temp: 40  # °C
    high_uv: 8  # UV index
    
  risk_weights:
    normal: 0.1
    rain: 0.3
    heavy_rain: 0.7
    heatwave: 0.8
    
  polling_interval: 300  # seconds
```

3. **Traffic Agent** (`agent_traffic.py`)
#### Monitors vehicle density using YOLOv8 and assesses congestion risk
#### Key Features
 ```bash
class TrafficAgent:
    """Smart-city traffic monitoring and congestion detection agent"""
    
    def __init__(self, camera_urls=None, model_path="models/yolov8n.pt"):
        self.camera_urls = camera_urls or ["rtsp://traffic-cam-1", "rtsp://traffic-cam-2"]
        self.model = YOLO(model_path)
        self.congestion_threshold = 15  # vehicles per 100m
        
    def detect_vehicles(self, frame):
        """Detect vehicles in CCTV frame using YOLOv8"""
        results = self.model(frame, conf=0.5, iou=0.45)
        detections = []
        
        for result in results:
            for box in result.boxes:
                detection = {
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": box.conf.item(),
                    "class": int(box.cls.item()),
                    "class_name": self.model.names[int(box.cls.item())]
                }
                detections.append(detection)
        
        return detections
    
    def estimate_congestion(self, detections, road_length=100):
        """Estimate traffic congestion level"""
        vehicle_count = len([d for d in detections if d["class_name"] in ["car", "truck", "bus"]])
        
        # Normalize to vehicles per 100 meters
        density = (vehicle_count / road_length) * 100
        
        # Calculate congestion score (0-1)
        if density <= 8:
            congestion_score = 0.2 + (density / 8) * 0.2  # 0.2-0.4
        elif density <= 15:
            congestion_score = 0.4 + ((density - 8) / 7) * 0.3  # 0.4-0.7
        else:
            congestion_score = 0.7 + min(0.3, (density - 15) / 20)  # 0.7-1.0
            
        return {
            "vehicle_count": vehicle_count,
            "density_per_100m": round(density, 2),
            "congestion_score": round(congestion_score, 3),
            "congestion_level": self.get_congestion_level(congestion_score)
        }
    
    def get_congestion_level(self, score):
        """Convert congestion score to qualitative level"""
        if score < 0.4:
            return "Low"
        elif score < 0.7:
            return "Moderate"
        else:
            return "High"
    
    def generate_traffic_recommendations(self, congestion_data, weather_risk=None):
        """Generate traffic management recommendations"""
        recommendations = []
        
        if congestion_data["congestion_level"] == "High":
            recommendations.append({
                "action": "reroute_traffic",
                "priority": "high",
                "details": "Divert vehicles to alternative routes",
                "estimated_impact": "Reduce congestion by 30-40%"
            })
            
        if weather_risk and weather_risk > 0.6:
            recommendations.append({
                "action": "reduce_speed_limits",
                "priority": "medium",
                "details": "Lower speed limits due to adverse weather",
                "estimated_impact": "Improve safety in wet conditions"
            })
            
        return recommendations
```
5. **Safety Agent** (`agent_safety.py`)
#### Detects fire and smoke hazards using YOLO11 Flare Guard
#### Key Capabilities
```bash
class SafetyAgent:
    """Smart-city safety monitoring and hazard detection agent"""
    
    def __init__(self, camera_urls=None, model_path="models/flare_guard.pt"):
        self.camera_urls = camera_urls or ["rtsp://safety-cam-1", "rtsp://safety-cam-2"]
        self.model = YOLO(model_path)
        self.hazard_threshold = 0.5
        
    def detect_hazards(self, frame):
        """Detect fire and smoke hazards in CCTV frame"""
        results = self.model(frame, conf=0.4, iou=0.3)
        hazards = {"fire": [], "smoke": [], "combined_risk": 0}
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = self.model.names[class_id]
                confidence = box.conf.item()
                
                if confidence >= self.hazard_threshold:
                    hazard_data = {
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": confidence,
                        "type": class_name
                    }
                    
                    if class_name == "fire":
                        hazards["fire"].append(hazard_data)
                    elif class_name == "smoke":
                        hazards["smoke"].append(hazard_data)
        
        # Calculate combined risk score
        fire_risk = sum([h["confidence"] for h in hazards["fire"]]) / max(len(hazards["fire"]), 1)
        smoke_risk = sum([h["confidence"] for h in hazards["smoke"]]) / max(len(hazards["smoke"]), 1)
        
        # Weighted combination (fire is more critical)
        hazards["combined_risk"] = round(0.7 * fire_risk + 0.3 * smoke_risk, 3)
        
        return hazards
    
    def determine_emergency_level(self, hazard_data):
        """Determine emergency response level based on hazard data"""
        combined_risk = hazard_data["combined_risk"]
        
        if combined_risk >= 0.8:
            return {
                "level": "CRITICAL",
                "response": "Immediate evacuation + full emergency response",
                "notifications": ["fire_department", "police", "hospitals", "public_alert"]
            }
        elif combined_risk >= 0.6:
            return {
                "level": "HIGH",
                "response": "Emergency services dispatch + area evacuation",
                "notifications": ["fire_department", "police", "local_authorities"]
            }
        elif combined_risk >= 0.4:
            return {
                "level": "MEDIUM",
                "response": "Investigation team dispatch + precautionary measures",
                "notifications": ["fire_department", "building_management"]
            }
        else:
            return {
                "level": "LOW",
                "response": "Monitor situation + prepare for escalation",
                "notifications": ["security_team"]
            }
```
   
7. **SORA Governance** (`sora_governance.py`)
#### Core governance with MAE-based LLM selection and policy enforcement
#### Governance Component
```bash
class SORAGovernance:
    """Central governance layer for multi-agent smart-city system"""
    
    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.trust_engine = AdaptiveTrustEngine()
        self.llm_selector = MAESelector()
        self.blockchain_client = self.init_blockchain_client()
        self.reference_values = self.load_reference_baselines()
        
    def validate_agent_output(self, agent_id, llm_outputs, context_data):
        """Validate agent outputs against governance policies"""
        validation_results = {
            "agent": agent_id,
            "timestamp": datetime.now().isoformat(),
            "validated_outputs": {},
            "policy_violations": [],
            "recommended_actions": []
        }
        
        # Apply policy gates (S1 from Table 3)
        for llm_name, outputs in llm_outputs.items():
            if "error" in outputs:
                continue
                
            # Check risk-trust gate
            if not self.check_risk_trust_gate(outputs["R"], outputs["T"], agent_id):
                validation_results["policy_violations"].append({
                    "llm": llm_name,
                    "violation": "risk_trust_gate_failed",
                    "details": f"R={outputs['R']}, T={outputs['T']} outside acceptable range"
                })
                continue
                
            # Check domain-specific thresholds
            if not self.check_domain_thresholds(agent_id, outputs["R"], outputs["T"]):
                validation_results["policy_violations"].append({
                    "llm": llm_name,
                    "violation": "domain_threshold_exceeded",
                    "details": "Output exceeds domain-specific safety thresholds"
                })
                continue
                
            validation_results["validated_outputs"][llm_name] = outputs
            
        return validation_results
    
    def select_best_llm(self, validated_outputs, reference_values):
        """Select LLM with minimum MAE to reference values"""
        if not validated_outputs:
            return None
            
        mae_scores = {}
        
        for llm_name, outputs in validated_outputs.items():
            # Calculate MAE (Definition in Section 5.3.2)
            risk_mae = abs(outputs["R"] - reference_values["R_ref"])
            trust_mae = abs(outputs["T"] - reference_values["T_ref"])
            total_mae = (risk_mae + trust_mae) / 2
            
            mae_scores[llm_name] = {
                "total_mae": round(total_mae, 4),
                "risk_mae": round(risk_mae, 4),
                "trust_mae": round(trust_mae, 4)
            }
        
        # Select LLM with minimum total MAE
        selected_llm = min(mae_scores.items(), key=lambda x: x[1]["total_mae"])[0]
        
        # Prepare feedback for non-selected LLMs
        feedback = {}
        for llm_name, scores in mae_scores.items():
            if llm_name != selected_llm:
                feedback[llm_name] = {
                    "ΔR": reference_values["R_ref"] - validated_outputs[llm_name]["R"],
                    "ΔT": reference_values["T_ref"] - validated_outputs[llm_name]["T"],
                    "adjustment_factor": 0.5,  # 50% adjustment as per policy S3
                    "suggestion": f"Adjust risk by {reference_values['R_ref'] - validated_outputs[llm_name]['R']:.3f}, "
                                f"trust by {reference_values['T_ref'] - validated_outputs[llm_name]['T']:.3f}"
                }
        
        return {
            "selected_llm": selected_llm,
            "mae_scores": mae_scores,
            "feedback": feedback,
            "selection_reason": f"Minimum MAE ({mae_scores[selected_llm]['total_mae']})"
        }
    
    def enforce_cross_domain_policies(self, agent_decisions):
        """Enforce cross-domain coordination rules"""
        cross_domain_checks = {
            "weather_traffic": self.check_weather_traffic_coordination,
            "traffic_safety": self.check_traffic_safety_coordination,
            "weather_safety": self.check_weather_safety_coordination
        }
        
        enforcement_results = {
            "timestamp": datetime.now().isoformat(),
            "cross_domain_valid": True,
            "violations": [],
            "coordinated_actions": []
        }
        
        # Apply each cross-domain check
        for check_name, check_function in cross_domain_checks.items():
            result = check_function(agent_decisions)
            if not result["valid"]:
                enforcement_results["cross_domain_valid"] = False
                enforcement_results["violations"].append({
                    "check": check_name,
                    "details": result["details"]
                })
            else:
                enforcement_results["coordinated_actions"].append({
                    "check": check_name,
                    "actions": result.get("actions", [])
                })
        
        return enforcement_results
    
    def check_weather_traffic_coordination(self, decisions):
        """Ensure traffic actions consider weather conditions"""
        weather_decision = decisions.get("weather")
        traffic_decision = decisions.get("traffic")
        
        if not weather_decision or not traffic_decision:
            return {"valid": True, "details": "Missing decision data"}
        
        weather_risk = weather_decision.get("risk_score", 0)
        traffic_action = traffic_decision.get("recommended_action", "")
        
        # Policy: Don't reroute traffic during high weather risk
        if "reroute" in traffic_action.lower() and weather_risk > 0.6:
            return {
                "valid": False,
                "details": f"Cannot reroute traffic during high weather risk (R={weather_risk})"
            }
        
        return {"valid": True, "details": "Weather-traffic coordination valid"}
    
    def log_governance_decision(self, decision_data):
        """Anchor governance decisions to SORA Blockchain"""
        log_entry = {
            "decision_id": f"gov_{datetime.now().timestamp()}",
            "timestamp": datetime.now().isoformat(),
            "agent_decisions": decision_data.get("agent_decisions", {}),
            "selected_llms": decision_data.get("selected_llms", {}),
            "cross_domain_validation": decision_data.get("cross_domain_validation", {}),
            "enforcement_actions": decision_data.get("enforcement_actions", []),
            "ecosystem_metrics": decision_data.get("ecosystem_metrics", {}),
            "metadata": {
                "governance_version": "SORA-ATMAS-v1.0",
                "policy_version": "2025.1"
            }
        }
        
        try:
            txid = self.blockchain_client.publish(
                "GovDecisions",
                log_entry["decision_id"],
                {'json': log_entry}
            )
            log_entry["blockchain_txid"] = txid
            return log_entry
        except Exception as e:
            print(f"Error logging to blockchain: {e}")
            log_entry["blockchain_error"] = str(e)
            return log_entry
```

# 📐 Trust & Risk Model Mathematics  
**Formal Definitions from the SORA-ATMAS Framework**

---

## Definition 1: Environmental Risk

Environmental risk quantifies deviations from expected operating conditions across continuous signals, capacity constraints, and discrete hazard events.

\[
R^{i}_{\text{Env}}(t)=
\begin{cases}
\displaystyle \frac{1}{n}\sum_{k=1}^{n}\mathbb{I}\big(|x_k(t)-\mu_k|>\theta_k\big), & \text{continuous signals} \\
\mathbb{I}\big(\text{Load}(t)>\theta_{\text{cap}}\big), & \text{capacity / volume conditions} \\
\mathbb{I}\big(\text{HazardEvents}(t)\ge 1\big), & \text{discrete hazard events}
\end{cases}
\]

### Parameters
- \( n \): Number of monitored environmental parameters  
- \( x_k(t) \): Normalized observation of parameter \(k\) at time \(t\)  
- \( \mu_k \): Expected baseline value for parameter \(k\)  
- \( \theta_k \): Acceptable deviation threshold for parameter \(k\)  
- \( \text{Load}(t) \in [0,1] \): Normalized utilization level  
- \( \theta_{\text{cap}} \): Maximum safe capacity threshold  
- \( \text{HazardEvents}(t) \): Count of verified hazard events  
---

## Definition 2: History-Reputation Trust (HRT)

\[
T^{i}_{\text{HRT}}(t)=
\begin{cases}
T_0, & t=t_0 \\
\delta \, T^{i}_{\text{HRT}}(t-\Delta T)
+(1-\delta)\big(\omega_p\, s(t)+\omega_r\, T^{i}_{\text{Rep}}(t)\big), & \text{otherwise}
\end{cases}
\]
---

## Definition 3: Service Risk

\[
R^{i}_{\text{Service}}(t)=1-T^{i}_{\text{HRT}}(t-\Delta T), \qquad
R^{i}_{\text{Service}}(t_0)=0.5
\]
---

## Definition 4: Overall Agent Risk

\[
R_i(t)=\lambda_i\, R^{i}_{\text{Env}}(t)
+\big(1-\lambda_i\big)R^{i}_{\text{Service}}(t)
\]
---

## Definition 5: Contextual Trust

\[
T^{i}_{\text{Ctx}}(t)
=\min\!\left(
T_{\text{base}}
\prod_{k=1}^{n_i}
\big(M_{i,k}(t)\big)^{w_{i,k}},
\;1.0
\right)
\]
---

## Definition 6: Overall Trust

\[
T^{i}_{\text{Overall}}(t)
=
w_{\text{HRT}}(t)\, T^{i}_{\text{HRT}}(t)
+
w_{C}(t)\, T^{i}_{\text{Ctx}}(t)
\]
---
Dynamic weights:

\[
w_{C}(t) = 0.5 + 0.2\, R_i(t)
\]
---



## Definition 7: Ecosystem Metrics

\[
T_{\text{Ecosystem}}(t)
=\frac{1}{|A(t)|}\sum_{i\in A(t)}T^{i}_{\text{Overall}}(t)
\]
---
\[
R_{\text{Ecosystem}}(t)
=\max_{i\in A(t)} R_i(t)
\]
---

## 🚀 Running the System

```bash
# Terminal 1: Weather Agent
python agent_weather.py --city Islamabad --interval 300 --enable-blockchain

# Terminal 2: Traffic Agent
python agent_traffic.py --interval 60 --enable-blockchain

# Terminal 3: Safety Agent
python agent_safety.py --interval 30 --enable-blockchain

# Terminal 4: SORA Governance
python sora_governance.py --agents weather traffic safety --dashboard --dashboard-port 8050

# Terminal 5: Monitoring Dashboard
streamlit run monitor_dashboard.py --server.port 8501
```

## 📊 Performance Benchmarks

| Metric             | Value              | Notes                              |
|--------------------|--------------------|------------------------------------|
| Throughput         | 13.8–17.2 req/s    | Varies with workload               |
| Governance Delay   | 21–92 ms/req       | SORA validation overhead           |
| MAE Reduction      | ≈35%               | Compared to single-LLM baseline    |

## 🔌 API Documentation

# Agent API Methods
**WeatherAgent.fetch_weather_data(latitude, longitude)**
```bash
def fetch_weather_data(self, latitude, longitude):
    """
    Fetch real-time weather data from OpenMeteo API.
    
    Args:
        latitude (float): Geographic latitude
        longitude (float): Geographic longitude
        
    Returns:
        dict: Weather data including temperature, precipitation, humidity, etc.
        
    Example:
        >>> agent = WeatherAgent()
        >>> data = agent.fetch_weather_data(33.6844, 73.0479)
        >>> print(data['temperature'])
        28.5
    """
```
**TrafficAgent.process_frame(frame, confidence_threshold=0.5)** 
```bash
def process_frame(self, frame, confidence_threshold=0.5):
    """
    Process a single frame for vehicle detection and congestion analysis.
    
    Args:
        frame (numpy.ndarray): Input image frame (BGR format)
        confidence_threshold (float): Minimum confidence for detection (0-1)
        
    Returns:
        dict: Detection results including bounding boxes, counts, and congestion score
        
    Raises:
        ValueError: If frame is None or empty
        RuntimeError: If model inference fails
        
    Example:
        >>> agent = TrafficAgent()
        >>> frame = cv2.imread('traffic.jpg')
        >>> results = agent.process_frame(frame)
        >>> print(results['vehicle_count'])
        23
    """
```
**SafetyAgent.detect_hazards(frame, emergency_threshold=0.7)**
```bash
def detect_hazards(self, frame, emergency_threshold=0.7):
    """
    Detect fire and smoke hazards in a surveillance frame.
    
    Args:
        frame (numpy.ndarray): Input image frame
        emergency_threshold (float): Confidence threshold for emergency alerts
        
    Returns:
        tuple: (hazard_data, emergency_level, recommended_actions)
        
    Notes:
        - emergency_threshold=0.7 corresponds to "HIGH" emergency level
        - Returns empty results if no hazards detected
    """
```
# SORA Governance API Methods
**SORAGovernance.validate_and_select(agent_data, llm_outputs)**
```bash
def validate_and_select(self, agent_data, llm_outputs):
    """
    Validate agent outputs and select best LLM using MAE-based selection.
    
    Args:
        agent_data (dict): Raw agent data including observations and context
        llm_outputs (dict): Dictionary of LLM outputs {llm_name: {R, T, explanation}}
        
    Returns:
        dict: Selection results with validation, MAE scores, and feedback
        
    Process:
        1. Apply risk-trust gate (policy S1)
        2. Compute MAE for each LLM (policy S2)
        3. Select LLM with minimum MAE
        4. Generate error-directed feedback for others (policy S3)
        5. Apply cross-domain constraints if applicable
        
    Example:
        >>> governance = SORAGovernance()
        >>> results = governance.validate_and_select(weather_data, llm_outputs)
        >>> print(results['selected_llm'])
        'GPT-4'
    """
```
**SORAGovernance.enforce_ecosystem_policies(agent_decisions)**
```bash
def enforce_ecosystem_policies(self, agent_decisions):
    """
    Enforce system-wide policies across all agents.
    
    Args:
        agent_decisions (dict): Dictionary of decisions from all active agents
        
    Returns:
        dict: Enforcement results including violations and coordinated actions
        
    Policies Enforced:
        - S4: Joint actuation when ≥2 agents have R > 0.80
        - S5: City-wide escalation when ecosystem risk > 0.70
        - S6: Hysteresis and cooldown to prevent oscillations
        
    Example:
        >>> decisions = {
        ...     'weather': {'R': 0.85, 'T': 0.65, 'action': 'issue_flood_advisory'},
        ...     'traffic': {'R': 0.78, 'T': 0.58, 'action': 'reroute_traffic'}
        ... }
        >>> enforcement = governance.enforce_ecosystem_policies(decisions)
        >>> print(enforcement['coordinated_actions'])
        ['Joint flood-traffic coordination activated']
    """
```
# Blockchain API Methods
**BlockchainClient.publish_to_stream(stream_name, key, data)**
```bash
def publish_to_stream(self, stream_name, key, data):
    """
    Publish data to a blockchain stream with automatic JSON serialization.
    
    Args:
        stream_name (str): Name of the blockchain stream
        key (str): Unique key for data retrieval
        data (dict): JSON-serializable data to publish
        
    Returns:
        str: Transaction ID if successful
        
    Raises:
        BlockchainError: If publishing fails
        StreamNotFound: If specified stream doesn't exist
        
    Example:
        >>> client = BlockchainClient('agentic', 9741)
        >>> txid = client.publish_to_stream(
        ...     'WeatherAgentLogs',
        ...     'weather_123456',
        ...     {'temperature': 28.5, 'risk': 0.3}
        ... )
        >>> print(f"Published with TX: {txid}")
    """
```
**BlockchainClient.query_stream(stream_name, count=10, start=-10)**
```bash
def query_stream(self, stream_name, count=10, start=-10):
    """
    Query items from a blockchain stream.
    
    Args:
        stream_name (str): Name of the blockchain stream
        count (int): Number of items to retrieve
        start (int): Starting position (negative for from end)
        
    Returns:
        list: Stream items with data and metadata
        
    Example:
        >>> client = BlockchainClient('sora', 9743)
        >>> items = client.query_stream('GovDecisions', count=5)
        >>> for item in items:
        ...     print(item['data']['decision_id'])
    """
```
# REST API Endpoints (Optional)
## If running with web server:
# Start REST API server

```bash
python api_server.py --port 8080 --host 0.0.0.0
```
## 🐛 Troubleshooting

Common issues, solutions, and debugging tips provided.

# Common Issues & Solutions
1. MultiChain Connection Issues
```bash
# Check if daemons are running
ps aux | grep multichaind | grep -v grep

# Check blockchain status
multichain-cli agentic getinfo
multichain-cli sora getinfo

# Check RPC configuration
cat ~/.multichain/agentic/multichain.conf
cat ~/.multichain/sora/multichain.conf

# Common errors and fixes:
# Error: "Could not connect to RPC server"
# Solution: Restart daemon with RPC flags
pkill multichaind
multichaind agentic -daemon -rpcallowip=127.0.0.1
multichaind sora -daemon -rpcallowip=127.0.0.1

# Error: "Invalid RPC credentials"
# Solution: Regenerate passwords
openssl rand -base64 32 > keys/agentic_rpc.conf
openssl rand -base64 32 > keys/sora_rpc.conf
# Update multichain.conf files accordingly
```
2. LLM API Rate Limiting and Errors
```bash
# Implement robust LLM calling with retries
import time
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_llm_with_retry(prompt, model="gpt-4"):
    """Call LLM with exponential backoff retry logic"""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        return response.choices[0].message.content
    except openai.error.RateLimitError:
        print("Rate limit hit, waiting before retry...")
        time.sleep(30)  # Wait 30 seconds for rate limit reset
        raise  # Re-raise to trigger retry
    except openai.error.APIError as e:
        print(f"API error: {e}")
        time.sleep(10)
        raise
```
3. YOLO Model Loading Issues
```bash
# Check CUDA availability for GPU acceleration
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Download models if missing
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/sayedgamal99/Real-Time-Smoke-Fire-Detection-YOLO11/releases/download/v1.0/flare_guard.pt

# Test model loading
python -c "
from ultralytics import YOLO
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
model = YOLO('models/yolov8n.pt')
print('YOLOv8 loaded successfully')
print(f'Model device: {model.device}')
"
```
4. Google Sheets API Authentication Issues
```bash
# Set up Google Cloud credentials
# 1. Enable Google Sheets API at https://console.cloud.google.com/apis/library/sheets.googleapis.com
# 2. Create a service account and download JSON credentials
# 3. Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"

# Test authentication
python -c "
from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
creds = service_account.Credentials.from_service_account_file(
    '$GOOGLE_APPLICATION_CREDENTIALS', scopes=SCOPES)
service = build('sheets', 'v4', credentials=creds)
print('Google Sheets API authenticated successfully')
"
```
5. Python Package Version Conflicts
```bash
# Create fresh virtual environment
deactivate
rm -rf sora_env
python3 -m venv sora_env
source sora_env/bin/activate

# Install with version constraints
pip install --upgrade pip
pip install "torch>=2.0.0" "torchvision>=0.15.0"
pip install "ultralytics>=8.0.0" "opencv-python>=4.7.0"
pip install "openai>=1.0.0" "xai-api>=0.1.0" "deepseek-sdk>=0.1.0"
pip install "xgboost>=1.7.0" "streamlit>=1.28.0"
pip install "multichain-python>=1.0.0"

# Generate requirements.txt for reproducibility
pip freeze > requirements.txt
```
6. Memory and Resource Issues
```bash
# Monitor system resources
watch -n 5 'echo "=== System Resources ===" && \
free -h && echo "=== GPU Memory ===" && \
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>/dev/null || echo "No GPU" && \
echo "=== Process Memory ===" && \
ps aux --sort=-%mem | head -10'

# Configure agent resource limits in config/resource_limits.yaml:
# agent_weather:
#   max_memory_mb: 1024
#   max_cpu_percent: 50
# agent_traffic:
#   max_memory_mb: 2048
#   max_cpu_percent: 70
#   use_gpu: true
```
# Debug Mode Activation
```bash
# Enable comprehensive debugging
import logging
import sys

def setup_debug_logging():
    """Configure detailed logging for debugging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/debug.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers to DEBUG
    loggers = ['agent_weather', 'agent_traffic', 'agent_safety', 
               'sora_governance', 'multichain']
    for logger_name in loggers:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)
    
    print("Debug logging enabled. Check logs/debug.log for details.")

# Test individual components in debug mode
if __name__ == "__main__":
    setup_debug_logging()
    
    # Test weather agent
    from agent_weather import WeatherAgent
    weather_agent = WeatherAgent(debug=True)
    test_data = weather_agent.fetch_weather_data(33.6844, 73.0479)
    logging.debug(f"Weather test data: {test_data}")
    
    # Test blockchain connection
    from multichain import MultiChainClient
    try:
        mc = MultiChainClient("127.0.0.1", 9741, "multichainrpc", 
                             open('keys/agentic_rpc.conf').read().strip())
        info = mc.getinfo()
        logging.debug(f"Blockchain info: {info}")
    except Exception as e:
        logging.error(f"Blockchain error: {e}")
```

## 📈 System Monitoring
# Health Check Dashboard
```bash
# health_dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import multichain
import requests
import json

class SORAMonitor:
    """Real-time monitoring dashboard for SORA-ATMAS"""
    
    def __init__(self):
        self.setup_page()
        self.load_config()
        
    def setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="SORA-ATMAS Monitor",
            page_icon="🏙️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🏙️ SORA-ATMAS Smart City Governance Monitor")
        st.markdown("Real-time monitoring of adaptive trust management system")
        
    def load_config(self):
        """Load system configuration"""
        try:
            with open('config/monitor_config.json', 'r') as f:
                self.config = json.load(f)
        except:
            self.config = {
                "blockchain_ports": {"agentic": 9741, "sora": 9743},
                "agent_endpoints": {
                    "weather": "http://localhost:8001/status",
                    "traffic": "http://localhost:8002/status", 
                    "safety": "http://localhost:8003/status"
                },
                "refresh_interval": 5
            }
    
    def display_system_overview(self):
        """Display system overview metrics"""
        st.header("📊 System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Agents Active", "3/3", "100%")
        with col2:
            st.metric("Governance Delay", "52 ms", "-3 ms")
        with col3:
            st.metric("Throughput", "15.2 req/s", "+0.3")
        with col4:
            st.metric("Ecosystem Trust", "0.78", "+0.02")
    
    def display_blockchain_status(self):
        """Display blockchain status"""
        st.header("⛓️ Blockchain Status")
        
        try:
            # Connect to blockchains
            agentic_mc = multichain.MultiChainClient(
                "127.0.0.1", self.config["blockchain_ports"]["agentic"], 
                "multichainrpc", open('keys/agentic_rpc.conf').read().strip()
            )
            sora_mc = multichain.MultiChainClient(
                "127.0.0.1", self.config["blockchain_ports"]["sora"],
                "multichainrpc", open('keys/sora_rpc.conf').read().strip()
            )
            
            agentic_info = agentic_mc.getinfo()
            sora_info = sora_mc.getinfo()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Agentic Blockchain")
                st.metric("Blocks", agentic_info.get("blocks", 0))
                st.metric("Connections", agentic_info.get("connections", 0))
                st.metric("Version", agentic_info.get("version", "Unknown"))
                
            with col2:
                st.subheader("SORA Blockchain")
                st.metric("Blocks", sora_info.get("blocks", 0))
                st.metric("Connections", sora_info.get("connections", 0))
                st.metric("Version", sora_info.get("version", "Unknown"))
                
        except Exception as e:
            st.error(f"Blockchain connection error: {e}")
    
    def display_agent_status(self):
        """Display agent status"""
        st.header("🤖 Agent Status")
        
        # Sample agent data - replace with actual API calls
        agents = [
            {"name": "Weather Agent", "status": "RUNNING", "last_update": "2 seconds ago",
             "metrics": {"temperature": "28.5°C", "risk": "0.23", "regime": "Normal"}},
            {"name": "Traffic Agent", "status": "RUNNING", "last_update": "5 seconds ago",
             "metrics": {"congestion": "Medium", "vehicles": "18", "risk": "0.45"}},
            {"name": "Safety Agent", "status": "RUNNING", "last_update": "1 second ago",
             "metrics": {"hazards": "0", "emergency": "None", "risk": "0.12"}}
        ]
        
        for agent in agents:
            with st.expander(f"{agent['name']} - {agent['status']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Last Update:**", agent["last_update"])
                with col2:
                    for metric, value in agent["metrics"].items():
                        st.write(f"**{metric.title()}:** {value}")
    
    def display_trust_risk_metrics(self):
        """Display trust and risk metrics over time"""
        st.header("📈 Trust & Risk Analytics")
        
        # Sample time series data
        times = pd.date_range(start='2025-01-02 10:00', periods=60, freq='1min')
        data = {
            'Time': times,
            'Ecosystem Trust': [0.7 + 0.1 * (i/60) for i in range(60)],
            'Ecosystem Risk': [0.3 + 0.4 * abs((i-30)/60) for i in range(60)],
            'Weather Trust': [0.8 - 0.2 * abs((i-20)/60) for i in range(60)],
            'Traffic Risk': [0.4 + 0.3 * (i/60) for i in range(60)]
        }
        df = pd.DataFrame(data)
        
        # Create Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Time'], y=df['Ecosystem Trust'],
                                mode='lines', name='Ecosystem Trust',
                                line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=df['Time'], y=df['Ecosystem Risk'],
                                mode='lines', name='Ecosystem Risk',
                                line=dict(color='red', width=2)))
        fig.add_trace(go.Scatter(x=df['Time'], y=df['Weather Trust'],
                                mode='lines', name='Weather Trust',
                                line=dict(color='blue', width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=df['Time'], y=df['Traffic Risk'],
                                mode='lines', name='Traffic Risk',
                                line=dict(color='orange', width=2, dash='dash')))
        
        fig.update_layout(
            title="Trust & Risk Metrics Over Time",
            xaxis_title="Time",
            yaxis_title="Score (0-1)",
            hovermode="x unified",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_governance_decisions(self):
        """Display recent governance decisions"""
        st.header("⚖️ Recent Governance Decisions")
        
        # Sample decisions
        decisions = [
            {"time": "10:15:23", "agent": "Weather", "decision": "APPROVE",
             "details": "Issue heatwave advisory", "llm_selected": "GPT-4"},
            {"time": "10:14:51", "agent": "Traffic", "decision": "RESTRICT",
             "details": "Reroute traffic (weather risk > 0.6)", "llm_selected": "Grok"},
            {"time": "10:13:12", "agent": "Safety", "decision": "APPROVE",
             "details": "No hazards detected", "llm_selected": "DeepSeek"},
            {"time": "10:12:45", "agent": "Cross-Domain", "decision": "COORDINATE",
             "details": "Weather-Traffic joint action", "llm_selected": "N/A"}
        ]
        
        df = pd.DataFrame(decisions)
        st.dataframe(df, use_container_width=True)
    
    def run(self):
        """Run the monitoring dashboard"""
        self.display_system_overview()
        
        col1, col2 = st.columns(2)
        with col1:
            self.display_blockchain_status()
        with col2:
            self.display_agent_status()
        
        self.display_trust_risk_metrics()
        self.display_governance_decisions()
        
        # Auto-refresh
        st.sidebar.header("Settings")
        refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 1, 30, 5)
        st.sidebar.info(f"Next refresh in {refresh_rate} seconds")
        
        # Force refresh
        if st.sidebar.button("Refresh Now"):
            st.rerun()

if __name__ == "__main__":
    monitor = SORAMonitor()
    monitor.run()
```
# Command-line Monitoring Tools
```bash
# Real-time log monitoring
tail -f logs/agent_weather.log logs/agent_traffic.log logs/agent_safety.log

# Blockchain transaction monitor
watch -n 5 'echo "=== Blockchain Transactions ===" && \
multichain-cli agentic liststreamitems WeatherAgentLogs 1 | jq ".[0].data.json" && \
multichain-cli sora liststreamitems GovDecisions 1 | jq ".[0].data.json"'

# System resource monitor
cat > system_monitor.sh << 'EOF'
#!/bin/bash
echo "=== SORA-ATMAS System Monitor ==="
echo "Timestamp: $(date)"
echo ""
echo "=== CPU Usage ==="
top -bn1 | grep "Cpu(s)" | awk '{print "User: " $2 "% System: " $4 "% Idle: " $8 "%"}'
echo ""
echo "=== Memory Usage ==="
free -h | awk 'NR==2{print "Total: " $2 " Used: " $3 " Free: " $4}'
echo ""
echo "=== Disk Usage ==="
df -h / | awk 'NR==2{print "Used: " $5 " of " $2}'
echo ""
echo "=== Process Count ==="
ps aux | grep -E "agent|sora|multichain" | grep -v grep | wc -l | xargs echo "Active processes:"
EOF
chmod +x system_monitor.sh
watch -n 10 ./system_monitor.sh
```
# Alerting System
```bash
# alert_system.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from datetime import datetime

class AlertSystem:
    """Alert and notification system for SORA-ATMAS"""
    
    def __init__(self):
        self.setup_logging()
        self.load_alert_config()
        
    def setup_logging(self):
        """Setup alert logging"""
        self.logger = logging.getLogger('alert_system')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('logs/alerts.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
    
    def load_alert_config(self):
        """Load alert configuration"""
        self.config = {
            "email": {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender": "sora.alerts@yourdomain.com",
                "recipients": ["admin@yourdomain.com", "ops@yourdomain.com"]
            },
            "thresholds": {
                "critical_risk": 0.8,
                "high_risk": 0.6,
                "low_trust": 0.5,
                "blockchain_error": 3  # Consecutive errors
            },
            "cooldown_periods": {
                "critical": 300,  # 5 minutes
                "high": 600,      # 10 minutes
                "medium": 1800,   # 30 minutes
            }
        }
    
    def send_alert(self, level, title, message, metadata=None):
        """Send alert through configured channels"""
        timestamp = datetime.now().isoformat()
        alert_id = f"alert_{timestamp}"
        
        alert_data = {
            "id": alert_id,
            "timestamp": timestamp,
            "level": level,
            "title": title,
            "message": message,
            "metadata": metadata or {}
        }
        
        # Log alert
        self.logger.info(f"ALERT {level}: {title} - {message}")
        
        # Send email if enabled
        if self.config["email"]["enabled"] and level in ["CRITICAL", "HIGH"]:
            self.send_email_alert(alert_data)
        
        # Log to blockchain for audit trail
        self.log_to_blockchain(alert_data)
        
        return alert_id
    
    def send_email_alert(self, alert_data):
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config["email"]["sender"]
            msg['To'] = ", ".join(self.config["email"]["recipients"])
            msg['Subject'] = f"[SORA-ATMAS {alert_data['level']}] {alert_data['title']}"
            
            body = f"""
            SORA-ATMAS Alert Notification
            =============================
            
            Alert Level: {alert_data['level']}
            Time: {alert_data['timestamp']}
            Alert ID: {alert_data['id']}
            
            Title: {alert_data['title']}
            
            Message:
            {alert_data['message']}
            
            Metadata:
            {alert_data['metadata']}
            
            ---
            This is an automated alert from the SORA-ATMAS Governance System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config["email"]["smtp_server"], 
                                 self.config["email"]["smtp_port"])
            server.starttls()
            # Note: In production, use environment variables or secure config
            # server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email alert sent for {alert_data['id']}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def log_to_blockchain(self, alert_data):
        """Log alert to blockchain for immutable record"""
        try:
            # Implementation depends on your blockchain setup
            pass
        except Exception as e:
            self.logger.error(f"Failed to log alert to blockchain: {e}")
    
    def check_system_health(self):
        """Perform system health checks and send alerts if needed"""
        checks = [
            self.check_blockchain_health,
            self.check_agent_health,
            self.check_governance_health,
            self.check_resource_health
        ]
        
        for check_function in checks:
            try:
                result = check_function()
                if not result["healthy"]:
                    self.send_alert(
                        level=result.get("level", "MEDIUM"),
                        title=result["title"],
                        message=result["message"],
                        metadata=result.get("metadata", {})
                    )
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
    
    def check_blockchain_health(self):
        """Check blockchain health"""
        # Implementation
        return {"healthy": True}
    
    def check_agent_health(self):
        """Check agent health"""
        # Implementation
        return {"healthy": True}
    
    def check_governance_health(self):
        """Check governance health"""
        # Implementation
        return {"healthy": True}
    
    def check_resource_health(self):
        """Check system resources"""
        # Implementation
        return {"healthy": True}

# Usage example
if __name__ == "__main__":
    alert_system = AlertSystem()
    
    # Example alert
    alert_system.send_alert(
        level="HIGH",
        title="Traffic Congestion Critical",
        message="Traffic congestion exceeded threshold with R=0.85",
        metadata={
            "agent": "traffic",
            "risk_score": 0.85,
            "location": "Main Street Intersection",
            "recommended_action": "Reroute traffic"
        }
    )
```

## 📚 Citation

```bibtex
@article{sora_atmas_2025,
  title={SORA-ATMAS: Adaptive Trust Management and Multi-LLM Aligned Governance for Future Smart Cities},
  author={Antuley, Usama and Siddiqui, Shahbaz and Team},
  journal={arXiv preprint},
  year={2025},
  url={https://github.com/Usama-Antuley/SORA-ATMAS},
  doi={10.48550/arXiv.2501.XXXXX}
}
```

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.

##  Acknowledgments

Usama Antuley, Shahbaz Siddiqui, Sufian Hameed, Waqas Arif, Syed Attique Shah,  Smart City Research Group, open-source contributors, and technology partners.
