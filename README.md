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

```bash
git clone https://github.com/Usama-Antuley/SORA-ATMAS-Adaptive-Trust-Management-and-Multi-LLM-Aligned-Governance-for-Future-Smart-Cities.git
cd SORA-ATMAS-Adaptive-Trust-Management-and-Multi-LLM-Aligned-Governance-for-Future-Smart-Cities/Multichain

python3 -m venv sora_env
source sora_env/bin/activate  # Linux/Mac
# sora_env\Scripts\activate  # Windows
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

## ⛓️ Agentic & SORA Blockchain Setup

### Step 1: Install MultiChain

```bash
cd /tmp
wget https://www.multichain.com/download/multichain-2.3.3.tar.gz
tar -xvzf multichain-2.3.3.tar.gz
cd multichain-2.3.3
sudo mv multichaind multichain-cli multichain-util /usr/local/bin/
```

### Step 2–5: Create chains, configure ports, start daemons, set up RPC, test connections

*(See detailed scripts in the original content.)*

## 📊 Blockchain Streams Configuration

Use `create_sora_streams.py` to initialize all required streams on both blockchains.

## 🔧 Core Modules Detailed

1. **Weather Agent** (`agent_weather.py`)
2. **Traffic Agent** (`agent_traffic.py`)
3. **Safety Agent** (`agent_safety.py`)
4. **SORA Governance** (`sora_governance.py`)

*(See code snippets and descriptions in the original content)*

# 📐 Trust & Risk Model Mathematics  
**Formal Definitions from the SORA-ATMAS Framework**

## Definition 1: Environmental Risk

Environmental risk quantifies deviations from expected operating conditions across continuous signals, capacity constraints, and discrete hazard events.

```math
R^{i}_{\text{Env}}(t)=
\begin{cases}
\displaystyle \frac{1}{n}\sum_{k=1}^{n}\mathbb{I}\big(|x_k(t)-\mu_k|>\theta_k\big), & \text{continuous signals} \\
\mathbb{I}\big(\text{Load}(t)>\theta_{\text{cap}}\big), & \text{capacity / volume conditions} \\
\mathbb{I}\big(\text{HazardEvents}(t)\ge 1\big), & \text{discrete hazard events}
\end{cases}
```

## Definition 2: History-Reputation Trust (HRT)

```math
T^{i}_{\text{HRT}}(t)=
\begin{cases}
T_0, & t=t_0 \\
\delta \, T^{i}_{\text{HRT}}(t-\Delta T)
+(1-\delta)\big(\omega_p\, s(t)+\omega_r\, T^{i}_{\text{Rep}}(t)\big), & \text{otherwise}
\end{cases}
```
## Definition 3: Service Risk
```math
R^{i}_{\text{Service}}(t)=1-T^{i}_{\text{HRT}}(t-\Delta T), \qquad
R^{i}_{\text{Service}}(t_0)=0.5
```

## Definition 4: Overall Agent Risk
```math
R_i(t)=\lambda_i\, R^{i}_{\text{Env}}(t)
+\big(1-\lambda_i\big)R^{i}_{\text{Service}}(t)
```

## Definition 5: Contextual Trust
```math
T^{i}_{\text{Ctx}}(t)
=\min\!\left(
T_{\text{base}}
\prod_{k=1}^{n_i}
\big(M_{i,k}(t)\big)^{w_{i,k}},
\;1.0
\right)
```

## Definition 6: Overall Trust
```math
T^{i}_{\text{Overall}}(t)
=
w_{\text{HRT}}(t)\, T^{i}_{\text{HRT}}(t)
+
w_{C}(t)\, T^{i}_{\text{Ctx}}(t)
```

Dynamic weights:
```math
w_{C}(t) = 0.5 + 0.2\, R_i(t)
```
## Definition 7: Ecosystem Metrics
```math
T_{\text{Ecosystem}}(t)
=\frac{1}{|A(t)|}\sum_{i\in A(t)}T^{i}_{\text{Overall}}(t)

R_{\text{Ecosystem}}(t)
=\max_{i\in A(t)} R_i(t)
```

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

### Agent API Methods
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
### SORA Governance API Methods
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
### Blockchain API Methods
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
### REST API Endpoints (Optional)
#### If running with web server:
#### Start REST API server

```bash
python api_server.py --port 8080 --host 0.0.0.0
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
