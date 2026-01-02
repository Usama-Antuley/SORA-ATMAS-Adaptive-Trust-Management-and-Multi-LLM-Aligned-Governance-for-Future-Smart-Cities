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

## 📐 Trust & Risk Model Mathematics

Formal definitions and Python implementations of environmental risk, history-reputation trust, contextual trust, overall trust, and ecosystem metrics.

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

Detailed examples for agent and governance API methods *(see original content)*.

## 🐛 Troubleshooting

Common issues, solutions, and debugging tips provided.

## 📈 System Monitoring

Streamlit-based dashboard, command-line monitors, and alerting system.

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
