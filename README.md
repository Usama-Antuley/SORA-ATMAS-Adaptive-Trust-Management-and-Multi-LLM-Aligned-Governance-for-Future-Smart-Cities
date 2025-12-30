
# SORA-ATMAS: Adaptive Trust Management and Multi-LLM Aligned Governance for Future Smart Cities

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Blockchain](https://img.shields.io/badge/blockchain-MultiChain-orange)
![Governance](https://img.shields.io/badge/governance-Multi--LLM-green)

SORA-ATMAS is a comprehensive framework for smart-city disaster response. It integrates real-time environmental sensing, dynamic multi-dimensional trust/risk computation, and a dual-chain blockchain architecture to ensure resilient, policy-aligned governance across multiple LLM agents (GPT, Grok, DeepSeek).

---

## 📋 Table of Contents
- [System Overview](#-system-overview)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation Guide](#-installation-guide)
- [Blockchain Setup](#-blockchain-setup)
- [Stream Configuration](#-stream-configuration)
- [Core Modules](#-core-modules)
- [Trust & Risk Model](#-trust--risk-model)
- [Running the System](#-running-the-system)
- [Performance Benchmarks](#-performance-benchmarks)
- [Troubleshooting](#-troubleshooting)
- [Citation & Support](#-support)

---

## 🎯 System Overview
This implementation provides a complete adaptive governance framework for smart cities using:
* **Real-time Monitoring:** Integration with OpenWeatherMap and various sensor APIs.
* **Multi-dimensional Trust:** Computation based on Historical, Reputation, and Contextual data.
* **Multi-LLM Reasoning:** Federated alignment between GPT, Grok, and DeepSeek.
* **Dual MultiChain Blockchain:** Immutable local logs (Agentic Chain) and global governance records (SORA Chain).
* **GRC Policy Enforcement:** Automated escalation and fallback mechanisms for safe decision-making.

**Key Innovation:** The SORA governance layer validates agent outputs, selects the optimal LLM via Mean Absolute Error (MAE) analysis, and ensures cross-domain safety (e.g., verifying weather conditions before authorizing traffic rerouting).

---

## 🏗️ Architecture



```text
+------------------------------+   +------------------------------+
|       AI Agentic Layer       |   |     SORA Governance Layer    |
|                              |   |                              |
|  Weather Agent  ──▶ LLMs ◀────┼──▶│ Best LLM Selection (MAE)    |
|  Traffic Agent  ──▶ LLMs ◀────┼──▶│ Comparison & Feedback       |
|  Safety Agent   ──▶ LLMs ◀────┼──▶│ GRC Policies                |
|                              |   | │ Adaptive Trust & Risk       |
|  Agentic Chain (Local Logs)  |   | │ SORA Chain & Global Alerts  |
+------------------------------+   +------------------------------+
               Disaster Management System
📋 PrerequisitesSystem RequirementsOS: Ubuntu 20.04/22.04 LTS (Recommended) or Windows 10/11 (WSL2)RAM: 4GB minimum (8GB recommended)Storage: 10GB free spaceSoftware RequirementsBash# Install dependencies on Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git wget curl build-essential libssl-dev
🔧 Installation Guide1. Clone & Environment SetupBashgit clone [https://github.com/Shahbazdefender/smartcity-trust-framework-2025.git](https://github.com/Shahbazdefender/smartcity-trust-framework-2025.git)
cd smartcity-trust-framework-2025

# Create virtual environment
python3 -m venv trust_env
source trust_env/bin/activate  # Windows: .	rust_env\Scriptsctivate

# Install Python dependencies
pip install --upgrade pip
pip install requests numpy pandas matplotlib cryptography
2. API ConfigurationObtain an API key from OpenWeatherMap and set it:Bashecho 'export OPENWEATHER_API_KEY="your_actual_api_key_here"' >> ~/.bashrc
source ~/.bashrc
⛓️ MultiChain Blockchain Setup1. Install MultiChainBashcd /tmp
wget [https://www.multichain.com/download/multichain-2.3.3.tar.gz](https://www.multichain.com/download/multichain-2.3.3.tar.gz)
tar -xvzf multichain-2.3.3.tar.gz
cd multichain-2.3.3
sudo mv multichaind multichain-cli /usr/local/bin/
2. Initialize ChainsWe use four separate chains: weather, transport, sf, and bank.Bashmkdir -p ~/blockchains/{weather,transport,sf,bank}
for chain in weather transport sf bank; do
    multichain-util create $chain
done

# Configure RPC ports to avoid conflicts
sed -i 's/default-rpc-port = 9744/default-rpc-port = 9724/' ~/.multichain/weather/params.dat
# (Repeat similar port shifts for transport, sf, and bank)
3. Start Daemons & Configure RPCBashmultichaind weather -daemon
# Repeat for other chains...

# Set credentials
for chain in weather transport sf bank; do
    echo "rpcuser=multichainrpc" > ~/.multichain/$chain/multichain.conf
    echo "rpcpassword=$(openssl rand -base64 32)" >> ~/.multichain/$chain/multichain.conf
    echo "rpcallowip=127.0.0.1" >> ~/.multichain/$chain/multichain.conf
done
📊 Stream ConfigurationInitialize the required metadata streams:Bash# This uses the internal stream list: 
# ["Service Registration", "TrustStream", "LocalRule", "LocalPolicies", etc.]
python create_streams.py
📐 Trust & Risk Model MathematicsEnvironmental Risk$$R\_{	ext{Env}}^i(t) =
egin{cases}
rac{1}{n}\sum \mathbb{I}(|x\_k - \mu\_k| \> 	heta\_k) & 	ext{continuous} \
\mathbb{I}(	ext{Load}(t) \> 	heta\_{	ext{cap}}) & 	ext{capacity} \
\mathbb{I}(	ext{HazardEvents}(t) \ge 1) & 	ext{discrete}
\end{cases}$$
History–Reputation Trust (HRT)$$T\_{	ext{HRT}}^i(t) =
egin{cases}
0.5 & t=t\_0 \
\delta T\_{	ext{HRT}}^i(t-\Delta T) + (1-\delta)(\omega\_p s(t) + \omega\_r T\_{	ext{Rept}}^i(t)) & 	ext{otherwise}
\end{cases}$$
Overall Risk & Contextual TrustService Risk: $R_{	ext{Service}}^i(t) = 1 - T_{	ext{HRT}}^i(t-\Delta T)$Overall Risk: $R^i(t) = \lambda_i R_{	ext{Env}}^i(t) + (1-\lambda_i) R_{	ext{Service}}^i(t)$Contextual Trust: $T_{	ext{Ctx}}^i(t) = \min\!\left(T_{	ext{base}} \prod (M_{i,k}(t))^{w_{i,k}}, 1.0ight)$Risk-Adaptive Overall Trust$$T_{	ext{Overall}}^i(t) = (0.5 - 0.2 R^i(t)) T_{	ext{HRT}}^i(t) + (0.5 + 0.2 R^i(t)) T_{	ext{Ctx}}^i(t)$$
🚀 Running the SystemService Registration: python updated_Registration.pyStart Governance: python main.pyRun Benchmarks: python main.py --benchmarkMonitor Status: ./monitor_system.sh📊 Performance BenchmarksRequestsThroughput (req/s)Exec Time (ms)Gov Delay (ms)10017.2582150016.36132100015.26552200013.87292🐛 TroubleshootingMultiChain Errors: Check if daemons are running: ps aux | grep multichaind.API Timeouts: Ensure OPENWEATHER_API_KEY is valid and you have active credits.Permissions: If database/logs fail, run chmod -R 755 ~/.multichain/.Verbose Logging: Change logging level to DEBUG in main.py for granular output.🆘 SupportIssues: Please use the [GitHub Issues] tab for bug reports.Health Check: Run python health_check.py to verify all blockchain connections.Logs: Monitor real-time logs via tail -f logs/*.json.
