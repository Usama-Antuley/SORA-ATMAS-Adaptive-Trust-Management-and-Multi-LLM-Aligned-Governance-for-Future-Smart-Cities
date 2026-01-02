"""
SORA-ATMAS Multi-LLM Agent with Agentic Chain Blockchain Integration
Run with: python llm_brain.py
"""
import os
import json
import pandas as pd
import time
import multichain
from datetime import datetime
from openai import OpenAI
from xai_sdk import Client as XAIClient

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BASE_DIR = "RawData"
RESULT_DIR = "Results"
BLOCKCHAIN_DIR = "BlockchainLogs"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(BLOCKCHAIN_DIR, exist_ok=True)

MODELS = ["gpt-4o-nano", "deepseek-r1", "grok"]

PROMPT_FILES = {
    "Weather": "WeatherPrompt.txt",
    "Traffic": "Traffic.txt",
    "Safety": "Safety.txt"
}

# SORA-ATMAS Paper Context
PAPER_CONTEXT = """
SORA-ATMAS: Adaptive Trust Management and Multi-LLM Aligned Governance for Future Smart Cities
Abstract
The rapid evolution of smart cities has increased reliance on intelligent interconnected services to optimize infrastructure, resources, and citizen well-being. Agentic AI enables autonomous decision-making and adaptive coordination so urban systems can respond in real time to dynamic conditions. In transportation, integrating traffic data, weather forecasts, and safety sensors enables dynamic rerouting and faster hazard response. However, deploying such intelligence introduces significant governance, risk, and compliance (GRC) challenges. Evaluation of SORA-ATMAS with three domain agents (Weather, Traffic, Safety) demonstrated that its governance policies effectively steer multiple LLMs (GPT, Grok, DeepSeek) towards domain-optimized, policy-aligned outputs, producing an average MAE reduction of 35% across agents. Results showed stable weather monitoring, effective handling of high-risk traffic plateaus (R ≈0.85), and adaptive trust regulation in Safety scenarios (τt = 0.65). Runtime profiling confirmed scalability with governance delays under 100 ms. These findings validate SORA-ATMAS as a regulation-aligned, context-aware, and verifiable governance framework.

Introduction
Agentic AI plays a vital role in urban development by enabling autonomous decision-making and proactive coordination. In smart cities, it addresses challenges in energy, transport, and safety through real-time data exchange and automation. However, without strong governance, opaque decisions can lead to severe consequences — e.g., misinterpreting sensor data during adverse weather and causing citywide disruption. SORA-ATMAS addresses this by enforcing accountability, transparency, and GRC compliance via adaptive trust, multi-LLM alignment, and auditable escalation.
"""

# --------------------------------------------------
# AGENTIC CHAIN CONFIGURATION
# --------------------------------------------------
AGENTIC_CHAIN_CONFIG = {
    "host": "127.0.0.1",
    "port": 9724,  # Using weather chain as Agentic Chain
    "rpcuser": "multichainrpc",
    "rpcpassword": "2xYe2PpKbiCpuXfHVhJPDUiVuus3k1dinKfMriSCD6dx",
    "stream_name": "Agentic_Decisions",
    "chain_name": "agentic_chain"
}

# Initialize Agentic Chain client
def init_agentic_chain():
    """Initialize MultiChain client for Agentic Chain"""
    try:
        agentic_client = multichain.MultiChainClient(
            AGENTIC_CHAIN_CONFIG["host"],
            AGENTIC_CHAIN_CONFIG["port"],
            AGENTIC_CHAIN_CONFIG["rpcuser"],
            AGENTIC_CHAIN_CONFIG["rpcpassword"]
        )
        
        # Test connection
        info = agentic_client.getinfo()
        print(f"✓ Connected to Agentic Chain: {info.get('chainname', 'Unknown')}")
        print(f"  Blocks: {info.get('blocks', 'N/A')}, Version: {info.get('version', 'N/A')}")
        
        # Create stream if it doesn't exist
        streams = agentic_client.liststreams()
        stream_names = [s.get('name', '') for s in streams]
        
        if AGENTIC_CHAIN_CONFIG["stream_name"] not in stream_names:
            print(f"Creating stream: {AGENTIC_CHAIN_CONFIG['stream_name']}")
            agentic_client.create(AGENTIC_CHAIN_CONFIG["stream_name"])
        
        return agentic_client
        
    except Exception as e:
        print(f"✗ Agentic Chain connection failed: {e}")
        print("  Will continue without blockchain logging")
        return None

# Initialize blockchain client
agentic_client = init_agentic_chain()

# API Clients (set these in your GitHub repo secrets or locally)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
deepseek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
xai_client = XAIClient(api_key=os.getenv("XAI_API_KEY"))

# --------------------------------------------------
# BLOCKCHAIN HELPER FUNCTIONS
# --------------------------------------------------
def log_to_agentic_chain(agent_id: str, model: str, row_data: dict, llm_output: dict):
    """
    Log LLM decision to Agentic Chain (for agent-level provenance)
    According to paper: "Agent proposals are initially logged on their local Agentic Blockchain for provenance"
    """
    if not agentic_client:
        return None
    
    try:
        # Create blockchain record
        chain_record = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "model": model,
            "row_index": row_data.get("index", "N/A"),
            "row_timestamp": row_data.get("timestamp", "N/A"),
            "llm_input_summary": {
                k: v for k, v in row_data.items() 
                if k not in ['index'] and not isinstance(v, dict)
            },
            "llm_output": {
                k: v for k, v in llm_output.items()
                if k not in ['model', 'error']
            },
            "log_type": "agent_decision",
            "chain": AGENTIC_CHAIN_CONFIG["chain_name"]
        }
        
        # Publish to Agentic Chain
        tx_id = agentic_client.publish(
            AGENTIC_CHAIN_CONFIG["stream_name"],
            f"{agent_id}_{model}_{int(time.time())}",
            {"json": chain_record}
        )
        
        print(f"  ✓ Logged to Agentic Chain (TX: {tx_id})")
        
        # Also save locally for backup
        local_file = os.path.join(BLOCKCHAIN_DIR, f"agentic_log_{int(time.time())}.json")
        with open(local_file, 'w') as f:
            json.dump(chain_record, f, indent=2)
            
        return tx_id
        
    except Exception as e:
        print(f"  ✗ Blockchain logging failed: {e}")
        return None

def save_batch_to_agentic_chain(agent_id: str, model: str, results: list):
    """
    Save batch of results to Agentic Chain for efficiency
    """
    if not agentic_client or not results:
        return None
    
    try:
        batch_record = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "model": model,
            "batch_size": len(results),
            "results_summary": {
                "first_timestamp": results[0].get("timestamp", "N/A"),
                "last_timestamp": results[-1].get("timestamp", "N/A"),
                "risk_range": f"{min(r.get('R_Overall', 0) for r in results):.3f}-{max(r.get('R_Overall', 0) for r in results):.3f}",
                "trust_range": f"{min(r.get('T_Overall', 0) for r in results):.3f}-{max(r.get('T_Overall', 0) for r in results):.3f}"
            },
            "log_type": "agent_batch",
            "chain": AGENTIC_CHAIN_CONFIG["chain_name"]
        }
        
        tx_id = agentic_client.publish(
            AGENTIC_CHAIN_CONFIG["stream_name"],
            f"batch_{agent_id}_{model}_{int(time.time())}",
            {"json": batch_record}
        )
        
        print(f"  ✓ Batch logged to Agentic Chain (TX: {tx_id})")
        return tx_id
        
    except Exception as e:
        print(f"  ✗ Batch blockchain logging failed: {e}")
        return None

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def get_csv_path(agent_folder):
    folder = os.path.join(BASE_DIR, agent_folder)
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        return None
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not csv_files:
        print(f"No CSV file in {folder}")
        return None
    return os.path.join(folder, csv_files[0])  # use the first CSV found

def load_prompt_with_context(file_name):
    if not os.path.exists(file_name):
        print(f"Prompt file missing: {file_name}")
        return "You are a helpful agent."
    with open(file_name, "r", encoding="utf-8") as f:
        agent_prompt = f.read().strip()
    return (
        "You are operating within the SORA-ATMAS governance framework for smart-city disaster management.\n"
        "This system requires strict policy compliance, explainability, and auditable decisions.\n\n"
        "RESEARCH CONTEXT:\n" + PAPER_CONTEXT + "\n\n"
        "YOUR AGENT PROMPT:\n" + agent_prompt
    )

def serialize_row(agent_id, row):
    row_dict = row.to_dict()
    lines = [f"{k}: {v}" for k, v in row_dict.items() if pd.notna(v)]
    return "Current input row:\n" + "\n".join(lines)

def call_llm(model_name, base_prompt, row_str, agent_id, row_index, row_data):
    output_schema = {
        "Weather": "timestamp, temp_c, humidity_pct, wind_kmh, precip_mmph, cloud_pct, uv_index, predicted_label, R_Env, T_Rept, s_t, T_HRT, R_Service, R_Overall, T_Ctx, w_HRT, w_C, T_Overall, Final Action, Final Comment",
        "Traffic": "timestamp, vehicle_count_per_100m, latitude, longitude, message, R_Env, T_Rept, s_t, T_HRT, R_Service, R_Overall, T_Ctx, w_HRT, w_C, T_Overall, Final Action, Final Comment",
        "Safety": "timestamp, class, confidence, camera_id, gps_lat, gps_lon, SmokeEvents, R_Env, T_Rept, s_t, T_HRT, R_Service, R_Overall, T_Ctx, w_HRT, w_C, T_Overall, Final Action, Final Comment"
    }
    schema = output_schema[agent_id]
    adapted_prompt = (
        base_prompt + "\n\n=== CURRENT ROW ===\n" + row_str +
        "\n\nINSTRUCTIONS: Compute ALL metrics. Round to 3 decimal places. Clip [0,1]. "
        "OUTPUT ONLY a single valid JSON with these exact keys (in order):\n" + schema +
        "\nNo extra text."
    )
    
    start_time = time.time()
    
    try:
        if model_name == "gpt-4o-nano":
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Using gpt-4o-mini instead of gpt-4o-nano
                messages=[{"role": "system", "content": adapted_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            
        elif model_name == "deepseek-r1":
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": adapted_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            
        elif model_name == "grok":
            chat = xai_client.chat.create(model="grok-beta", temperature=0.0)
            chat.append({"role": "system", "content": adapted_prompt})
            response = chat.sample()
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            result = json.loads(content)
        
        # Add metadata
        result["model"] = model_name
        result["processing_time_ms"] = int((time.time() - start_time) * 1000)
        result["row_index"] = row_index
        
        # Log to Agentic Chain
        log_to_agentic_chain(agent_id, model_name, row_data, result)
        
        return result
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "model": model_name,
            "row_index": row_index,
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }
        
        # Log error to blockchain
        log_to_agentic_chain(agent_id, model_name, row_data, error_result)
        
        print(f"Error [{agent_id}-{model_name}]: {e}")
        return error_result

# --------------------------------------------------
# PROCESS ONE AGENT
# --------------------------------------------------
def process_agent(agent_id, folder_name):
    csv_path = get_csv_path(folder_name)
    if not csv_path:
        return
    
    print(f"\n[{agent_id}] Found: {csv_path}")
    if agentic_client:
        print(f"  Agentic Chain: Ready for logging")
    else:
        print(f"  Agentic Chain: Not connected")
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()
    
    # Add index column for tracking
    df["row_index"] = df.index
    
    base_prompt = load_prompt_with_context(PROMPT_FILES[agent_id])
    results = {m: [] for m in MODELS}
    
    print(f"[{agent_id}] Processing {len(df)} rows...")
    
    for idx, row in df.iterrows():
        row_data = row.to_dict()
        row_str = serialize_row(agent_id, row)
        
        print(f"  Row {idx+1}/{len(df)}: ", end="")
        
        for model in MODELS:
            result = call_llm(model, base_prompt, row_str, agent_id, idx, row_data)
            results[model].append(result)
        
        print(f"Completed")
    
    # Save Excel results
    for model, data in results.items():
        out_df = pd.DataFrame(data)
        filename = f"{agent_id}_Results_{model.replace('-', '')}.xlsx"
        save_path = os.path.join(RESULT_DIR, filename)
        out_df.to_excel(save_path, index=False)
        print(f"  Saved: {save_path} ({len(data)} rows)")
        
        # Log batch summary to blockchain
        save_batch_to_agentic_chain(agent_id, model, data)
    
    # Generate summary report
    generate_summary_report(agent_id, results)

def generate_summary_report(agent_id: str, results: dict):
    """Generate summary report of agent processing"""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "agent_id": agent_id,
        "total_rows": 0,
        "models": {},
        "blockchain_status": "connected" if agentic_client else "disconnected"
    }
    
    if results:
        for model, data in results.items():
            successful = [r for r in data if "error" not in r]
            errors = [r for r in data if "error" in r]
            
            summary["models"][model] = {
                "total": len(data),
                "successful": len(successful),
                "errors": len(errors),
                "avg_processing_time_ms": int(
                    sum(r.get("processing_time_ms", 0) for r in successful) / len(successful)
                ) if successful else 0
            }
        
        summary["total_rows"] = len(next(iter(results.values()), []))
    
    # Save summary
    summary_file = os.path.join(RESULT_DIR, f"{agent_id}_Summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Summary saved: {summary_file}")

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SORA-ATMAS Multi-LLM Agent with Agentic Chain Integration")
    print("=" * 60)
    
    # Check if required prompt files exist
    missing_prompts = []
    for agent, prompt_file in PROMPT_FILES.items():
        if not os.path.exists(prompt_file):
            missing_prompts.append(prompt_file)
    
    if missing_prompts:
        print(f"Warning: Missing prompt files: {missing_prompts}")
    
    # Process agents
    agents_to_process = [
        ("Weather", "weather"),
        ("Traffic", "traffic"),
        ("Safety", "safety")
    ]
    
    for agent_id, folder_name in agents_to_process:
        process_agent(agent_id, folder_name)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Results saved in: {RESULT_DIR}/")
    print(f"Blockchain logs in: {BLOCKCHAIN_DIR}/")
    
    if agentic_client:
        try:
            # Query blockchain to verify entries
            items = agentic_client.liststreamitems(AGENTIC_CHAIN_CONFIG["stream_name"])
            print(f"Total blockchain entries: {len(items)}")
            print("Recent entries:")
            for item in items[-3:]:  # Show last 3 entries
                print(f"  - {item.get('key', 'N/A')} at {item.get('blocktime', 'N/A')}")
        except Exception as e:
            print(f"Could not query blockchain: {e}")
