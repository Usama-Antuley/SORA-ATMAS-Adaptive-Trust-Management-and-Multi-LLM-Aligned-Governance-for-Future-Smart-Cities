"""
SORA-ATMAS Multi-LLM Agent 
Run with: python llm_brain.py
"""
import os
import json
import pandas as pd
from openai import OpenAI
from xai_sdk import Client as XAIClient

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BASE_DIR = "agent"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

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
The rapid evolution of smart cities has increased reliance on intelligent interconnected services...
(Your full abstract + introduction here - same as before)
"""

# API Clients (set these in your GitHub repo secrets or locally)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
deepseek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
xai_client = XAIClient(api_key=os.getenv("XAI_API_KEY"))

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

def call_llm(model_name, base_prompt, row_str, agent_id):
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
    try:
        if model_name == "gpt-4o-nano":
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",  # cheaper & works the same
                messages=[{"role": "system", "content": adapted_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        elif model_name == "deepseek-r1":
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": adapted_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        elif model_name == "grok":
            chat = xai_client.chat.create(model="grok-beta", temperature=0.0)
            chat.append({"role": "system", "content": adapted_prompt})
            response = chat.sample()
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content)
    except Exception as e:
        print(f"Error [{agent_id}-{model_name}]: {e}")
        return {"error": str(e)}

# --------------------------------------------------
# PROCESS ONE AGENT
# --------------------------------------------------
def process_agent(agent_id, folder_name):
    csv_path = get_csv_path(folder_name)
    if not csv_path:
        return
    print(f"\n[{agent_id}] Found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()
    base_prompt = load_prompt_with_context(PROMPT_FILES[agent_id])
    results = {m: [] for m in MODELS}
    print(f"[{agent_id}] Processing {len(df)} rows...")

    for _, row in df.iterrows():
        row_str = serialize_row(agent_id, row)
        for model in MODELS:
            result = call_llm(model, base_prompt, row_str, agent_id)
            result["model"] = model
            results[model].append(result)

    for model, data in results.items():
        out_df = pd.DataFrame(data)
        filename = f"{agent_id}_Results_{model.replace('-', '')}.xlsx"
        save_path = os.path.join(RESULT_DIR, filename)
        out_df.to_excel(save_path, index=False)
        print(f"Saved: {save_path} ({len(data)} rows)")

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    print("Starting SORA-ATMAS LLM processing on GitHub\n")
    process_agent("Weather", "weather")
    process_agent("Traffic", "traffic")
    process_agent("Safety", "safety")
    print("\nDONE! Check the 'results' folder for your LLM outputs.")
    print("You can download the whole folder or individual Excel files directly on GitHub.")
