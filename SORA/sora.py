# -*- coding: utf-8 -*-
"""SORA.ipynb

SORA Governance Engine with Dual-Chain Blockchain Integration
- Agentic Chain (port 9724): Logs agent-level proposals for provenance
- SORA Chain (port 7202): Logs governance decisions for global auditability
"""

import os
import json
import time
import pandas as pd
import datetime
from typing import Dict, List, Tuple, Optional, Any
from multichain import MultiChainClient

class SORA:
    """
    SORA (Security & Operational Response Agent) Governance Engine.
    Implements validation, selection, feedback, policy enforcement, and consolidated alerting
    for a multi-agent smart-city disaster management system.
    """

    def __init__(self,
                 theta_r: float = 0.7,
                 theta_t: float = 0.5,
                 tau_t: Dict[str, float] = None,
                 epsilon_r: float = 0.1,
                 epsilon_t: float = 0.1,
                 mae_tie_tolerance: float = 0.01,
                 hysteresis: float = 0.05,
                 cooldown: int = 300) -> None:
        """
        Initialize SORA with policy configurations and blockchain integration.
        """
        self.theta_r = theta_r
        self.theta_t = theta_t
        self.tau_t = tau_t or {
            "Weather": 0.60,
            "Traffic": 0.55,
            "Safety": 0.65
        }
        self.epsilon_r = epsilon_r
        self.epsilon_t = epsilon_t
        self.mae_tie_tolerance = mae_tie_tolerance
        self.hysteresis = hysteresis
        self.cooldown = cooldown
        
        # Blockchain state tracking
        self.last_escalation_time: Optional[datetime.datetime] = None
        self.previous_escalation: bool = False
        
        # --- BLOCKCHAIN CONFIGURATION ---
        # According to paper: "Agentic Blockchain at the edge" and "SORA-Blockchain at governance layer"
        self.AGENTIC_CHAIN_CONFIG = {
            "host": "127.0.0.1",
            "port": 9724,  # Using weather chain as Agentic Chain
            "rpcuser": "multichainrpc",
            "rpcpassword": "2xYe2PpKbiCpuXfHVh..................",
            "stream_name": "Agentic_Decisions",
            "chain_type": "Agentic"
        }
        
        self.SORA_CHAIN_CONFIG = {
            "host": "127.0.0.1",
            "port": 7202,  # Using transport chain as SORA Chain
            "rpcuser": "multichainrpc",
            "rpcpassword": "GLKF9X2pi7pLqMcTYE...............",  # Use your actual password
            "stream_name": "SORA_Governance",
            "chain_type": "SORA"
        }
        
        # Initialize blockchain clients
        self.agentic_client = self._init_chain_client(self.AGENTIC_CHAIN_CONFIG)
        self.sora_client = self._init_chain_client(self.SORA_CHAIN_CONFIG)
        
        # Create blockchain logs directory
        os.makedirs("SORA_Blockchain_Logs", exist_ok=True)
        
    def _init_chain_client(self, config: Dict) -> Optional[MultiChainClient]:
        """
        Initialize MultiChain client for a blockchain.
        Returns client if successful, None if failed.
        """
        try:
            client = MultiChainClient(
                config["host"],
                config["port"],
                config["rpcuser"],
                config["rpcpassword"]
            )
            
            # Test connection
            info = client.getinfo()
            print(f"✓ Connected to {config['chain_type']} Chain: {info.get('chainname', 'Unknown')}")
            
            # Create stream if it doesn't exist
            streams = client.liststreams()
            stream_names = [s.get('name', '') for s in streams]
            
            if config["stream_name"] not in stream_names:
                print(f"  Creating stream: {config['stream_name']}")
                client.create(config["stream_name"])
            
            return client
            
        except Exception as e:
            print(f"✗ {config['chain_type']} Chain connection failed: {e}")
            return None
    
    # --------------------------------------------------------------------
    # BLOCKCHAIN ANCHORING METHODS (Dual-Chain Architecture)
    # --------------------------------------------------------------------
    
    def anchor_to_agentic_chain(self, agent_id: str, agent_packet: Dict[str, Any], 
                               llm_outputs: List[Dict[str, Any]]) -> str:
        """
        Anchor agent proposal to Agentic Chain for provenance.
        According to paper: "Agent proposals are initially logged on their local Agentic Blockchain for provenance"
        """
        if not self.agentic_client:
            print("  [Agentic Chain] Not available - skipping anchoring")
            return "no_agentic_chain"
        
        try:
            agentic_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "agent_id": agent_id,
                "agent_packet_summary": {
                    "R_overall": agent_packet.get("R_overall", 0),
                    "T_overall": agent_packet.get("T_overall", 0),
                    "metadata": agent_packet.get("metadata", {})
                },
                "llm_outputs_count": len(llm_outputs),
                "llm_models": [out.get("model", "unknown") for out in llm_outputs],
                "log_type": "agent_proposal",
                "chain": "Agentic",
                "sora_phase": "pre_validation"
            }
            
            # Publish to Agentic Chain
            tx_id = self.agentic_client.publish(
                self.AGENTIC_CHAIN_CONFIG["stream_name"],
                f"{agent_id}_proposal_{int(time.time())}",
                {"json": agentic_record}
            )
            
            print(f"  ✓ Anchored {agent_id} proposal to Agentic Chain (TX: {tx_id})")
            
            # Save local backup
            self._save_local_blockchain_log("Agentic", agentic_record)
            
            return tx_id
            
        except Exception as e:
            print(f"  ✗ Agentic Chain anchoring failed: {e}")
            return f"error_{str(e)}"
    
    def anchor_to_sora_chain(self, governance_decision: Dict[str, Any], 
                            agent_id: str = None, 
                            decision_type: str = "agent_decision") -> str:
        """
        Anchor governance decision to SORA Chain for global auditability.
        According to paper: "only SORA's final validation decisions are anchored on the SORA Blockchain for global auditability"
        """
        if not self.sora_client:
            print("  [SORA Chain] Not available - skipping anchoring")
            return "no_sora_chain"
        
        try:
            sora_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "agent_id": agent_id,
                "decision_type": decision_type,
                "decision": governance_decision,
                "policy_version": "1.0",
                "policy_thresholds": {
                    "theta_r": self.theta_r,
                    "theta_t": self.theta_t,
                    "tau_t": self.tau_t
                },
                "log_type": "governance_validation",
                "chain": "SORA",
                "sora_phase": "post_validation"
            }
            
            # Publish to SORA Chain
            tx_id = self.sora_client.publish(
                self.SORA_CHAIN_CONFIG["stream_name"],
                f"sora_gov_{decision_type}_{int(time.time())}",
                {"json": sora_record}
            )
            
            print(f"  ✓ Anchored {decision_type} to SORA Chain (TX: {tx_id})")
            
            # Save local backup
            self._save_local_blockchain_log("SORA", sora_record)
            
            return tx_id
            
        except Exception as e:
            print(f"  ✗ SORA Chain anchoring failed: {e}")
            return f"error_{str(e)}"
    
    def anchor_ecosystem_to_sora_chain(self, ecosystem_metrics: Dict[str, Any]) -> str:
        """
        Anchor ecosystem evaluation to SORA Chain.
        According to paper: "Ecosystem-level enforcement uses aggregated metrics"
        """
        if not self.sora_client:
            return "no_sora_chain"
        
        try:
            ecosystem_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "ecosystem_metrics": ecosystem_metrics,
                "policy_thresholds": {
                    "joint_actuation": ">=2 agents with R > 0.80",
                    "escalation": "R_ecosystem > 0.70 AND T_ecosystem >= 0.60",
                    "hysteresis": self.hysteresis,
                    "cooldown": self.cooldown
                },
                "log_type": "ecosystem_evaluation",
                "chain": "SORA",
                "sora_phase": "ecosystem_monitoring"
            }
            
            tx_id = self.sora_client.publish(
                self.SORA_CHAIN_CONFIG["stream_name"],
                f"ecosystem_{int(time.time())}",
                {"json": ecosystem_record}
            )
            
            print(f"  ✓ Anchored ecosystem evaluation to SORA Chain (TX: {tx_id})")
            self._save_local_blockchain_log("SORA", ecosystem_record)
            
            return tx_id
            
        except Exception as e:
            print(f"  ✗ Ecosystem anchoring failed: {e}")
            return f"error_{str(e)}"
    
    def _save_local_blockchain_log(self, chain_type: str, record: Dict[str, Any]) -> None:
        """Save blockchain record locally for backup"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"SORA_Blockchain_Logs/{chain_type}_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(record, f, indent=2)
        except Exception as e:
            print(f"  Note: Could not save local backup: {e}")
    
    def compute_sora_reference(self, agent_packet: Dict[str, Any]) -> Tuple[float, float]:
        """
        Recompute reference risk (R_ref) and trust (T_ref) using Definitions 1-7.
        """
        agent_id = agent_packet["agent_id"]
        metadata = agent_packet["metadata"]
        
        if agent_id == "Weather":
            predicted_label = metadata.get("predicted_label", "Unknown")
            r_map = {"Normal": 0.0, "Flood": 0.9, "Heatwave": 0.7}
            r_ref = r_map.get(predicted_label, 0.5)
            t_ref = self.tau_t["Weather"]
            
        elif agent_id == "Traffic":
            vehicle_count = metadata.get("vehicle_count", 0)
            r_ref = min(vehicle_count / 50.0, 1.0)
            t_ref = self.tau_t["Traffic"]
            
        elif agent_id == "Safety":
            fire_detected = metadata.get("fire_detected", False)
            smoke_detected = metadata.get("smoke_detected", False)
            if fire_detected and smoke_detected:
                r_ref = 1.0
            elif fire_detected:
                r_ref = 0.9
            elif smoke_detected:
                r_ref = 0.8
            else:
                r_ref = 0.0
            t_ref = self.tau_t["Safety"]
        else:
            raise ValueError(f"Unknown agent_id: {agent_id}")
            
        return r_ref, t_ref
    
    def admission_gate(self, agent_packet: Dict[str, Any]) -> bool:
        """
        S1: Validate agent packet for admission.
        """
        r_overall = agent_packet["R_overall"]
        t_overall = agent_packet["T_overall"]
        agent_id = agent_packet["agent_id"]
        
        r_ref, t_ref = self.compute_sora_reference(agent_packet)
        delta_r = abs(r_ref - r_overall)
        delta_t = abs(t_ref - t_overall)
        
        if t_overall < self.tau_t[agent_id]:
            print(f"  Rejection: T_overall {t_overall} < τ_T {self.tau_t[agent_id]} for {agent_id}")
            return False
            
        if delta_r > self.epsilon_r or delta_t > self.epsilon_t:
            print(f"  Rejection: ΔR {delta_r} > ε_R {self.epsilon_r} or ΔT {delta_t} > ε_T {self.epsilon_t} for {agent_id}")
            return False
            
        return True
    
    def select_llm(self, llm_outputs: List[Dict[str, Any]], r_ref: float, t_ref: float) -> Dict[str, Any]:
        """
        S2: Select best LLM output using MAE.
        """
        if not llm_outputs:
            raise ValueError("No LLM outputs provided.")
        
        def mae(r: float, t: float) -> float:
            return 0.5 * (abs(r - r_ref) + abs(t - t_ref))
        
        scored = [(mae(out["R"], out["T"]), abs(out["R"] - r_ref), out) for out in llm_outputs]
        min_mae = min(s[0] for s in scored)
        candidates = [s for s in scored if abs(s[0] - min_mae) <= self.mae_tie_tolerance]
        candidates.sort(key=lambda s: s[1])
        selected = candidates[0][2]
        
        # Safety fallback
        if selected["T"] < self.theta_t:
            qualifiers = [out for out in llm_outputs if out["T"] >= self.theta_t]
            if qualifiers:
                qualifiers.sort(key=lambda out: abs(out["T"] - t_ref))
                selected = qualifiers[0]
        
        return selected
    
    def generate_feedback(self, llm_output: Dict[str, Any], r_ref: float, t_ref: float) -> Dict[str, Any]:
        """
        S3: Generate corrective feedback for an LLM output.
        """
        delta_r = r_ref - llm_output["R"]
        delta_t = t_ref - llm_output["T"]
        corrected_r = max(0.0, min(1.0, llm_output["R"] + 0.5 * delta_r))
        corrected_t = max(0.0, min(1.0, llm_output["T"] + 0.5 * delta_t))
        
        return {
            "model": llm_output["model"],
            "delta_r": delta_r,
            "delta_t": delta_t,
            "corrected_r": corrected_r,
            "corrected_t": corrected_t,
            "explanation": f"Applied 50% correction based on reference (Definitions 1-7)."
        }
    
    def policy_decision(self, r: float, t: float) -> str:
        """
        Policy decision based on R and T.
        """
        if t < self.theta_t:
            return "deny"
        if r > self.theta_r:
            return "restrict"
        return "approve"
    
    
    def process_agent(self, agent_packet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single agent packet through S1-S3 with dual-chain anchoring.
        """
        print(f"\n[Processing {agent_packet['agent_id']}]")
        print("-" * 40)
        
        # Step 1: Anchor agent proposal to Agentic Chain (before validation)
        agent_tx = self.anchor_to_agentic_chain(
            agent_packet["agent_id"],
            agent_packet,
            agent_packet["llm_outputs"]
        )
        
        # Step 2: Admission gate (S1)
        if not self.admission_gate(agent_packet):
            # Log rejection to SORA Chain
            rejection_record = {
                "agent_id": agent_packet["agent_id"],
                "decision": "rejected",
                "reason": "Failed admission gate (S1)",
                "R_overall": agent_packet.get("R_overall", 0),
                "T_overall": agent_packet.get("T_overall", 0)
            }
            self.anchor_to_sora_chain(rejection_record, agent_packet["agent_id"], "rejection")
            return None
        
        # Step 3: Compute reference and select LLM (S2)
        r_ref, t_ref = self.compute_sora_reference(agent_packet)
        selected = self.select_llm(agent_packet["llm_outputs"], r_ref, t_ref)
        
        # Step 4: Policy decision
        decision = self.policy_decision(selected["R"], selected["T"])
        
        # Step 5: Generate feedback for all LLMs (S3)
        feedbacks = [self.generate_feedback(out, r_ref, t_ref) for out in agent_packet["llm_outputs"]]
        
        # Create governance output
        gov_output = {
            "agent_id": agent_packet["agent_id"],
            "timestamp": agent_packet.get("timestamp", datetime.datetime.now().isoformat()),
            "decision": decision,
            "justification": (
                f"Risk driver: {selected['R']:.3f} (ref {r_ref:.3f}). "
                f"Trust driver: {selected['T']:.3f} (ref {t_ref:.3f}). "
                f"Agent domain: {agent_packet['agent_id']}."
            ),
            "selected_model": selected["model"],
            "risk": selected["R"],
            "trust": selected["T"],
            "risk_reference": r_ref,
            "trust_reference": t_ref,
            "feedback_summary": {
                "models_corrected": len(feedbacks),
                "avg_delta_r": sum(f["delta_r"] for f in feedbacks) / len(feedbacks),
                "avg_delta_t": sum(f["delta_t"] for f in feedbacks) / len(feedbacks)
            },
            "agentic_tx": agent_tx
        }
        
        # Step 6: Anchor governance decision to SORA Chain
        sora_tx = self.anchor_to_sora_chain(gov_output, agent_packet["agent_id"], "agent_decision")
        gov_output["sora_tx"] = sora_tx
        
        print(f"  Decision: {decision}")
        print(f"  Selected: {selected['model']} (R={selected['R']:.3f}, T={selected['T']:.3f})")
        
        return gov_output
    
    def ecosystem_evaluation(self, agent_decisions: List[Dict[str, Any]], 
                           current_time: datetime.datetime) -> Dict[str, Any]:
        """
        S4-S6: Evaluate cross-agent ecosystem with blockchain logging.
        """
        if not agent_decisions:
            return {"R_ecosystem": 0.0, "T_ecosystem": 0.0, "escalation": False}
        
        r_values = [d["risk"] for d in agent_decisions]
        t_values = [d["trust"] for d in agent_decisions]
        r_eco = max(r_values)
        t_eco = sum(t_values) / len(t_values)
        high_r_count = sum(1 for r in r_values if r > 0.80)
        joint_actuation = high_r_count >= 2
        
        # Hysteresis: lower threshold if previous escalation
        effective_r_threshold = 0.70 - self.hysteresis if self.previous_escalation else 0.70
        effective_t_threshold = 0.60
        
        # Cooldown check
        if self.last_escalation_time and (current_time - self.last_escalation_time).total_seconds() < self.cooldown:
            escalation = False
        else:
            escalation = r_eco > effective_r_threshold and t_eco >= effective_t_threshold
        
        if escalation:
            self.last_escalation_time = current_time
            self.previous_escalation = True
        else:
            self.previous_escalation = False
        
        ecosystem_metrics = {
            "R_ecosystem": r_eco,
            "T_ecosystem": t_eco,
            "joint_actuation": joint_actuation,
            "escalation": escalation,
            "high_risk_agents": high_r_count,
            "agents_count": len(agent_decisions),
            "timestamp": current_time.isoformat()
        }
        
        # Anchor ecosystem evaluation to SORA Chain
        self.anchor_ecosystem_to_sora_chain(ecosystem_metrics)
        
        return ecosystem_metrics
    
    def generate_city_alert(self,
                           agent_decisions: List[Dict[str, Any]],
                           timestamp: str,
                           location: Dict[str, Any],
                           ecosystem: Dict[str, Any]) -> Dict[str, Any]:

        # Determine primary hazard (Safety precedence)
        weather_dec = next((d for d in agent_decisions if d["agent_id"] == "Weather"), None)
        traffic_dec = next((d for d in agent_decisions if d["agent_id"] == "Traffic"), None)
        safety_dec = next((d for d in agent_decisions if d["agent_id"] == "Safety"), None)
        
        primary_type = "None"
        confidence = "Low"
        primary_action = "Monitor"
        
        # Safety
        if safety_dec and safety_dec["risk"] > 0.7:
            primary_type = "Fire" if safety_dec["risk"] >= 0.9 else "Smoke"
            confidence = "High" if safety_dec["trust"] >= 0.7 else "Medium" if safety_dec["trust"] >= 0.5 else "Low"
            primary_action = "Evacuate" if safety_dec["risk"] > 0.9 else "Dispatch"
        
        # Secondary
        secondary = []
        if weather_dec and weather_dec["risk"] > 0.5:
            secondary.append({
                "type": "Weather",
                "severity": f"{weather_dec['risk']:.2f}",
                "action": "Advisory" if weather_dec["risk"] > 0.7 else "None"
            })
        if traffic_dec and traffic_dec["risk"] > 0.5:
            secondary.append({
                "type": "Traffic",
                "severity": f"{traffic_dec['risk']:.2f}",
                "action": "Reroute" if traffic_dec["risk"] > 0.7 else "None"
            })
        
        gov_decision = "Emergency Escalation" if ecosystem["escalation"] else "Advisory" if max(d["risk"] for d in agent_decisions) > 0.5 else "No Action"
        
        policy_just = (
            f"Escalation based on R_ecosystem {ecosystem['R_ecosystem']:.2f} > 0.70 "
            f"and T_ecosystem {ecosystem['T_ecosystem']:.2f} >= 0.60 "
            f"(with hysteresis {self.hysteresis})."
        )
        
        final_directive = (
            f"Immediate {primary_action} to {location['zone']} due to {primary_type}. "
            f"Reroute traffic if applicable. Monitor secondary conditions."
        )
        
        alert = {
            "timestamp": timestamp,
            "location": location,
            "primary_hazard": {
                "type": primary_type,
                "confidence": confidence,
                "action": primary_action
            },
            "secondary_conditions": secondary,
            "ecosystem_metrics": {
                "R_ecosystem": ecosystem["R_ecosystem"],
                "T_ecosystem": ecosystem["T_ecosystem"],
                "joint_actuation": ecosystem["joint_actuation"],
                "escalation": ecosystem["escalation"]
            },
            "governance_decision": gov_decision,
            "policy_justification": policy_just,
            "final_directive": final_directive,
            "agent_decisions_summary": [
                {"agent": d["agent_id"], "risk": d["risk"], "trust": d["trust"], "decision": d["decision"]}
                for d in agent_decisions
            ]
        }
        
        # Anchor city alert to SORA Chain
        self.anchor_to_sora_chain(alert, decision_type="city_alert")
        
        return alert


def load_agent_results(agent_name: str, results_dir: str = "Results") -> Dict[str, Any]:
    """
    Load agent results from Excel files generated by LLM processing.
    """
    models = [
        ("gpt-4o-nano", f"{agent_name}_Results_gpt4onano.xlsx"),
        ("deepseek-r1", f"{agent_name}_Results_deepseekr1.xlsx"),
        ("grok", f"{agent_name}_Results_grok.xlsx"),
    ]
    
    llm_outputs = []
    r_total = 0.0
    t_total = 0.0
    count = 0
    metadata = {}
    
    for model_name, filename in models:
        filepath = os.path.join(results_dir, filename)
        if not os.path.exists(filepath):
            print(f"  File not found: {filepath}")
            continue
        
        try:
            df = pd.read_excel(filepath)
            if len(df) == 0:
                print(f"  Empty file: {filename}")
                continue
            
            row = df.iloc[0]  # first row
            
            # Find R_Overall and T_Overall columns (flexible name matching)
            r_col = next((c for c in df.columns if "r_overall" in c.lower()), None)
            t_col = next((c for c in df.columns if "t_overall" in c.lower()), None)
            
            if r_col is None or t_col is None:
                print(f"  Warning: Missing R/T columns in {filename}")
                continue
            
            r = float(row[r_col])
            t = float(row[t_col])
            
            llm_outputs.append({
                "model": model_name,
                "R": r,
                "T": t,
                "explanation": row.get("Final Comment", "No explanation")
            })
            
            r_total += r
            t_total += t
            count += 1
            
            # Extract metadata based on agent type
            if agent_name == "Weather" and "predicted_label" in df.columns:
                metadata["predicted_label"] = str(row["predicted_label"])
            elif agent_name == "Traffic" and "vehicle_count_per_100m" in df.columns:
                metadata["vehicle_count"] = float(row["vehicle_count_per_100m"])
            elif agent_name == "Safety":
                if "class" in df.columns:
                    class_val = str(row["class"]).lower()
                    metadata["fire_detected"] = "fire" in class_val
                    metadata["smoke_detected"] = "smoke" in class_val
            
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
    
    if count == 0:
        print(f"  ERROR: No data loaded for {agent_name}")
        # Fallback values
        return {
            "agent_id": agent_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "R_overall": 0.5,
            "T_overall": 0.6,
            "llm_outputs": [],
            "metadata": {}
        }
    
    return {
        "agent_id": agent_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "R_overall": r_total / count,
        "T_overall": t_total / count,
        "llm_outputs": llm_outputs,
        "metadata": metadata
    }


if __name__ == "__main__":
    print("=" * 70)
    print("SORA-ATMAS Governance Engine with Dual-Chain Blockchain Integration")
    print("=" * 70)
    
    # Initialize SORA with default policy parameters
    sora = SORA()
    
    # Check blockchain connectivity
    print("\n[Blockchain Status]")
    if sora.agentic_client:
        print(f"  Agentic Chain: ✓ Connected (port {sora.AGENTIC_CHAIN_CONFIG['port']})")
    else:
        print(f"  Agentic Chain: ✗ Disconnected")
    
    if sora.sora_client:
        print(f"  SORA Chain: ✓ Connected (port {sora.SORA_CHAIN_CONFIG['port']})")
    else:
        print(f"  SORA Chain: ✗ Disconnected")
    
    # Load agent results 
    weather_packet = load_agent_results("Weather")
    traffic_packet = load_agent_results("Traffic")
    safety_packet = load_agent_results("Safety")
    
    agent_packets = [weather_packet, traffic_packet, safety_packet]
    agent_decisions = []
    
    # Process each agent
    for packet in agent_packets:
        decision = sora.process_agent(packet)
        if decision:
            agent_decisions.append(decision)
        else:
            print(f"  {packet['agent_id']}: Rejected by admission gate")
    
    current_time = datetime.datetime.now()
    ecosystem = sora.ecosystem_evaluation(agent_decisions, current_time)
    
    alert = sora.generate_city_alert(agent_decisions, timestamp, location, ecosystem)
    
    if sora.sora_client:
        try:
            items = sora.sora_client.liststreamitems(sora.SORA_CHAIN_CONFIG["stream_name"])
            print(f"Total SORA Chain entries: {len(items)}")
            print("Recent entries:")
            for item in items[-5:]:
                data = item.get('data', {}).get('json', {})
                print(f"  - {data.get('decision_type', 'unknown')} at {data.get('timestamp', 'N/A')}")
        except Exception as e:
            print(f"Could not query SORA Chain: {e}")
    
    print("\nProcessing complete. Check 'SORA_Blockchain_Logs/' for local backups.")
