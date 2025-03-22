import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

openai.api_key = "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"  # Set your API key as an env variable

app = Flask(__name__)
CORS(app)

##############################################################################
# 1) GPT Function Schemas (No numeric libs—GPT handles everything)
##############################################################################

# Parse XRD
parse_data_schema = {
    "name": "parse_xrd_data",
    "description": "Parses raw XRD text data into a list of {two_theta, intensity}. (No numeric libs, GPT does it)",
    "parameters": {
        "type": "object",
        "properties": {
            "parsed_data": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "two_theta": {"type": "number"},
                        "intensity": {"type": "number"}
                    },
                    "required": ["two_theta", "intensity"]
                }
            }
        },
        "required": ["parsed_data"]
    }
}

# Detect peaks
peak_detection_schema = {
    "name": "detect_peaks",
    "description": "Identifies major peaks. GPT uses advanced logic (no numeric libs).",
    "parameters": {
        "type": "object",
        "properties": {
            "peaks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "two_theta": {"type": "number"},
                        "intensity": {"type": "number"}
                    },
                    "required": ["two_theta", "intensity"]
                }
            }
        },
        "required": ["peaks"]
    }
}

# Pattern Decomposition (Overlapping peaks)
pattern_decomp_schema = {
    "name": "pattern_decomposition",
    "description": "GPT does advanced peak fitting & pattern decomposition, purely text-based no HPC libs.",
    "parameters": {
        "type": "object",
        "properties": {
            "fitted_peaks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "two_theta": {"type": "number"},
                        "intensity": {"type": "number"},
                        "fwhm": {"type": "number"}
                    },
                    "required": ["two_theta", "intensity", "fwhm"]
                }
            }
        },
        "required": ["fitted_peaks"]
    }
}

# Phase identification
phase_id_schema = {
    "name": "phase_identification",
    "description": "GPT decides possible phases from 2theta peaks, purely from internal knowledge. No DB libs.",
    "parameters": {
        "type": "object",
        "properties": {
            "phases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "phase_name": {"type": "string"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["phase_name", "confidence"]
                }
            }
        },
        "required": ["phases"]
    }
}

# Rietveld-like quantification
quant_schema = {
    "name": "quantitative_analysis",
    "description": "GPT simulates Rietveld refinement (no numeric HPC). Returns phase amounts, etc.",
    "parameters": {
        "type": "object",
        "properties": {
            "quant_results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "phase_name": {"type": "string"},
                        "weight_percent": {"type": "number"},
                        "lattice_params": {"type": "string"},
                        "crystallite_size_nm": {"type": "number"},
                        "confidence_score": {"type": "number"}
                    },
                    "required": ["phase_name","weight_percent","lattice_params","crystallite_size_nm","confidence_score"]
                }
            }
        },
        "required": ["quant_results"]
    }
}

# Error detection
error_detection_schema = {
    "name": "error_detection",
    "description": "Detect anomalies purely with GPT. (No mention of Age/Income).",
    "parameters": {
        "type": "object",
        "properties": {
            "issues_found": {
                "type": "array",
                "items": {"type": "string"}
            },
            "suggested_actions": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["issues_found", "suggested_actions"]
    }
}

# Generate final text-based report
report_schema = {
    "name": "generate_final_report",
    "description": "GPT merges all results into a final text-based summary with no numeric libs.",
    "parameters": {
        "type": "object",
        "properties": {
            "report_text": {"type": "string"}
        },
        "required": ["report_text"]
    }
}

# Cluster analysis (multi-file)
cluster_schema = {
    "name": "cluster_files_gpt",
    "description": "GPT does cluster analysis across multiple patterns purely from internal knowledge. No numeric HPC.",
    "parameters": {
        "type": "object",
        "properties": {
            "clusters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "cluster_label": {"type": "string"},
                        "explanation": {"type": "string"}
                    },
                    "required": ["filename","cluster_label","explanation"]
                }
            }
        },
        "required": ["clusters"]
    }
}

# Simulation
simulation_schema = {
    "name": "simulate_pattern_gpt",
    "description": "GPT simulates a pattern from structure, no numeric HPC.",
    "parameters": {
        "type": "object",
        "properties": {
            "parsed_data": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "two_theta": {"type": "number"},
                        "intensity": {"type": "number"}
                    },
                    "required": ["two_theta","intensity"]
                }
            }
        },
        "required": ["parsed_data"]
    }
}

##############################################################################
# 2) GPT Call Helper (No numeric libs)
##############################################################################

import re

def call_gpt(prompt, functions=None, function_call="auto", max_tokens=2000):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            functions=functions,
            function_call=function_call,
            temperature=0.0,
            max_tokens=max_tokens
        )
        print("=== RAW GPT RESPONSE ===")
        print(response)
        return response
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None

def safe_json_loads(s):
    s = s or ""
    cleaned = re.sub(r'[\x00-\x1F\x7F]', '', s)
    try:
        return json.loads(cleaned)
    except:
        return {}

##############################################################################
# 3) Single-File Pipeline with GPT for Everything
##############################################################################

def run_gpt_pipeline(raw_text):
    # 1) parse
    parse_prompt = f"Convert lines to (two_theta, intensity). Use parse_xrd_data:\n```\n{raw_text}\n```"
    parse_resp = call_gpt(parse_prompt, [parse_data_schema], {"name":"parse_xrd_data"})
    parsed_data = []
    if parse_resp:
        fc = parse_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "parse_xrd_data":
            args = safe_json_loads(fc["arguments"])
            parsed_data = args.get("parsed_data", [])

    # 2) detect peaks
    detect_prompt = f"We have {parsed_data}. Identify peaks with detect_peaks."
    detect_resp = call_gpt(detect_prompt, [peak_detection_schema], {"name":"detect_peaks"})
    peaks = []
    if detect_resp:
        fc = detect_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "detect_peaks":
            args = safe_json_loads(fc["arguments"])
            peaks = args.get("peaks", [])

    # 3) pattern decomposition
    pattern_prompt = f"Given these raw peaks {peaks}, do advanced pattern decomposition => pattern_decomposition."
    pattern_resp = call_gpt(pattern_prompt, [pattern_decomp_schema], {"name":"pattern_decomposition"})
    fitted_peaks = []
    if pattern_resp:
        fc = pattern_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "pattern_decomposition":
            args = safe_json_loads(fc["arguments"])
            fitted_peaks = args.get("fitted_peaks", [])

    # 4) phase identification
    phase_prompt = f"Given fitted_peaks={fitted_peaks}, identify phases => phase_identification"
    phase_resp = call_gpt(phase_prompt, [phase_id_schema], {"name":"phase_identification"})
    phases = []
    if phase_resp:
        fc = phase_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "phase_identification":
            args = safe_json_loads(fc["arguments"])
            phases = args.get("phases", [])

    # 5) quant
    quant_prompt = f"Phases: {phases}. Do Rietveld-like quant => quantitative_analysis"
    quant_resp = call_gpt(quant_prompt, [quant_schema], {"name":"quantitative_analysis"})
    quant_results = []
    if quant_resp:
        fc = quant_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "quantitative_analysis":
            args = safe_json_loads(fc["arguments"])
            quant_results = args.get("quant_results", [])

    # 6) error detection
    error_prompt = f"Check anomalies in numeric data => error_detection. Data: {parsed_data}"
    error_resp = call_gpt(error_prompt, [error_detection_schema], {"name":"error_detection"})
    issues_found = []
    suggested_actions = []
    if error_resp:
        fc = error_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "error_detection":
            args = safe_json_loads(fc["arguments"])
            issues_found = args.get("issues_found", [])
            suggested_actions = args.get("suggested_actions", [])

    # 7) final report
    final_prompt = f"""
    Summarize all. 
    parsed_data={parsed_data}, peaks={peaks}, fitted_peaks={fitted_peaks}, 
    phases={phases}, quant={quant_results}, issues={issues_found}, suggestions={suggested_actions}.
    => generate_final_report
    """
    report_resp = call_gpt(final_prompt, [report_schema], {"name":"generate_final_report"})
    final_report = ""
    if report_resp:
        fc = report_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "generate_final_report":
            args = safe_json_loads(fc["arguments"])
            final_report = args.get("report_text","")

    return {
        "parsedData": parsed_data,
        "peaks": peaks,
        "fittedPeaks": fitted_peaks,
        "phases": phases,
        "quantResults": quant_results,
        "issuesFound": issues_found,
        "suggestedActions": suggested_actions,
        "finalReport": final_report
    }

##############################################################################
# 4) Flask Endpoints
##############################################################################

@app.route('/api/analyze', methods=['POST'])
def analyze_xrd():
    """
    Single-file approach, letting GPT do everything.
    """
    f = request.files.get('xrdFile')
    if not f:
        return jsonify({'error': 'No file uploaded'}), 400

    raw_text = f.read().decode('utf-8', errors='ignore')
    result = run_gpt_pipeline(raw_text)
    return jsonify(result), 200

@app.route('/api/simulate', methods=['POST'])
def simulate_pattern():
    """
    GPT-based pattern simulation, no numeric libs. 
    Accepts JSON { 'structure': ... } 
    => returns a simulated parse_xrd_data.
    """
    data = request.get_json()
    if not data or 'structure' not in data:
        return jsonify({'error': 'No structure provided'}), 400

    struct_info = data['structure']
    prompt = f"Simulate an XRD pattern from structure: {struct_info} => simulate_pattern_gpt"
    sim_resp = call_gpt(prompt, [simulation_schema], {"name":"simulate_pattern_gpt"})
    parsed_data = []
    if sim_resp:
        fc = sim_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "simulate_pattern_gpt":
            args = safe_json_loads(fc["arguments"])
            parsed_data = args.get("parsed_data", [])

    final_report = f"Synthetic pattern from GPT, purely AI. Found {len(parsed_data)} points."
    return jsonify({
        "parsedData": parsed_data,
        "finalReport": final_report
    }), 200

@app.route('/api/cluster', methods=['POST'])
def cluster_analysis():
    """
    GPT-based cluster of multiple .xy. 
    We'll parse each file with GPT parse_xrd_data, then let GPT group them => cluster_files_gpt
    """
    cluster_files = request.files.getlist('clusterFiles')
    if not cluster_files:
        return jsonify({'error':'No files for cluster analysis'}), 400

    # We'll parse each file with GPT parse_xrd_data, store the results, then feed them back to GPT for cluster
    pattern_summaries = []
    for f in cluster_files:
        text = f.read().decode('utf-8', errors='ignore')
        parse_prompt = f"Parse lines => parse_xrd_data:\n```\n{text}\n```"
        parse_resp = call_gpt(parse_prompt, [parse_data_schema], {"name":"parse_xrd_data"})
        parsed_data = []
        if parse_resp:
            fc = parse_resp["choices"][0]["message"].get("function_call")
            if fc and fc["name"]=="parse_xrd_data":
                args = safe_json_loads(fc["arguments"])
                parsed_data = args.get("parsed_data", [])

        # We'll just store filename + parsed_data
        pattern_summaries.append({
            "filename": f.filename,
            "parsed_data": parsed_data
        })

    # Now pass them to GPT to cluster
    cluster_prompt = f"""
    We have multiple patterns (filename + data). Cluster them => cluster_files_gpt
    Patterns: {pattern_summaries}
    """
    cluster_resp = call_gpt(cluster_prompt, [cluster_schema], {"name":"cluster_files_gpt"})
    clusters = []
    if cluster_resp:
        fc = cluster_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"]=="cluster_files_gpt":
            args = safe_json_loads(fc["arguments"])
            clusters = args.get("clusters", [])

    final_report = f"GPT-based cluster for {len(cluster_files)} files. Pure AI approach."
    return jsonify({
        "clusters": clusters,
        "finalReport": final_report
    }), 200

@app.route('/api/instrument-upload', methods=['POST'])
def instrument_upload():
    """
    For zero-click from instrument => calls run_gpt_pipeline
    """
    f = request.files.get('xrdFile')
    if not f:
        return jsonify({'error': 'No xrdFile provided'}), 400

    text = f.read().decode('utf-8', errors='ignore')
    result = run_gpt_pipeline(text)
    # Could notify user
    return jsonify(result), 200

##############################################################################
# 5) Run
##############################################################################

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
