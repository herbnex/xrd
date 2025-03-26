import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

##############################################################################
# 0) Configure OpenAI Key
##############################################################################

openai.api_key = os.getenv("OPENAI_API_KEY")  # or hard-code for local testing
app = Flask(__name__)
CORS(app)

##############################################################################
# 1) GPT Function Schemas
##############################################################################

# Existing schemas
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
                    "required": [
                        "phase_name",
                        "weight_percent",
                        "lattice_params",
                        "crystallite_size_nm",
                        "confidence_score"
                    ]
                }
            }
        },
        "required": ["quant_results"]
    }
}

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
                    "required": ["filename", "cluster_label", "explanation"]
                }
            }
        },
        "required": ["clusters"]
    }
}

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
                    "required": ["two_theta", "intensity"]
                }
            }
        },
        "required": ["parsed_data"]
    }
}

##############################################################################
# Additional function schemas for advanced steps
##############################################################################
background_sub_schema = {
    "name": "background_subtraction",
    "description": "GPT-based background subtraction. Returns corrected_data with background removed.",
    "parameters": {
        "type": "object",
        "properties": {
            "corrected_data": {
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
        "required": ["corrected_data"]
    }
}

smoothing_schema = {
    "name": "smooth_data",
    "description": "GPT-based smoothing. Returns smoothed_data with peaks preserved as best as possible.",
    "parameters": {
        "type": "object",
        "properties": {
            "smoothed_data": {
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
        "required": ["smoothed_data"]
    }
}

kalpha2_schema = {
    "name": "kalpha2_stripping",
    "description": "Strips out Kα2 from Cu or other sources. Returns stripped_data.",
    "parameters": {
        "type": "object",
        "properties": {
            "stripped_data": {
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
        "required": ["stripped_data"]
    }
}

##############################################################################
# 2) GPT Call Helper
##############################################################################

def call_gpt(prompt, functions=None, function_call="auto", max_tokens=2000):
    """
    Wrapper to call GPT with function calling.
    """
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
# 3) Single-File Pipeline with GPT for Everything (Advanced)
##############################################################################

def run_gpt_pipeline(raw_text):
    # ---------------------------------------------------------
    # (1) PARSE
    # ---------------------------------------------------------
    parse_prompt = (
        "Convert lines to (two_theta, intensity). "
        "Even if there's a header or metadata, extract numeric pairs. "
        "Use parse_xrd_data:\n"
        f"```\n{raw_text}\n```"
    )
    parse_resp = call_gpt(parse_prompt, [parse_data_schema], {"name": "parse_xrd_data"})
    parsed_data = []
    if parse_resp:
        fc = parse_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "parse_xrd_data":
            args = safe_json_loads(fc["arguments"])
            parsed_data = args.get("parsed_data", [])
    if not parsed_data:
        return {"error": "No valid data found", "finalReport": "No data to analyze."}

    # ---------------------------------------------------------
    # (2) BACKGROUND SUBTRACTION
    # ---------------------------------------------------------
    bg_prompt = (
        "Perform background subtraction on the parsed data => background_subtraction.\n"
        f"parsed_data={parsed_data}"
    )
    bg_resp = call_gpt(bg_prompt, [background_sub_schema], {"name": "background_subtraction"})
    bg_corrected_data = parsed_data
    if bg_resp:
        fc = bg_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "background_subtraction":
            args = safe_json_loads(fc["arguments"])
            bg_corrected_data = args.get("corrected_data", parsed_data)

    # ---------------------------------------------------------
    # (3) SMOOTHING
    # ---------------------------------------------------------
    smoothing_prompt = (
        "Apply smoothing while preserving peaks => smooth_data.\n"
        f"bg_corrected_data={bg_corrected_data}"
    )
    smooth_resp = call_gpt(smoothing_prompt, [smoothing_schema], {"name": "smooth_data"})
    smoothed_data = bg_corrected_data
    if smooth_resp:
        fc = smooth_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "smooth_data":
            args = safe_json_loads(fc["arguments"])
            smoothed_data = args.get("smoothed_data", bg_corrected_data)

    # ---------------------------------------------------------
    # (4) Kα2 STRIPPING
    # ---------------------------------------------------------
    ka2_prompt = (
        "Strip Kα2 from the smoothed data => kalpha2_stripping.\n"
        f"smoothed_data={smoothed_data}"
    )
    ka2_resp = call_gpt(ka2_prompt, [kalpha2_schema], {"name": "kalpha2_stripping"})
    stripped_data = smoothed_data
    if ka2_resp:
        fc = ka2_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "kalpha2_stripping":
            args = safe_json_loads(fc["arguments"])
            stripped_data = args.get("stripped_data", smoothed_data)

    # ---------------------------------------------------------
    # (5) PEAK DETECTION
    # ---------------------------------------------------------
    detect_prompt = (
        "We have the Kα1-only data. Identify major peaks => detect_peaks.\n"
        f"data={stripped_data}"
    )
    detect_resp = call_gpt(detect_prompt, [peak_detection_schema], {"name": "detect_peaks"})
    peaks = []
    if detect_resp:
        fc = detect_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "detect_peaks":
            args = safe_json_loads(fc["arguments"])
            peaks = args.get("peaks", [])

    # ---------------------------------------------------------
    # (6) PATTERN DECOMPOSITION
    # ---------------------------------------------------------
    pattern_prompt = (
        "Given these detected peaks, do advanced pattern decomposition => pattern_decomposition.\n"
        f"peaks={peaks}"
    )
    pattern_resp = call_gpt(pattern_prompt, [pattern_decomp_schema], {"name": "pattern_decomposition"})
    fitted_peaks = []
    if pattern_resp:
        fc = pattern_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "pattern_decomposition":
            args = safe_json_loads(fc["arguments"])
            fitted_peaks = args.get("fitted_peaks", [])

    # ---------------------------------------------------------
    # (7) PHASE IDENTIFICATION
    # ---------------------------------------------------------
    example_phases = """
"phases": [
  { "phase_name": "Quartz", "confidence": 0.8 },
  { "phase_name": "Feldspar", "confidence": 0.7 }
]
"""
    phase_prompt = (
        "We have fitted peaks from an XRD pattern. "
        "Identify 1-4 possible phases. Example:\n"
        + example_phases +
        "\nfitted_peaks={fitted_peaks}\n"
        "=> phase_identification"
    )
    phase_resp = call_gpt(phase_prompt, [phase_id_schema], {"name": "phase_identification"})
    phases = []
    if phase_resp:
        fc = phase_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "phase_identification":
            args = safe_json_loads(fc["arguments"])
            phases = args.get("phases", [])

    # fallback if empty
    if not phases:
        fallback_prompt = (
            "No phases returned. Guess 2-3 typical phases => phase_identification.\n"
            f"fitted_peaks={fitted_peaks}"
        )
        fb_resp = call_gpt(fallback_prompt, [phase_id_schema], {"name": "phase_identification"})
        if fb_resp:
            fc2 = fb_resp["choices"][0]["message"].get("function_call")
            if fc2 and fc2["name"] == "phase_identification":
                args2 = safe_json_loads(fc2["arguments"])
                maybe_phases = args2.get("phases", [])
                if maybe_phases:
                    phases = maybe_phases

    # ---------------------------------------------------------
    # (8) QUANTITATIVE ANALYSIS (Rietveld-like)
    # ---------------------------------------------------------
    example_quant = """
"quant_results": [
  {
    "phase_name": "Quartz",
    "weight_percent": 30,
    "lattice_params": "a=4.9134 Å, c=5.4051 Å",
    "crystallite_size_nm": 50,
    "confidence_score": 0.8
  }
]
"""
    quant_prompt = (
        "Given these identified phases, do a Rietveld-like quant => quantitative_analysis.\n"
        f"Example:\n{example_quant}\nphases={phases}"
    )
    quant_resp = call_gpt(quant_prompt, [quant_schema], {"name": "quantitative_analysis"})
    quant_results = []
    if quant_resp:
        fc = quant_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "quantitative_analysis":
            args = safe_json_loads(fc["arguments"])
            quant_results = args.get("quant_results", [])

    # ---------------------------------------------------------
    # (9) ERROR DETECTION
    # ---------------------------------------------------------
    error_prompt = (
        "Check anomalies in numeric data. Just note them => error_detection.\n"
        f"parsed_data={parsed_data}"
    )
    error_resp = call_gpt(error_prompt, [error_detection_schema], {"name": "error_detection"})
    issues_found = []
    suggested_actions = []
    if error_resp:
        fc = error_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "error_detection":
            args = safe_json_loads(fc["arguments"])
            issues_found = args.get("issues_found", [])
            suggested_actions = args.get("suggested_actions", [])

    # ---------------------------------------------------------
    # (10) FINAL REPORT
    # ---------------------------------------------------------
    final_prompt = (
        "Create a final text-based summary of all results => generate_final_report.\n"
        f"parsed_data={len(parsed_data)}, fitted_peaks={len(fitted_peaks)}, phases={phases}, quant={quant_results}, "
        f"issues={issues_found}, suggestions={suggested_actions}"
    )
    report_resp = call_gpt(final_prompt, [report_schema], {"name": "generate_final_report"})
    final_report = ""
    if report_resp:
        fc = report_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "generate_final_report":
            args = safe_json_loads(fc["arguments"])
            final_report = args.get("report_text", "")

    # Return everything
    return {
        "parsedData": parsed_data,
        "bgCorrectedData": bg_corrected_data,
        "smoothedData": smoothed_data,
        "strippedData": stripped_data,
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
    Single-file approach, letting GPT do everything (advanced steps).
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
    Accepts JSON { 'structure': ... } => returns a simulated parse_xrd_data.
    """
    data = request.get_json()
    if not data or 'structure' not in data:
        return jsonify({'error': 'No structure provided'}), 400

    struct_info = data['structure']
    prompt = (
        "Simulate an XRD pattern from this structure. "
        "Return a parse_xrd_data function call with 2theta and intensity. "
        f"=> simulate_pattern_gpt\nStructure info:\n{struct_info}"
    )
    sim_resp = call_gpt(prompt, [simulation_schema], {"name": "simulate_pattern_gpt"})
    parsed_data = []
    if sim_resp:
        fc = sim_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "simulate_pattern_gpt":
            args = safe_json_loads(fc["arguments"])
            parsed_data = args.get("parsed_data", [])

    final_report = f"Synthetic pattern from GPT. Found {len(parsed_data)} points."
    return jsonify({"parsedData": parsed_data, "finalReport": final_report}), 200

@app.route('/api/cluster', methods=['POST'])
def cluster_analysis():
    """
    GPT-based cluster of multiple .xy or any extension.
    We'll parse each file with parse_xrd_data, then feed them back to GPT => cluster_files_gpt
    """
    cluster_files = request.files.getlist('clusterFiles')
    if not cluster_files:
        return jsonify({'error':'No files for cluster analysis'}), 400

    pattern_summaries = []
    for f in cluster_files:
        text = f.read().decode('utf-8', errors='ignore')
        parse_prompt = f"Parse lines => parse_xrd_data:\n```\n{text}\n```"
        parse_resp = call_gpt(parse_prompt, [parse_data_schema], {"name": "parse_xrd_data"})
        parsed_data = []
        if parse_resp:
            fc = parse_resp["choices"][0]["message"].get("function_call")
            if fc and fc["name"] == "parse_xrd_data":
                args = safe_json_loads(fc["arguments"])
                parsed_data = args.get("parsed_data", [])

        pattern_summaries.append({
            "filename": f.filename,
            "parsed_data": parsed_data
        })

    cluster_prompt = (
        "We have multiple patterns (filename + data). Group them into clusters => cluster_files_gpt.\n"
        f"Patterns: {pattern_summaries}"
    )
    cluster_resp = call_gpt(cluster_prompt, [cluster_schema], {"name": "cluster_files_gpt"})
    clusters = []
    if cluster_resp:
        fc = cluster_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"] == "cluster_files_gpt":
            args = safe_json_loads(fc["arguments"])
            clusters = args.get("clusters", [])

    final_report = f"GPT-based cluster for {len(cluster_files)} files. Pure AI approach."
    return jsonify({"clusters": clusters, "finalReport": final_report}), 200

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
    return jsonify(result), 200

##############################################################################
# 5) Run
##############################################################################

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
