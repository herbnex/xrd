import os
import json
import re
import numpy as np
import pywt  # For wavelet-based background subtraction
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit

app = Flask(__name__)
CORS(app)

##############################################################################
# 0) Configure OpenAI Key
##############################################################################

openai.api_key = os.getenv("OPENAI_API_KEY")  # or hard-code for local testing

##############################################################################
# 1) GPT Function Schemas
##############################################################################

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

recommend_params_schema = {
    "name": "recommend_numeric_params",
    "description": "GPT suggests numeric processing parameters (bgMethod, polyOrder, wavelet, smoothingMethod, etc.) based on the data range, intensities, etc.",
    "parameters": {
        "type": "object",
        "properties": {
            "recommended_settings": {
                "type": "object",
                "properties": {
                    "bgMethod": {"type": "string"},
                    "polyOrder": {"type": "number"},
                    "wavelet": {"type": "string"},
                    "smoothingMethod": {"type": "string"},
                    "smoothingWindow": {"type": "number"},
                    "kalphaFraction": {"type": "number"},
                    "enableIterativeRefinement": {"type": "boolean"}
                },
                "required": [
                    "bgMethod",
                    "polyOrder",
                    "wavelet",
                    "smoothingMethod",
                    "smoothingWindow",
                    "kalphaFraction",
                    "enableIterativeRefinement"
                ]
            }
        },
        "required": ["recommended_settings"]
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
# 2) GPT Call Helper Functions
##############################################################################

def call_gpt(prompt, functions=None, function_call="auto", max_tokens=2000):
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            functions=functions,
            function_call=function_call,
            temperature=0.0,
            max_tokens=max_tokens
        )
        return resp
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None

def safe_json_loads(s):
    s = s or ""
    cleaned = re.sub(r'[\x00-\x1F\x7F]', '', s)
    try:
        return json.loads(cleaned)
    except Exception as e:
        print(f"JSON parse error: {e}")
        return {}

##############################################################################
# 3) Numeric Enhancements for XRD Data Processing
##############################################################################

def wavelet_bg_subtraction(x, y, wavelet='db4', level=1, iteration=2):
    signal = y.copy()
    for _ in range(iteration):
        coeffs = pywt.wavedec(signal, wavelet, mode='smooth')
        coeffs[0] = coeffs[0] * 0.7  
        new_signal = pywt.waverec(coeffs, wavelet, mode='smooth')
        if len(new_signal) >= len(signal):
            signal = new_signal[:len(signal)]
        else:
            signal[:len(new_signal)] = new_signal
    corrected = y - signal
    corrected = np.clip(corrected, 0, None)
    return corrected

def iterative_poly_bg(x, y, max_iter=3, order=3):
    signal = y.copy()
    for _ in range(max_iter):
        coeffs = np.polyfit(x, signal, order)
        p = np.poly1d(coeffs)
        bg = p(x)
        corrected = signal - bg
        signal = np.clip(corrected, 0, None)
    return signal

def advanced_background_subtraction(data, settings):
    """
    Chooses background subtraction method based on final 'settings' (merged user & GPT).
    """
    x = np.array([d["two_theta"] for d in data])
    y = np.array([d["intensity"] for d in data])

    bg_method = settings.get("bgMethod", "iterative_poly")
    if bg_method == "wavelet":
        wavelet = settings.get("wavelet", "db4")
        iteration = settings.get("waveletIterations", 2)
        removed = wavelet_bg_subtraction(x, y, wavelet, 1, iteration)
        final = y - removed
    else:
        poly_order = settings.get("polyOrder", 3)
        max_iter = settings.get("maxIter", 3)
        final = iterative_poly_bg(x, y, max_iter, poly_order)

    final = np.clip(final, 0, None)
    return [{"two_theta": float(x[i]), "intensity": float(final[i])} for i in range(len(data))]

def apply_smoothing(data, settings):
    method = settings.get("smoothingMethod", "savitzky_golay")
    window = settings.get("smoothingWindow", 5)

    x = np.array([d["two_theta"] for d in data])
    y = np.array([d["intensity"] for d in data])

    if len(data) < window:
        return data

    if method == "savitzky_golay":
        poly_order = settings.get("smoothingPolyOrder", 2)
        if window % 2 == 0:
            window += 1
        if window > len(y):
            window = len(y) - 1 if len(y) % 2 == 0 else len(y)
        sm = savgol_filter(y, window_length=window, polyorder=poly_order)
    else:
        kernel = np.ones(window) / window
        sm = np.convolve(y, kernel, mode='same')

    return [{"two_theta": float(x[i]), "intensity": float(sm[i])} for i in range(len(data))]

def strip_kalpha2(data, fraction=0.05):
    out = []
    for d in data:
        out.append({
            "two_theta": d["two_theta"],
            "intensity": d["intensity"] * (1 - fraction)
        })
    return out

def apply_calibration(data, settings):
    offset = settings.get("calibrationOffset", 0.0)
    intensityScale = settings.get("intensityScale", 1.0)
    out = []
    for d in data:
        out.append({
            "two_theta": d["two_theta"] + offset,
            "intensity": d["intensity"] * intensityScale
        })
    return out

def pseudo_voigt(x, x0, amplitude, sigma, fraction):
    lor = amplitude * (sigma ** 2) / ((x - x0) ** 2 + sigma ** 2)
    gau = amplitude * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    return fraction * lor + (1 - fraction) * gau

def iterative_refinement(x, y, peak_guesses):
    def model_func(x, *params):
        total = np.zeros_like(x)
        for i in range(0, len(params), 4):
            x0 = params[i]
            amp = params[i+1]
            sigma = params[i+2]
            frac = params[i+3]
            total += pseudo_voigt(x, x0, amp, sigma, frac)
        return total

    p0 = []
    for (x0, amp, sigma, frac) in peak_guesses:
        p0.extend([x0, amp, sigma, frac])

    try:
        popt, pcov = curve_fit(model_func, x, y, p0=p0)
    except Exception as e:
        print(f"Curve fit failed: {e}")
        popt = p0
        pcov = np.eye(len(p0))

    fitted = model_func(x, *popt)
    residual = y - fitted
    w = 1.0 / (np.sqrt(y + 1e-9))
    wrss = np.sum((w * residual) ** 2)
    wrss_tot = np.sum((w * y) ** 2)
    Rwp = np.sqrt(wrss / wrss_tot)
    Rp = np.sum(np.abs(residual)) / np.sum(np.abs(y))

    fitted_peaks = []
    for i in range(0, len(popt), 4):
        x0 = popt[i]
        amp = popt[i+1]
        sigma = popt[i+2]
        frac = popt[i+3]
        fwhm = 2.0 * sigma
        fitted_peaks.append({
            "two_theta": float(x0),
            "intensity": float(amp),
            "fwhm": float(fwhm)
        })

    return fitted_peaks, float(Rwp), float(Rp), np.sqrt(np.diag(pcov)).tolist()

##############################################################################
# 4) Master Pipeline: Advanced Numeric + GPT Integration
##############################################################################

def run_advanced_pipeline(raw_text, user_settings):
    """
    Main pipeline. We do:
      1) GPT parse
      2) GPT "recommend_numeric_params" to fill in missing user settings
      3) Apply calibration, background, smoothing, etc.
      4) Numeric peak detection & iterative refinement
      5) GPT phase ID, quant, error detection, final report
    """
    # ------------------1) GPT parse------------------
    parse_prompt = (
        "Parse lines => parse_xrd_data.\n```\n" + raw_text + "\n```"
    )
    parse_resp = call_gpt(parse_prompt, [parse_data_schema], function_call={"name": "parse_xrd_data"})
    parsed_data = []
    if parse_resp:
        fc = parse_resp["choices"][0]["message"].get("function_call")
        if fc and fc.get("name") == "parse_xrd_data":
            args = safe_json_loads(fc.get("arguments"))
            parsed_data = args.get("parsed_data", [])
    if not parsed_data:
        return {"error": "No valid data found.", "finalReport": "No data."}

    # ------------------2) GPT recommend numeric params (if user hasn't set them)-----
    # We'll feed GPT some basic stats about data range, length, intensities
    # Then GPT returns recommended_settings
    # We'll merge user_settings with recommended ones (user overrides GPT).
    xvals = [d["two_theta"] for d in parsed_data]
    yvals = [d["intensity"] for d in parsed_data]
    if len(xvals)>0:
        data_min = min(xvals)
        data_max = max(xvals)
        intensity_min = min(yvals)
        intensity_max = max(yvals)
    else:
        data_min = 0
        data_max = 0
        intensity_min = 0
        intensity_max = 0

    rec_prompt = (
        f"Data range: {data_min:.2f} to {data_max:.2f} (deg), length={len(xvals)}.\n"
        f"Intensity range: {intensity_min:.2f} to {intensity_max:.2f}.\n"
        "Recommend numeric XRD processing parameters => recommend_numeric_params.\n"
        "Output must have: bgMethod, polyOrder, wavelet, smoothingMethod, smoothingWindow, kalphaFraction, enableIterativeRefinement.\n"
    )
    rec_resp = call_gpt(rec_prompt, [recommend_params_schema], function_call={"name":"recommend_numeric_params"})
    recommended_settings = {}
    if rec_resp:
        fc = rec_resp["choices"][0]["message"].get("function_call")
        if fc and fc.get("name") == "recommend_numeric_params":
            args = safe_json_loads(fc.get("arguments"))
            recommended_settings = args.get("recommended_settings", {})

    # merge user + GPT recommended
    # user overrides GPT if conflict
    final_settings = dict(recommended_settings)
    for k,v in user_settings.items():
        final_settings[k] = v

    # For clarity, ensure some defaults
    final_settings.setdefault("bgMethod", "iterative_poly")
    final_settings.setdefault("polyOrder", 3)
    final_settings.setdefault("wavelet", "db4")
    final_settings.setdefault("smoothingMethod", "savitzky_golay")
    final_settings.setdefault("smoothingWindow", 5)
    final_settings.setdefault("kalphaFraction", 0.05)
    final_settings.setdefault("enableIterativeRefinement", True)
    final_settings.setdefault("calibrationOffset", 0.0)
    final_settings.setdefault("intensityScale", 1.0)

    # ------------------3) Calibration + BG + Smoothing + Kalpha-----------
    c_data = apply_calibration(parsed_data, final_settings)
    bg_data = advanced_background_subtraction(c_data, final_settings)
    sm_data = apply_smoothing(bg_data, final_settings)
    kap_data = strip_kalpha2(sm_data, final_settings["kalphaFraction"])

    # ------------------4) Numeric Peak Detection & Refinement-------------
    x_vals = np.array([d["two_theta"] for d in kap_data])
    y_vals = np.array([d["intensity"] for d in kap_data])

    # Adjust these detection parameters if you like
    peak_indices, properties = find_peaks(y_vals, prominence=50, distance=2)
    numeric_peaks = []
    for idx in peak_indices:
        numeric_peaks.append({"two_theta": float(x_vals[idx]), "intensity": float(y_vals[idx])})

    Rwp, Rp = 0.0, 0.0
    final_fitted_peaks = []
    if numeric_peaks and final_settings.get("enableIterativeRefinement", True):
        guesses = []
        for pk in numeric_peaks:
            # initial guess for each peak
            guesses.append((pk["two_theta"], pk["intensity"], 0.1, 0.5))
        fitted_peaks, Rwp, Rp, _ = iterative_refinement(x_vals, y_vals, guesses)
        final_fitted_peaks = fitted_peaks
    else:
        # fallback GPT pattern decomposition
        pattern_prompt = (
            "Given these numeric peaks, do advanced pattern decomposition => pattern_decomposition.\n"
            f"peaks={numeric_peaks}"
        )
        pattern_resp = call_gpt(pattern_prompt, [pattern_decomp_schema], function_call={"name": "pattern_decomposition"})
        if pattern_resp:
            fc = pattern_resp["choices"][0]["message"].get("function_call")
            if fc and fc.get("name") == "pattern_decomposition":
                args = safe_json_loads(fc.get("arguments"))
                final_fitted_peaks = args.get("fitted_peaks", [])

    # ------------------5) GPT Phase ID, Quant, Error Detection, Final Report-----
    # Phase ID
    phase_prompt = (
        "We have final fitted peaks. Identify possible phases => phase_identification.\n"
        f"fitted_peaks={final_fitted_peaks}"
    )
    phase_resp = call_gpt(phase_prompt, [phase_id_schema], function_call={"name": "phase_identification"})
    phases = []
    if phase_resp:
        fc = phase_resp["choices"][0]["message"].get("function_call")
        if fc and fc.get("name") == "phase_identification":
            args = safe_json_loads(fc.get("arguments"))
            phases = args.get("phases", [])

    # Quant
    quant_results = []
    if phases:
        quant_prompt = (
            "Given these phases => quantitative_analysis.\n"
            f"phases={phases}"
        )
        quant_resp = call_gpt(quant_prompt, [quant_schema], function_call={"name": "quantitative_analysis"})
        if quant_resp:
            fc = quant_resp["choices"][0]["message"].get("function_call")
            if fc and fc.get("name") == "quantitative_analysis":
                args = safe_json_loads(fc.get("arguments"))
                quant_results = args.get("quant_results", [])

    # Error detection
    error_prompt = (
        "Check anomalies => error_detection. parsed_data below.\n"
        f"parsed_data={parsed_data}"
    )
    error_resp = call_gpt(error_prompt, [error_detection_schema], function_call={"name": "error_detection"})
    issues_found = []
    suggested_actions = []
    if error_resp:
        fc = error_resp["choices"][0]["message"].get("function_call")
        if fc and fc.get("name") == "error_detection":
            args = safe_json_loads(fc.get("arguments"))
            issues_found = args.get("issues_found", [])
            suggested_actions = args.get("suggested_actions", [])

    # Final report
    final_prompt = (
        "Create final text-based summary => generate_final_report.\n"
        f"parsed_data_count={len(parsed_data)}, "
        f"fitted_peaks_count={len(final_fitted_peaks)}, "
        f"Rwp={Rwp}, Rp={Rp}, "
        f"phases={phases}, "
        f"quant={quant_results}, "
        f"issues={issues_found}, "
        f"suggestions={suggested_actions}, "
        f"userSettings={user_settings}, recommendedSettings={recommended_settings}"
    )
    rep_resp = call_gpt(final_prompt, [report_schema], function_call={"name": "generate_final_report"})
    final_report = ""
    if rep_resp:
        fc = rep_resp["choices"][0]["message"].get("function_call")
        if fc and fc.get("name") == "generate_final_report":
            args = safe_json_loads(fc.get("arguments"))
            final_report = args.get("report_text", "")

    return {
        "parsedData": parsed_data,
        "calibratedData": c_data,
        "bgCorrectedData": bg_data,
        "smoothedData": sm_data,
        "strippedData": kap_data,
        "peaks_found_numeric": numeric_peaks,
        "fittedPeaks": final_fitted_peaks,
        "phases": phases,
        "quantResults": quant_results,
        "issuesFound": issues_found,
        "suggestedActions": suggested_actions,
        "Rwp": Rwp,
        "Rp": Rp,
        "finalReport": final_report,
        # helpful to see what GPT recommended
        "recommendedSettings": recommended_settings,
        "finalSettings": final_settings
    }

##############################################################################
# 5) Flask Endpoints
##############################################################################

@app.route('/api/analyze', methods=['POST'])
def analyze_xrd():
    """
    Endpoint to analyze XRD data.
    Expects a file in 'xrdFile' and optional JSON settings.
    """
    f = request.files.get('xrdFile')
    if not f:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        user_settings = json.loads(request.form.get('settings', '{}'))
    except Exception as e:
        return jsonify({'error': f'Error parsing settings: {e}'}), 400

    raw_text = f.read().decode('utf-8', errors='ignore')
    result = run_advanced_pipeline(raw_text, user_settings)
    return jsonify(result), 200

@app.route('/api/simulate', methods=['POST'])
def simulate_pattern():
    data = request.get_json()
    if not data or 'structure' not in data:
        return jsonify({'error': 'No structure provided'}), 400

    struct_info = data['structure']
    prompt = (
        "Simulate an XRD pattern => simulate_pattern_gpt.\n"
        f"Structure:\n{struct_info}"
    )
    sim = call_gpt(prompt, [simulation_schema], function_call={"name": "simulate_pattern_gpt"})
    parsed_data = []
    if sim:
        fc = sim["choices"][0]["message"].get("function_call")
        if fc and fc.get("name") == "simulate_pattern_gpt":
            args = safe_json_loads(fc.get("arguments"))
            parsed_data = args.get("parsed_data", [])
    final_report = f"GPT-based synthetic pattern. Found {len(parsed_data)} points."
    return jsonify({"parsedData": parsed_data, "finalReport": final_report}), 200

@app.route('/api/cluster', methods=['POST'])
def cluster_analysis():
    cluster_files = request.files.getlist('clusterFiles')
    if not cluster_files:
        return jsonify({'error': 'No files for cluster analysis'}), 400

    pattern_summaries=[]
    for f in cluster_files:
        txt = f.read().decode('utf-8', errors='ignore')
        parse_prompt = f"Parse => parse_xrd_data:\n```\n{txt}\n```"
        parse_resp = call_gpt(parse_prompt, [parse_data_schema], function_call={"name":"parse_xrd_data"})
        pd_data=[]
        if parse_resp:
            fc = parse_resp["choices"][0]["message"].get("function_call")
            if fc and fc.get("name")=="parse_xrd_data":
                args = safe_json_loads(fc.get("arguments"))
                pd_data = args.get("parsed_data",[])
        pattern_summaries.append({"filename":f.filename,"parsed_data":pd_data})

    cluster_prompt = (
        "We have multiple patterns => cluster_files_gpt.\n"
        f"Patterns: {pattern_summaries}"
    )
    cresp = call_gpt(cluster_prompt, [cluster_schema], function_call={"name":"cluster_files_gpt"})
    clusters=[]
    if cresp:
        fc = cresp["choices"][0]["message"].get("function_call")
        if fc and fc.get("name")=="cluster_files_gpt":
            args = safe_json_loads(fc.get("arguments"))
            clusters=args.get("clusters",[])

    final_report = f"GPT-based cluster for {len(cluster_files)} files."
    return jsonify({"clusters":clusters,"finalReport":final_report}),200

@app.route('/api/instrument-upload', methods=['POST'])
def instrument_upload():
    f = request.files.get('xrdFile')
    if not f:
        return jsonify({'error': 'No xrdFile provided'}), 400
    raw = f.read().decode('utf-8', errors='ignore')
    user_settings = {}
    result = run_advanced_pipeline(raw, user_settings)
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
