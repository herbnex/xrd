import os
import json
import re
import numpy as np
import pywt                      # For wavelet-based background subtraction (example)
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

app = Flask(__name__)
CORS(app)

##############################################################################
# 0) Configure OpenAI Key
##############################################################################

openai.api_key = os.getenv("OPENAI_API_KEY")  # or hard-code for local testing

##############################################################################
# 1) GPT Function Schemas (Unchanged)
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
# 2) GPT Call Helper
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
    except:
        return {}

##############################################################################
# 3) Numeric Enhancements (Advanced BG, Wavelet, Savitzky-Golay, Pseudo-Voigt, etc.)
##############################################################################

# ----- Advanced BG Subtraction -----
def wavelet_bg_subtraction(x, y, wavelet='db4', level=1, iteration=2):
    """
    Example wavelet-based background removal approach (very simplistic).
    We'll do a few wavelet decompositions, zero out some lower-frequency comps, then reconstruct.
    """
    # For real wavelet-based BG, you'd do something more elaborate
    # Here: do a small iterative approach
    import pywt

    signal = y.copy()
    for _ in range(iteration):
        coeffs = pywt.wavedec(signal, wavelet, mode='smooth')
        # reduce the approximation coeffs to remove background
        # e.g. scale them by some fraction
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
    """
    Iterative polynomial background subtraction:
    1) Fit polynomial
    2) Subtract
    3) Clip negative
    4) Possibly re-fit ignoring negative residual regions
    """
    signal = y.copy()
    for _ in range(max_iter):
        coeffs = np.polyfit(x, signal, order)
        p = np.poly1d(coeffs)
        bg = p(x)
        corrected = signal - bg
        corrected = np.clip(corrected, 0, None)
        signal = corrected
    return signal

# Example function that picks approach based on user “bgMethod”
def advanced_background_subtraction(data, settings):
    """
    settings might contain e.g.:
      {
        "bgMethod": "wavelet" or "iterative_poly",
        "wavelet": "db4",
        "waveletLevel": 2,
        "polyOrder": 3
      }
    """
    x = np.array([d["two_theta"] for d in data])
    y = np.array([d["intensity"] for d in data])

    bg_method = settings.get("bgMethod", "iterative_poly")

    if bg_method == "wavelet":
        wavelet = settings.get("wavelet", "db4")
        waveletLevel = settings.get("waveletLevel", 1)
        iteration = settings.get("iteration", 2)
        # do wavelet approach
        removed = wavelet_bg_subtraction(x, y, wavelet, waveletLevel, iteration)
        final = y - removed
        final = np.clip(final, 0, None)
    else:
        # default iterative poly
        order = settings.get("polyOrder", 3)
        max_iter = settings.get("maxIter", 3)
        final = iterative_poly_bg(x, y, max_iter, order)

    # build new data
    out = []
    for i in range(len(data)):
        out.append({"two_theta": float(x[i]), "intensity": float(final[i])})
    return out


# ----- Sophisticated Smoothing -----
def apply_smoothing(data, settings):
    """
    Could have e.g. smoothingMethod: "savitzky_golay" or "moving_average"
    """
    method = settings.get("smoothingMethod", "savitzky_golay")
    window = settings.get("smoothingWindow", 5)
    x = np.array([d["two_theta"] for d in data])
    y = np.array([d["intensity"] for d in data])

    if len(data) < window:
        return data

    if method == "savitzky_golay":
        poly_order = settings.get("smoothingPolyOrder", 2)
        # Ensure window <= len(data)
        if window > len(y):
            window = len(y) - (1 if len(y)%2==0 else 0)
        sm = savgol_filter(y, window_length=window, polyorder=poly_order)
    else:
        # fallback: simple moving average
        kernel = np.ones(window)/window
        sm = np.convolve(y, kernel, mode='same')

    out = []
    for i in range(len(data)):
        out.append({"two_theta": float(x[i]), "intensity": float(sm[i])})
    return out


# ----- Kα2 Stripping (naive or advanced)
def strip_kalpha2(data, fraction=0.05):
    out = []
    for d in data:
        out.append({
            "two_theta": d["two_theta"],
            "intensity": float(d["intensity"]*(1-fraction))
        })
    return out


# ----- Calibration Correction -----
def apply_calibration(data, settings):
    """
    Example: add or subtract a known offset to 2theta, or scale intensities,
    e.g. from measuring a standard sample offset of 0.1 deg in 2theta
    """
    offset = settings.get("calibrationOffset", 0.0)
    intensityScale = settings.get("intensityScale", 1.0)
    out = []
    for d in data:
        out.append({
            "two_theta": d["two_theta"]+offset,
            "intensity": float(d["intensity"]*intensityScale)
        })
    return out


# ----- Non-linear fitting with pseudo-Voigt (example for iterative refinement) -----
def pseudo_voigt(x, x0, amplitude, sigma, fraction):
    """
    A simplified pseudo-Voigt = fraction * Lorentzian + (1-fraction)*Gaussian
    """
    lor = amplitude * (sigma**2)/((x - x0)**2 + sigma**2)
    gau = amplitude * np.exp(-((x - x0)**2)/(2*sigma**2))
    return fraction*lor + (1-fraction)*gau

def iterative_refinement(x, y, peak_guesses):
    """
    Suppose we have multiple peaks (peak_guesses) as initial guesses.
    We'll do multi-peak fitting by summing pseudo-Voigt for each peak and
    do non-linear least squares. Then we compute Rwp, Rp, etc.
    This is just an example approach.
    """
    # Combine guess parameters
    # each peak guess: (x0, amplitude, sigma, fraction)
    # param_count = 4 * len(peak_guesses)
    import itertools

    def model_func(x, *params):
        # params in blocks of 4
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

    # bounds, etc. can be refined
    try:
        popt, pcov = curve_fit(model_func, x, y, p0=p0)
    except:
        popt = p0
        pcov = np.eye(len(p0))

    fitted = model_func(x, *popt)
    # compute R-factors
    residual = y - fitted
    # Weighted residual sum of squares
    w = 1/(np.sqrt(y+1e-9))  # a naive weighting
    wrss = np.sum((w*residual)**2)
    wrss_tot = np.sum((w*y)**2)
    Rwp = np.sqrt(wrss/wrss_tot)
    Rp = np.sum(abs(residual))/np.sum(abs(y))
    # basic uncertainty
    # diagonal of pcov is approximate variance of parameters
    param_stderr = np.sqrt(np.diag(pcov))

    # build result
    # each peak gets final (x0, intensity=amp, fwhm?), etc.
    # approximate FWHM from sigma
    fitted_peaks = []
    for i in range(0, len(popt),4):
        x0 = popt[i]
        amp = popt[i+1]
        sigma = popt[i+2]
        frac = popt[i+3]
        # approximate FWHM for pseudo-Voigt
        # naive formula: FWHM ~ 2 * sigma
        fwhm = 2*sigma
        fitted_peaks.append({
            "two_theta": float(x0),
            "intensity": float(amp),
            "fwhm": float(fwhm)
        })

    return fitted_peaks, float(Rwp), float(Rp), param_stderr.tolist()


##############################################################################
# 4) Master Pipeline (Advanced Numeric + GPT)
##############################################################################

def run_advanced_pipeline(raw_text, settings):
    """
    settings = {
      "bgMethod": "iterative_poly" or "wavelet",
      ...
      "smoothingMethod": "savitzky_golay" or "average",
      ...
      "calibrationOffset": 0.0,
      ...
      "enableIterativeRefinement": true/false,
      ...
    }

    We'll parse with GPT, then do advanced numeric steps,
    then optionally do iterative refinement for final peaks.
    We'll pass final peaks to GPT for phase ID, quant, final report, etc.
    """
    # 1) GPT parse
    parse_prompt = (
        "Parse lines => parse_xrd_data. \n```\n"+raw_text+"\n```"
    )
    parse_resp = call_gpt(parse_prompt, [parse_data_schema], {"name":"parse_xrd_data"})
    parsed_data = []
    if parse_resp:
        fc = parse_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"]=="parse_xrd_data":
            args = safe_json_loads(fc["arguments"])
            parsed_data = args.get("parsed_data", [])
    if not parsed_data:
        return {"error":"No valid data found.","finalReport":"No data."}

    # 2) calibration correction
    c_data = apply_calibration(parsed_data, settings)

    # 3) advanced background subtraction
    bg_data = advanced_background_subtraction(c_data, settings)

    # 4) smoothing
    sm_data = apply_smoothing(bg_data, settings)

    # 5) Kα2 strip
    kap_data = strip_kalpha2(sm_data, settings.get("kalphaFraction", 0.05))

    # 6) GPT peak detection
    detect_prompt = (
        "We have advanced processed data. Identify major peaks => detect_peaks.\n"
        f"data={kap_data}"
    )
    detect_resp = call_gpt(detect_prompt, [peak_detection_schema], {"name":"detect_peaks"})
    peaks = []
    if detect_resp:
        fc = detect_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"]=="detect_peaks":
            args = safe_json_loads(fc["arguments"])
            peaks = args.get("peaks", [])

    # 7) If user wants iterative refinement with pseudo-Voigt
    Rwp = 0
    Rp = 0
    final_fitted_peaks = []
    if settings.get("enableIterativeRefinement", False) and len(peaks)>0:
        # build some initial guesses
        # e.g. for each GPT peak: x0=peak["two_theta"], amplitude=peak["intensity"], sigma=0.1, frac=0.5
        x = np.array([d["two_theta"] for d in kap_data])
        y = np.array([d["intensity"] for d in kap_data])

        # initial guesses
        peak_guesses = []
        for pk in peaks:
            peak_guesses.append((pk["two_theta"], pk["intensity"], 0.1, 0.5))

        fitted_peaks, Rwp, Rp, param_stderr = iterative_refinement(x, y, peak_guesses)
        final_fitted_peaks = fitted_peaks
    else:
        # fallback to GPT decomposition
        pattern_prompt = (
            "Given these peaks, do advanced pattern decomposition => pattern_decomposition.\n"
            f"peaks={peaks}"
        )
        pattern_resp = call_gpt(pattern_prompt, [pattern_decomp_schema], {"name":"pattern_decomposition"})
        if pattern_resp:
            fc = pattern_resp["choices"][0]["message"].get("function_call")
            if fc and fc["name"]=="pattern_decomposition":
                args = safe_json_loads(fc["arguments"])
                final_fitted_peaks = args.get("fitted_peaks", [])

    # 8) GPT phase ID
    phase_prompt = (
        "We have final fitted peaks. Identify possible phases => phase_identification.\n"
        f"fitted_peaks={final_fitted_peaks}"
    )
    phase_resp = call_gpt(phase_prompt, [phase_id_schema], {"name":"phase_identification"})
    phases = []
    if phase_resp:
        fc = phase_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"]=="phase_identification":
            args = safe_json_loads(fc["arguments"])
            phases = args.get("phases", [])

    # 9) GPT quant
    quant_prompt = (
        "Given these phases => quantitative_analysis."
        f"\nphases={phases}"
    )
    quant_resp = call_gpt(quant_prompt, [quant_schema], {"name":"quantitative_analysis"})
    quant_results = []
    if quant_resp:
        fc = quant_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"]=="quantitative_analysis":
            args = safe_json_loads(fc["arguments"])
            quant_results = args.get("quant_results", [])

    # 10) GPT error detection
    error_prompt = (
        "Check anomalies => error_detection. parsed_data below.\n"
        f"parsed_data={parsed_data}"
    )
    error_resp = call_gpt(error_prompt, [error_detection_schema], {"name":"error_detection"})
    issues_found = []
    suggested_actions = []
    if error_resp:
        fc = error_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"]=="error_detection":
            args = safe_json_loads(fc["arguments"])
            issues_found = args.get("issues_found", [])
            suggested_actions = args.get("suggested_actions", [])

    # 11) GPT final report
    final_prompt = (
        "Create final text-based summary => generate_final_report.\n"
        f"parsed_data_count={len(parsed_data)}, fitted_peaks_count={len(final_fitted_peaks)}, Rwp={Rwp}, Rp={Rp}, phases={phases}, quant={quant_results}, issues={issues_found}, suggestions={suggested_actions}"
    )
    rep_resp = call_gpt(final_prompt, [report_schema], {"name":"generate_final_report"})
    final_report = ""
    if rep_resp:
        fc = rep_resp["choices"][0]["message"].get("function_call")
        if fc and fc["name"]=="generate_final_report":
            args = safe_json_loads(fc["arguments"])
            final_report = args.get("report_text","")

    return {
        "parsedData": parsed_data,
        "calibratedData": c_data,
        "bgCorrectedData": bg_data,
        "smoothedData": sm_data,
        "strippedData": kap_data,
        "peaks": peaks,
        "fittedPeaks": final_fitted_peaks,
        "phases": phases,
        "quantResults": quant_results,
        "issuesFound": issues_found,
        "suggestedActions": suggested_actions,
        "Rwp": Rwp,
        "Rp": Rp,
        "finalReport": final_report
    }

##############################################################################
# 5) Flask Endpoints
##############################################################################

@app.route('/api/analyze', methods=['POST'])
def analyze_xrd():
    """
    Accepts:
      - xrdFile (the .xy or .txt)
      - JSON body with 'settings' for advanced numeric steps
    """
    f = request.files.get('xrdFile')
    if not f:
        return jsonify({'error':'No file uploaded'}),400

    # parse optional JSON settings
    settings = json.loads(request.form.get('settings','{}'))

    raw_text = f.read().decode('utf-8', errors='ignore')
    result = run_advanced_pipeline(raw_text, settings)
    return jsonify(result),200

@app.route('/api/simulate', methods=['POST'])
def simulate_pattern():
    data = request.get_json()
    if not data or 'structure' not in data:
        return jsonify({'error':'No structure provided'}),400

    struct_info = data['structure']
    prompt = (
        "Simulate an XRD pattern => simulate_pattern_gpt.\n"
        f"Structure:\n{struct_info}"
    )
    sim = call_gpt(prompt, [simulation_schema], {"name":"simulate_pattern_gpt"})
    parsed_data = []
    if sim:
        fc = sim["choices"][0]["message"].get("function_call")
        if fc and fc["name"]=="simulate_pattern_gpt":
            args = safe_json_loads(fc["arguments"])
            parsed_data = args.get("parsed_data",[])
    final_report = f"GPT-based synthetic pattern. Found {len(parsed_data)} points."
    return jsonify({"parsedData": parsed_data, "finalReport":final_report}),200

@app.route('/api/cluster', methods=['POST'])
def cluster_analysis():
    cluster_files = request.files.getlist('clusterFiles')
    if not cluster_files:
        return jsonify({'error':'No files for cluster analysis'}),400

    pattern_summaries=[]
    for f in cluster_files:
        txt = f.read().decode('utf-8',errors='ignore')
        parse_prompt = f"Parse => parse_xrd_data:\n```\n{txt}\n```"
        parse_resp = call_gpt(parse_prompt, [parse_data_schema], {"name":"parse_xrd_data"})
        pd=[]
        if parse_resp:
            fc = parse_resp["choices"][0]["message"].get("function_call")
            if fc and fc["name"]=="parse_xrd_data":
                args = safe_json_loads(fc["arguments"])
                pd=args.get("parsed_data",[])
        pattern_summaries.append({"filename":f.filename,"parsed_data":pd})

    cluster_prompt = (
        f"We have multiple patterns => cluster_files_gpt.\n"
        f"Patterns: {pattern_summaries}"
    )
    cresp = call_gpt(cluster_prompt, [cluster_schema], {"name":"cluster_files_gpt"})
    clusters=[]
    if cresp:
        fc=cresp["choices"][0]["message"].get("function_call")
        if fc and fc["name"]=="cluster_files_gpt":
            args = safe_json_loads(fc["arguments"])
            clusters=args.get("clusters",[])

    final_report = f"GPT-based cluster for {len(cluster_files)} files."
    return jsonify({"clusters":clusters,"finalReport":final_report}),200

@app.route('/api/instrument-upload', methods=['POST'])
def instrument_upload():
    f = request.files.get('xrdFile')
    if not f:
        return jsonify({'error':'No xrdFile provided'}),400
    raw = f.read().decode('utf-8', errors='ignore')
    # maybe pass default settings
    settings = {}
    result = run_advanced_pipeline(raw, settings)
    return jsonify(result),200


if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
