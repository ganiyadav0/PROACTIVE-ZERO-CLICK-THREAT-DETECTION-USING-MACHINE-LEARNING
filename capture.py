from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json
import random
import time
from datetime import datetime

app = Flask(__name__)

# ─── Load Models ────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

try:
    anomaly_model  = joblib.load(os.path.join(MODEL_DIR, 'anomaly_xgb.pkl'))
    attack_model   = joblib.load(os.path.join(MODEL_DIR, 'attack_xgb.pkl'))
    scaler_anomaly = joblib.load(os.path.join(MODEL_DIR, 'scaler_anomaly.pkl'))
    scaler_attack  = joblib.load(os.path.join(MODEL_DIR, 'scaler_attack.pkl'))
    le_attack      = joblib.load(os.path.join(MODEL_DIR, 'le_attack.pkl'))
    feature_cols   = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))
    MODELS_LOADED  = True
    print("✅ Models loaded successfully")
except Exception as e:
    MODELS_LOADED = False
    print(f"⚠️  Models not found, running in DEMO mode: {e}")

# ─── MITRE ATT&CK Mapping ───────────────────────────────────────────────────
MITRE_MAP = {
    'DDoS':                  {'id': 'T1498',    'name': 'Network Denial of Service',        'tactic': 'Impact'},
    'DoS Hulk':              {'id': 'T1499',    'name': 'Endpoint Denial of Service',        'tactic': 'Impact'},
    'DoS GoldenEye':         {'id': 'T1499',    'name': 'Endpoint Denial of Service',        'tactic': 'Impact'},
    'DoS Slowhttptest':      {'id': 'T1499.001','name': 'OS Exhaustion Flood',               'tactic': 'Impact'},
    'DoS slowloris':         {'id': 'T1499.001','name': 'OS Exhaustion Flood',               'tactic': 'Impact'},
    'PortScan':              {'id': 'T1046',    'name': 'Network Service Discovery',         'tactic': 'Discovery'},
    'FTP-Patator':           {'id': 'T1110',    'name': 'Brute Force',                       'tactic': 'Credential Access'},
    'SSH-Patator':           {'id': 'T1110.001','name': 'Password Guessing',                 'tactic': 'Credential Access'},
    'Bot':                   {'id': 'T1071',    'name': 'Application Layer Protocol',        'tactic': 'Command and Control'},
    'Web Attack_Brute Force':{'id': 'T1110',    'name': 'Brute Force',                       'tactic': 'Credential Access'},
    'Web Attack_XSS':        {'id': 'T1059.007','name': 'JavaScript Execution',              'tactic': 'Execution'},
    'ZERO_CLICK':            {'id': 'T1203',    'name': 'Exploitation for Client Execution', 'tactic': 'Execution'},
    'SAFE':                  {'id': '—',         'name': 'No threat detected',                'tactic': '—'},
}

SEVERITY = {
    'SAFE':                   'safe',
    'ZERO_CLICK':             'critical',
    'DDoS':                   'high',
    'DoS Hulk':               'high',
    'DoS GoldenEye':          'high',
    'DoS Slowhttptest':       'medium',
    'DoS slowloris':          'medium',
    'PortScan':               'medium',
    'FTP-Patator':            'medium',
    'SSH-Patator':            'medium',
    'Bot':                    'high',
    'Web Attack_Brute Force': 'medium',
    'Web Attack_XSS':         'medium',
}

# ─── In-memory event log ─────────────────────────────────────────────────────
event_log = []

# ─── Prediction helper ───────────────────────────────────────────────────────
def predict_row(row_df_or_dict):
    """Run the two-stage prediction on a single-row DataFrame or feature dict."""
    if isinstance(row_df_or_dict, dict):
        row_df = pd.DataFrame([row_df_or_dict])
    else:
        row_df = row_df_or_dict

    # Align columns – fill any missing with 0
    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0.0
    data = row_df[feature_cols]

    # Stage 1 – anomaly
    scaled1   = scaler_anomaly.transform(data)
    anom_prob = float(anomaly_model.predict_proba(scaled1)[0][1])

    if anom_prob < 0.5:
        return 'SAFE', anom_prob, 1.0

    # Stage 2 – classification
    scaled2  = scaler_attack.transform(data)
    probs    = attack_model.predict_proba(scaled2)[0]
    max_prob = float(max(probs))
    pred_idx = int(np.argmax(probs))

    if max_prob < 0.4:
        return 'ZERO_CLICK', anom_prob, max_prob

    label = le_attack.inverse_transform([pred_idx])[0]
    return label, anom_prob, max_prob


def build_event(label, anom_prob, conf, src_ip=None, dst_ip=None, port=None):
    mitre = MITRE_MAP.get(label, MITRE_MAP['ZERO_CLICK'])
    sev   = SEVERITY.get(label, 'medium')
    return {
        'timestamp':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'label':      label,
        'severity':   sev,
        'anom_prob':  round(anom_prob * 100, 1),
        'confidence': round(conf * 100, 1),
        'mitre_id':   mitre['id'],
        'mitre_name': mitre['name'],
        'tactic':     mitre['tactic'],
        'src_ip':     src_ip or f"192.168.{random.randint(1,10)}.{random.randint(1,254)}",
        'dst_ip':     dst_ip or f"10.0.{random.randint(0,5)}.{random.randint(1,254)}",
        'port':       port or random.randint(1024, 65535),
        'source':     'simulate',
    }

# ─── Demo simulation ─────────────────────────────────────────────────────────
DEMO_LABELS = ['SAFE','SAFE','SAFE','SAFE','PortScan','DDoS','Bot',
               'SSH-Patator','ZERO_CLICK','Web Attack_XSS','DoS Hulk']

def demo_event():
    label = random.choice(DEMO_LABELS)
    anom  = 0.05 if label == 'SAFE' else random.uniform(0.6, 0.99)
    conf  = 1.0  if label == 'SAFE' else random.uniform(0.55, 0.98)
    return build_event(label, anom, conf)


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           models_loaded=MODELS_LOADED,
                           attack_classes=list(MITRE_MAP.keys()))

@app.route('/live')
def live_page():
    return render_template('live.html', models_loaded=MODELS_LOADED)

@app.route('/api/status')
def status():
    total  = len(event_log)
    alerts = sum(1 for e in event_log if e['severity'] != 'safe')
    crits  = sum(1 for e in event_log if e['severity'] == 'critical')
    label_counts = {}
    for e in event_log:
        label_counts[e['label']] = label_counts.get(e['label'], 0) + 1
    return jsonify({
        'total': total, 'alerts': alerts, 'critical': crits,
        'label_counts': label_counts,
        'models_loaded': MODELS_LOADED
    })

@app.route('/api/events')
def get_events():
    limit = int(request.args.get('limit', 50))
    return jsonify(list(reversed(event_log[-limit:])))

@app.route('/api/simulate', methods=['POST'])
def simulate():
    ev = demo_event()
    event_log.append(ev)
    return jsonify(ev)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    try:
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
    except Exception as e:
        return jsonify({'error': f'CSV parse error: {e}'}), 400

    results = []
    if MODELS_LOADED:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            return jsonify({'error': f'Missing columns: {missing[:5]}...'}), 400
        for i in range(min(len(df), 200)):
            row = df.iloc[[i]]
            try:
                label, ap, cp = predict_row(row)
                ev = build_event(label, ap, cp)
                event_log.append(ev)
                results.append(ev)
            except Exception as e:
                results.append({'error': str(e), 'row': i})
    else:
        for i in range(min(len(df), 200)):
            ev = demo_event()
            ev['row'] = i
            event_log.append(ev)
            results.append(ev)

    summary = {}
    for r in results:
        lbl = r.get('label', 'error')
        summary[lbl] = summary.get(lbl, 0) + 1
    return jsonify({'results': results, 'summary': summary, 'total': len(results)})

@app.route('/api/clear', methods=['POST'])
def clear_events():
    event_log.clear()
    return jsonify({'ok': True})


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE CAPTURE API  (new)
# ══════════════════════════════════════════════════════════════════════════════

def _live_predict_fn(feat_dict):
    """Adapter: feature dict → (label, anom_prob, conf)"""
    if MODELS_LOADED:
        return predict_row(feat_dict)
    # demo fallback
    label = random.choice(DEMO_LABELS)
    return label, (0.05 if label == 'SAFE' else random.uniform(0.6, 0.99)), random.uniform(0.55, 0.98)


@app.route('/api/live/interfaces')
def live_interfaces():
    """Return available Npcap interfaces."""
    try:
        from capture import list_interfaces, SCAPY_AVAILABLE
        if not SCAPY_AVAILABLE:
            return jsonify({'error': 'Scapy/Npcap not installed', 'interfaces': []})
        ifaces = list_interfaces()
        return jsonify({'interfaces': ifaces})
    except Exception as e:
        return jsonify({'error': str(e), 'interfaces': []})


@app.route('/api/live/start', methods=['POST'])
def live_start():
    """Start live packet capture."""
    data  = request.get_json(force=True) or {}
    iface = data.get('iface', '')
    pkt_filter = data.get('filter', 'ip')

    if not iface:
        return jsonify({'error': 'No interface specified'}), 400

    try:
        from capture import start_capture, capture_stats
        if capture_stats['running']:
            return jsonify({'error': 'Already running'}), 409
        ok, msg = start_capture(iface, _live_predict_fn, pkt_filter)
        if ok:
            return jsonify({'ok': True, 'message': msg})
        return jsonify({'error': msg}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/live/stop', methods=['POST'])
def live_stop():
    """Stop live packet capture."""
    try:
        from capture import stop_capture
        stop_capture()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/live/status')
def live_status():
    """Return capture engine stats + buffer size."""
    try:
        from capture import capture_stats, capture_buffer, MAX_BUFFER
        return jsonify({
            **capture_stats,
            'buffer_size':  len(capture_buffer),
            'buffer_max':   MAX_BUFFER,
        })
    except Exception as e:
        return jsonify({
            'running': False, 'error': str(e),
            'buffer_size': 0, 'buffer_max': 500,
            'packets_seen': 0, 'flows_flushed': 0,
        })


@app.route('/api/live/events')
def live_events():
    """
    Return new events from the live capture ring-buffer.
    Pass ?since=<ISO-timestamp> to get only newer events.
    Returns max 50 events per poll.
    """
    try:
        from capture import capture_buffer
        since = request.args.get('since', '')
        events = list(capture_buffer)

        if since:
            try:
                events = [e for e in events if e['timestamp'] > since]
            except Exception:
                pass

        # Also push live events into the main event_log
        for ev in events[-50:]:
            if ev not in event_log:
                event_log.append(ev)
        if len(event_log) > 2000:
            del event_log[:len(event_log) - 2000]

        return jsonify(events[-50:])
    except Exception as e:
        return jsonify([])


@app.route('/api/live/buffer/clear', methods=['POST'])
def live_buffer_clear():
    """Clear the live capture buffer."""
    try:
        from capture import capture_buffer
        capture_buffer.clear()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
