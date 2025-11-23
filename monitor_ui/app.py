import os
import json
from flask import Flask, render_template, abort,jsonify

app = Flask(__name__)
LOG_DIR = "../logs" # Assumes the UI is next to the main project folder


@app.template_filter('pretty_json')
def pretty_json_filter(value):
    """Formats a Python dict or list into a pretty-printed JSON string for display."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=False)
    # For strings that are already JSON, try to format them too
    try:
        if isinstance(value, str):
            parsed = json.loads(value)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass # Not a valid JSON string, return as is
    return value

@app.route('/')
def dashboard():
    """Shows a list of all available run traces."""
    runs = []
    for filename in sorted(os.listdir(LOG_DIR), reverse=True):
        if filename.endswith(".json"):
            # You could parse the JSON here to get status, cost, etc. for the dashboard
            runs.append({
                "id": filename.replace(".json", ""),
                "timestamp": filename.split('_')[2] # Basic parsing
            })
    return render_template('dashboard.html', runs=runs)

@app.route('/trace/<run_id>')
def trace_view(run_id):
    """Renders the initial HTML shell for the trace view."""
    json_path = os.path.join(LOG_DIR, run_id + ".json")
    if not os.path.exists(json_path):
        abort(404, "Trace not found")
        
    # The initial render can be minimal, as JS will populate it
    return render_template('trace_view.html', run_id=run_id)


@app.route('/api/trace/<run_id>')
def get_trace_data(run_id):
    """Returns the raw trace data as JSON."""
    json_path = os.path.join(LOG_DIR, run_id + ".json")
    if not os.path.exists(json_path):
        return jsonify({"error": "Trace not found"}), 404
        
    with open(json_path, 'r', encoding='utf-8') as f:
        trace_data = json.load(f)
        
    # We also need to know if the run is still in progress.
    # A simple heuristic: if the last event is not "Final Report", it's probably running.
    is_running = True
    if trace_data and trace_data[-1].get("type") == "phase" and trace_data[-1].get("name") == "Final Report":
        is_running = False

    return jsonify({"trace": trace_data, "is_running": is_running})

if __name__ == '__main__':
    app.run(debug=True, port=5001)