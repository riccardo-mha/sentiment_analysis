from flask import Flask, render_template, request, jsonify, send_from_directory
import analysis_logic
import re
import os
import json

app = Flask(__name__)

# --- Route 1: La Pagina Principale (ora dinamica) ---
@app.route('/')
def index():
    """
    Legge il database JSON e renderizza dinamicamente la pagina dei report.
    """
    json_db_file = "reports_data.json"
    reports = {}
    if os.path.exists(json_db_file):
        with open(json_db_file, "r", encoding="utf-8") as f:
            try:
                reports = json.load(f)
            except json.JSONDecodeError:
                reports = {}
    
    # Passa i dati dei report al template HTML
    return render_template('reports.html', reports=reports)

# --- Route 2: Serve i file dei report individuali e altri file statici ---
@app.route('/<path:filename>')
def serve_static_files(filename):
    """
    Serve i report generati (es. 3nkFtJMCs1Q_report.html).
    """
    return send_from_directory('.', filename)

# --- Route 3: L'Endpoint per l'Analisi ---
@app.route('/analyze', methods=['POST'])
def analyze_video():
    """
    Riceve un URL di YouTube, esegue l'analisi completa e restituisce il nome del file del report.
    """
    data = request.get_json()
    youtube_url = data.get('url')

    if not youtube_url:
        return jsonify({'error': 'URL è richiesto'}), 400

    video_id_match = re.search(r'(?:v=|\/|embed\/|youtu.be\/)([0-9A-Za-z_-]{11})', youtube_url)
    if not video_id_match:
        return jsonify({'error': 'URL di YouTube non valido'}), 400
    
    video_id = video_id_match.group(1)

    try:
        print(f"--- Inizio analisi per Video ID: {video_id} ---")
        
        video_details = analysis_logic.get_video_details(analysis_logic.YOUTUBE_API_KEY, video_id)
        if not video_details or video_details["title"] == "Unknown Video":
            # CORREZIONE: L'indentazione qui è corretta (4 spazi).
            return jsonify({'error': 'Impossibile recuperare i dettagli del video. Controlla l-ID del video e la tua chiave API.'}), 404

        video_comments = analysis_logic.get_video_comments(analysis_logic.YOUTUBE_API_KEY, video_id)
        if video_comments is None:
            # CORREZIONE: L'indentazione qui è corretta.
            return jsonify({'error': 'Impossibile recuperare i commenti. Potrebbero essere disabilitati o la quota della chiave API è esaurita.'}), 500

        analysis_data = analysis_logic.analyze_comments_vader(video_comments)
        
        summaries = {
            'positive': analysis_logic.summarize_with_ollama(analysis_data['positive_comments'], 'positive'),
            'negative': analysis_logic.summarize_with_ollama(analysis_data['negative_comments'], 'negative'),
            'neutral': analysis_logic.summarize_with_ollama(analysis_data['neutral_comments'], 'neutral')
        }
        
        insight_data = analysis_logic.extract_insights(analysis_data['positive_comments'])

        strategic_conclusion = analysis_logic.generate_strategic_conclusion(
            len(video_comments), analysis_data, summaries
        )
        
        report_file_name = analysis_logic.generate_html_report(
            video_id, video_details, len(video_comments), analysis_data, 
            insight_data, summaries, strategic_conclusion
        )

        analysis_logic.update_landing_page(video_id, video_details['title'])
        
        print(f"--- Analisi completata. Report generato: {report_file_name} ---")
        return jsonify({'report_url': report_file_name})

    except Exception as e:
        print(f"Errore durante l'analisi: {e}")
        return jsonify({'error': f'Errore interno del server: {e}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

