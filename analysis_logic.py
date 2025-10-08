# Import the required libraries
import re
import json
import requests
from collections import Counter
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import webbrowser
import os

# --- Pre-load NLTK data ---
nltk.download('vader_lexicon', quiet=True)


# --- CONFIGURATION ---
from youtubeapi import YOUTUBE_API_KEY
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral" 


def get_video_details(api_key, video_id):
    """
    Fetches details for a specific video like title and channel.
    """
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        if not response['items']:
            return {"title": "Unknown Video", "channelTitle": "Unknown Channel"}
        snippet = response['items'][0]['snippet']
        return {
            "title": snippet.get('title', 'No Title'),
            "channelTitle": snippet.get('channelTitle', 'No Channel Title')
        }
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred while fetching video details: {e.content}")
        return {"title": "Unknown Video", "channelTitle": "Unknown Channel"}


def get_video_comments(api_key, video_id):
    """
    Fetches all top-level comments from a YouTube video using the YouTube Data API.
    """
    comments = []
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        request = youtube.commentThreads().list(part="snippet", videoId=video_id, textFormat="plainText", maxResults=100)
        while request:
            response = request.execute()
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
            request = youtube.commentThreads().list_next(request, response)
        return comments
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content}")
        if e.resp.status == 403:
            print("This might be because comments are disabled for the video or your API key has an issue.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def analyze_comments_vader(comments):
    """
    Performs a fast, initial sentiment classification using VADER.
    """
    analyzer = SentimentIntensityAnalyzer()
    analysis_results = {'positive_comments': [], 'negative_comments': [], 'neutral_comments': []}
    for comment in comments:
        scores = analyzer.polarity_scores(comment)
        compound_score = scores['compound']
        sentiment_data = {'text': comment, 'score': compound_score}
        if compound_score >= 0.05:
            analysis_results['positive_comments'].append(sentiment_data)
        elif compound_score <= -0.05:
            analysis_results['negative_comments'].append(sentiment_data)
        else:
            analysis_results['neutral_comments'].append(sentiment_data)
    return analysis_results

def summarize_with_ollama(comments, category):
    """
    Uses a local Ollama model to summarize the key themes in a list of comments.
    """
    if not comments:
        return f"<p>No {category} comments to analyze.</p>"
    comment_texts = "\n".join([f"- \"{c['text']}\"" for c in comments[:30]])
    prompt = (
        f"You are a TECHNO DJ analyst. Based ONLY on the following '{category}' comments, "
        "summarize the key themes in 2-3 concise bullet points. Format the output as simple HTML bullet points using <ul> and <li> tags.\n\n"
        f"Comments:\n{comment_texts}"
    )
    payload = { "model": OLLAMA_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False }
    headers = {"Content-Type": "application/json"}
    try:
        print(f"  > Sending {category} comments to local Ollama model for summarization...")
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result['message']['content']
    except Exception as e:
        return f"<p class='text-red-400'>Error communicating with Ollama: {e}</p>"

def generate_strategic_conclusion(total_comments, results, summaries):
    """
    Uses Ollama to generate strategic advice based on the full analysis.
    """
    print("  > Sending full report to local Ollama model for strategic conclusion...")
    num_positive = len(results['positive_comments'])
    num_negative = len(results['negative_comments'])
    clean_positive_summary = re.sub('<[^<]+?>', '', summaries['positive'])
    clean_negative_summary = re.sub('<[^<]+?>', '', summaries['negative'])
    prompt = (
        "You are a Techno DJ strategy consultant. I will provide you with a sentiment analysis report for a video. "
        "Your task is to provide 3-4 actionable, strategic suggestions for the mathame dj duo. "
        "Focus on what's working (strengths to double down on), what isn't (weaknesses to address), and how they can improve audience engagement or content strategy. "
        "Format the output as an HTML unordered list (`<ul>` and `<li>` tags).\n\n"
        f"--- ANALYSIS REPORT ---\n"
        f"Total Comments: {total_comments}\n"
        f"Positive Comments: {num_positive} ({num_positive/total_comments:.1%})\n"
        f"Negative Comments: {num_negative} ({num_negative/total_comments:.1%})\n\n"
        f"Key Positive Themes:\n{clean_positive_summary}\n\n"
        f"Key Negative Themes:\n{clean_negative_summary}\n"
        "--- END REPORT ---\n\n"
        "Strategic Suggestions:"
    )
    payload = { "model": OLLAMA_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result['message']['content']
    except Exception as e:
        return f"<p class='text-red-400'>Failed to generate strategic conclusion: {e}</p>"

def extract_insights(positive_comments):
    """
    Extracts timestamps and common keywords from positive comments.
    """
    timestamp_regex = re.compile(r'\b(\d{1,2}:\d{2}(?::\d{2})?)\b')
    all_timestamp_mentions = [] 
    all_positive_text = ""
    for comment_data in positive_comments:
        text = comment_data['text']
        all_positive_text += text + " "
        found_stamps = timestamp_regex.findall(text)
        for ts in found_stamps:
            all_timestamp_mentions.append({'timestamp': ts, 'comment': text})
    unique_timestamps_map = {}
    for mention in all_timestamp_mentions:
        ts = mention['timestamp']
        if ts not in unique_timestamps_map:
            unique_timestamps_map[ts] = {'comment': mention['comment'], 'count': 0}
        unique_timestamps_map[ts]['count'] += 1
    top_timestamps_with_comments = []
    for ts, data in unique_timestamps_map.items():
        top_timestamps_with_comments.append({'timestamp': ts, 'comment': data['comment'], 'count': data['count']})
    top_timestamps_with_comments.sort(key=lambda x: x['count'], reverse=True)
    top_timestamps_with_comments = top_timestamps_with_comments[:5]
    words = re.findall(r'\b\w{4,}\b', all_positive_text.lower())
    stopwords = set(['this', 'that', 'with', 'what', 'from', 'your', 'have', 'just', 'like', 'love', 'video'])
    meaningful_words = [word for word in words if word not in stopwords]
    return {
        'top_timestamps_with_comments': top_timestamps_with_comments,
        'top_keywords': Counter(meaningful_words).most_common(10)
    }

def update_landing_page(video_id, video_title):
    """
    Creates or updates a landing page (reports.html) by managing a JSON database of all generated reports.
    """
    json_db_file = "reports_data.json"
    reports = {}
    if os.path.exists(json_db_file):
        with open(json_db_file, "r", encoding="utf-8") as f:
            try: reports = json.load(f)
            except json.JSONDecodeError: reports = {}
    
    # Only add the new report if a video_id is provided
    if video_id and video_title:
        reports[video_id] = video_title
    
    with open(json_db_file, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=4)
    print(f"✅ Reports database updated: {json_db_file}")

def generate_html_report(video_id, video_details, total_comments, results, insights, summaries, strategic_conclusion):
    """
    Generates a visually stunning HTML report from the analysis data.
    """
    report_file_name = f"{video_id}_report.html"
    num_positive = len(results['positive_comments'])
    num_negative = len(results['negative_comments'])
    num_neutral = len(results['neutral_comments'])
    top_3_positive = sorted(results['positive_comments'], key=lambda x: x['score'], reverse=True)[:3]
    top_3_negative = sorted(results['negative_comments'], key=lambda x: x['score'])[:3]
    positive_scores = [c['score'] for c in results['positive_comments']]
    negative_scores = [c['score'] for c in results['negative_comments']]
    score_bins = {'-1.0 to -0.6': 0, '-0.6 to -0.2': 0, 'Neutral (-0.2 to 0.2)': num_neutral, '0.2 to 0.6': 0, '0.6 to 1.0': 0}
    for score in negative_scores:
        if -1.0 <= score < -0.6: score_bins['-1.0 to -0.6'] += 1
        elif -0.6 <= score < -0.2: score_bins['-0.6 to -0.2'] += 1
    for score in positive_scores:
        if 0.2 <= score < 0.6: score_bins['0.2 to 0.6'] += 1
        elif 0.6 <= score <= 1.0: score_bins['0.6 to 1.0'] += 1
    chart_labels = list(score_bins.keys())
    chart_data = list(score_bins.values())
    def create_reaction_html(comments, positive=True):
        if not comments: return f"<li>Nessuna reazione {'positiva' if positive else 'negativa'} significativa trovata.</li>"
        return "".join([f"""<li class="mb-2"><span class="font-semibold text-white">(Punteggio: {c['score']:.2f})</span> "{c['text'][:100]}{'...' if len(c['text']) > 100 else ''}"</li>""" for c in comments])
    def create_insights_list_html(items, not_found_text):
        if not items: return f"<li>{not_found_text}</li>"
        return "".join([f"""<li class="mb-2"><span class="font-bold text-teal-400">{item['timestamp']}</span> (menzionato {item['count']} volte)</li>""" for item in items])
    def create_keywords_html(keywords):
        if not keywords: return "<p>Nessuna parola chiave prominente trovata.</p>"
        max_count = keywords[0][1] if keywords else 1
        tags = []
        for keyword, count in keywords:
            size_class = 'text-lg'
            if count > max_count * 0.7: size_class = 'text-2xl'
            elif count > max_count * 0.4: size_class = 'text-xl'
            tags.append(f'<span class="inline-block bg-gray-700 text-teal-300 rounded-full px-3 py-1 m-1 font-semibold {size_class}">{keyword}</span>')
        return "".join(tags)
    moment_players_html = []
    for i, moment in enumerate(insights['top_timestamps_with_comments']):
        parts = moment['timestamp'].split(':')
        start_seconds = 0
        if len(parts) == 2: start_seconds = int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3: start_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        moment_players_html.append(f"""
        <div class="card p-4 rounded-lg">
            <div class="w-full aspect-w-16 aspect-h-9 mx-auto mb-3">
                <iframe src="https://www.youtube.com/embed/{video_id}?start={start_seconds}&controls=1&rel=0&modestbranding=1" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
            </div>
            <p class="text-gray-400 text-sm italic text-center">"{moment['comment'][:150]}..." <span class="text-teal-400 font-semibold">({moment['timestamp']})</span></p>
        </div>
        """)
    moment_players_html_str = "\n".join(moment_players_html)
    html_template = f"""
    <!DOCTYPE html>
    <html lang="it">
    <head>
        <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Report di Analisi del Sentiment di YouTube</title>
        <script src="https://cdn.tailwindcss.com"></script><script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Inter', sans-serif; background-color: #000000; color: #d1d5db; }}
            .hero-gradient {{ background-color: #1f2937; }} .card {{ background-color: #1f2937; border: 1px solid #374151; }}
            .section-title {{ border-bottom: 2px solid #14b8a6; padding-bottom: 8px; }} .chart-container {{ height: 400px; width: 100%; }}
            .aspect-w-16 {{ position: relative; width: 100%; }} .aspect-h-9 {{ padding-bottom: 56.25%; }}
            .aspect-w-16 iframe, .aspect-w-16 > div {{ position: absolute; width: 100%; height: 100%; top: 0; left: 0; }}
        </style>
    </head>
    <body class="antialiased">
        <header class="bg-black/80 backdrop-blur-md shadow-lg sticky top-0 z-50"><div class="container mx-auto px-6 py-3"><h1 class="text-xl font-bold text-white">Report di Analisi del Sentiment di YouTube</h1></div></header>
        <main>
            <section class="text-center py-20 px-6 hero-gradient">
                <p class="text-teal-400 font-semibold">{video_details['channelTitle']}</p>
                <h2 class="text-4xl md:text-5xl font-extrabold text-white mb-4 tracking-tight">{video_details['title']}</h2>
                <div class="max-w-4xl mx-auto my-8"><div class="w-full aspect-w-16 aspect-h-9"><iframe src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen class="rounded-lg shadow-2xl"></iframe></div></div>
                <p class="text-lg text-gray-400 max-w-3xl mx-auto">Un'analisi del sentiment basata sull'IA di {total_comments} commenti di YouTube.</p>
            </section>
            <div class="container mx-auto px-6 py-16">
                <section id="statistics" class="mb-16">
                    <h3 class="text-4xl font-bold text-center mb-10 section-title text-white">Statistiche Visive</h3>
                    <div class="grid md:grid-cols-5 gap-8">
                        <div class="md:col-span-2 card p-6 rounded-lg"><h4 class="text-xl font-bold text-center text-teal-400 mb-4">Ripartizione del Sentiment</h4><div class="chart-container mx-auto" style="height: 350px;"><canvas id="sentimentDoughnutChart"></canvas></div></div>
                        <div class="md:col-span-3 card p-6 rounded-lg"><h4 class="text-xl font-bold text-center text-teal-400 mb-4">Distribuzione del Punteggio di Sentiment</h4><div class="chart-container mx-auto" style="height: 350px;"><canvas id="scoreDistributionBarChart"></canvas></div></div>
                    </div>
                </section>
                <section id="summaries" class="mb-16">
                    <h3 class="text-4xl font-bold text-center mb-10 section-title text-white">Riepiloghi Tematici Generati dall'IA</h3>
                    <div class="grid md:grid-cols-3 gap-8">
                        <div class="card p-6 rounded-lg"><h4 class="text-xl font-bold text-green-400 mb-4">Temi Positivi</h4><div class="text-gray-300 space-y-2">{summaries['positive']}</div></div>
                        <div class="card p-6 rounded-lg"><h4 class="text-xl font-bold text-red-400 mb-4">Temi Negativi</h4><div class="text-gray-300 space-y-2">{summaries['negative']}</div></div>
                        <div class="card p-6 rounded-lg"><h4 class="text-xl font-bold text-gray-400 mb-4">Temi Neutrali</h4><div class="text-gray-300 space-y-2">{summaries['neutral']}</div></div>
                    </div>
                </section>
                <section id="insights" class="mb-16">
                     <h3 class="text-4xl font-bold text-center mb-10 section-title text-white">Approfondimenti</h3>
                     <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                        <div class="card p-6 rounded-lg"><h4 class="text-xl font-bold text-teal-400 mb-3">❤️ Le 3 Migliori Reazioni Positive</h4><ul class="text-gray-400 space-y-2">{create_reaction_html(top_3_positive, positive=True)}</ul></div>
                        <div class="card p-6 rounded-lg"><h4 class="text-xl font-bold text-teal-400 mb-3">💔 Le 3 Migliori Reazioni Negative</h4><ul class="text-gray-400 space-y-2">{create_reaction_html(top_3_negative, positive=False)}</ul></div>
                        <div class="card p-6 rounded-lg lg:col-span-1"><h4 class="text-xl font-bold text-teal-400 mb-3">⏱️ Momenti Migliori (Timestamp)</h4><ul class="text-gray-400 space-y-2">{create_insights_list_html(insights['top_timestamps_with_comments'], 'Nessun timestamp trovato.')}</ul></div>
                     </div>
                     <div class="card p-8 rounded-lg mt-8"><h4 class="text-xl font-bold text-teal-400 mb-4 text-center">Argomenti Caldi nei Commenti Positivi</h4><div class="text-center">{create_keywords_html(insights['top_keywords'])}</div></div>
                </section>
                <section id="comment-moments" class="mb-16">
                    <h3 class="text-4xl font-bold text-center mb-10 section-title text-white">Momenti Chiave Riferiti nei Commenti</h3>
                    <div id="moments-container" class="grid md:grid-cols-2 gap-8">{moment_players_html_str}</div>
                </section>
                <section id="conclusion">
                     <h3 class="text-4xl font-bold text-center mb-10 section-title text-white">Conclusioni Strategiche e Suggerimenti</h3>
                     <div class="max-w-4xl mx-auto card p-8 rounded-lg"><div class="text-gray-300 space-y-3">{strategic_conclusion}</div></div>
                </section>
            </div>
        </main>
        <footer class="bg-gray-900 text-white mt-16"><div class="container mx-auto px-6 py-8 text-center"><p class="text-gray-500">&copy; 2025 Analisi del Sentiment IA. Report generato localmente.</p></div></footer>
        <script>
            const doughnutCtx = document.getElementById('sentimentDoughnutChart').getContext('2d');
            const barCtx = document.getElementById('scoreDistributionBarChart').getContext('2d');
            Chart.defaults.color = '#d1d5db'; Chart.defaults.font.family = "'Inter', sans-serif";
            new Chart(doughnutCtx, {{ type: 'doughnut', data: {{ labels: ['Positivo', 'Negativo', 'Neutrale'], datasets: [{{ label: 'Ripartizione del Sentiment', data: [{num_positive}, {num_negative}, {num_neutral}], backgroundColor: ['#2dd4bf', '#f87171', '#9ca3af'], borderColor: '#1f2937', borderWidth: 4, hoverOffset: 8 }}] }}, options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ position: 'bottom', labels: {{ padding: 20, font: {{ size: 14 }} }} }} }} }} }});
            new Chart(barCtx, {{ type: 'bar', data: {{ labels: {json.dumps(chart_labels)}, datasets: [{{ label: 'Numero di Commenti', data: {json.dumps(chart_data)}, backgroundColor: [ 'rgba(248, 113, 113, 0.6)', 'rgba(251, 146, 60, 0.6)', 'rgba(156, 163, 175, 0.6)', 'rgba(52, 211, 153, 0.6)', 'rgba(45, 212, 191, 0.6)', ], borderColor: [ '#f87171', '#fb923c', '#9ca3af', '#34d399', '#2dd4bf' ], borderWidth: 2 }}] }}, options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ beginAtZero: true, grid: {{ color: '#374151' }}, ticks: {{ precision: 0 }} }}, x: {{ grid: {{ display: false }} }} }}, plugins: {{ legend: {{ display: false }} }} }} }});
        </script>
    </body>
    </html>
    """
    with open(report_file_name, "w", encoding='utf-8') as f:
        f.write(html_template)
    print(f"\n✅ HTML report generated: {report_file_name}")
    return report_file_name

