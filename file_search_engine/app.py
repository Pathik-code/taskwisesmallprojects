from flask import Flask, request, jsonify, render_template
from search_engine import FileSearchEngine
import time
from datetime import datetime, timedelta

app = Flask(__name__)
engine = FileSearchEngine("/")  # Set your root directory here

@app.route('/')
def index():
    return render_template('search.html')

@app.route('/search')

def search():
    method = request.args.get('method', 'name')
    query = request.args.get('query', '')
    use_index = request.args.get('useIndex', 'true') == 'true'
    compare = request.args.get('comparePerformance', 'false') == 'true'

    if not use_index:
        engine.index_files()

    performance_data = {}
    results = []

    if compare:
        # Measure performance without index
        start_time = time.time()
        results_no_index = perform_search(method, query, False)
        performance_data['noIndex'] = (time.time() - start_time) * 1000

        # Measure performance with index
        start_time = time.time()
        index_time = time.time()
        if use_index:
            engine.index_files()
        index_time = (time.time() - index_time) * 1000

        search_time = time.time()
        results = perform_search(method, query, True)
        search_time = (time.time() - search_time) * 1000

        performance_data['withIndex'] = {
            'total': index_time + search_time,
            'index': index_time,
            'search': search_time
        }
    else:
        results = perform_search(method, query, use_index)

    return jsonify({
        'results': results,
        'performance': performance_data if compare else None
    })

def perform_search(method, query, use_index):
    if method == 'name':
        return engine.search_by_name(query)
    elif method == 'date':
        try:
            days = int(query)
            start_date = datetime.now() - timedelta(days=days)
            return engine.search_by_date(start_date=start_date)
        except ValueError:
            return []
    elif method == 'size':
        try:
            size_mb = float(query)
            return engine.search_by_size(min_size=size_mb * 1_000_000)
        except ValueError:
            return []
    elif method == 'type':
        return engine.search_by_type(query)
    elif method == 'regex':
        return engine.search_by_metadata(query)
    return []

if __name__ == '__main__':
    app.run(debug=True)
