import argparse
import time
from search_engine import FileSearchEngine
from datetime import datetime, timedelta

def format_results(results, time_taken):

    output = f"\nFound {len(results)} results in {time_taken:.4f} seconds:\n"
    for r in results:
        output += f"- {r['name']} ({r['path']})\n"
        output += f"  Size: {r['size']/1024:.1f}KB, Modified: {r['modified']}\n"
    return output

def main():
    parser = argparse.ArgumentParser(description='File Search Engine CLI')
    parser.add_argument('--root', '-r', required=True, help='Root directory to search')
    parser.add_argument('--method', '-m', choices=['name', 'date', 'size', 'type', 'regex'],
                       help='Search method')
    parser.add_argument('--query', '-q', help='Search query')
    parser.add_argument('--no-index', action='store_true',
                       help='Search without indexing (slower but real-time)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare search times with and without indexing')

    args = parser.parse_args()

    engine = FileSearchEngine(args.root)

    if args.compare:
        # Compare search times
        print(f"\nComparing search times for query: {args.query}")

        # Without indexing
        start_time = time.time()
        results_no_index = engine.search_by_metadata(f".*{args.query}.*")
        time_no_index = time.time() - start_time

        # With indexing
        start_time = time.time()
        engine.index_files()
        index_time = time.time() - start_time

        start_time = time.time()
        results_with_index = engine.search_by_name(args.query)
        search_time = time.time() - start_time

        print(f"\nResults:")
        print(f"Without indexing: {time_no_index:.4f} seconds")
        print(f"With indexing: {index_time:.4f} + {search_time:.4f} = {index_time + search_time:.4f} seconds")
        print(f"Found {len(results_with_index)} results")
        return

    if not args.no_index:
        start_time = time.time()
        engine.index_files()
        index_time = time.time() - start_time
        print(f"Indexing completed in {index_time:.4f} seconds")

    start_time = time.time()

    if args.method == 'name':
        results = engine.search_by_name(args.query)
    elif args.method == 'date':
        days = int(args.query)
        start_date = datetime.now() - timedelta(days=days)
        results = engine.search_by_date(start_date=start_date)
    elif args.method == 'size':
        size_mb = float(args.query)
        results = engine.search_by_size(max_size=size_mb * 1024 * 1024)
    elif args.method == 'type':
        results = engine.search_by_type(args.query)
    elif args.method == 'regex':
        results = engine.search_by_metadata(args.query)
    else:
        print("Please specify a search method")
        return

    search_time = time.time() - start_time
    print(format_results(results, search_time))

if __name__ == '__main__':
    main()
