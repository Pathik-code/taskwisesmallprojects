from trie import Trie
from graph import FileGraph
from file_utils import get_file_metadata, search_by_regex
import os
import time
from datetime import datetime, timedelta

class FileSearchEngine:
    def __init__(self, root_dir):
        self.trie = Trie()
        self.graph = FileGraph()
        self.root_dir = root_dir
        self.metadata_cache = {}
        self.last_indexed = None

    def index_files(self):
        """Index all files in the root directory."""
        self.graph.build_graph(self.root_dir)
        self.metadata_cache.clear()

        for dirpath, filenames in self.graph.adjacency_list.items():
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                self.trie.insert(filename.lower(), file_path)
                self.metadata_cache[file_path] = get_file_metadata(file_path)

        self.last_indexed = datetime.now()

    def search_by_name(self, prefix, case_sensitive=False):
        """Search files by name prefix."""
        if not case_sensitive:
            prefix = prefix.lower()
        results = self.trie.search(prefix)
        return [self.metadata_cache[path] for path in results]

    def search_by_date(self, start_date=None, end_date=None, date_type='modified'):
        """Search files by date range."""
        results = []
        for metadata in self.metadata_cache.values():
            file_date = datetime.fromisoformat(metadata[date_type])
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue
            results.append(metadata)
        return results

    def search_by_size(self, min_size=None, max_size=None):
        """Search files by size range (in bytes)."""
        results = []
        for metadata in self.metadata_cache.values():
            size = metadata['size']
            if min_size and size < min_size:
                continue
            if max_size and size > max_size:
                continue
            results.append(metadata)
        return results

    def search_by_type(self, extension):
        """Search files by extension."""
        if not extension.startswith('.'):
            extension = '.' + extension
        extension = extension.lower()
        return [
            metadata for metadata in self.metadata_cache.values()
            if metadata['extension'] == extension
        ]

    def search_by_metadata(self, regex_pattern):
        return search_by_regex(self.root_dir, regex_pattern)

    def search_by_path(self, dir_name, file_name):
        """Search for files in specific directory matching the file name"""
        results = []
        for dirpath, filenames in self.graph.adjacency_list.items():
            if dir_name.lower() in dirpath.lower():
                for filename in filenames:
                    if file_name.lower() in filename.lower():
                        results.append(os.path.join(dirpath, filename))
        return results

    def traverse_directory(self, method="bfs"):
        if method == "bfs":
            return self.graph.bfs(self.root_dir)
        elif method == "dfs":
            return self.graph.dfs(self.root_dir)

if __name__ == "__main__":
    # Example usage
    engine = FileSearchEngine("/path/to/search")
    engine.index_files()

    # Search examples
    print("\nSearch by name:")
    results = engine.search_by_name("test")
    for r in results[:5]: print(f"- {r['name']}")

    print("\nSearch by date (last 7 days):")
    week_ago = datetime.now() - timedelta(days=7)
    results = engine.search_by_date(start_date=week_ago)
    for r in results[:5]: print(f"- {r['name']} ({r['modified']})")

    print("\nSearch by size (1MB to 10MB):")
    results = engine.search_by_size(1_000_000, 10_000_000)
    for r in results[:5]: print(f"- {r['name']} ({r['size']/1_000_000:.1f}MB)")

    print("\nSearch Python files:")
    results = engine.search_by_type('.py')
    for r in results[:5]: print(f"- {r['name']}")
