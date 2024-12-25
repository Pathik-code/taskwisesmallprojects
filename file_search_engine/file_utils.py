import os
import re
import time
from datetime import datetime

def get_file_metadata(file_path):
    """Get detailed file metadata."""
    stats = os.stat(file_path)
    return {
        'name': os.path.basename(file_path),
        'path': file_path,
        'size': stats.st_size,
        'created': datetime.fromtimestamp(stats.st_ctime).isoformat(),
        'modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
        'accessed': datetime.fromtimestamp(stats.st_atime).isoformat(),
        'extension': os.path.splitext(file_path)[1].lower(),
        'is_hidden': os.path.basename(file_path).startswith('.')
    }

def search_by_regex(root_dir, pattern):
    """Search files using regex pattern."""
    matches = []
    regex = re.compile(pattern)

    for root, _, files in os.walk(root_dir):
        for file in files:
            if regex.match(file):
                full_path = os.path.join(root, file)
                matches.append(get_file_metadata(full_path))

    return matches
