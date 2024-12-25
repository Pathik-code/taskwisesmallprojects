import os
from collections import deque, defaultdict

class FileGraph:
    def __init__(self):
        self.adjacency_list = defaultdict(list)

    def build_graph(self, root_dir: str) -> None:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            self.adjacency_list[dirpath].extend(filenames)
            for dirname in dirnames:
                full_path = os.path.join(dirpath, dirname)
                self.adjacency_list[dirpath].append(dirname)

    def bfs(self, start_dir: str) -> list:
        visited = set()
        queue = deque([start_dir])
        result = []

        while queue:
            current_dir = queue.popleft()
            if current_dir not in visited:
                visited.add(current_dir)
                result.extend([os.path.join(current_dir, f)
                             for f in self.adjacency_list[current_dir]])

                # Add subdirectories to queue
                for item in self.adjacency_list[current_dir]:
                    full_path = os.path.join(current_dir, item)
                    if os.path.isdir(full_path):
                        queue.append(full_path)

        return result

    def dfs(self, start_dir: str) -> list:
        visited = set()
        result = []

        def dfs_recursive(current_dir):
            if current_dir not in visited:
                visited.add(current_dir)
                result.extend([os.path.join(current_dir, f)
                             for f in self.adjacency_list[current_dir]])

                for item in self.adjacency_list[current_dir]:
                    full_path = os.path.join(current_dir, item)

                    if os.path.isdir(full_path):
                        dfs_recursive(full_path)

        dfs_recursive(start_dir)
        return result
