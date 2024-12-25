class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.paths = set()

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, path: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.paths.add(path)
        node.is_end = True

    def search(self, prefix: str) -> list:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return list(node.paths)
