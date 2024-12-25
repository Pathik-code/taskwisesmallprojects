# Advanced File Search Engine

## Overview

The Advanced File Search Engine is a Python-based tool designed to efficiently search files within directories by their names, extensions, or metadata. It leverages advanced data structures such as Tries for prefix searches and Graphs for directory traversal, along with regex-based search capabilities.

---

## Features

1. **Prefix-based File Search:**

   - Implements a Trie data structure to allow fast searching of files by name prefixes.

2. **Regex-based File Search:**

   - Supports advanced search patterns using regular expressions to locate files based on names or extensions.

3. **Graph-based Directory Traversal:**

   - Uses BFS and DFS algorithms to traverse the directory structure and list files.

4. **File Metadata Retrieval:**

   - Extracts and displays metadata, including file size, extension, and last modified date.

5. **Scalable Design:**

   - Modular components for indexing files, searching, and traversing directories, making it suitable for large-scale use cases.

---

## Project Structure

```plaintext
file_search_engine/
├── search_engine.py    # Main module to execute the program
├── trie.py             # Module for Trie implementation
├── graph.py            # Module for graph traversal logic
├── file_utils.py       # Module for file operations and metadata handling
└── README.md           # Documentation
```

---

## Getting Started

### Prerequisites

- Python 3.x
- Basic knowledge of file systems and Python modules

### Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd file_search_engine
   ```

2. Ensure Python is installed:

   ```bash
   python3 --version
   ```

---

## Usage

### 1. Index Files

Run the main script to index files within the specified root directory.

```bash
python3 search_engine.py
```

### 2. Search by Name

Search files starting with a specific prefix:

```python
search_engine.search_by_name("test")
```

### 3. Regex Search

Search files matching a regex pattern:

```python
search_engine.search_by_metadata(r".+\.txt")
```

### 4. Directory Traversal

Perform BFS or DFS traversal of directories:

```python
search_engine.traverse_directory(method="bfs")
```

---

## Implementation Details

### 1. **Trie**

- Efficient prefix-based searching of file names.
- Files matching the prefix are stored as a list in each Trie node.

### 2. **Graph Traversal**

- BFS and DFS algorithms to explore directory structures and retrieve file paths.
- Built using adjacency lists.

### 3. **File Metadata Handling**

- Retrieves file metadata such as name, extension, size, and last modified timestamp.
- Regex matching for complex search patterns.

---

## Example

```plaintext
Root Directory: /example_dir
Files:
- test1.txt
- test2.py
- example.txt

Search Prefix: "test"
Output:
- /example_dir/test1.txt
- /example_dir/test2.py

Regex Pattern: ".+\.txt"
Output:
- /example_dir/test1.txt
- /example_dir/example.txt
```

---

## Future Enhancements

1. Add support for indexing large files with a database.
2. Include a web-based interface for user-friendly searches.
3. Optimize performance for real-time updates to the directory.
4. Add parallel processing for faster indexing.

---

## Contributors

- Pathik ([Your GitHub](https://github.com/Pathik-code))
