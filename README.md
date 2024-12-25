# ...existing code...

## File Search Engine Features

The File Search Engine provides advanced file searching capabilities:

### Search Methods

1. **Search by Name**
```python
results = engine.search_by_name("test")
```

2. **Search by Date Range**
```python
from datetime import datetime, timedelta
week_ago = datetime.now() - timedelta(days=7)
results = engine.search_by_date(start_date=week_ago)
```

3. **Search by File Size**
```python
# Search files between 1MB and 10MB
results = engine.search_by_size(1_000_000, 10_000_000)
```

4. **Search by File Type**
```python

# Search all Python files
results = engine.search_by_type('.py')
```

5. **Search by Regular Expression**
```python
results = engine.search_by_regex(r".*\.txt$")
```

### Example Script

```python
from file_search_engine import FileSearchEngine

# Initialize engine
engine = FileSearchEngine("/path/to/search")
engine.index_files()

# Search for Python files modified in the last week
week_ago = datetime.now() - timedelta(days=7)
recent_files = engine.search_by_date(start_date=week_ago)
python_files = engine.search_by_type('.py')

# Get intersection of results
recent_python_files = [f for f in recent_files if f in python_files]
```

# ...existing code...
