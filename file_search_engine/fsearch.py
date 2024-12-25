#!/usr/bin/env python3
import argparse
import sys
from search_engine import FileSearchEngine

def main():
    parser = argparse.ArgumentParser(description='File Search Tool')
    parser.add_argument('-d', '--directory', default='.',
                       help='Directory to search in')
    parser.add_argument('-n', '--name', help='Search by file name pattern')
    parser.add_argument('-r', '--regex', help='Search by regex pattern')
    parser.add_argument('-p', '--path', help='Search by path pattern')
    parser.add_argument('--gui', action='store_true',
                       help='Launch GUI terminal interface')

    args = parser.parse_args()

    if args.gui:
        import tkinter as tk
        from search_ui import TerminalSearchUI
        root = tk.Tk()
        app = TerminalSearchUI(root)
        root.mainloop()
        return

    search_engine = FileSearchEngine(args.directory)
    search_engine.index_files()

    if args.name:
        results = search_engine.search_by_name(args.name)
        print("\nFiles matching name pattern:")
        for result in results:
            print(f"- {result}")

    if args.regex:
        results = search_engine.search_by_metadata(args.regex)
        print("\nFiles matching regex pattern:")
        for result in results:
            print(f"- {result}")

    if args.path:
        if '/' in args.path:
            dir_name, file_name = args.path.rsplit('/', 1)
        else:
            dir_name, file_name = '.', args.path
        results = search_engine.search_by_path(dir_name, file_name)
        print("\nFiles matching path pattern:")
        for result in results:
            print(f"- {result}")

if __name__ == "__main__":
    main()
