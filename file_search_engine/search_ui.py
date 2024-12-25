import tkinter as tk
from tkinter import ttk, messagebox
from search_engine import FileSearchEngine
import os

class TerminalSearchUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FileSearch Terminal")
        self.root.configure(bg='#1e1e1e')

        # Terminal style configuration
        self.style = ttk.Style()
        self.style.configure('Terminal.TFrame', background='#1e1e1e')
        self.style.configure('Terminal.TLabel',
                           background='#1e1e1e',
                           foreground='#00ff00',
                           font=('Courier', 10))

        # Initialize search engine
        self.current_directory = os.getcwd()
        self.search_engine = FileSearchEngine(self.current_directory)
        self.search_engine.index_files()

        self._create_terminal_ui()

    def _create_terminal_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, style='Terminal.TFrame')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Output area
        self.output_area = tk.Text(main_frame,
                                 bg='#1e1e1e',
                                 fg='#00ff00',
                                 font=('Courier', 10),
                                 insertbackground='#00ff00')
        self.output_area.pack(fill='both', expand=True)
        self.output_area.configure(state='disabled')

        # Command input frame
        cmd_frame = ttk.Frame(main_frame, style='Terminal.TFrame')
        cmd_frame.pack(fill='x', pady=(5, 0))

        # Command prompt
        prompt_label = ttk.Label(cmd_frame,
                               text="fsearch> ",
                               style='Terminal.TLabel')
        prompt_label.pack(side='left')

        # Command entry
        self.cmd_entry = tk.Entry(cmd_frame,
                                bg='#1e1e1e',
                                fg='#00ff00',
                                insertbackground='#00ff00',
                                font=('Courier', 10),
                                bd=0)
        self.cmd_entry.pack(side='left', fill='x', expand=True)
        self.cmd_entry.bind('<Return>', self._handle_command)

        # Display welcome message and help
        self._display_welcome()

    def _handle_command(self, event):
        command = self.cmd_entry.get()
        self.cmd_entry.delete(0, tk.END)

        if not command:
            return

        # Log command
        self._append_output(f"\nfsearch> {command}\n")

        # Parse command
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        # Handle commands
        if cmd == 'help':
            self._show_help()
        elif cmd == 'cd':
            self._change_directory(args)
        elif cmd == 'find':
            self._find_files(args)
        elif cmd == 'search':
            self._search_content(args)
        elif cmd == 'ls':
            self._list_directory()
        elif cmd == 'clear':
            self._clear_output()
        elif cmd == 'exit':
            self.root.quit()
        else:
            self._append_output("Error: Unknown command. Type 'help' for available commands.\n")

    def _display_welcome(self):
        welcome_msg = """
╔════════════════════════════════════════════╗
║           File Search Terminal             ║
╚════════════════════════════════════════════╝

Type 'help' for available commands.
Current directory: {}

""".format(self.current_directory)
        self._append_output(welcome_msg)

    def _show_help(self):
        help_text = """
Available Commands:
------------------
help              Show this help message
cd <path>         Change current directory
find <pattern>    Find files by name pattern
search <regex>    Search files by content
ls                List current directory
clear             Clear terminal output
exit              Exit the application

Examples:
---------
find *.py         Find all Python files
search "TODO:"    Search for TODO comments
cd ..             Go to parent directory
"""
        self._append_output(help_text)

    def _append_output(self, text):
        self.output_area.configure(state='normal')
        self.output_area.insert(tk.END, text)
        self.output_area.see(tk.END)
        self.output_area.configure(state='disabled')

    # ... rest of the implementation methods ...

if __name__ == "__main__":
    root = tk.Tk()
    app = TerminalSearchUI(root)
    root.mainloop()
