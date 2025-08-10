import tkinter as tk
from tkinter import ttk, scrolledtext
import csv
import json
from pathlib import Path
import traceback
import re

# --- Configuration ---
# The script is in D:\RED\LangOverride, and the data files are in D:\RED.
# So, we need to go one directory up from the script's location.
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent

CSV_EN_PATH = DATA_DIR / "English.csv"
CSV_JA_PATH = DATA_DIR / "Japanese.csv"
JSON_OVERRIDE_PATH = DATA_DIR / "languageOverrides.json"


class LanguageToolApp:
    def __init__(self, root):
        """Initializes the GUI application."""
        self.root = root
        self.root.title("Language File Updater")
        self.root.geometry("600x450") # Slightly taller for better log view
        self.root.resizable(False, True)

        # Create and place widgets
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.update_button = ttk.Button(
            self.main_frame, text="Update Language CSVs", command=self.run_update_process
        )
        self.update_button.pack(pady=5, fill=tk.X)

        self.log_area = scrolledtext.ScrolledText(
            self.main_frame, wrap=tk.WORD, state="disabled", height=20
        )
        self.log_area.pack(pady=5, fill=tk.BOTH, expand=True)
        
        # Configure text tags for colored logs
        self.log_area.tag_config("INFO", foreground="black")
        self.log_area.tag_config("SUCCESS", foreground="green")
        self.log_area.tag_config("ERROR", foreground="red")
        self.log_area.tag_config("WARN", foreground="orange")
        
        self.log("INFO", "Ready. Click the button to start the update process.")

    def log(self, level, message):
        """Adds a message to the log area with a specific color."""
        self.log_area.config(state="normal")
        self.log_area.insert(tk.END, f"{message}\n", level)
        self.log_area.config(state="disabled")
        self.log_area.see(tk.END) # Auto-scroll to the bottom
        self.root.update_idletasks() # Refresh the GUI immediately

    def run_update_process(self):
        """The main logic for updating the files."""
        self.update_button.config(state="disabled")
        self.log_area.config(state="normal")
        self.log_area.delete(1.0, tk.END) # Clear previous logs
        self.log_area.config(state="disabled")

        try:
            self.log("INFO", "--- Starting update process ---")

            # 1. Read CSV files into dictionaries for fast lookup
            self.log("INFO", f"Reading {CSV_EN_PATH.name}...")
            en_dict = self.read_csv_to_dict(CSV_EN_PATH)
            self.log("INFO", f" -> Found {len(en_dict)} entries.")

            self.log("INFO", f"Reading {CSV_JA_PATH.name}...")
            ja_dict = self.read_csv_to_dict(CSV_JA_PATH)
            self.log("INFO", f" -> Found {len(ja_dict)} entries.")

            # 2. Read and FIX the override JSON
            self.log("INFO", f"Reading and fixing {JSON_OVERRIDE_PATH.name}...")
            with open(JSON_OVERRIDE_PATH, "r", encoding="utf-8") as f:
                broken_json_string = f.read()

            # --- The Magic Regex Fix ---
            def fix_newlines_in_text(match):
                content_with_newlines = match.group(1)
                # Replace actual newlines with escaped '\n'
                fixed_content = content_with_newlines.replace('\n', '\\n').replace('\r', '')
                return f'"text": "{fixed_content}"'

            # Find all "text": "..." blocks, even multi-line ones, and apply the fix
            fixed_json_string = re.sub(r'"text":\s*"((?:\\"|[^"])*)"', fix_newlines_in_text, broken_json_string, flags=re.DOTALL)
            
            try:
                # Parse the FIXED string
                override_data = json.loads(fixed_json_string)
                self.log("INFO", " -> JSON fixed and parsed successfully.")
            except json.JSONDecodeError as e:
                self.log("ERROR", f"Failed to parse JSON even after regex fix: {e}")
                char_index = e.pos
                context = 30
                snippet = fixed_json_string[max(0, char_index - context):char_index + context]
                self.log("WARN", f" -> Problem area (around char {char_index}): ...{snippet}...")
                raise # Stop the process

            # 3. Apply overrides
            self.log("INFO", "Applying English overrides...")
            en_overrides = override_data.get("languageOverridesConfig", {}).get("overrides", {}).get("English", {}).get("overrideEntries", [])
            updated_en_count = self.apply_overrides(en_dict, en_overrides)
            self.log("INFO", f" -> {updated_en_count} English entries updated or added.")

            self.log("INFO", "Applying Japanese overrides...")
            ja_overrides = override_data.get("languageOverridesConfig", {}).get("overrides", {}).get("Japanese", {}).get("overrideEntries", [])
            updated_ja_count = self.apply_overrides(ja_dict, ja_overrides)
            self.log("INFO", f" -> {updated_ja_count} Japanese entries updated or added.")

            # 4. Write the updated dictionaries back to CSV files
            self.log("INFO", f"Writing updated data to {CSV_EN_PATH.name}...")
            self.write_dict_to_csv(CSV_EN_PATH, en_dict)
            self.log("INFO", " -> English file saved.")
            
            self.log("INFO", f"Writing updated data to {CSV_JA_PATH.name}...")
            self.write_dict_to_csv(CSV_JA_PATH, ja_dict)
            self.log("INFO", " -> Japanese file saved.")

            self.log("SUCCESS", "\n--- Update process completed successfully! ---")

        except Exception as e:
            self.log("ERROR", f"\n--- AN ERROR OCCURRED ---")
            self.log("ERROR", f"Error Type: {type(e).__name__}")
            self.log("ERROR", f"Message: {e}")
            self.log("WARN", "\n--- Traceback ---")
            self.log("WARN", traceback.format_exc())
        
        finally:
            self.update_button.config(state="normal")
    
    def read_csv_to_dict(self, file_path):
        """Reads a 2-column CSV file into a dictionary, using UPPERCASE keys."""
        data_dict = {}
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                # Read the header to know the column names
                header = [h.upper() for h in next(reader)] 
            except StopIteration:
                return {} # Return empty dict for empty file
            
            # Find the index of KEY and TEXT columns
            try:
                key_index = header.index('KEY')
                text_index = header.index('TEXT')
            except ValueError:
                self.log("ERROR", f"CSV file {file_path.name} must have 'KEY' and 'TEXT' columns.")
                raise
                
            for row in reader:
                if len(row) > max(key_index, text_index):
                    data_dict[row[key_index]] = row[text_index]
        return data_dict

    def apply_overrides(self, data_dict, override_list):
        """Updates a dictionary with a list of override entries."""
        count = 0
        if override_list:
            for entry in override_list:
                # The JSON uses "key", but our dictionary uses the UPPERCASE "KEY"
                # We use the JSON's "key" to find and update the entry in our dict.
                data_dict[entry["key"]] = entry["text"]
                count += 1
        return count

    def write_dict_to_csv(self, file_path, data_dict):
        """Writes a dictionary back to a 2-column CSV file."""
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["KEY", "TEXT"]) # Write header
            for key, value in data_dict.items():
                writer.writerow([key, value])


if __name__ == "__main__":
    root = tk.Tk()
    app = LanguageToolApp(root)
    root.mainloop()