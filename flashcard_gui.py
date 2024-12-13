import site, sys
if site.USER_SITE is None:
    site.USER_SITE = sys.prefix

#arch -arm64 pyinstaller flashcard_creator.spec

from image_processor import process_image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext, simpledialog
from PIL import Image, ImageTk

class FlashcardCreatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Flashcard Creator")
        self.root.geometry("800x900")
        
        # State variables
        self.header_assignments = {}
        self.image_paths = []  # Changed to list for multiple images
        self.processed_table = None
        self.header_row = None
        self.heteronym_enabled = tk.BooleanVar(value=False)
        self.is_traditional = tk.BooleanVar(value=False)
        
        self.create_gui()
        self.disable_processing_widgets()
    
    def create_gui(self):
    # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Character type selection
        char_type_frame = ttk.LabelFrame(main_frame, text="Character Type", padding="10")
        char_type_frame.grid(row=0, column=0, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Radiobutton(char_type_frame, 
                        text="Traditional Chinese", 
                        variable=self.is_traditional, 
                        value=True).grid(row=0, column=0, padx=10)
        ttk.Radiobutton(char_type_frame, 
                        text="Simplified Chinese", 
                        variable=self.is_traditional, 
                        value=False).grid(row=0, column=1, padx=10)
        
        # Image selection section
        image_frame = ttk.LabelFrame(main_frame, text="Step 1: Select Images", padding="10")
        image_frame.grid(row=1, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Add buttons for image management
        button_frame = ttk.Frame(image_frame)
        button_frame.grid(row=0, column=0, pady=5)
        
        ttk.Button(button_frame, text="Add Images", command=self.add_images).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_image).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Clear All", command=self.clear_images).grid(row=0, column=2, padx=5)
        
        # Listbox for selected images
        self.image_listbox = tk.Listbox(image_frame, width=70, height=5)
        self.image_listbox.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Processing section
        process_frame = ttk.LabelFrame(main_frame, text="Step 2: Process Images", padding="10")
        process_frame.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Create a frame for processing controls
        process_controls = ttk.Frame(process_frame)
        process_controls.grid(row=0, column=0, pady=5)
        
        self.process_button = ttk.Button(process_controls, text="Process Images", 
                                        command=self.process_images)
        self.process_button.grid(row=0, column=0, padx=5)
        
        # Add heteronym checkbox
        ttk.Checkbutton(process_controls, 
                        text="Enable heteronym selection", 
                        variable=self.heteronym_enabled).grid(row=0, column=1, padx=5)
        
        self.process_label = ttk.Label(process_frame, text="")
        self.process_label.grid(row=1, column=0, padx=10, pady=5)
        
        # Header assignment section
        self.header_frame = ttk.LabelFrame(main_frame, text="Step 3: Assign Attributes", padding="10")
        self.header_frame.grid(row=3, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Left side (Side 1)
        self.side1_list = tk.Listbox(self.header_frame, width=30, height=10)
        self.side1_list.grid(row=0, column=0, padx=5)
        ttk.Label(self.header_frame, text="Side 1").grid(row=1, column=0)
        
        # Buttons frame
        button_frame = ttk.Frame(self.header_frame)
        button_frame.grid(row=0, column=1, padx=10)
        
        self.move_right_btn = ttk.Button(button_frame, text="→", command=lambda: self.move_header("side1", "side2"))
        self.move_right_btn.grid(row=0, column=0, pady=2)
        
        self.move_left_btn = ttk.Button(button_frame, text="←", command=lambda: self.move_header("side2", "side1"))
        self.move_left_btn.grid(row=1, column=0, pady=2)
        
        self.disable_btn = ttk.Button(button_frame, text="Disable", command=self.disable_header)
        self.disable_btn.grid(row=2, column=0, pady=2)
        
        self.enable_btn = ttk.Button(button_frame, text="Re-enable", command=self.enable_header)
        self.enable_btn.grid(row=3, column=0, pady=2)
        
        # Right side (Side 2)
        self.side2_list = tk.Listbox(self.header_frame, width=30, height=10)
        self.side2_list.grid(row=0, column=2, padx=5)
        ttk.Label(self.header_frame, text="Side 2").grid(row=1, column=2)
        
        # Disabled headers
        self.disabled_list = tk.Listbox(self.header_frame, width=30, height=5)
        self.disabled_list.grid(row=2, column=0, columnspan=3, pady=10)
        ttk.Label(self.header_frame, text="Disabled Headers").grid(row=3, column=0, columnspan=3)
        
        # Export section
        export_frame = ttk.LabelFrame(main_frame, text="Step 4: Export", padding="10")
        export_frame.grid(row=4, column=0, pady=10, sticky=(tk.W, tk.E))

        self.quizlet_button = ttk.Button(export_frame, text="Export Flashcards", 
                                        command=self.export_to_quizlet)
        self.quizlet_button.grid(row=0, column=0, padx=5, pady=5)


    def add_images(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        for file in files:
            if file not in self.image_paths:
                self.image_paths.append(file)
                self.image_listbox.insert(tk.END, file)
        
        if self.image_paths:
            self.process_button.state(['!disabled'])
            self.process_label.config(text="Click 'Process Images' to start OCR")

    def remove_image(self):
        selection = self.image_listbox.curselection()
        if selection:
            idx = selection[0]
            self.image_paths.pop(idx)
            self.image_listbox.delete(idx)
            
            if not self.image_paths:
                self.process_button.state(['disabled'])
                self.process_label.config(text="")

    def clear_images(self):
        self.image_paths.clear()
        self.image_listbox.delete(0, tk.END)
        self.process_button.state(['disabled'])
        self.process_label.config(text="")
    

    def prompt_for_heteronym(self, char, heteronyms):
        """Create a dialog for user to select correct pronunciation"""
        options = "\n".join([f"{i+1}. {p}" for i, p in enumerate(heteronyms)])
        while True:
            choice = simpledialog.askstring(
                "Select Pronunciation",
                f"Select the correct pronunciation for '{char}':\n\n{options}\n\nEnter number (1-{len(heteronyms)}):",
                parent=self.root
            )
            
            if choice is None:  # User clicked Cancel
                return heteronyms[0]  # Default to first pronunciation
                
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(heteronyms):
                    return heteronyms[choice_idx]
            except ValueError:
                pass
            
    @staticmethod
    def headers_are_similar(header1, header2):
        """
        Check if two headers are similar enough to be considered the same.
        Very strict matching to only catch minor OCR errors.
        """
        # Convert both to lowercase and remove spaces
        h1 = header1.lower().replace(" ", "")
        h2 = header2.lower().replace(" ", "")
        
        # If they're exactly the same after normalization, return True
        if h1 == h2:
            return True
        
        # First letter must match for headers to be considered similar
        if h1[0] != h2[0]:
            return False
            
        # If one is completely contained in the other, they're likely the same
        if h1 in h2 or h2 in h1:
            # But only if the length difference is very small (1-2 characters)
            return abs(len(h1) - len(h2)) <= 2
        
        # Count differences
        differences = 0
        # If lengths are too different, they're not similar
        if abs(len(h1) - len(h2)) > 2:
            return False
            
        # Compare characters
        for i in range(min(len(h1), len(h2))):
            if h1[i] != h2[i]:
                differences += 1
                if differences > 2:  # More than 2 character differences
                    return False
                    
        return True
    
    def process_images(self):
        if not self.image_paths:
            return
            
        try:
            self.process_label.config(text="Processing images... This may take a bit...")
            self.root.update()
            
            # Process each image and collect results
            all_tables = []
            all_headers = set()
            
            for img_path in self.image_paths:
                # Only pass callback if heteronym selection is enabled
                callback = self.prompt_for_heteronym if self.heteronym_enabled.get() else None
                table = process_image(
                    img_path, 
                    heteronym_callback=callback,
                    is_traditional=self.is_traditional.get()
                )
                if table and len(table) > 0:
                    headers = table[0]
                    all_headers.update(headers)
                    all_tables.append(table)
            
            if not all_tables:
                raise Exception("No valid data found in any images")
            
            # Merge tables
            merged_table = self.merge_tables(all_tables)
            
            if merged_table and len(merged_table) > 0:
                self.processed_table = merged_table
                self.header_row = merged_table[0]
                # Initialize all headers to side 1
                for header in self.header_row:
                    self.header_assignments[header] = "side1"
                self.update_lists()
                self.enable_header_widgets()
                self.process_label.config(text=f"Successfully processed! Found {len(merged_table)-1} total rows of data")
            else:
                self.process_label.config(text="No data found in images")
                
        except Exception as e:
            self.process_label.config(text=f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to process images: {str(e)}")

    

    def merge_tables(self, tables):
        """Merge multiple tables based on matching headers"""
        if not tables:
            return None
        
            
        # Get all unique headers while handling similar names
        all_headers = []
        header_map = {}  # Maps similar headers to canonical version
        
        for table in tables:
            headers = table[0]
            for header in headers:
                # Check if this header is similar to any existing one
                found_match = False

                for existing_header in all_headers:
                    if self.headers_are_similar(header, existing_header):
                        header_map[header] = existing_header
                        found_match = True
                        break
                
                if not found_match:
                    all_headers.append(header)
                    header_map[header] = header
        

        # Create merged table with unified headers
        merged_table = [all_headers]
        
        # For each table, map its data to the complete header structure
        for table in tables:
            headers = table[0]
            # Create header index mapping using the normalized headers
            header_indices = {header_map[header]: i for i, header in enumerate(headers)}
            
            # Process each data row
            for row_idx in range(1, len(table)):
                row = table[row_idx]
                new_row = ["Not found"] * len(all_headers)
                
                # Map values to their correct positions in the merged table
                for merged_idx, merged_header in enumerate(all_headers):
                    if merged_header in header_indices:
                        orig_idx = header_indices[merged_header]
                        if orig_idx < len(row):
                            new_row[merged_idx] = row[orig_idx]
                
                merged_table.append(new_row)
        
        return merged_table

    def disable_processing_widgets(self):
        """Disable all widgets except image selection"""
        self.process_button.state(['disabled'])
        self.move_right_btn.state(['disabled'])
        self.move_left_btn.state(['disabled'])
        self.disable_btn.state(['disabled'])
        self.enable_btn.state(['disabled'])
        self.quizlet_button.state(['disabled'])
    
    def enable_header_widgets(self):
        """Enable widgets for header assignment"""
        self.move_right_btn.state(['!disabled'])
        self.move_left_btn.state(['!disabled'])
        self.disable_btn.state(['!disabled'])
        self.enable_btn.state(['!disabled'])
        self.quizlet_button.state(['!disabled'])
    
    def move_header(self, from_side, to_side):
        source_list = self.side1_list if from_side == "side1" else self.side2_list
        selection = source_list.curselection()
        if selection:
            header = source_list.get(selection[0])
            self.header_assignments[header] = to_side
            self.update_lists()
    
    def disable_header(self):
        for lst in [self.side1_list, self.side2_list]:
            selection = lst.curselection()
            if selection:
                header = lst.get(selection[0])
                self.header_assignments[header] = "disabled"
                self.update_lists()
                break
    
    def enable_header(self):
        selection = self.disabled_list.curselection()
        if selection:
            header = self.disabled_list.get(selection[0])
            self.header_assignments[header] = "side1"
            self.update_lists()
    
    def update_lists(self):
        for lst in [self.side1_list, self.side2_list, self.disabled_list]:
            lst.delete(0, tk.END)
        
        for header, assignment in self.header_assignments.items():
            if assignment == "side1":
                self.side1_list.insert(tk.END, header)
            elif assignment == "side2":
                self.side2_list.insert(tk.END, header)
            elif assignment == "disabled":
                self.disabled_list.insert(tk.END, header)
    
    def show_text_popup(self, text_content):
        """Show a popup window with the text content and a copy button"""
        popup = tk.Toplevel(self.root)
        popup.title("Flashcard Text")
        popup.geometry("600x400")
        
        # Create and pack a frame
        frame = ttk.Frame(popup, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a text area with scrollbars
        text_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=60, height=20)
        text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        text_area.insert(tk.END, text_content)
        
        # Function to copy all text
        def copy_text():
            popup.clipboard_clear()
            popup.clipboard_append(text_area.get("1.0", tk.END))
            copy_button.config(text="Copied!")
            popup.after(1500, lambda: copy_button.config(text="Copy All"))
        
        # Add copy button
        copy_button = ttk.Button(frame, text="Copy All", command=copy_text)
        copy_button.pack(pady=5)

    def export_to_quizlet(self):
        if not self.processed_table or len(self.processed_table) < 2:
            messagebox.showerror("Error", "No data to export")
            return
        
        header_indices = {header: idx for idx, header in enumerate(self.header_row)}
        
        # Generate the text content
        text_content = ""
        for row in self.processed_table[1:]:  # Skip header row
            side1_text = []
            side2_text = []
            
            # Collect all text for each side based on header assignments
            for header, idx in header_indices.items():
                if self.header_assignments.get(header) == "side1":
                    if row[idx] != "Not found":
                        side1_text.append(str(row[idx]).strip())
                elif self.header_assignments.get(header) == "side2":
                    if row[idx] != "Not found":
                        side2_text.append(str(row[idx]).strip())
            
            # Only add the line if both sides have content
            if side1_text and side2_text:
                text_content += f"{', '.join(side1_text)}\t{', '.join(side2_text)};\n"
        
        # Create dialog to choose export method
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Options")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (
            self.root.winfo_rootx() + self.root.winfo_width()//2 - 150,
            self.root.winfo_rooty() + self.root.winfo_height()//2 - 75
        ))
        
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Choose export method:").pack(pady=(0, 10))
        
        def save_to_file():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")]
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                messagebox.showinfo("Success", "Flashcards exported successfully!")
            dialog.destroy()
        
        def show_in_popup():
            dialog.destroy()
            self.show_text_popup(text_content)
        
        ttk.Button(frame, text="Save to File", command=save_to_file).pack(pady=5, expand=True)
        ttk.Button(frame, text="Show in Popup", command=show_in_popup).pack(pady=5, expand=True)
        ttk.Button(frame, text="Cancel", command=dialog.destroy).pack(pady=5, expand=True)

def main():
    root = tk.Tk()
    app = FlashcardCreatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
