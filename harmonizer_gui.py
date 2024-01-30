import tkinter as tk
from tkinter import ttk, simpledialog
import pandas as pd


class HarmonizerGUI:
    def __init__(self, harmonized_df):
        self.harmonized_df = harmonized_df
        self.root = tk.Tk()
        self.root.title("Harmonizer")

        # Create a Label
        self.label = ttk.Label(self.root, text="Please enter a column name:")
        self.label.grid(column=0, row=0)

        # Create a Combobox for input
        self.input_text = ttk.Combobox(self.root, values=harmonized_df.columns.tolist())
        self.input_text.grid(column=1, row=0)

        # Create a Spinbox to select the number of results
        self.num_results = ttk.Spinbox(self.root, from_=1, to=100)
        self.num_results.grid(column=2, row=0)

        # Create a Text widget to display the results
        self.output = tk.Text(self.root)
        self.output.grid(column=0, row=1, columnspan=3)

        # Bind the Combobox and Spinbox to the on_value_change function
        self.input_text.bind("<<ComboboxSelected>>", self.on_value_change)
        self.num_results.bind("<Return>", self.on_value_change)

    def on_value_change(self, event):
        # Get the input value
        search_column = self.input_text.get()
        # Check if the input value is a column in the DataFrame
        if search_column in self.harmonized_df.columns:
            # Get the top n rows ordered by score
            top_rows = self.harmonized_df[[search_column]].nlargest(
                int(self.num_results.get()), search_column
            )
            # Convert the DataFrame to a string and set it as the value of the output widget
            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, top_rows.to_string())

    def get_user_input(self, prompt: str) -> str:
        # Create a simple dialog that asks the user for input and returns it
        user_input = simpledialog.askstring("Input", prompt, parent=self.root)
        return user_input

    def display_message(self, message: str):
        # Insert the message at the end of the Text widget and scroll to the end
        self.output.insert(tk.END, message + "\n")
        self.output.see(tk.END)

    def run(self):
        self.root.mainloop()


# Usage:
# harmonizer = YourHarmonizerClass()
# gui = HarmonizerGUI(harmonizer)
# gui.run()
