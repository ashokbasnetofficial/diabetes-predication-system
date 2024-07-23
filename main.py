import tkinter as tk
from tkinter import filedialog
import os
from tkinter import messagebox
import pandas as pd
from calculation import run_calculation
from plot_data_summary import plot_summary_data

# function to handle file selection
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    if file_path:
        analysis_btn.config(state=tk.NORMAL)  # enable the "OK" button if a file was selected
        file_name = os.path.basename(file_path)  # get the file name without the path
        file_label.config(text=f"Selected File: {file_name}")  # update the label text
        global selected_file
        selected_file = file_path

# Check condition for analysis summary generator
def summary_gen_condition():
    if os.path.exists("./summary_data"):
        if os.listdir("./summary_data") == ['knn.csv', 'logistic.csv', 'svm.csv']:
            df1 = pd.read_csv("./summary_data/knn.csv")
            df2 = pd.read_csv("./summary_data/logistic.csv")
            df3 = pd.read_csv("./summary_data/svm.csv")
            if df1.shape[0] >= 2 and df2.shape[0] >=2 and df3.shape[0] >= 2:
                return True
            else:
                return False
        else:
            return False
    else:
        return False
# Function to handle the "OK" button click
def perform_analysis():

    if selected_file:
        run_calculation(selected_file)
        reset_button_click()

def generate_summary():
    if summary_gen_condition():
        plot_summary_data()
    else:
         messagebox.showinfo("Analysis Summary", "You should analyze at least two datasets to generate an analysis summary")

# Function to handle the "Reset" button click
def reset_button_click():
    global selected_file
    selected_file = None
    file_label.config(text="No file selected")
    analysis_btn.config(state=tk.DISABLED)

# Delete created summary csv files, work as reset
def delete_files():
    user_ans = messagebox.askokcancel("Delete!!", "Delete all the previous Analysis Data")
    files = ["svm", "logistic"]
    if user_ans and os.path.exists("./summary_data"):
        if os.listdir("./summary_data") == ['logistic.csv', 'svm.csv']:
            for i in files:
                if os.path.exists(f"./summary_data/{i}.csv"):
                    os.remove(f"./summary_data/{i}.csv")

# Create a tkinter window
window = tk.Tk()
window.title("Comparative Analysis")
window.configure(background="#5FE88D")

# Set the size of the window
window.geometry("450x550")

# Create a label
comp_label = tk.Label(text="For the Analysis of Algorithm:", font=('Times', 18), background="#5FE88D")
comp_label.pack(anchor="nw", padx=10, pady=10)

# Create a label to show the selected file
file_label = tk.Label(text="No file selected")
file_label.pack(pady=10)

# Create a button to select file
select_button = tk.Button(text="Select CSV File", font=('Times', 12), command=select_file)
select_button.pack(pady=10)

# Create an "OK" button to run the calculation
analysis_btn = tk.Button(text="Perform Analysis", command=perform_analysis, state=tk.DISABLED)
analysis_btn.pack(pady=10)

# Create a "Reset" button to clear the selected file path
reset_button = tk.Button(text="Reset", command=reset_button_click)
reset_button.pack(pady=10)

summary_label = tk.Label(text="For creating summary of Analysis:", font=('Times', 18), background="#5FE88D")
summary_label.pack(anchor="nw", padx=10, pady=10)

# Create an "OK" button to run the calculation
summary_gen = tk.Button(text="Generate Summary of Analysis", command=generate_summary, state=tk.NORMAL)
summary_gen.pack(pady=10)

# Create an "OK" button to run the calculation
delete_files_btn = tk.Button(text="Remove all summary data", command=delete_files, state=tk.NORMAL)
delete_files_btn.pack(pady=10)

# Run the main event loop
window.mainloop()
