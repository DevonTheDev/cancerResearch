import sys
import os
import traceback
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QTextEdit, QWidget, QTabWidget
)
from PyQt5.QtCore import Qt
from glob import glob

from scripting import gene_tab, AnalysisThread
from scripting.MachineLearning import MLWorker

class GeneDrugApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analysis_thread = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Gene-Drug Correlation Analysis")
        self.setGeometry(100, 100, 800, 600)

        # Main layout with tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.setup_analysis_tab()
        self.setup_data_tab()

    def setup_analysis_tab(self):
        self.analysis_tab = QWidget()
        self.tabs.addTab(self.analysis_tab, "Run Analysis")

        layout = QVBoxLayout()
        self.analysis_tab.setLayout(layout)

        try:
            # Attempt to set the Raw Data directory by looking for it in the CWD
            self.raw_data_dir = os.path.join(os.getcwd(), "Raw Data")
            self.raw_data_label = QLabel("Raw Data Directory Found")
            self.raw_data_path = QLabel(f"Selected Directory: {self.raw_data_dir}")
            self.raw_data_button = QPushButton("Change Directory")
        except Exception as e:
            print(f"Error getting Raw Data Directory Automatically: {str(e)}")
            self.raw_data_label = QLabel("Raw Data Directory not found. Please select one.")
            self.raw_data_path = QLabel("No directory selected.")
            self.raw_data_button = QPushButton("Select Directory")

        self.raw_data_button.clicked.connect(self.select_raw_data_dir)

        layout.addWidget(self.raw_data_label)
        layout.addWidget(self.raw_data_button)
        layout.addWidget(self.raw_data_path)

        self.gene_names_label = QLabel("Enter Gene Names (comma-separated):")
        self.gene_names_input = QTextEdit("ABCB1,ABCG2,ABCC1,ABCC2,ABCC3,ABCC4")
        layout.addWidget(self.gene_names_label)
        layout.addWidget(self.gene_names_input)

        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_button)

        self.output_label = QLabel("")
        self.output_label.setAlignment(Qt.AlignTop)
        layout.addWidget(self.output_label)

    def setup_data_tab(self):
        """Sets up tabs for viewing Pearson Correlations, Properties Analysis, and ML results."""

        # Tab definitions: { tab_variable_name: ("Tab Title", "Button Label", associated function) }
        tab_definitions = {
            "data_tab": ("View Pearson Correlations", "Load Pearson Correlations", self.load_pearson_correlations),
            "properties_tab": ("View Properties Analysis", "Load Properties Analysis", self.load_properties_analysis),
            "ml_tab": ("View Machine Learning Analysis", "Load Machine Learning Analysis", self.add_ML_tab)
        }

        # Dictionary to store tab widgets
        self.tabs_dict = {}

        for tab_name, (tab_title, button_text, function) in tab_definitions.items():
            tab, _ = self.create_tab_with_button(button_text, function)
            self.tabs.addTab(tab, tab_title)
            self.tabs_dict[tab_name] = tab  # Store tab reference

        # Pearson Correlations & Properties have subtabs
        self.pearson_subtabs = QTabWidget()
        self.tabs_dict["data_tab"].layout().insertWidget(0, self.pearson_subtabs)

        self.properties_subtabs = QTabWidget()
        self.tabs_dict["properties_tab"].layout().insertWidget(0, self.properties_subtabs)

        self.ml_subtabs = QTabWidget()
        self.tabs_dict["ml_tab"].layout().insertWidget(0, self.ml_subtabs)

    def create_tab_with_button(self, button_text, function):
        """Creates a tab with a button and returns the tab widget and its layout."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        button = QPushButton(button_text)
        button.clicked.connect(function)

        layout.addWidget(button)
        return tab, layout

    def select_raw_data_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.raw_data_path.setText(f"Selected Directory: {directory}")
            self.raw_data_dir = directory

    def run_analysis(self):
        if not hasattr(self, "raw_data_dir") or not self.raw_data_dir:
            self.output_label.setText("Error: Please select a raw data directory.")
            return

        gene_names = [gene.strip() for gene in self.gene_names_input.toPlainText().split(",")]
        self.analysis_thread = AnalysisThread.AnalysisThread(self.raw_data_dir, gene_names)
        self.analysis_thread.progress.connect(self.update_progress)
        self.analysis_thread.finished.connect(self.analysis_completed)
        self.analysis_thread.start()

    def update_progress(self, message):
        self.output_label.setText(message)

    def analysis_completed(self, message):
        self.output_label.setText(message)
        self.analysis_thread = None

    def load_pearson_correlations(self):
        try:
            if not hasattr(self, "raw_data_dir") or not self.raw_data_dir:
                self.output_label.setText("Error: Please select a raw data directory.")
                return

            base_dir = os.path.dirname(self.raw_data_dir)

            # Search for the "pearson_correlations" folder
            pearson_folders = glob(os.path.join(base_dir, "Processed_Data", "*pearson_correlations*"))
            if not pearson_folders:
                self.output_label.setText("Error: No folder found containing 'pearson_correlations'.")
                return

            # Use the first matching folder (if multiple are found)
            processed_data_dir = pearson_folders[0]

            # Get gene names from input
            gene_names = [gene.strip() for gene in self.gene_names_input.toPlainText().split(",")]

            for gene in gene_names:
                # Search for CSV files matching the gene name within the pearson_correlations folder
                pattern = os.path.join(processed_data_dir, f"*{gene}*.csv")
                matching_files = glob(pattern)

                if matching_files:  # If matching files are found
                    for file_path in matching_files:
                        data_frame = pd.read_csv(file_path)
                        data_frame.sort_values(by="Pearson_Correlation", ascending=False, inplace=True)
                        self.add_gene_tab(gene, data_frame)
                else:
                    self.output_label.setText(f"No file found for gene: {gene} in folder: {processed_data_dir}")

        except Exception as e:
            self.output_label.setText(f"Error loading data: {str(e)}")
            print(traceback.format_exc())

    def load_properties_analysis(self):
        try:
            if not hasattr(self, "raw_data_dir") or not self.raw_data_dir:
                self.output_label.setText("Error: Please select a raw data directory.")
                return
            
            base_dir = os.path.dirname(self.raw_data_dir)

            # Search for gene_top_bottom_results folder
            properties_folders = glob(os.path.join(base_dir, "Processed_Data", "*gene_top_bottom_results*"))
            if not properties_folders:
                self.output_label.setText("Error: No folder found containing 'gene_top_bottom_results'.")
                return
            
            # Use the first matching folder (if multiple are found)
            processed_data_dir = properties_folders[0]

            # Get gene names from input
            gene_names = [gene.strip() for gene in self.gene_names_input.toPlainText().split(",")]

            for gene in gene_names:
                # Search for CSV files matching the gene name within the gene_top_bottom_results folder
                pattern = os.path.join(processed_data_dir, f"*{gene}*_properties_analysis.csv")
                matching_files = glob(pattern)

                if matching_files:
                    for file_path in matching_files:
                        data_frame = pd.read_csv(file_path)
                        data_frame.sort_values(by="Effect Size", ascending=False, inplace=True)
                        self.add_properties_tab(gene, data_frame)
                else:
                    self.output_label.setText(f"No file found for gene: {gene} in folder: {processed_data_dir}")
        
        except Exception as e:
            self.output_label.setText(f"Error loading data: {str(e)}")
            print(traceback.format_exc())

    def add_gene_tab(self, gene, data_frame):
        gene_tab_widget = gene_tab.GeneTab(gene, data_frame)
        self.pearson_subtabs.addTab(gene_tab_widget, gene)

    def add_properties_tab(self, gene, data_frame):
        gene_tab_widget = gene_tab.PropertiesTab(gene, data_frame)
        self.properties_subtabs.addTab(gene_tab_widget, gene)

    def add_ML_tab(self):
        """Runs the selected ML model in a separate thread and updates the UI."""
        self.ml_worker = MLWorker.MLWorker()

        self.ml_worker.finished.connect(lambda results: self.on_ml_finished(results))

        self.ml_worker.start()

    def on_ml_finished(self, ml_results):
        """Handles ML results after the worker thread has completed."""
        if not ml_results:
            self.output_label.setText(f"Error: No ML results returned.")
            return

        ml_tab_widget = gene_tab.MLResultsTab(ml_results)

        self.ml_subtabs.addTab(ml_tab_widget, "ML Results")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = GeneDrugApp()
    main_window.show()
    sys.exit(app.exec_())