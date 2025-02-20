import logging
import sys
import os
import traceback
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QTextEdit, QWidget, QTabWidget, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from glob import glob

from scripting import mainScript
from scripting import gene_tab
from scripting import pearson_correlation_analyser
from scripting import random_forest, elastic_net
from scripting.file_manager import OrderedFolderCreator

class AnalysisThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, raw_data_dir, gene_names):
        super().__init__()
        self.raw_data_dir = raw_data_dir
        self.gene_names = gene_names

        # Reset folder numbering by removing the index file
        index_file = "folder_index.json"
        if os.path.exists(index_file):
            os.remove(index_file)

        self.folder_creator = OrderedFolderCreator()

    def run(self):
        try:
            base_dir = os.path.dirname(self.raw_data_dir)
            processed_data_dir = os.path.join(base_dir, "Processed_Data")
            os.makedirs(processed_data_dir, exist_ok=True)

            # Handle merged data
            merged_data_folder = self.folder_creator.create_folder(processed_data_dir, "merged_data")
            merged_data_path = os.path.join(merged_data_folder, "merged_data.csv")
            drug_data_path = os.path.join(self.raw_data_dir, "drug_id2_PD.csv")

            if os.path.exists(merged_data_path):
                self.progress.emit(f"Merged data file exists at {merged_data_path}. Skipping merge step.")
                merged_data = pd.read_csv(merged_data_path)
            else:
                self.progress.emit("Merging data...")
                merged_data = mainScript.merge_abc_and_drug_data(self.raw_data_dir)
                if merged_data is None:
                    self.progress.emit("Error: Missing required raw data files.")
                    return
                merged_data.to_csv(merged_data_path, index=False)
                self.progress.emit(f"Merged data saved at {merged_data_path}")

            if not os.path.exists(drug_data_path):
                self.progress.emit("Error: 'drug_id2_PD.csv' file is missing.")
                return

            # Handle correlation results
            correlation_results_folder = self.folder_creator.create_folder(processed_data_dir, "pearson_correlations")
            self.progress.emit("Calculating correlations...")
            merged_properties_path = mainScript.calculate_correlations(
                merged_data_path, drug_data_path, self.gene_names, correlation_results_folder, self.folder_creator
            )

            # Analyze top and bottom drugs
            self.progress.emit("Analyzing top and bottom drugs...")
            output_directory = self.folder_creator.create_folder(processed_data_dir, "gene_top_bottom_results")
            for gene_name in self.gene_names:
                self.progress.emit(f"Analyzing gene: {gene_name}")
                analyzer = pearson_correlation_analyser.PearsonCorrelationAnalyzer(merged_properties_path, gene_name)
                analyzer.save_top_bottom_drugs(output_directory)

            self.finished.emit(f"Analysis completed. Results saved in {processed_data_dir}")
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
            print(traceback.format_exc())

class MLWorker(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, use_random_forest=True, exclude_autocorr=False):
        super().__init__()
        self.use_random_forest = use_random_forest
        self.exclude_autocorr = exclude_autocorr

    def run(self):
        if self.use_random_forest:
            logging.info("Running Random Forest model...")
            ml_results = random_forest.run_ml_model(exclude_autocorr=self.exclude_autocorr)
        else:
            logging.info("Running Elastic Net model...")
            ml_results = elastic_net.run_ml_pipeline()
        self.finished.emit(ml_results)

class GeneDrugApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analysis_thread = None
        self.exclude_autocorr = False
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

        self.raw_data_label = QLabel("Raw Data Directory not found. Please select one.")
        self.raw_data_button = QPushButton("Browse")
        self.raw_data_button.clicked.connect(self.select_raw_data_dir)

        self.raw_data_path = QLabel("No directory selected.")

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
            "rf_tab": ("View Random Forest Analysis", "Load Random Forest Analysis", lambda: self.add_ML_tab(True)),
            "en_tab": ("View Elastic Net Analysis", "Load Elastic Net Analysis", lambda: self.add_ML_tab(False)),
        }

        # Dictionary to store tab widgets
        self.tabs_dict = {}

        for tab_name, (tab_title, button_text, function) in tab_definitions.items():
            tab, tab_layout = self.create_tab_with_button(tab_title, button_text, function)
            self.tabs.addTab(tab, tab_title)
            self.tabs_dict[tab_name] = tab  # Store tab reference

        # Pearson Correlations & Properties have subtabs, ML tabs do not
        self.pearson_subtabs = QTabWidget()
        self.tabs_dict["data_tab"].layout().insertWidget(0, self.pearson_subtabs)

        self.properties_subtabs = QTabWidget()
        self.tabs_dict["properties_tab"].layout().insertWidget(0, self.properties_subtabs)

        # Separate Machine Learning Subtabs
        self.rf_ml_subtabs = QTabWidget()
        self.exclude_autocorr_checkbox = QCheckBox("Exclude Extra Features")
        self.exclude_autocorr_checkbox.setChecked(False)
        self.exclude_autocorr_checkbox.stateChanged.connect(self.toggle_autocorr_exclusion)
        self.tabs_dict["rf_tab"].layout().insertWidget(0, self.exclude_autocorr_checkbox)
        self.tabs_dict["rf_tab"].layout().insertWidget(1, self.rf_ml_subtabs)

        self.en_ml_subtabs = QTabWidget()
        self.tabs_dict["en_tab"].layout().insertWidget(0, self.en_ml_subtabs)

    def toggle_autocorr_exclusion(self, state):
        self.exclude_autocorr = state == Qt.Checked

    def create_tab_with_button(self, tab_title, button_text, function):
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
        self.analysis_thread = AnalysisThread(self.raw_data_dir, gene_names)
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

    def add_ML_tab(self, use_random_forest):
        """Runs the selected ML model (Random Forest or Elastic Net) in a separate thread and updates the UI."""
        model_name = "Random Forest" if use_random_forest else "Elastic Net"
        self.output_label.setText(f"Running {model_name} Model... Please wait.")

        # Create MLWorker Thread
        self.ml_worker = MLWorker(use_random_forest, self.exclude_autocorr)

        # ✅ Corrected signal connection to pass `use_random_forest`
        self.ml_worker.finished.connect(lambda results: self.on_ml_finished(results, use_random_forest))

        self.ml_worker.start()

    def on_ml_finished(self, ml_results, use_random_forest):
        """Handles ML results after the worker thread has completed."""
        if not ml_results:
            self.output_label.setText("Error: No ML results returned.")
            return

        # ✅ Use `use_random_forest` instead of relying on ml_results
        target_subtab = self.rf_ml_subtabs if use_random_forest else self.en_ml_subtabs

        # Initialize MLResultsTab and add to UI
        ml_tab_widget = gene_tab.MLResultsTab()
        target_subtab.addTab(ml_tab_widget, f"{'Random Forest' if use_random_forest else 'Elastic Net'} Results")

        self.output_label.setText(f"{'Random Forest' if use_random_forest else 'Elastic Net'} Model Analysis Loaded.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = GeneDrugApp()
    main_window.show()
    sys.exit(app.exec_())