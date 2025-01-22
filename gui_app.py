import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QTextEdit, QWidget, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os
import pandas as pd
import traceback

from scripting import mainScript
from scripting import gene_tab
from scripting import pearson_correlation_analyser
from scripting.file_manager import OrderedFolderCreator

class AnalysisThread(QThread):
    progress = pyqtSignal(str)  # Signal to send progress updates to the UI
    finished = pyqtSignal(str)  # Signal to indicate completion

    def __init__(self, raw_data_dir, gene_names):
        super().__init__()
        self.raw_data_dir = raw_data_dir
        self.gene_names = gene_names

        # Remove the index file for each run to reset folder numbering by file_manager
        index_file = "folder_index.json"
        if os.path.exists(index_file):
            os.remove(index_file)

        self.folder_creator = OrderedFolderCreator() # Create an instance of OrderedFolderCreator for folder creation

    def run(self):
        try:
            base_dir = os.path.dirname(self.raw_data_dir)
            processed_data_dir = os.path.join(base_dir, "Processed_Data")

            # Ensure directories exist
            os.makedirs(processed_data_dir, exist_ok=True)

            # Create ordered folders and define file paths
            merged_data_folder = self.folder_creator.create_folder(processed_data_dir, "merged_data")
            merged_data_path = os.path.join(merged_data_folder, "merged_data.csv")
            drug_data_path = os.path.join(self.raw_data_dir, "drug_id2_PD.csv")

            # Check if merged data file already exists
            if os.path.exists(merged_data_path):
                self.progress.emit(f"Merged data file already exists at {merged_data_path}. Skipping merge step.")
                merged_data = pd.read_csv(merged_data_path)
            else:
                self.progress.emit("Merging data...")
                merged_data = mainScript.merge_abc_and_drug_data(self.raw_data_dir)
                if merged_data is None:
                    self.progress.emit("Error: Required raw data files are missing.")
                    return
                merged_data.to_csv(merged_data_path, index=False)
                self.progress.emit(f"Merged data saved at {merged_data_path}")

            # Check if drug data file exists
            if not os.path.exists(drug_data_path):
                self.progress.emit("Error: 'drug_id2_PD.csv' file is missing.")
                return

            # Create folder for correlation results
            correlation_results_folder = self.folder_creator.create_folder(processed_data_dir, "pearson_correlations")

            # Calculate correlations
            self.progress.emit("Calculating correlations...")
            mainScript.calculate_correlations(merged_data_path, drug_data_path, self.gene_names, correlation_results_folder, self.folder_creator)

            # Analyze and save top and bottom drugs
            self.progress.emit("Analyzing top and bottom drugs...")
            output_directory = self.folder_creator.create_folder(processed_data_dir, "gene_top_bottom_results")
            for gene_name in self.gene_names:
                self.progress.emit(f"Analyzing gene: {gene_name}")
                analyzer = pearson_correlation_analyser.PearsonCorrelationAnalyzer(correlation_results_folder, gene_name)
                analyzer.save_top_bottom_drugs(output_directory)

            self.progress.emit("Top and bottom drug analysis completed.")

            # Notify success
            self.finished.emit(f"Analysis completed. Results saved in {processed_data_dir}")
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
            print(traceback.format_exc())

class GeneDrugApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.analysis_thread = None

    def initUI(self):
        # Set up the main window
        self.setWindowTitle("Gene-Drug Correlation Analysis")
        self.setGeometry(100, 100, 800, 600)

        # Main layout with tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: Analysis Configuration
        self.analysis_tab = QWidget()
        self.tabs.addTab(self.analysis_tab, "Run Analysis")

        self.analysis_layout = QVBoxLayout()
        self.analysis_tab.setLayout(self.analysis_layout)

        # Raw data directory
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            raw_data_dir = os.path.join(script_dir, "Raw Data")
            self.raw_data_label = QLabel(f"Raw Data Directory Found: {raw_data_dir}" if os.path.exists(raw_data_dir) else "Raw Data Directory not found. Please select one.")
            self.raw_data_path = QLabel(f"Selected Directory: {raw_data_dir}" if os.path.exists(raw_data_dir) else "No directory selected.")
            if os.path.exists(raw_data_dir):
                self.raw_data_dir = raw_data_dir
        except Exception as e:
            self.raw_data_label = QLabel(f"Error locating Raw Data Directory: {e}")

        self.analysis_layout.addWidget(self.raw_data_label)

        self.raw_data_button = QPushButton("Browse")
        self.raw_data_button.clicked.connect(self.select_raw_data_dir)
        self.analysis_layout.addWidget(self.raw_data_button)

        self.analysis_layout.addWidget(self.raw_data_path)

        # Gene names input
        self.gene_names_label = QLabel("Enter Gene Names (comma-separated):")
        self.analysis_layout.addWidget(self.gene_names_label)

        self.gene_names_input = QTextEdit("ABCB1,ABCG2,ABCC1,ABCC2,ABCC3,ABCC4")
        self.analysis_layout.addWidget(self.gene_names_input)

        # Run analysis button
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        self.analysis_layout.addWidget(self.run_button)

        # Output label
        self.output_label = QLabel("")
        self.output_label.setAlignment(Qt.AlignTop)
        self.analysis_layout.addWidget(self.output_label)

        # Tab 2: Display Pearson Correlations
        self.data_tab = QWidget()
        self.tabs.addTab(self.data_tab, "View Pearson Correlations")

        self.data_layout = QVBoxLayout()
        self.data_tab.setLayout(self.data_layout)

        self.subtabs = QTabWidget()
        self.data_layout.addWidget(self.subtabs)

        # Load data button for the tab
        self.load_data_button = QPushButton("Load Pearson Correlations")
        self.load_data_button.clicked.connect(self.load_pearson_correlations)
        self.data_layout.addWidget(self.load_data_button)

    def select_raw_data_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.raw_data_path.setText(f"Selected Directory: {directory}")
            self.raw_data_dir = directory

    def run_analysis(self):
        raw_data_dir = getattr(self, "raw_data_dir", None)
        if not raw_data_dir:
            self.output_label.setText("Error: Please select a raw data directory.")
            return

        gene_names = [
            gene.strip()
            for gene in self.gene_names_input.toPlainText().split(",")
        ]

        # Initialize and start the analysis thread
        self.analysis_thread = AnalysisThread(raw_data_dir, gene_names)
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
            # Get the processed_data directory
            raw_data_dir = getattr(self, "raw_data_dir", None)
            if not raw_data_dir:
                self.output_label.setText("Error: Please select a raw data directory.")
                return

            base_dir = os.path.dirname(raw_data_dir)
            processed_data_dir = os.path.join(base_dir, "Processed_Data", "pearson_correlations")

            # Load data for each gene
            gene_names = [
                gene.strip()
                for gene in self.gene_names_input.toPlainText().split(",")
            ]

            for gene in gene_names:
                file_path = os.path.join(processed_data_dir, f"{gene}_pearson_correlations.csv")
                if os.path.exists(file_path):
                    data_frame = pd.read_csv(file_path)
                    data_frame = data_frame.reindex(data_frame['Pearson_Correlation'].sort_values(ascending=False).index)
                    self.add_gene_tab(gene, data_frame)
                else:
                    self.output_label.setText(f"File not found for gene: {gene}")

        except Exception as e:
            self.output_label.setText(f"Error loading data: {str(e)}")
            print(traceback.format_exc())

    def add_gene_tab(self, gene, data_frame):
        # Add a new GeneTab to the subtabs
        genes_tab = gene_tab.GeneTab(gene, data_frame)
        self.subtabs.addTab(genes_tab, gene)

    @staticmethod
    def getRawDataDirectory():
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            raw_data_dir = os.path.join(script_dir, "Raw Data")
            return raw_data_dir
        except Exception as e:
            return None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = GeneDrugApp()
    ex.show()
    sys.exit(app.exec_())