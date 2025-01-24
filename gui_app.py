import sys
import os
import traceback
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QTextEdit, QWidget, QTabWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from scripting import mainScript
from scripting import gene_tab
from scripting import pearson_correlation_analyser
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
        self.data_tab = QWidget()
        self.tabs.addTab(self.data_tab, "View Pearson Correlations")

        layout = QVBoxLayout()
        self.data_tab.setLayout(layout)

        self.subtabs = QTabWidget()
        layout.addWidget(self.subtabs)

        self.load_data_button = QPushButton("Load Pearson Correlations")
        self.load_data_button.clicked.connect(self.load_pearson_correlations)
        layout.addWidget(self.load_data_button)

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
            processed_data_dir = os.path.join(base_dir, "Processed_Data", "pearson_correlations")

            gene_names = [gene.strip() for gene in self.gene_names_input.toPlainText().split(",")]

            for gene in gene_names:
                file_path = os.path.join(processed_data_dir, f"{gene}_pearson_correlations.csv")
                if os.path.exists(file_path):
                    data_frame = pd.read_csv(file_path)
                    data_frame.sort_values(by="Pearson_Correlation", ascending=False, inplace=True)
                    self.add_gene_tab(gene, data_frame)
                else:
                    self.output_label.setText(f"File not found for gene: {gene}")

        except Exception as e:
            self.output_label.setText(f"Error loading data: {str(e)}")
            print(traceback.format_exc())

    def add_gene_tab(self, gene, data_frame):
        gene_tab_widget = gene_tab.GeneTab(gene, data_frame)
        self.subtabs.addTab(gene_tab_widget, gene)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = GeneDrugApp()
    main_window.show()
    sys.exit(app.exec_())