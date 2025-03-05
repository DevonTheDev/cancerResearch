import os
import traceback
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal
from scripting import mainScript, pearson_correlation_analyser
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