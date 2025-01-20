from gui_app import GeneDrugApp
import os
import pandas as pd
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QLabel, QScrollArea, QDialog
import pubchempy as pcp


class DrugInfo(QWidget):
    def __init__(self):
        super().__init__()

    def onRowDoubleClicked(self, gene_name, drug_id):
        try:
            file_path = self.get_file_path(gene_name)
            data_frame = pd.read_csv(file_path)

            extracted_row = data_frame[data_frame['id'] == drug_id]
            if extracted_row.empty:
                raise ValueError(f"No data found for drug_id: {drug_id}.")

            pubchem_cid = self.get_pubchem_cid(extracted_row)
            smiles = extracted_row['smiles'].values[0]
            inchi_key = extracted_row['InChIKey'].values[0]

            compound_info = self.fetch_pubchem_data(pubchem_cid)
            self.display_drug_info(gene_name, pubchem_cid, smiles, inchi_key, compound_info)

        except Exception as e:
            self.show_error(str(e))

    def get_file_path(self, gene_name):
        raw_data_dir = os.path.dirname(GeneDrugApp.getRawDataDirectory())
        processed_data_dir = os.path.join(raw_data_dir, "Processed_Data", "pearson_correlations")

        if not os.path.exists(processed_data_dir):
            raise FileNotFoundError(f"Directory not found: {processed_data_dir}")

        file_path = os.path.join(processed_data_dir, f"{gene_name}_properties_merged.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found for gene: {gene_name} at {file_path}")

        return file_path

    def get_pubchem_cid(self, row):
        pubchem_cid = row['pubchem_cid'].values[0]
        if pd.isna(pubchem_cid):
            raise ValueError("PubChem CID is missing.")
        return str(int(pubchem_cid))

    def fetch_pubchem_data(self, pubchem_cid):
        try:
            compound = pcp.Compound.from_cid(pubchem_cid)
            return {
                'MolecularWeight': compound.molecular_weight,
                'CanonicalSMILES': compound.canonical_smiles,
                'IUPACName': compound.iupac_name,
            }
        except Exception as e:
            print(f"Error fetching PubChem data: {e}")
            return {}

    def display_drug_info(self, gene_name, pubchem_cid, smiles, inchi_key, compound_info):
        dialog = QDialog()
        dialog.setWindowTitle(f"Drug Info: {pubchem_cid}")
        layout = QVBoxLayout(dialog)

        # Add labels for extracted data
        layout.addWidget(QLabel(f"Gene Name: {gene_name}"))
        layout.addWidget(QLabel(f"PubChem CID: {pubchem_cid}"))
        layout.addWidget(QLabel(f"SMILES: {smiles}"))
        layout.addWidget(QLabel(f"InChIKey: {inchi_key}"))

        # Add additional compound information
        if compound_info:
            layout.addWidget(QLabel(f"Molecular Weight: {compound_info.get('MolecularWeight', 'N/A')}"))
            layout.addWidget(QLabel(f"Canonical SMILES: {compound_info.get('CanonicalSMILES', 'N/A')}"))
            layout.addWidget(QLabel(f"IUPAC Name: {compound_info.get('IUPACName', 'N/A')}"))
        else:
            layout.addWidget(QLabel("No additional information found."))

        # Show dialog
        dialog.exec_()

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)