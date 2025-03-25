from gui_app import GeneDrugApp
import os
import pandas as pd
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QLabel, QDialog
import pubchempy as pcp

class DrugInfo(QWidget):
    def __init__(self):
        super().__init__()

    def onRowDoubleClicked(self, gene_name, drug_id):
        try:
            data_frame = pd.read_csv(self.get_file_path(gene_name))
            extracted_row = self.get_extracted_row(data_frame, drug_id)
            pubchem_cid, smiles, inchi_key = self.extract_drug_info(extracted_row)
            compound_info = self.fetch_pubchem_data(pubchem_cid)
            self.display_drug_info(gene_name, pubchem_cid, smiles, inchi_key, compound_info)
        except Exception as e:
            self.show_error(str(e))

    def get_file_path(self, gene_name):
        raw_data_dir = os.path.dirname(GeneDrugApp.getRawDataDirectory())
        file_path = os.path.join(raw_data_dir, "Processed_Data", "pearson_correlations", f"{gene_name}_properties_merged.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path

    def get_extracted_row(self, data_frame, drug_id):
        extracted_row = data_frame[data_frame['id'] == drug_id]
        if extracted_row.empty:
            raise ValueError(f"No data found for drug_id: {drug_id}.")
        return extracted_row

    def extract_drug_info(self, row):
        pubchem_cid = row['pubchem_cid'].values[0]
        if pd.isna(pubchem_cid):
            raise ValueError("PubChem CID is missing.")
        return str(int(pubchem_cid)), row['smiles'].values[0], row['InChIKey'].values[0]

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

        # Display extracted and additional data
        labels = [
            f"Gene Name: {gene_name}",
            f"PubChem CID: {pubchem_cid}",
            f"SMILES: {smiles}",
            f"InChIKey: {inchi_key}",
            f"Molecular Weight: {compound_info.get('MolecularWeight', 'N/A')}",
            f"Canonical SMILES: {compound_info.get('CanonicalSMILES', 'N/A')}",
            f"IUPAC Name: {compound_info.get('IUPACName', 'N/A')}"
        ]
        for label in labels:
            layout.addWidget(QLabel(label))

        dialog.exec_()

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)