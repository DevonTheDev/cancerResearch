from collections import Counter
import os
import re
import pandas as pd
import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.FilterCatalog as FilterCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from sklearn.decomposition import PCA

class MLFolderFinder:
    def __init__(self, target_folder="cancerResearch"):
        self.target_folder = target_folder
        self.parent_dir = self.find_parent_directory()
        self.processed_folder = os.path.join(
            self.parent_dir, "Processed_Data", "3_properties_merged", "ml_processed_properties"
        )
        os.makedirs(self.processed_folder, exist_ok=True)

    def find_parent_directory(self):
        """Finds the absolute path of the specified parent directory, ensuring it returns the target folder itself."""
        current_dir = os.path.abspath(__file__)  # Get current script path
        
        while True:
            parent_dir, last_folder = os.path.split(current_dir)  # Split path into parent & last directory
            if last_folder == self.target_folder:
                return current_dir  # Return "cancerResearch" directory itself
            if not last_folder:  # Stop if we reach the root directory
                raise FileNotFoundError(f"Parent folder '{self.target_folder}' not found in path hierarchy.")
            current_dir = parent_dir  # Move up one level

    def get_processed_folder(self):
        """Returns the path to the processed folder."""
        return self.processed_folder
    

# Constants
FIXED_COLUMNS = ["Drug", "Pearson_Correlation", "smiles"]  # Columns to validate CSV structure
DROPPED_COLUMNS = ["ExactMolWt", "Phi", "NumValenceElectrons", "MaxAbsPartialCharge", "MaxEStateIndex", "MaxAbsEStateIndex", "MaxPartialCharge", "MinAbsEStateIndex", "MinAbsPartialCharge", "MinEStateIndex", "MinPartialCharge", "MolMR", "NumRadicalElectrons", "NumUnspecifiedAtomStereoCenters", "NumValenceElectrons", "fr_Ndealkylation1", "fr_Ndealkylation2", "fr_allylic_oxid", "fr_azide", "fr_azo", "fr_barbitur", "fr_diazo", "fr_epoxide", "fr_isocyan", "fr_isothiocyan", "fr_nitrile", "fr_nitro", "fr_nitro_arom", "fr_nitro_arom_nonothro", "fr_nitroso", "qed", "Ipc", "AvgIpc", "BalabanJ", "BertzCT", "HallKierAlpha"]  # Columns to drop from properties
DROPPED_PREFIXES = ["BCUT2D_", "Chi", "EState_VSA", "FpDensityMorgan", "PEOE_VSA", "SMR_VSA", "VSA_EState"]
CUTOFF_PERCENT = 0.025  # Percentage of top and bottom data to select

# Directories
parent_dir = MLFolderFinder().parent_dir
processed_folder = os.path.join(parent_dir, "Processed_Data", "3_properties_merged")
output_folder = os.path.join(processed_folder, "ml_processed_properties")
os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

# Labels for readability of AUTOCORR columns
property_blocks = [
    ("GasteigerPartialCharge", 1, 8),
    ("Electronegativity", 9, 16),
    ("CovalentRadius", 17, 24),
    ("VDWRadius", 25, 32),
    ("AtomicMass", 33, 40),
    ("Polarizability", 41, 48),
    ("IonizationPotential", 49, 56),
    ("ElectronAffinity", 57, 64),
    ("Hardness", 65, 72),
    ("Softness", 73, 80),
    ("Electrophilicity", 81, 88),
    ("NMRShieldingConstant", 89, 96),
    ("EStateRelatedIndex1", 97, 104),
    ("EStateRelatedIndex2", 105, 112),
    ("EStateRelatedIndex3", 113, 120),
    ("EStateRelatedIndex4", 121, 128),
    ("EStateRelatedIndex5", 129, 136),
    ("FormalCharge", 137, 144),
    ("HBondDonors", 145, 152),
    ("HBondAcceptors", 153, 160),
    ("Aromaticity", 161, 168),
    ("PiElectronCount", 169, 176),
    ("Lipophilicity", 177, 184),
    ("MolarRefractivity", 185, 192),
]

# Labels for readability of Kappa columns
kappa_descriptions = {
    1: "Molecular_Size_Flexibility",
    2: "Branching_Degree",
    3: "Cyclicity_Ring_Complexity"
}

feature_smiles = {
    "fr_indole": "[nH]1cccc2c1cccc2",
    "fr_quinoline": "c1ccc2ncccc2c1",
    "fr_isoquinoline": "c1ccc2c(nccc2)c1",
    "fr_purine": "c1ncnc2ncnc12",
    "fr_pyrimidine": "c1cncnc1",
    "fr_pyrazole": "c1cn[nH]c1",
    "fr_pyridazine": "c1cnccn1",
    "fr_pyrazine": "c1cncnc1",
    "fr_triazole": "c1nnn[cH]1",
    "fr_oxadiazole(1,2,4)": "c1nocn1",
    "fr_oxadiazole(1,3,4)": "c1nnco1",
    "fr_thiadiazole(1,2,4)": "c1nscn1",
    "fr_thiadiazole(1,3,4)": "c1nncs1",
    "fr_tetrazine": "c1nnnnc1",
    "fr_benzofuran": "c1ccc2occc2c1",
    "fr_benzothiophene": "c1ccc2sccc2c1",
    "fr_benzimidazole": "c1ccc2[nH]c[nH]c2c1",
    "fr_thiazolidinedione": "O=C1CSC(=O)N1",
    "fr_napthalene": "c1ccc2ccccc2c1",
    "fr_s_double_bond_o": "[S]=O"
}

# Precompile SMARTS patterns into RDKit molecules
feature_patterns = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in feature_smiles.items()
}

# Load PAINS filters
params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
pains_catalog = FilterCatalog.FilterCatalog(params)

class MLFileCleaner:
    @staticmethod
    def load_and_process_csv(csv_file):
        """Loads a CSV file, filters data, and selects relevant columns."""
        file_path = os.path.join(processed_folder, csv_file)

        try:
            df = pd.read_csv(file_path)

            # Check for missing required columns
            missing_columns = [col for col in FIXED_COLUMNS if col not in df.columns]
            if missing_columns:
                print(f"Skipping {csv_file} - Missing columns: {missing_columns}")
                return None

            # Select top and bottom percentile rows
            cutoff = max(1, int(CUTOFF_PERCENT * len(df)))
            filtered_df = pd.concat([df.iloc[:cutoff], df.iloc[-cutoff:]])

            # Initialize processed_df with fixed columns and label
            processed_df = filtered_df[FIXED_COLUMNS].copy()
            processed_df["Label"] = processed_df["Pearson_Correlation"].apply(lambda x: 0 if x < 0 else 1)

            # Process additional properties if available
            if len(df.columns) > 15:
                additional_columns = filtered_df.iloc[:, 15:].copy()
                max_float = np.finfo(np.float32).max

                def is_valid_column(col):
                    return col.notna().all() and np.isfinite(col).all() and (col.abs() < max_float).all()

                # Clean additional columns
                additional_columns = additional_columns.loc[:, additional_columns.apply(is_valid_column)]
                additional_columns = additional_columns.loc[:, ~additional_columns.columns.str.startswith(tuple(DROPPED_PREFIXES))]
                additional_columns = additional_columns.drop(columns=DROPPED_COLUMNS, errors='ignore')
                additional_columns = additional_columns.loc[:, additional_columns.nunique() >= 3]

                # Rename AUTOCORR2D and Kappa columns
                renamed = {}
                autocorr_pattern = re.compile(r"AUTOCORR2D_(\d{1,3})$")
                kappa_pattern = re.compile(r"Kappa(\d{1,2})$")

                for col in additional_columns.columns:
                    match = autocorr_pattern.match(col)
                    if match:
                        index = int(match.group(1))
                        for label, start, end in property_blocks:
                            if start <= index <= end:
                                lag = index - start
                                renamed[col] = f"{col}_{label}@{lag}_lag"
                                break

                    match = kappa_pattern.search(col)
                    if match:
                        index = int(match.group(1))
                        if index in kappa_descriptions:
                            renamed[col] = f"{col}_{kappa_descriptions[index]}"

                additional_columns.rename(columns=renamed, inplace=True)
                processed_df = pd.concat([processed_df, additional_columns], axis=1)

                # Add derived features
                if {"MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"}.issubset(processed_df.columns):
                    lipinski_results = processed_df.apply(lipinski_violations, axis=1)
                    processed_df["rule_of_three_violations"] = processed_df.apply(rule_of_three_violations, axis=1)
                    processed_df = pd.concat([processed_df, lipinski_results], axis=1)

                if {"TPSA", "MolWt"}.issubset(processed_df.columns):
                    processed_df["Norm_TPSA"] = processed_df.apply(normalised_tpsa, axis=1)

                if "smiles" in filtered_df.columns:
                    smiles_subset = filtered_df["smiles"].iloc[:len(processed_df)].copy()

                    heterocycle_df = pd.DataFrame(smiles_subset.apply(count_heterocycles).tolist(), index=processed_df.index)
                    processed_df = pd.concat([processed_df, heterocycle_df], axis=1)

                    processed_df["PAINS_Filter_Pass"] = smiles_subset.apply(passes_pains_filter)
                    processed_df = processed_df[processed_df["PAINS_Filter_Pass"]].copy()
                    smiles_subset = smiles_subset.loc[processed_df.index]

                    bond_df = pd.DataFrame(smiles_subset.apply(count_double_triple_bonds).tolist(), index=processed_df.index)
                    processed_df = pd.concat([processed_df, bond_df], axis=1)

                    processed_df["IHD"] = smiles_subset.apply(calculate_ihd)

                    processed_df["fr_disulfide_bonds"] = smiles_subset.apply(count_disulfide_bonds)
                    processed_df["passes_veber_rule"] = processed_df.apply(passes_veber_rule, axis=1)
                    processed_df["flexibility_index"] = processed_df.apply(calculate_flexibility_index, axis=1)

                    # Get SMILES list
                    smiles_list = processed_df["smiles"].dropna().tolist().copy()

                    # Get the dominant Bemis–Murcko scaffold
                    dominant_scaffolds = get_frequent_scaffolds(smiles_list)

                    for idx, scaffold in enumerate(dominant_scaffolds):
                        processed_df[f"fr_scaffold_{idx}"] = processed_df["smiles"].apply(lambda sm: has_scaffold(sm, scaffold))

                    # Save the scaffold SMILES string to a file
                    with open(os.path.join(output_folder, f"{csv_file}_dominant_scaffold.smiles.txt"), "w") as f:
                        for scaffold in dominant_scaffolds:
                            f.write(f'{scaffold} \n')

                    # Calc AUTOCORR3D values
                    autocorr3d_data = smiles_subset.apply(get_autocorr3d).tolist()
                    autocorr3d_df = pd.DataFrame(autocorr3d_data, index=processed_df.index)
                    autocorr3d_df.columns = [f"AUTOCORR3D_{i}" for i in range(1, 1 + autocorr3d_df.shape[1])]
                    processed_df = pd.concat([processed_df, autocorr3d_df], axis=1)

                    # Calc planarity score and drop drugs where NaN to allow ML models to run
                    processed_df["planarity_score"] = smiles_subset.apply(planarity_score)
                    processed_df = processed_df.dropna(subset=["planarity_score"])

                columns_to_keep = ["PAINS_Filter_Pass"]

                label_col = processed_df["Label"]
                early_cols = processed_df.iloc[:, :3]
                filter_cols_all = processed_df.iloc[:, 3:]
                filter_cols = filter_cols_all.loc[:, (filter_cols_all.nunique() >= 3) | (filter_cols_all.columns.isin(columns_to_keep))]
                processed_df = pd.concat([early_cols, filter_cols], axis=1)

                # Reinsert Label at column index 1
                processed_df = processed_df.drop(columns=["Label"], errors="ignore")
                processed_df.insert(loc=2, column="Label", value=label_col)

            # Save and return path
            processed_file_path = os.path.join(output_folder, csv_file)
            processed_df.to_csv(processed_file_path, index=False)
            print(f"Processed and saved: {csv_file}")
            return processed_file_path

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            return None

    @staticmethod
    def run_file_clean():
        """Processes all CSV files in the input directory."""
        csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]

        for csv_file in csv_files:
            MLFileCleaner.load_and_process_csv(csv_file)  # Fixed function call

# ----------------------------------------------------------------------------------
# FUNCTIONS TO OBTAIN ADDITIONAL PROPERTIES FOR USE
# ----------------------------------------------------------------------------------

def lipinski_violations(row):
    """Returns number of Lipinski violations and compliance status for a single row."""
    violations = 0
    if row.get("MolWt", 0) > 500:
        violations += 1
    if row.get("MolLogP", 0) > 5:
        violations += 1
    if row.get("NumHDonors", 0) > 5:
        violations += 1
    if row.get("NumHAcceptors", 0) > 10:
        violations += 1
    meets_lipinski = 0 if violations >= 2 else 1
    return pd.Series([violations, meets_lipinski], index=["Lipinski_Violations", "Lipinski_Compliant"])

def normalised_tpsa(row):
    """Returns normalised TPSA value"""
    return row["TPSA"]/row["MolWt"]

def count_heterocycles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid Smiles: {smiles}")
        return {name: 0 for name in feature_patterns}  # Return 0s if SMILES is invalid

    return {
        name: len(mol.GetSubstructMatches(pattern))
        for name, pattern in feature_patterns.items()
    }

def passes_pains_filter(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False  # Treat invalid SMILES as failing the filter
    return not pains_catalog.HasMatch(mol)

def count_double_triple_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"num_double_bonds": 0, "num_triple_bonds": 0}
    
    double_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE)
    triple_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE)
    
    return {
        "num_double_bonds": double_bonds,
        "num_triple_bonds": triple_bonds
    }

def calculate_ihd(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0

    atom_counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())

    C = atom_counts.get("C", 0)
    H = atom_counts.get("H", 0)
    N = atom_counts.get("N", 0)
    halogens = sum(atom_counts.get(x, 0) for x in ["F", "Cl", "Br", "I"])

    ihd = C - (H + halogens) / 2 + N / 2 + 1
    return round(ihd, 4)

def count_disulfide_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return 0

    disulfide_pattern = Chem.MolFromSmarts("[S]-[S]")
    matches = mol.GetSubstructMatches(disulfide_pattern)

    # Each disulfide bond involves 2 sulfur atoms, but RDKit will return each match once
    return len(matches)

def passes_veber_rule(row):
    if row.get("TPSA") is not None and row.get("NumRotatableBonds") is not None:
        if row["TPSA"] <= 140 and row["NumRotatableBonds"] <= 10:
            return 1
    return 0
    
def calculate_flexibility_index(row):
    num_rotatable_bonds = row.get("NumRotatableBonds")
    heavy_atom_count = row.get("HeavyAtomCount")

    if num_rotatable_bonds is None or heavy_atom_count is None or heavy_atom_count == 0:
        return 0.0

    return round(num_rotatable_bonds / heavy_atom_count, 4)

def rule_of_three_violations(row):
    mol_wt = row.get("MolWt")
    log_p = row.get("MolLogP")
    h_donors = row.get("NumHDonors")
    h_acceptors = row.get("NumHAcceptors")
    rotatable_bonds = row.get("NumRotatableBonds")

    violations = 0

    if mol_wt is None or mol_wt > 300:
        violations += 1
    if log_p is None or log_p > 3:
        violations += 1
    if h_donors is None or h_donors > 3:
        violations += 1
    if h_acceptors is None or h_acceptors > 3:
        violations += 1
    if rotatable_bonds is None or rotatable_bonds > 3:
        violations += 1

    return violations

# Returns a list of scaffolds that 2 or more molecules share
def get_frequent_scaffolds(smiles_list, min_count=2):
    scaffolds = []
    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold:
                scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=True)
                if scaffold_smiles.strip():  # This ensures the SMILES string is not empty
                    scaffolds.append(scaffold_smiles)

    if not scaffolds:
        return []

    scaffold_counts = Counter(scaffolds)
    frequent_scaffolds = [smi for smi, count in scaffold_counts.items() if count >= min_count]

    return frequent_scaffolds

def has_scaffold(smiles, scaffold_smarts):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = Chem.MolFromSmiles(scaffold_smarts)

    if mol and scaffold:
        return len(mol.GetSubstructMatches(scaffold))
    else:
        return 0
    
def planarity_score(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.NaN

    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG()) # Generate 3D coordinates
        AllChem.UFFOptimizeMolecule(mol)
    except:
        return np.NaN

    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

    pca = PCA(n_components=3)
    pca.fit(coords)

    planarity_score = pca.explained_variance_ratio_[2] # Get Z-axis of PCA graph

    return float(planarity_score)

def get_autocorr3d(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        return list(rdMolDescriptors.CalcAUTOCORR3D(mol))
    except:
        return [np.nan] * 80  # Adjust if RDKit returns a different length