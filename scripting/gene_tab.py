import logging
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QPushButton, QDialog, QTabWidget, QScrollArea, QLabel, QFormLayout
)
from PyQt5.QtGui import QColor, QBrush, QFont
from PyQt5.QtCore import Qt
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os

class BaseTab(QWidget):
    def __init__(self, data_frame):
        super().__init__()
        self.data_frame = data_frame
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        self.table = QTableWidget(len(self.data_frame), len(self.data_frame.columns))
        layout.addWidget(self.table)

        # Set up table headers and appearance
        self.table.setHorizontalHeaderLabels(self.data_frame.columns)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setFont(QFont("Arial", 10, QFont.Bold))
        self.table.verticalHeader().setVisible(False)  # Hide row numbers
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(self.get_table_stylesheet())

        # Populate table and finalize
        self.populate_table()
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()

    def populate_table(self):
        # Dynamically determine numeric columns
        numeric_column_indexes = [
            i for i, dtype in enumerate(self.data_frame.dtypes) if np.issubdtype(dtype, np.number)
        ]

        for row_index, row_data in self.data_frame.iterrows():
            for col_index, value in enumerate(row_data):
                item = QTableWidgetItem()
                try:
                    if col_index in numeric_column_indexes:  # Numeric sorting for numeric columns
                        item.setData(Qt.EditRole, float(value))  # Properly set numeric data
                    else:
                        item.setData(Qt.EditRole, str(value))  # For non-numeric columns, use string data
                except ValueError:
                    item.setData(Qt.EditRole, 0.0)  # Fallback value for non-convertible data

                item.setText(str(value))  # Set the display text for the cell
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Make all cells read-only

                # Apply alternating row colors
                color = QColor("#e6f7ff") if row_index % 2 == 0 else QColor("#ffffff")
                item.setBackground(QBrush(color))

                self.table.setItem(row_index, col_index, item)

    @staticmethod
    def get_table_stylesheet():
        """Returns the shared stylesheet for the table."""
        return """
            QTableWidget {
                gridline-color: #ccc;
                font-family: Arial;
                font-size: 10pt;
                background-color: #f9f9f9;
            }
            QHeaderView::section {
                background-color: #0078D7;
                color: white;
                padding: 4px;
                font-size: 10pt;
                border: 1px solid #ccc;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """


class GeneTab(BaseTab):
    def __init__(self, gene, data_frame):
        self.gene = gene
        super().__init__(data_frame)


class PropertiesTab(BaseTab):
    def __init__(self, gene, data_frame):
        self.gene = gene
        super().__init__(data_frame)

    def initUI(self):
        layout = QVBoxLayout(self)

        # Add a button above the table
        self.button = QPushButton("View Volcano Plot", self)
        self.button.setStyleSheet("font-size: 12pt; padding: 5px; background-color: #bdbdbd; color: black;")
        self.button.clicked.connect(self.show_volcano_plot)
        layout.addWidget(self.button)

        self.table = QTableWidget(len(self.data_frame), len(self.data_frame.columns))
        layout.addWidget(self.table)

        # Set up table headers and appearance
        self.table.setHorizontalHeaderLabels(self.data_frame.columns)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setFont(QFont("Arial", 10, QFont.Bold))
        self.table.verticalHeader().setVisible(False)  # Hide row numbers
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(self.get_table_stylesheet())

        # Populate table and finalize
        self.populate_table()
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()

    def show_volcano_plot(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Volcano Plot for {self.gene}")
        dialog.resize(800, 600)

        layout = QVBoxLayout(dialog)

        # Define test types with their respective colors
        test_types = {
            "t-test": {"color": "blue", "label": "T-Test"},
            "Mann-Whitney U": {"color": "green", "label": "Mann-Whitney U"}
        }

        # Create a single figure for both test types
        fig, ax = plt.subplots()

        # Plot data for each test type
        for test_type, properties in test_types.items():
            data = self.data_frame[self.data_frame["Test Type"] == test_type]
            if data.empty:
                continue

            ax.scatter(
                data["Effect Size"],
                -np.log10(data["P-Value"]),
                color=properties["color"],
                label=properties["label"],  # Add legend label
                alpha=0.7  # Set transparency for better visualization
            )

        # Set plot titles and labels
        ax.set_title(f"Volcano Plot for {self.gene}")
        ax.set_xlabel("Effect Size")
        ax.set_ylabel("-log10(P-Value)")

        # Add a legend
        ax.legend(title="Test Type")

        # Embed the figure into the dialog
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        dialog.exec_()

import os
import logging
import joblib
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QTabWidget,
    QPushButton, QScrollArea, QFormLayout, QHeaderView, QDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns

class MLResultsTab(QWidget):
    def __init__(self, model_type):
        """
        A PyQt5 class that loads all trained ML models and displays results for each gene in separate tabs.
        Supports Random Forest, XGBoost, and Neural Networks.
        """
        super().__init__()
        self.models = {}  # Store models by gene name
        self.results = {}  # Store extracted results per gene
        self.model_type = model_type  # "random_forest", "xg_boost", or "neural_network"

        self.load_models_from_parent_directory(self.model_type)
        self.initUI()

    def load_models_from_parent_directory(self, model_type, force_reload=False):
        """Searches the appropriate directory for trained models and loads them."""
        model_dirs = {
            "random_forest": "random_forest_models",
            "xg_boost": "xgboost_models",
            "neural_network": "neural_net_models",
        }

        parent_dir = os.path.join(os.path.dirname(os.getcwd()), "cancerResearch", "ml_models", model_dirs[model_type])
        logging.info(f"Searching for models in: {parent_dir}")

        if force_reload:
            self.models.clear()
            self.results.clear()

        if model_type in ["random_forest", "xg_boost"]:  # Joblib models
            for file in os.listdir(parent_dir):
                if file.endswith(".joblib") and file.startswith(f"{model_type}_"):
                    gene_name = file.replace(f"{model_type}_", "").replace(".joblib", "").replace("_properties_merged", "")
                    model_path = os.path.join(parent_dir, file)

                    try:
                        model_data = joblib.load(model_path)
                        self.models[gene_name] = model_data["model"]
                        self.results[gene_name] = {
                            "model_type": model_type,
                            "model_path": model_path,
                            "feature_importances": model_data.get("feature_importances", []),
                            "selected_features": model_data.get("selected_features", []),
                            "accuracy": model_data.get("accuracy", None),
                            "hyperparameters": model_data.get("best_params", {})
                        }
                        logging.info(f"Loaded {model_type} model for gene: {gene_name}")

                    except Exception as e:
                        logging.error(f"Failed to load {file}: {e}")

        elif model_type == "neural_network":  # TensorFlow .h5 models
            for file in os.listdir(parent_dir):
                if file.endswith(".h5") and file.startswith("mlp_"):
                    gene_name = file.replace("mlp_", "").replace(".h5", "")
                    model_path = os.path.join(parent_dir, file)

                    try:
                        model = tf.keras.models.load_model(model_path)
                        self.models[gene_name] = model
                        self.results[gene_name] = {
                            "model_type": "neural_network",
                            "model_path": model_path,
                            "accuracy": None,  # Neural network models might not store accuracy inside the .h5 file
                            "feature_importances": [],  # No direct feature importances for MLP
                            "selected_features": [],
                            "hyperparameters": {}
                        }
                        logging.info(f"Loaded Neural Network model for gene: {gene_name}")

                    except Exception as e:
                        logging.error(f"Failed to load {file}: {e}")

        if force_reload:
            self.refresh_UI()

    def refresh_UI(self):
        """Refreshes the UI by reloading all tabs."""
        self.tab_widget.clear()
        for gene, result in self.results.items():
            self.add_gene_tab(gene, result)

    def initUI(self):
        """Initializes the UI with tabs for each gene's ML results."""
        layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        for gene, result in self.results.items():
            self.add_gene_tab(gene, result)

    def add_gene_tab(self, gene, result):
        """Creates a separate tab for each gene's ML results."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        # Model Type Display
        model_type_label = QLabel(f"Model Type: {result['model_type'].replace('_', ' ').title()}")
        tab_layout.addWidget(model_type_label)

        # Performance Metrics
        metric_label = QLabel(
            f"Model Accuracy: {result['accuracy']:.4f}" if result["accuracy"] is not None else "Accuracy: Not Available"
        )
        tab_layout.addWidget(metric_label)

        # Feature Importance Table (only for Random Forest and XGBoost)
        if result["feature_importances"]:
            feature_table = QTableWidget(len(result["selected_features"]), 2)
            feature_table.setHorizontalHeaderLabels(["Feature", "Importance"])
            feature_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

            sorted_features = sorted(zip(result["selected_features"], result["feature_importances"]),
                                     key=lambda x: abs(x[1]), reverse=True)

            for i, (feature, value) in enumerate(sorted_features):
                feature_table.setItem(i, 0, QTableWidgetItem(str(feature)))
                feature_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))

            tab_layout.addWidget(QLabel("Feature Importance"))
            tab_layout.addWidget(feature_table)

        # Model Properties Table
        properties_widget = self.create_model_properties_table(result)
        tab_layout.addWidget(QLabel("Model Properties"))
        tab_layout.addWidget(properties_widget)

        # Visualization Button (for feature importance, only for RF/XGB)
        if result["feature_importances"]:
            button_feature_importance = QPushButton("Feature Importance", self)
            button_feature_importance.clicked.connect(lambda: self.show_feature_importance(gene))
            tab_layout.addWidget(button_feature_importance)

        self.tab_widget.addTab(tab, gene)

    def create_model_properties_table(self, result):
        """Creates a scrollable widget to display model properties."""
        scroll_area = QScrollArea()
        container = QWidget()
        layout = QFormLayout(container)

        for key, value in result.items():
            if key in ["feature_importances", "selected_features"]:
                continue
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    layout.addRow(QLabel(f"{key} - {sub_key}"), QLabel(str(sub_value)))
            else:
                layout.addRow(QLabel(key), QLabel(str(value)))

        scroll_area.setWidget(container)
        scroll_area.setWidgetResizable(True)
        return scroll_area

    def show_feature_importance(self, gene):
        """Displays a feature importance plot for a specific gene."""
        if gene not in self.results:
            return

        result = self.results[gene]
        feature_values = result["feature_importances"]
        selected_features = np.array(result["selected_features"])

        sorted_features, sorted_values = zip(*sorted(zip(selected_features, feature_values), key=lambda x: abs(x[1]), reverse=True))

        dialog = self.create_dialog(f"Feature Importance - {gene}")
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.barplot(x=list(sorted_values), y=list(sorted_features), ax=ax)

        ax.set_title(f"Feature Importance ({gene})")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")

        self.add_canvas_to_dialog(dialog, fig)

    @staticmethod
    def create_dialog(title):
        """Creates a modal dialog for displaying plots."""
        dialog = QDialog()
        dialog.setWindowTitle(title)
        dialog.resize(1600, 1200)
        layout = QVBoxLayout(dialog)
        return dialog, layout

    @staticmethod
    def add_canvas_to_dialog(dialog_layout, fig):
        """Adds a matplotlib figure to the dialog."""
        dialog, layout = dialog_layout
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        dialog.exec_()