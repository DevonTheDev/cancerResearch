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

class MLResultsTab(QWidget):
    def __init__(self):
        """
        A PyQt5 class that loads all trained ML models (.joblib files) from the parent directory
        and displays results for each gene in separate tabs.
        """
        super().__init__()
        self.models = {}  # Store models by gene name
        self.results = {}  # Store extracted results per gene
        self.load_models_from_parent_directory()
        self.initUI()

    def load_models_from_parent_directory(self):
        """Searches the parent directory for .joblib files and loads trained models and metadata."""
        parent_dir = os.path.join(os.path.dirname(os.getcwd()), "cancerResearch")  # Get parent directory
        logging.info(f"Searching for joblib models in: {parent_dir}")

        for file in os.listdir(parent_dir):
            if file.endswith(".joblib"):
                # Determine model type (Random Forest or ElasticNet)
                model_type = "random_forest" if file.startswith("random_forest_") else "elasticnet"
                gene_name = file.replace(f"{model_type}_", "").replace(".joblib", "").replace("_properties_merged", "")
                model_path = os.path.join(parent_dir, file)

                try:
                    # Load the joblib dictionary (contains model and metadata)
                    model_data = joblib.load(model_path)

                    self.models[gene_name] = model_data["model"]  # Store model separately

                    # Store extracted results, adjusting for model type
                    if model_type == "random_forest":
                        self.results[gene_name] = {
                            "model_type": "Random Forest",
                            "model_path": model_path,
                            "feature_importances": model_data["feature_importances"],
                            "selected_features": model_data["selected_features"],
                            "accuracy": model_data.get("accuracy", None),
                            "hyperparameters": model_data.get("best_params", {}),
                        }
                    else:  # ElasticNet model
                        self.results[gene_name] = {
                            "model_type": "ElasticNet",
                            "model_path": model_path,
                            "feature_coefficients": model_data["feature_coefficients"],
                            "selected_features": model_data["selected_features"],
                            "r2_score": model_data.get("r2_score", None),
                            "mse": model_data.get("mse", None),
                            "hyperparameters": model_data.get("best_params", {}),
                        }

                    logging.info(f"Loaded {model_type} model for gene: {gene_name}")

                except Exception as e:
                    logging.error(f"Failed to load {file}: {e}")

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
        model_type_label = QLabel(f"Model Type: {result['model_type']}")
        tab_layout.addWidget(model_type_label)

        # Performance Metrics
        if result["model_type"] == "Random Forest":
            metric_label = QLabel(f"Model Accuracy: {result['accuracy']:.4f}" if result["accuracy"] is not None else "Accuracy: Not Available")
        else:  # ElasticNet
            metric_label = QLabel(
                f"R² Score: {result['r2_score']:.4f}, MSE: {result['mse']:.4f}" if result["r2_score"] is not None else "Metrics Not Available"
            )
        tab_layout.addWidget(metric_label)

        # Feature Importance Table
        feature_table = QTableWidget(len(result["selected_features"]), 2)
        feature_table.setHorizontalHeaderLabels(
            ["Feature", "Importance" if result["model_type"] == "Random Forest" else "Coefficient"]
        )
        feature_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Sort features by importance/coefficient
        feature_values = result["feature_importances"] if result["model_type"] == "Random Forest" else result["feature_coefficients"]
        sorted_features = sorted(zip(result["selected_features"], feature_values), key=lambda x: abs(x[1]), reverse=True)

        for i, (feature, value) in enumerate(sorted_features):
            feature_table.setItem(i, 0, QTableWidgetItem(str(feature)))
            feature_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))

        tab_layout.addWidget(QLabel("Feature Importance" if result["model_type"] == "Random Forest" else "Feature Coefficients"))
        tab_layout.addWidget(feature_table)

        # Model Properties Table
        properties_widget = self.create_model_properties_table(result)
        tab_layout.addWidget(QLabel("Model Properties"))
        tab_layout.addWidget(properties_widget)

        # Visualization Buttons
        button_feature_importance = QPushButton("Feature Importance", self)
        button_feature_importance.clicked.connect(lambda: self.show_feature_importance(gene))
        tab_layout.addWidget(button_feature_importance)

        self.tab_widget.addTab(tab, gene)

    def create_model_properties_table(self, result):
        """Creates a scrollable widget to display all extracted model properties."""
        scroll_area = QScrollArea()
        container = QWidget()
        layout = QFormLayout(container)

        for key, value in result.items():
            if key in ["feature_importances", "selected_features", "feature_coefficients"]:  # Skip redundant data
                continue

            if isinstance(value, dict):  # Handle hyperparameters separately
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

        # Get feature values based on model type
        feature_values = result["feature_importances"] if result["model_type"] == "Random Forest" else result["feature_coefficients"]
        selected_features = np.array(result["selected_features"])

        # Sort the selected features **along with** their feature values
        sorted_features, sorted_values = zip(*sorted(zip(selected_features, feature_values), key=lambda x: abs(x[1]), reverse=True))

        # Prepare visualization
        dialog = self.create_dialog(f"Feature Importance - {gene}")
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot directly using the sorted selected features
        sns.barplot(x=list(sorted_values), y=list(sorted_features), ax=ax)

        ax.set_title(f"Feature Importance ({gene})")
        ax.set_xlabel("Importance Score" if result["model_type"] == "Random Forest" else "Coefficient Value")
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