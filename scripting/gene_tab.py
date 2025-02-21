from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QPushButton, QDialog
)
from PyQt5.QtGui import QColor, QBrush, QFont
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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