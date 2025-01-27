from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QColor, QBrush, QFont
from PyQt5.QtCore import Qt
import numpy as np

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
                item = QTableWidgetItem(str(value))
                if col_index in numeric_column_indexes:  # Numeric sorting of numeric columns
                    try:
                        item.setData(Qt.EditRole, float(value))
                    except ValueError:
                        item.setData(Qt.EditRole, 0.0)  # Fallback value for non-convertible data
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