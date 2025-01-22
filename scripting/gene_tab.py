from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QColor, QBrush, QFont
from PyQt5.QtCore import Qt
from scripting import drug_info


class GeneTab(QWidget):
    def __init__(self, gene, data_frame):
        super().__init__()
        self.gene = gene
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
        self.table.setStyleSheet("""
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
        """)

        # Populate table and finalize
        self.populate_table()
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.table.cellDoubleClicked.connect(self.onRowDoubleClicked)

    def populate_table(self):
        for row_index, row_data in self.data_frame.iterrows():
            for col_index, value in enumerate(row_data):
                item = QTableWidgetItem(str(value) if col_index != 2 else "")
                if col_index == 2:  # Pearson_Correlation column (numeric sorting)
                    item.setData(Qt.EditRole, float(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable) # Make all cells read-only

                # Apply alternating row colors
                if row_index % 2 == 0:
                    item.setBackground(QBrush(QColor("#e6f7ff")))  # Light blue for alternating rows
                else:
                    item.setBackground(QBrush(QColor("#ffffff")))  # White for regular rows

                self.table.setItem(row_index, col_index, item)

    def onRowDoubleClicked(self, row, col):
        gene_name = self.table.item(row, 0).text()
        drug_id = self.table.item(row, 3).text()
        drug_info.DrugInfo().onRowDoubleClicked(gene_name, drug_id)