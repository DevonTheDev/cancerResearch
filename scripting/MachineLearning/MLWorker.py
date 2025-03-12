from scripting.MachineLearning import random_forest, xg_boost, neural_network
from PyQt5.QtCore import QThread, pyqtSignal

class MLWorker(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, model_type="random_forest"):
        super().__init__()
        self.model_type = model_type

    def run(self):
        print(f"Running {self.model_type.replace('_', ' ').title()} model...")

        if self.model_type == "random_forest":
            ml_results = random_forest.run_ml_model()
        elif self.model_type == "xg_boost":
            ml_results = xg_boost.run_ml_model()
        elif self.model_type == "neural_network":
            ml_results = neural_network.run_mlp()

        # Neural Network returns a list of all models, we need to convert it to a dict for emitting
        if isinstance(ml_results, list):
            ml_results = {d["model_path"]: {"model_path": d["model_path"], "accuracy": d["accuracy"]} for d in ml_results} if ml_results else None

        self.finished.emit(ml_results)