import os
import json

class OrderedFolderCreator:
    def __init__(self, index_file="folder_index.json"):
        """
        Initialize the OrderedFolderCreator with an optional index file.

        :param index_file: Path to the file that stores the current index.
        """
        self.index_file = index_file
        self.current_index = self._load_index()

    def _load_index(self):
        """
        Load the current index from the index file, or initialize it to 1 if the file doesn't exist.

        :return: The current index as an integer.
        """
        if os.path.exists(self.index_file):
            with open(self.index_file, "r") as f:
                return json.load(f).get("index", 1)
        return 1

    def _save_index(self):
        """
        Save the current index to the index file.
        """
        with open(self.index_file, "w") as f:
            json.dump({"index": self.current_index}, f)

    def create_folder(self, base_path, folder_name):
        """
        Create a folder with the current index prepended to the folder name.

        :param base_path: The base directory where the folder will be created.
        :param folder_name: The desired name of the folder (without the index).
        :return: The full path of the created folder.
        """
        # Ensure the base path exists
        os.makedirs(base_path, exist_ok=True)

        # Create the indexed folder name
        indexed_folder_name = f"{self.current_index:1d}_{folder_name}"
        full_path = os.path.join(base_path, indexed_folder_name)

        # Create the folder
        os.makedirs(full_path, exist_ok=True)

        # Increment and save the index
        self.current_index += 1
        self._save_index()

        return full_path