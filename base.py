import os
import json
import pandas as pd

# Constants for valid JSON and CSV file types
_JSON = ['AI_VIT_O', 'AI_VIT_X', 'by_id']
_CSV = ['figure_info_hand.csv', 'table_info_hand.csv', 'metadata.csv']

class Config:
    """
    Configuration class for handling file paths and loading data from JSON and CSV files.
    """
    def __init__(self, json_dir:str = '../upstage/json', csv_dir:str='./upstage/csv'):
        """
        Initializes the configuration with directories for JSON and CSV files.
        
        Parameters:
        - json_dir (str): Directory path where JSON files are stored.
        - csv_dir (str): Directory path where CSV files are stored.
        """
        self.json_dir = json_dir
        self.csv_dir = csv_dir
    
    def open_json(self, file:str):
        """
        Opens and loads a JSON file based on the specified file path.
        
        Parameters:
        - file (str): Path to the JSON file to open.
        
        Returns:
        - Dictionary containing the JSON data.
        """
        with open(file, 'r') as f:
            data = json.load(f)
        return data
    
    def get_json_files(self, mode:str):
        """
        Retrieves a list of JSON files based on the specified mode.
        
        Parameters:
        - mode (str): The mode specifying the subset of JSON files to retrieve.
        
        Returns:
        - List of JSON file names within the specified mode directory.
        
        Raises:
        - ValueError: If the specified mode is invalid.
        """
        if mode not in _JSON:
            raise ValueError(f"Invalid mode: {mode}. Choose from {_JSON}")
        return [os.path.join(self.json_dir, mode, filename) for filename in os.listdir(os.path.join(self.json_dir, mode))]
    
    def get_csv_files(self, file:str):
        """
        Retrieves the file path for a specified CSV file.
        
        Parameters:
        - file (str): Name of the CSV file to retrieve.
        
        Returns:
        - Path to the specified CSV file.
        
        Raises:
        - ValueError: If the specified file name is invalid.
        """
        if file not in _CSV:
            raise ValueError(f"Invalid file: {file}. Choose from {_CSV}")
        return os.path.join(self.csv_dir, file)
    
    def get_figure_info(self):
        """
        Loads and returns the figure information from the figure_info.csv file as a DataFrame.
        
        Returns:
        - DataFrame containing the data from figure_info.csv.
        """
        figure_info = self.get_csv_files('table_info_hand.csv')
        df = pd.read_csv(figure_info)
        return df
    
    def get_metadata(self):
        """
        Loads and returns the metadata from the metadata.csv file as a DataFrame.
        
        Returns:
        - DataFrame containing the data from metadata.csv.
        """
        metadata = self.get_csv_files('metadata.csv')
        df = pd.read_csv(metadata)
        return df
