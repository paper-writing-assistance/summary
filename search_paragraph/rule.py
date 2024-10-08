import os
import re
import json
import sys
import pandas as pd
from typing import Optional
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from base import _JSON, Config

class RuleSearch(Config):
    """
    A class for searching through rules based on specified criteria. It inherits from Config to use its configuration settings for file paths.
    """
    def __init__(self, 
                json_dir: str = '../upstage/json', 
                csv_dir: str = '../upstage/csv',
                mode: str = 'by_id',
                paper_id: Optional[str] = None, 
                is_figure: bool = True):
        """
        Initializes the RuleSearch object with directories for JSON and CSV, the mode of search, and optionally a specific paper ID.
        
        Parameters:
        - json_dir (str): Directory for JSON files.
        - csv_dir (str): Directory for CSV files.
        - mode (str): Mode of search, default is 'by_id'.
        - paper_id (Optional[str]): Specific paper ID for search in 'by_id' mode.
        
        Raises:
        - ValueError: If mode is 'by_id' but no paper_id is provided.
        - ValueError: If an invalid mode is provided.
        """
        super().__init__(json_dir, csv_dir, is_figure)
        self.mode = mode
        self.metadata = self.get_metadata()  # Load metadata
        self.figure_info = self.get_figure_info()  # Load figure information
        
        self.is_batch = mode != 'by_id'  # Determine if the search is in batch mode

        if self.is_batch:
            if mode not in _JSON:
                raise ValueError(f"Invalid mode: {mode}. Choose from {_JSON}")
            self.json_file_paths = self.get_json_files(mode)

        else:
            if paper_id is None:
                raise ValueError("Paper ID is required for 'by_id' mode.")
            # Find JSON file path for the specific paper ID
            self.json_file_paths = [self.metadata[self.metadata['id'] == paper_id]['json_file_path'].values[0]]
        
        # Extract paper IDs from the JSON file paths
        self.paper_ids = [self.metadata[self.metadata['json_file_path'] == path]['id'].values[0] for path in self.json_file_paths]

    def get_figure_number(self, caption_text):
        """
        Extracts and returns the figure number from the given caption text.
        
        Parameters:
        - caption_text (str): The caption text from which to extract the figure number.
        
        Returns:
        - str: The extracted figure number.
        
        Raises:
        - ValueError: If the caption text does not contain 'figure' or 'fig.'.
        """
        # Ensure caption_text is a string
        caption_text = str(caption_text).lower()
        
        # Define regex pattern to match 'figure ' or 'fig. ' followed by the figure number
        pattern = r'\b(figure|fig\.)\s*([^\s:.,]*)'
        
        # Search for the pattern in the caption text
        match = re.search(pattern, caption_text)
        
        if not match:
            raise ValueError("Caption text does not contain 'figure' or 'fig.'")
        
        # Extract the figure number
        prefix = match.group(1)
        figure_number = match.group(2)
        
        # Ensure the result always has "figure" as the prefix
        return f"figure {figure_number}"

    def get_table_number(self, caption_text):
        """
        Extracts and returns the table number from the given caption text.
        
        Parameters:
        - caption_text (str): The caption text from which to extract the table number.
        
        Returns:
        - str: The extracted table number.
        
        Raises:
        - ValueError: If the caption text does not contain 'table'.
        """
        # Ensure caption_text is a string
        caption_text = str(caption_text).lower()
        
        # Define regex pattern to match 'table ' followed by the table number
        pattern = r'\b(table)\s*([^\s:.,]*[^\s]*)'
        
        # Search for the pattern in the caption text
        match = re.search(pattern, caption_text)
        
        if not match:
            raise ValueError("Caption text does not contain 'table'")
        
        # Extract the table number
        prefix = match.group(1)
        table_number = match.group(2)
        
        # Check for ':' and '.' within the first 5 characters
        if ':' in table_number[:5] and '.' in table_number[:5]:
            table_number = table_number.split(':')[0]
        elif ':' in table_number[:5]:
            table_number = table_number.split(':')[0]
        elif '.' in table_number[:5]:
            table_number = table_number.split('.')[0]
        else:
            table_number = table_number.split()[0]  # Handle cases without '.' or ':'
        
        # Ensure the result always has "table" as the prefix
        return f"table {table_number}"

    
    def __search_paragraph__(self, element, search_word):
        """
        Private method to search within a paragraph element for a specific search word.
        
        Parameters:
        - element (dict): The JSON element, expected to be a paragraph.
        - search_word (str): The word or phrase to search for within the paragraph text.
        
        Returns:
        - str or None: Returns the paragraph ID if the search_word is found within the paragraph text; otherwise, returns None.
        """
        if search_word is None:
            return None
        # Only proceed if element is a paragraph
        if element["category"] != "paragraph":
            return None
        # If the search word is found in the paragraph text, return its ID
        if search_word in element['text'].lower():
            return element['id']
        # If the search word is not found, return None
        return None

    def search_paragraph(self):
        """
        Searches through paragraphs in each paper to find references to figure numbers.
        
        Returns:
        - dict: A dictionary where each key is a paper ID and each value is another dictionary mapping figure numbers to the IDs of paragraphs referencing those figures.
        """
        result_dict = defaultdict(lambda: defaultdict(tuple))
        # Iterate through each paper and its corresponding JSON file path
        for paper_id, json_file_path in zip(self.paper_ids, self.json_file_paths):
            # Get the caption texts for the current paper
            img_element_idx = self.figure_info.loc[self.figure_info['id'] == paper_id, 'img_element_idx'].reset_index(drop=True)
            caption_texts = self.figure_info.loc[self.figure_info['id'] == paper_id, 'caption']
            # Extract the figure numbers from the caption texts
            # Determine which function to use for extracting numbers based on self.is_figure
            if self.is_figure:
                extract_number_func = self.get_figure_number
            else:
                extract_number_func = self.get_table_number

            # Extract the figure or table numbers from the caption texts
            figure_numbers = [extract_number_func(caption) for caption in caption_texts]
            
            # Search for paragraphs referencing each figure number
            for idx, fig_num in enumerate(figure_numbers):
                paragraph_ids = []
                with open(json_file_path, 'r') as f:
                    json_data = json.load(f)
                for element in json_data['elements']:
                    paragraph_id = self.__search_paragraph__(element, fig_num)
                    
                    if paragraph_id:
                        paragraph_ids.append(paragraph_id)
                
                # Store the found paragraph IDs in the result dictionary
                if paper_id not in result_dict:
                    result_dict[paper_id] = {}
                result_dict[paper_id][img_element_idx[idx]] = (fig_num, ', '.join(map(str, paragraph_ids)))
                
        return result_dict

    def update_df(self, result_dict):
        """
        Updates the figure_info DataFrame with the results of the paragraph search.
        
        Parameters:
        - result_dict (dict): A dictionary mapping paper IDs to dictionaries that map figure numbers to paragraph IDs.
        """
        # Check if the columns exist, if not, initialize them
        if 'figure_number' not in self.figure_info.columns:
            self.figure_info['figure_number'] = None
        if 'rule_based_paragraphs' not in self.figure_info.columns:
            self.figure_info['rule_based_paragraphs'] = None

        # Iterate through the results dictionary to update the DataFrame
        for paper_id, fig_nums_paragraphs in result_dict.items():
            for img_element_idx, (fig_num, paragraph_ids) in fig_nums_paragraphs.items():
                # Update the DataFrame with the found paragraph IDs
                self.figure_info.loc[(self.figure_info['id'] == paper_id) & (self.figure_info['img_element_idx'] == img_element_idx), 'figure_number'] = fig_num
                self.figure_info.loc[(self.figure_info['id'] == paper_id) & (self.figure_info['figure_number'] == fig_num), 'rule_based_paragraphs'] = paragraph_ids

        # Save the updated DataFrame to a CSV file
        self.figure_info.to_csv(os.path.join(self.csv_dir, 'table_info_hand.csv'), index=False)

    def execute(self):
        """
        Executes the paragraph search process: searches paragraphs, updates the DataFrame, and returns the updated DataFrame.
        
        Returns:
        - DataFrame: The updated figure_info DataFrame including the rule-based paragraphs column.
        """
        # Perform the paragraph search
        result_dict = self.search_paragraph()
        # Update the DataFrame with the search results
        self.update_df(result_dict)
        # Return the updated DataFrame
        return self.figure_info
    
search = RuleSearch(mode='AI_VIT_X')
result_dict = search.execute()
print(result_dict)
# , paper_id='6298b6a2-0f92-11ef-8230-426932df3dcf'

breakpoint()