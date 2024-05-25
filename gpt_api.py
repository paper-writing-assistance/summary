import os
import requests
import openai
import json
import pandas as pd
from typing import Optional
from base import Config, _JSON

# Class GenerateTemplate for GPT-3 API

class Template:
    def __init__(self, system, prompt):
        if system:
            self.system = system
        else:
            self.system = "You are a professor who is good at explaining and very kind"
            
        if prompt:
            self.prompt = prompt
        else:
            self.prompt = """Please make a description to the figure/table of academic paper based on the following topic, caption, related paragraph contents. Ensure your responses are detailed, supportive, and easy to understand. 
            Title : [TITLE]
            Caption : [CAPTION]
            Paragraph : [PARAGRAPH]
            """
class GenerationTemplate:
    """
    Takes a prompt template and provides methods for filling in blanks. 
    The format is as follows. 
    [TITLE] : The title of the paper.
    [CAPTION] : The caption of the figure or table.
    [PARAGRAPH] : The paragraph text.
    """
    def __init__(self, template: Template):
        self.template = template 
        
    def fill(self, title, caption, paragraph):
        """
        Fills in the blanks in the template with the provided caption and paragraph.
        
        Parameters:
        - title (str): The title of the paper.
        - caption (str): The caption of the figure or table.
        - paragraph (str): The paragraph text.
        
        Returns:
        - str: The filled-in template.
        """
        filled_template = self.template.prompt.replace("[TITLE]", title).replace("[CAPTION]", caption).replace("[PARAGRAPH]", paragraph)
        return filled_template

# Class SummaryGenerator for GPT-3 API
class SummaryGenerator(GenerationTemplate):
    """
    Generates summaries using the GPT-3 API.
    """
    def __init__(self, df, api_key, mode):
        super().__init__()
        self.df = df
        self.api_key = api_key
        self.mode = mode
        self.metadata = self.get_metadata()
    
    def extract_caption(self, paper_id):
        """
        Extracts the caption from the DataFrame based on the paper ID.
        
        Parameters:
        - paper_id (str): The ID of the paper.
        
        Returns:
        - str: The caption of the figure or table.
        """
        caption = self.df[self.df['id'] == paper_id]['caption']
        return caption
    
    def get_paragraphs(self, paper_id):
        """
        Extracts the paragraph from the DataFrame based on the paper ID.
        
        Parameters:
        - paper_id (str): The ID of the paper.
        
        Returns:
        - str: The paragraph text.
        """
        if self.mode == "rule":
            criterion = "rule_based_paragraph_ids"
        elif self.mode == 'image':
            criterion = "img_similarity_paragraph_id"
        elif self.mode == 'text':
            criterion = "text_similarity_paragraph_id"
        else:
            raise ValueError("Mode must be either 'rule', 'image', or 'text'.")
        paragraph_id = self.df[self.df['id'] == paper_id][criterion]
        return paragraph_id
    
    def generate_template(self, paper_id, caption, paragraph_id):
        """
        Generates a template using the caption and paragraph.
        
        Parameters:
        - caption (str): The caption of the figure or table.
        - paragraph (str): The paragraph text.
        
        Returns:
        - str: The filled-in template.
        """
        # Get json data for the paragraph_id
        json_data = self.df[self.df['id'] == paper_id]['json_file_path']
        template = super().fill(caption, paragraph)
        return template
class SummaryGenerator(GenerationTemplate, Config):
    """
    Generates summaries using the GPT-3 API.
    """

    def __init__(self, json_dir: str = '../upstage/json', csv_dir: str = '../upstage/csv', mode: str = 'by_id', paper_id: Optional[str] = None, output_dir: str = './paragraph_info/output'):
        """
        Initializes the SimSearch object.

        Args:
        json_dir (str): Directory for JSON files.
        csv_dir (str): Directory for CSV files.
        mode (str): Mode of operation ('by_id' for individual paper, others for batch processing).
        paper_id (Optional[str]): Paper ID for 'by_id' mode.
        """
        super().__init__(json_dir, csv_dir)
        self.mode = mode
        self.output_dir = output_dir
        self.is_batch = mode != 'by_id'
        self.metadata, self.figure_info = self.load_metadata_and_figures()
        self.paper_id, self.json_file_path = self.prepare_file_paths(mode, paper_id)
        self.similar_paragraphs = self.initialize_similar_paragraphs()
        self.prepare_similar_paragraphs()

    def load_metadata_and_figures(self):
        """
        Loads metadata and figure information from the specified directories.

        Returns:
        Tuple[DataFrame, DataFrame]: A tuple containing metadata and figure information dataframes.
        """
        metadata = self.get_metadata()
        figure_info = self.get_figure_info()
        return metadata, figure_info

    def prepare_file_paths(self, mode, paper_id):
        """
        Prepares file paths based on the operating mode.

        Args:
        mode (str): Operation mode.
        paper_id (Optional[str]): Paper ID for 'by_id' mode.

        Returns:
        Tuple[List[str], List[str]]: A tuple containing lists of paper IDs and JSON file paths.
        """
        if self.is_batch:
            self.validate_mode(mode)
            json_file_path = self.get_json_files(mode)
            paper_id = [self.metadata[self.metadata['json_file_path'] == path]['id'].iloc[0] for path in json_file_path]
        else:
            self.validate_paper_id(paper_id)
            json_file_path = list(self.metadata[self.metadata['id'] == paper_id]['json_file_path'])
            paper_id = [paper_id]
        return paper_id, json_file_path

    def validate_mode(self, mode):
        """
        Validates the specified mode against allowed modes.

        Args:
        mode (str): Operation mode to validate.

        Raises:
        ValueError: If the mode is invalid.
        """
        if mode not in _JSON:
            raise ValueError(f"Invalid mode: {mode}. Choose from {_JSON}")

    def validate_paper_id(self, paper_id):
        """
        Validates the provided paper ID for 'by_id' mode.

        Args:
        paper_id (str): Paper ID to validate.

        Raises:
        ValueError: If the paper ID is not provided.
        """
        if paper_id is None:
            raise ValueError("Paper ID is required for 'by_id' mode.")

    def initialize_similar_paragraphs(self):
        """
        Initializes the structure for storing similar paragraphs.

        Returns:
        Dict: A dictionary to store similar paragraphs for each paper ID.
        """
        return {pid: {} for pid in self.paper_id}

    def prepare_similar_paragraphs(self):
        """
        Prepares the structure to hold similar paragraphs, organized by image element indices.
        """
        for pid in self.paper_id:
            img_element_idxs = self.figure_info[self.figure_info['id'] == pid]['img_element_idx']
            for idx in img_element_idxs:
                self.similar_paragraphs[pid][idx] = []
    
    def get_title(self, paper_id):
        """
        Retrieves the title for a given paper ID.

        Args:
        paper_id (str): The paper ID.

        Returns:
        str: The title of the paper.
        """
        title_data_path = '../upstage/info/title/title.json'
        
        # open the title.json file
        with open(title_data_path) as f:
            title_data = json.load(f)
        
        # iterate through each element in the list
        for paper in title_data:
            # check if the 'id' matches the given paper_id
            if paper['id'] == paper_id:
                return paper['text']
        
        # if no match is found, return None or an appropriate message
        return None
    
    def get_paragraphs(self, paper_id):
        """
        Retrieves summarized texts and figure information for a given paper ID and JSON file.

        Args:
        paper_id (str): The paper ID.
        json_file (str): The path to the JSON file.

        Returns:
        Tuple[Dict, Dict]: A tuple containing dictionaries for summarized texts and figure information.
        """  
        # get the paragraph_info/output/{paper_id}.csv file path
        path = os.path.join(self.output_dir, f'{paper_id}.csv')
        df = pd.read_csv(path)
                    
        summarized_texts = {}
        fig_info = {}
        
        for img_element_idx, paragraph_ids in self.similar_paragraphs[paper_id].items():
            paragraph_info = {}
            for paragraph_id in paragraph_ids:
                # if paragraph_id not in df['element_idx'], break and skip that paragraph
                if paragraph_id not in df['element_idx'].values:
                    continue
                
                paragraph_data = df[df['element_idx'] == paragraph_id]
                
                if paragraph_data.empty:
                    raise ValueError("Paragraph info not found")
                
                paragraph_info[paragraph_id] = paragraph_data['summarized_text'].iloc[0]
            
            summarized_texts[img_element_idx] = paragraph_info
            
            fig_info_data = self.figure_info[(self.figure_info['img_element_idx'] == img_element_idx) & (self.figure_info['id'] == paper_id)]
            if fig_info_data.empty:
                raise ValueError("Figure info not found")
            fig_info[img_element_idx] = fig_info_data
        
        return summarized_texts, fig_info
    
    def get_paragraph_text(self):
        """
        Identifies candidate paragraphs for each paper based on image element indices.
        """
        for idx, pid in enumerate(self.paper_id):
            with open(self.json_file_path[idx]) as f:
                json_data = json.load(f)
            for img_element_idx in self.similar_paragraphs[pid]:
                paragraph_ids = get_paragraph_ids(json_data, img_element_idx)
                self.similar_paragraphs[pid][img_element_idx] = paragraph_ids
        self.is_candidate = False

    def save_csv(self):
        """
        Saves the figure information as a CSV file in the configured directory.

        Returns:
        DataFrame: The figure information dataframe.
        """
        csv_path = os.path.join(self.csv_dir, 'figure_info_hand.csv')
        self.figure_info.to_csv(csv_path, index=False)
        return self.figure_info