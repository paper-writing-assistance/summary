from typing import Optional
import json
import os
from PIL import Image 
import pandas as pd
import numpy as np
import torch
import argparse

from transformers import CLIPProcessor, CLIPModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

from utils import get_paragraph_ids, get_summarized_text
from base import _JSON, Config

class SimSearch(Config):
    def __init__(self, json_dir: str = './json', csv_dir: str = './csv', mode: str = 'by_id', paper_id: Optional[str] = None):
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
                
    def get_summarized_texts_and_fig_info(self, paper_id, json_file):
        """
        Retrieves summarized texts and figure information for a given paper ID and JSON file.

        Args:
        paper_id (str): The paper ID.
        json_file (str): The path to the JSON file.

        Returns:
        Tuple[Dict, Dict]: A tuple containing dictionaries for summarized texts and figure information.
        """  
        with open(json_file) as f:
            json_data = json.load(f)
            
        summarized_texts = {}
        fig_info = {}
        
        for img_element_idx, paragraph_ids in self.similar_paragraphs[paper_id].items():
            summarized_texts[img_element_idx] = get_summarized_text(json_data, paragraph_ids)
            fig_info_data = self.figure_info[(self.figure_info['img_element_idx'] == img_element_idx) & (self.figure_info['id'] == paper_id)]
            if fig_info_data.empty:
                raise ValueError("Figure info not found")
            fig_info[img_element_idx] = fig_info_data
        
        return summarized_texts, fig_info
    
    def get_candidate_paragraphs(self):
        """
        Identifies candidate paragraphs for each paper based on image element indices.
        """
        for idx, pid in enumerate(self.paper_id):
            with open(self.json_file_path[idx]) as f:
                json_data = json.load(f)
            for img_element_idx in self.similar_paragraphs[pid]:
                paragraph_ids = get_paragraph_ids(json_data, img_element_idx)
                self.similar_paragraphs[pid][img_element_idx] = paragraph_ids
        self.is_candidate = True

    def save_csv(self):
        """
        Saves the figure information as a CSV file in the configured directory.

        Returns:
        DataFrame: The figure information dataframe.
        """
        csv_path = os.path.join(self.csv_dir, 'figure_info.csv')
        self.figure_info.to_csv(csv_path, index=False)
        return self.figure_info

class SearchStrategy:
    def execute(self, search: SimSearch):
        """
        Execute the search strategy.

        This method needs to be implemented by subclasses.

        Args:
        search (SimSearch): The SimSearch instance to execute the strategy on.
        """
        raise NotImplementedError

class TextSearchStrategy(SearchStrategy):
    def execute(self, search: SimSearch):
        """
        Executes a text-based search strategy.

        This method utilizes sentence embeddings to find the paragraph most similar to the caption of each figure.

        Args:
        search (SimSearch): The SimSearch instance containing the data for execution.
        """
        # Initialize the sentence transformer model
        model = SentenceTransformer('ABrinkmann/sbert_xtremedistil-l6-h256-uncased-mean-cosine-h32')
        
        # Iterate through each paper and its corresponding JSON file
        for pid, json_file in zip(search.paper_id, search.json_file_path):
            summarized_texts, fig_info = search.get_summarized_texts_and_fig_info(pid, json_file)
            
            # For each figure in a paper, compute the similarity between the figure caption and paragraphs
            for img_element_idx, texts in summarized_texts.items():
                summarized_texts_list = list(texts.values())
                embeddings = model.encode(summarized_texts_list, convert_to_tensor=True)
                caption = fig_info[img_element_idx]['caption'].iloc[0]
                caption_embedding = model.encode([caption], convert_to_tensor=True)
                
                # Compute cosine similarity scores and find the best matching paragraph
                cosine_scores = util.pytorch_cos_sim(caption_embedding, embeddings)[0]
                best_index = cosine_scores.argmax()
                best_paragraph_id = list(texts.keys())[best_index]
                
                # Update the figure information with the ID of the best matching paragraph
                search.figure_info.loc[(search.figure_info['img_element_idx'] == img_element_idx) & (search.figure_info['id'] == pid), 'text_similarity_paragraph_id'] = best_paragraph_id
                
        # Ensure the 'text_similarity_paragraph_id' column is of integer type
        search.figure_info['text_similarity_paragraph_id'] = search.figure_info['text_similarity_paragraph_id'].astype('Int64')
        return search.figure_info

class ImgSearchStrategy(SearchStrategy):
    def execute(self, search: SimSearch):
        """
        Executes an image-based search strategy.

        This method utilizes CLIP to find the paragraph most relevant to the image content.

        Args:
        search (SimSearch): The SimSearch instance containing the data for execution.
        """
        # Initialize the CLIP processor and model
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        
        # Iterate through each paper and its corresponding JSON file
        for paper_id, json_file in zip(search.paper_id, search.json_file_path):
            summarized_texts, fig_info = search.get_summarized_texts_and_fig_info(paper_id, json_file)
            
            # For each figure in a paper, compute the relevance between the figure image and paragraphs
            for img_element_idx, texts in summarized_texts.items():
                img_path = fig_info[img_element_idx]['image_path'].iloc[0]
                image = Image.open(img_path)
                
                paragraph_ids_list = list(texts.keys())
                summarized_texts_list = list(texts.values())
                inputs = processor(text=summarized_texts_list, images=image, return_tensors='pt', padding=True)
                outputs = model(**inputs)
                
                # Compute probabilities and find the best matching paragraph
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                best_index = probs.argmax()
                best_paragraph_id = paragraph_ids_list[best_index]
                
                # Update the figure information with the ID of the best matching paragraph
                search.figure_info.loc[(search.figure_info['img_element_idx'] == img_element_idx) & (search.figure_info['id'] == paper_id), 'img_similarity_paragraph_id'] = best_paragraph_id
                
        # Ensure the 'img_similarity_paragraph_id' column is of integer type
        search.figure_info['img_similarity_paragraph_id'] = search.figure_info['img_similarity_paragraph_id'].astype('Int64')  
        return search.figure_info

class Search(SimSearch):
    def __init__(self, strategy: SearchStrategy, **kwargs):
        """
        Initialize the Search instance with a specific strategy.

        Args:
        strategy (SearchStrategy): The strategy to use for the search.
        **kwargs: Additional keyword arguments for the SimSearch initializer.
        """
        super().__init__(**kwargs)
        self.strategy = strategy
        self.get_candidate_paragraphs()
        assert self.is_candidate, "Candidate paragraphs not found"

    def execute(self):
        """
        Executes the selected search strategy and saves the results as a CSV file.

        Returns:
        DataFrame: The figure information dataframe with the added similarity information.
        """
        self.strategy.execute(self)
        return self.save_csv()

strategy = ImgSearchStrategy()
search = Search(strategy = strategy, mode='AI_VIT_O')
result_dict = search.execute()