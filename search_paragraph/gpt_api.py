import os
import sys
import json
import openai
import pandas as pd
from typing import Optional, List, Dict, Tuple

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from base import Config, _JSON

class Template:
    def __init__(self, system: Optional[str] = None, prompt: Optional[str] = None):
        self.system = system or "You are a professor who is good at explaining and very kind"
        self.prompt = prompt or """Please make a description to the figure/table of academic paper based on the following topic, caption, related paragraph contents. Ensure your responses are detailed, supportive, and easy to understand. But do not make your response over 100words. The length is the most important. 
            Title : [TITLE]
            Caption : [CAPTION]
            Paragraph : [PARAGRAPH]
            
            Remember the 100 words limit.
            """
class GenerationTemplate:
    """
    Takes a prompt template and provides methods for filling in blanks.
    """
    def __init__(self, template: Template):
        self.template = template or Template()

    def fill(self, title: str, caption: str, paragraph: str) -> str:
        """
        Fills in the blanks in the template with the provided caption and paragraph.
        """
        paragraph_content = paragraph or ""
        return self.template.prompt.replace("[TITLE]", title).replace("[CAPTION]", caption).replace("[PARAGRAPH]", paragraph_content)

class SummaryGenerator(Config):
    """
    Generates summaries using the GPT-3 API.
    """
    def __init__(self, json_dir: str, csv_dir: str, mode: str, api_key: str, paper_id: Optional[str] = None):
        super().__init__(json_dir, csv_dir)
        self.generate_template = None
        self.api_key = api_key
        self.mode = mode
        self.is_batch = mode != 'by_id'
        self.json_dir = json_dir
        self.csv_dir = csv_dir
        self.metadata, self.figure_info = self.load_metadata_and_figures()
        self.paper_id, self.json_file_path = self.prepare_file_paths(mode, paper_id)
        self.similar_paragraphs = self.initialize_similar_paragraphs()
        self.prepare_similar_paragraphs()

    def load_metadata_and_figures(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads metadata and figure information from the specified directories.
        """
        metadata = pd.read_csv(os.path.join(self.csv_dir, 'metadata.csv'))
        figure_info = pd.read_csv(os.path.join(self.csv_dir, 'table_info_hand.csv'))
        return metadata, figure_info

    def prepare_file_paths(self, mode: str, paper_id: Optional[str]) -> Tuple[List[str], List[str]]:
        """
        Prepares file paths based on the operating mode.
        """
        if self.is_batch:
            self.validate_mode(mode)
            
            json_file_paths = [os.path.join(self.json_dir, self.mode, file) for file in os.listdir(os.path.join(self.json_dir, self.mode)) if file.endswith('.json')]
            paper_ids = [self.get_paper_id_from_path(path) for path in json_file_paths]
        else:
            self.validate_paper_id(paper_id)
            json_file_paths = list(self.metadata[self.metadata['id'] == paper_id]['json_file_path'])
            paper_ids = [paper_id]
        return paper_ids, json_file_paths

    def get_paper_id_from_path(self, path: str) -> str:
        """
        Extracts the paper ID from the given file path.
        """
        return self.metadata[self.metadata['json_file_path'] == path]['id'].iloc[0]

    def validate_mode(self, mode: str):
        """
        Validates the specified mode against allowed modes.
        """
        
        if mode not in _JSON:
            raise ValueError(f"Invalid mode: {mode}. Choose from {_JSON}")

    def validate_paper_id(self, paper_id: Optional[str]):
        """
        Validates the provided paper ID for 'by_id' mode.
        """
        if not paper_id:
            raise ValueError("Paper ID is required for 'by_id' mode.")

    def initialize_similar_paragraphs(self) -> Dict[str, Dict[int, List[str]]]:
        """
        Initializes the structure for storing similar paragraphs.
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

    def get_title(self, paper_id: str) -> str:
        """
        Retrieves the title for a given paper ID.
        """
        title_data_path = '../upstage/info/title/title.json'
        with open(title_data_path) as f:
            title_data = json.load(f)
        for paper in title_data:
            if paper['id'] == paper_id:
                return paper['text']
        return None

    def get_candidate_paragraphs(self, paper_id: str, img_element_idx: int) -> Dict[str, List[int]]:
        """
        Retrieves dictionary of paragraphs based on the paper ID and image element index.
        """
        df = self.figure_info[(self.figure_info['id'] == paper_id) & (self.figure_info['img_element_idx'] == img_element_idx)]
        if df.empty:
            return {
                'rule_based_paragraphs': [],
                'img_similarity_keyword_paragraph_id': None,
                'text_similarity_keyword_paragraph_id': None,
                'img_similarity_summarized_text_paragraph_id': None,
                'text_similarity_summarized_text_paragraph_id': None
            }

        row = df.iloc[0]

        return {
            'rule_based_paragraphs': self.parse_paragraph_ids(row['rule_based_paragraphs']),
            'img_similarity_keyword_paragraph_id': self.parse_paragraph_id(row['img_similarity_keyword_paragraph_id']),
            'text_similarity_keyword_paragraph_id': self.parse_paragraph_id(row['text_similarity_keyword_paragraph_id']),
            'img_similarity_summarized_text_paragraph_id': self.parse_paragraph_id(row['img_similarity_summarized_text_paragraph_id']),
            'text_similarity_summarized_text_paragraph_id': self.parse_paragraph_id(row['text_similarity_summarized_text_paragraph_id'])
        }

    def parse_paragraph_ids(self, paragraph_str: str) -> List[int]:
        """
        Parses a comma-separated string of paragraph IDs into a list of integers.
        """
        if pd.isna(paragraph_str):
            return []

        return [int(pid) for pid in paragraph_str.split(', ')]

    def parse_paragraph_id(self, paragraph_id) -> Optional[int]:
        """
        Parses a paragraph ID into an integer, or returns None if invalid.
        """
        return int(paragraph_id) if pd.notna(paragraph_id) else None

    def find_top_2_paragraphs(self, candidate_paragraphs: List[int]) -> List[int]:
        """
        Given a list of candidate paragraphs, returns the top 2 paragraphs based on the number of similar paragraphs.
        """
        paragraph_count = [(paragraph, candidate_paragraphs.count(paragraph)) for paragraph in set(candidate_paragraphs)]
        paragraph_count.sort(key=lambda x: (-x[1], x[0]))
        return [paragraph[0] for paragraph in paragraph_count[:2]]

    def choose_paragraph(self, related_paragraphs: Dict[str, List[int]]) -> Optional[List[int]]:
        """
        Given a dictionary of related paragraphs, returns the list paragraph ID of the chosen paragraph.
        """
        paragraph_ids = related_paragraphs['rule_based_paragraphs']

        if not paragraph_ids:
            paragraph_ids = [pid for pid in related_paragraphs.values() if pid]
        if len(paragraph_ids) > 2:
            return self.find_top_2_paragraphs(paragraph_ids)
        return paragraph_ids if paragraph_ids else None

    def get_paragraph_text(self, paper_id: str, paragraph_ids: List[int]) -> List[str]:
        """
        Identifies the paragraph text based on the paper ID and paragraph ID.
        """
        paragraph_text = []
        with open(self.metadata[self.metadata['id'] == paper_id]['json_file_path'].iloc[0]) as f:
            json_data = json.load(f)
            for paragraph in json_data['elements']:
                if not paragraph_ids:
                    return None
                if paragraph['id'] in paragraph_ids:
                    paragraph_text.append(paragraph['text'])
        return paragraph_text

    def execute_first(self):
        """
        Executes getting the final paragraph text for each image element.
        """

        for idx, pid in enumerate(self.paper_id):

            with open(self.json_file_path[idx]) as f:
                json_data = json.load(f)

            for img_element_idx in self.similar_paragraphs[pid]:
                candidate_paragraphs = self.get_candidate_paragraphs(pid, img_element_idx)
                chosen_paragraphs = self.choose_paragraph(candidate_paragraphs)
                paragraph_text = self.get_paragraph_text(pid, chosen_paragraphs)
                self.similar_paragraphs[pid][img_element_idx] = paragraph_text
        self.is_candidate = False
        

    def gpt_api(self, system: str, prompt: str) -> Optional[str]:
        """
        Calls the OpenAI API to generate a response based on the given system and prompt.
        """
        openai.api_key = self.api_key
        client = openai.Client(api_key=self.api_key)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=messages,
                max_tokens=400,
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def execute_final(self):
        assert not self.is_candidate
        for paper_id, img_element_idxs in self.similar_paragraphs.items():
            title = self.get_title(paper_id)
            for img_element_idx, paragraph_text in img_element_idxs.items():
                caption = self.figure_info.loc[(self.figure_info['id'] == paper_id) & (self.figure_info['img_element_idx'] == img_element_idx), 'caption'].iloc[0]
                if paragraph_text is None:
                    continue
                if len(paragraph_text) > 1:
                    paragraph_text = paragraph_text[0] + '\n' + paragraph_text[1]
                else:
                    paragraph_text = paragraph_text[0]
                self.generate_template = GenerationTemplate(Template())
                filled_prompt = self.generate_template.fill(title, caption, paragraph_text)
                response = self.gpt_api(self.generate_template.template.system, filled_prompt)
                self.figure_info.loc[(self.figure_info['id'] == paper_id) & (self.figure_info['img_element_idx'] == img_element_idx), 'gpt_summary'] = response
        self.save_csv()

    def save_csv(self):
        """
        Saves the figure information as a CSV file in the configured directory.

        Returns:
        DataFrame: The figure information dataframe.
        """
        csv_path = os.path.join(self.csv_dir, 'table_info_hand.csv')
        self.figure_info.to_csv(csv_path, index=False)
        return self.figure_info

api_key = "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"
summary_generator = SummaryGenerator(api_key=api_key, json_dir='../upstage/json', csv_dir='../upstage/csv', mode='AI_VIT_X')
# summary_generator = SummaryGenerator(api_key=api_key, json_dir='../upstage/json', csv_dir='../upstage/csv', mode='by_id', paper_id="6298b6a2-0f92-11ef-8230-426932df3dcf")
summary_generator.execute_first()
summary_generator.execute_final()
print(summary_generator.figure_info)
