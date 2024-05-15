import os
import requests
import openai

loaddotenv.load_dotenv()

# Class GenerateTemplate for GPT-3 API
class GenerationTemplate:
    """
    Takes a prompt template and provides methods for filling in blanks. 
    The format is as follows. 
    [CAPTION] : The caption of the figure or table.
    [PARAGRAPH] : The paragraph text.
    """
    def __init__(self, template):
        self.template = template 
        
    def fill(self, caption, paragraph):
        """
        Fills in the blanks in the template with the provided caption and paragraph.
        
        Parameters:
        - caption (str): The caption of the figure or table.
        - paragraph (str): The paragraph text.
        
        Returns:
        - str: The filled-in template.
        """
        filled_template = self.template.replace("[CAPTION]", caption).replace("[PARAGRAPH]", paragraph)
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
        self.metadata = 
    
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

