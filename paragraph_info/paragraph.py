import argparse
import os
from typing import Optional
import pandas as pd
import sys
import re
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from base import Config, _JSON

# Initialize the summarizer once, outside of the function, to improve efficiency
summarizer = pipeline("summarization", model="Falconsai/text_summarization")
model = T5ForConditionalGeneration.from_pretrained("Voicelab/vlt5-base-keywords")
tokenizer = T5Tokenizer.from_pretrained("Voicelab/vlt5-base-keywords")

class ParagraphInfo(Config):
    """
    A class for extracting information from paragraphs.
    """
    def __init__(self, 
                json_dir: str = '../upstage/json', 
                csv_dir: str = '../upstage/csv', 
                mode: str = 'by_id', 
                paper_id: Optional[str] = None, 
                method: str = 'keyword',
                save_dir: str = './paragraph_info/output'):
        """
        Initializes the ParagraphInfo object with directories for JSON and CSV files, the mode of search, a specific paper ID, and the search method.
        
        Parameters:
        - json_dir (str): Directory for JSON files.
        - csv_dir (str): Directory for CSV files.
        - mode (str): Mode of search, default is 'by_id'.
        - paper_id (Optional[str]): Specific paper ID for search in 'by_id' mode.
        - method (str): Method of search, default is 'keyword'.
        
        Raises:
        - ValueError: If an invalid mode is provided.
        """
        super().__init__(json_dir, csv_dir)
        self.mode = mode
        self.metadata = self.get_metadata()  # Load metadata
        self.figure_info = self.get_figure_info()  # Load figure information
        
        self.is_batch = mode != 'by_id'  # Determine if the search is in batch mode
        self.method = method
        self.save_dir = save_dir
        
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

    def make_summary(self, text_dict: dict):
        '''
        Given a dictionary of text, returns a summarized version of the text.
        '''
        # Perform summarization on texts 
        for key in list(text_dict.keys()):
            value = text_dict[key]
            # Skip summarization if the text is less than 300 characters
            if len(value) < 400:
                text_dict.pop(key)
                continue
            # get rid of '-\n' in the text
            value = value.replace('-\n', '')
            # get rid of '\n' in the text and replace with space
            value = value.replace('\n', ' ')
            # get rid of '[]' and things in between the brackets
            value = re.sub(r'\[.*?\]', '', value)
            # get rid of '()' and things in between the brackets
            value = re.sub(r'\(.*?\)', '', value)
            summary = summarizer(value, max_length=70, min_length=30, do_sample=False)
            text_dict[key] = summary[0]['summary_text']
            # Print the original and summarized texts for verification
            print(f"Original text: {value}")
            print(f"Summarized text: {summary[0]['summary_text']}")
        
        return text_dict
    
    def extract_keyword(self, text_dict: dict):
        '''
        Given a dictionary of text, returns a dictionary of keywords extracted from the text.
        '''
        task_prefix = "Keywords: "
        
        for key in list(text_dict.keys()):
            value = text_dict[key]
            if len(value) < 400:
                # Get rid of the key if the if the text is less than 400 characters
                text_dict.pop(key)
                continue
            
            input_text = task_prefix + value

            input_ids = tokenizer(
                input_text, return_tensors="pt", truncation=True
            ).input_ids
            outputs = model.generate(input_ids, max_length=70, early_stopping=True)
            keywords = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # iterate through keywords, make it lower case and remove if redundant
            keywords = [keyword.lower().strip().strip(',') for keyword in keywords.split()]
            # if the element string in the list has length of 1, remove it
            keywords = [keyword for keyword in keywords if len(keyword) > 1]
            # remove if there is the same word with different capitalization
            keywords = list(set(keywords))

            # make it a string
            keywords = ', '.join(keywords)
            text_dict[key] = keywords
            print(f"Original text: {value}")
            print(f"Keywords: {keywords}")
            print('====================')

        return text_dict
    
    def iterate_elements(self):
        '''
        Iterates through the elements of the JSON files and extracts the paragraph information.
        '''
        papers_dict = {}
        for idx, paper_id in enumerate(self.paper_ids):
            papers_dict[paper_id] = {}
            json_file = self.open_json(self.json_file_paths[idx])
            for element in json_file['elements']:
                if element['category'] == 'paragraph':
                    element_id = element['id']  
                    text = element['text']
                    print(f"Paragraph ID: {element_id}") 
                    print(f"Text: {text}")
                    print('====================')
                    papers_dict[paper_id][element_id] = text 

        return papers_dict

    def summary(self):
        '''
        Generates a summary of the paragraph information.
        '''
        if self.method != 'summary':
            raise ValueError("Invalid method. Choose 'summary' as the method.")
        
        text_dict = self.iterate_elements()
        
        for paper_id, dict in text_dict.items():

            summarized_text = self.make_summary(dict)
            text_dict[paper_id] = summarized_text
        
        return text_dict
    
    def keyword(self):
        '''
        Extracts keywords from the paragraph information.
        '''
        if self.method != 'keyword':
            raise ValueError("Invalid method. Choose 'keyword' as the method.")
        
        text_dict = self.iterate_elements()
        
        for paper_id, dict in text_dict.items():
            keywords = self.extract_keyword(dict)
            text_dict[paper_id] = keywords
            
        
        return text_dict


    def save_csv(self, text_dict: dict):
        '''
        Saves the paragraph information to a CSV file.
        '''
        # make directory if it does not exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        for paper_id, text_info in text_dict.items():
            # Convert the keys and values to lists and then create a DataFrame
            keys = list(text_info.keys())
            values = list(text_info.values())

            column_name = 'summarized_text' if self.method == 'summary' else 'keywords'

            new_df = pd.DataFrame({'element_idx': keys, column_name: values})

            # File path for the CSV
            file_path = os.path.join(self.save_dir, f"{paper_id}.csv")

            # Check if the file already exists
            if os.path.exists(file_path):
                # Read the existing CSV
                existing_df = pd.read_csv(file_path)

                # Check if the target column exists in the existing dataframe
                if column_name in existing_df.columns:
                    # Merge on 'element_idx' and update values in the existing column
                    merged_df = pd.merge(existing_df, new_df, on='element_idx', how='outer', suffixes=('', '_new'))
                    merged_df[column_name] = merged_df[column_name + '_new'].combine_first(merged_df[column_name])
                    merged_df.drop(columns=[column_name + '_new'], inplace=True)
                else:
                    # If the column does not exist, simply add the new column to the existing dataframe
                    merged_df = pd.merge(existing_df, new_df, on='element_idx', how='outer')
                
                merged_df.to_csv(file_path, index=False)
            else:
                # If the file does not exist, simply save the new dataframe
                new_df.to_csv(file_path, index=False)
        
        # print success message 
        print(f"CSV files saved successfully in {self.save_dir}")


    def run(self):
        '''
        Runs the paragraph information extraction process.
        '''
        if self.method == 'summary':
            text_dict = self.summary()

        elif self.method == 'keyword':
            text_dict = self.keyword()

        self.save_csv(text_dict)
        return text_dict

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Paragraph Information Extraction')
    # Add the arguments
    parser.add_argument('--json_dir', type=str, default='./upstage/json', help='Directory for JSON files')
    parser.add_argument('--csv_dir', type=str, default='./upstage/csv', help='Directory for CSV files')
    parser.add_argument('--mode', type=str, default='by_id', help='Mode of search, default is by_id')
    parser.add_argument('--paper_id', type=str, default=None, help='Specific paper ID for search in by_id mode')
    parser.add_argument('--method', type=str, default='keyword', help='Method of search, default is keyword')
    parser.add_argument('--save_dir', type=str, default='./summary/paragraph_info/output', help='Directory to save the output CSV files')
    # Parse the arguments
    args = parser.parse_args()
    return args

def main():
    # Parse the arguments
    args = parse_args()
    args_dict = vars(args)
    # Initialize the ParagraphInfo object
    paragraph_info = ParagraphInfo(**args_dict)
    # Run the paragraph information extraction process
    paragraph_info.run()
    
    
if __name__ == '__main__':
    main()