from transformers import pipeline

# Initialize the summarizer once, outside of the function, to improve efficiency
summarizer = pipeline("summarization", model="Falconsai/text_summarization")

def get_summarized_text(json, paragraph_ids):
    """
    Given a JSON object and a list of paragraph IDs, returns a dictionary mapping
    those IDs to their summarized text.
    """
    summarized_texts = {}
    for id in paragraph_ids:
        paragraph = json['elements'][id]
        text = paragraph['text']

        # Skip summarization if the text is less than 300 characters
        if len(text) < 300:
            continue

        # Perform summarization on texts longer than 100 characters
        summary = summarizer(text, max_length=70, min_length=30, do_sample=False)
        
        # Print the original and summarized texts for verification
        print(f"Original text: {text}")
        print(f"Summarized text: {summary[0]['summary_text']}")

        # Map paragraph ID to its summarized text
        summarized_texts[id] = summary[0]['summary_text']
    
    return summarized_texts

def get_paragraph_ids(json, image_element_id):
    """
    Given a JSON object and an image element ID, returns a list of paragraph IDs
    that are on the same, previous, or next page as the image element.
    """
    json_elements = json['elements']
    element = json_elements[image_element_id]
    page = element['page']
    
    # Find paragraph IDs on the same, previous, or next page as the image element
    paragraph_ids = [
        element['id'] for element in json_elements
        if element['page'] in [page, page+1, page-1] and element['category'] == 'paragraph'
    ]
    
    return paragraph_ids
