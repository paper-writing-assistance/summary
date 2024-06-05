import argparse
import os
from rule import RuleSearch
from similarity import Search, TextSearchStrategy, ImgSearchStrategy
from gpt_api import SummaryGenerator

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Paragraph Information Extraction')
    # Add the arguments
    parser.add_argument('--json_dir', type=str, default='../upstage/json', help='Directory for JSON files')
    parser.add_argument('--csv_dir', type=str, default='../upstage/csv', help='Directory for CSV files')
    parser.add_argument('--mode', type=str, default='by_id', help='Mode of search, default is by_id')
    parser.add_argument('--paper_id', type=str, default=None, help='Specific paper ID for search in by_id mode')
    parser.add_argument('--task', type=str, required=True, choices=[
                        'rule', 'text-keyword', 'text-summary', 'image-keyword', 'image-summary', 'gpt_summary'], 
                        help='Task to perform')
    parser.add_argument('--save_dir', type=str, default='../upstage/csv', help='Directory to save the output CSV files')
    parser.add_argument('--is_figure', action='store_true', help='Flag to indicate figure-based search (default is table-based)')
    parser.add_argument('--api_key', type=str, help='API key for GPT summarization', required=False)
    # Parse the arguments
    return parser.parse_args()

def main():
    # Parse the arguments
    args = parse_args()
    
    if args.task == 'rule':
        search = RuleSearch(json_dir=args.json_dir, csv_dir=args.csv_dir, mode=args.mode, paper_id=args.paper_id, is_figure=args.is_figure)
        result_dict = search.execute()
        print(result_dict)
    elif args.task in ['text-keyword', 'text-summary', 'image-keyword', 'image-summary']:
        if 'text' in args.task:
            strategy = TextSearchStrategy()
        elif 'image' in args.task:
            strategy = ImgSearchStrategy()
        
        search = Search(strategy=strategy, json_dir=args.json_dir, csv_dir=args.csv_dir, mode=args.mode, paper_id=args.paper_id, output_dir=args.save_dir)
        result_dict = search.execute()
        print(result_dict)
    elif args.task == 'gpt_summary':
        if not args.api_key:
            raise ValueError("API key is required for GPT summarization task")
        summary_generator = SummaryGenerator(api_key=args.api_key, json_dir=args.json_dir, csv_dir=args.csv_dir, mode=args.mode, paper_id=args.paper_id)
        summary_generator.execute_first()
        summary_generator.execute_final()
        print(summary_generator.figure_info)
    else:
        raise ValueError(f"Unknown task: {args.task}")

if __name__ == '__main__':
    main()