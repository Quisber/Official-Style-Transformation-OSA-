# файл для обработки текстовых массивов
import csv

def get_pair(target_id, file_path="corpus.tsv"):
     with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t') 

        for row in reader:
            if row['pair_id'] == target_id:
                return row['source_text'], row['target_text']
     return None, None
        

