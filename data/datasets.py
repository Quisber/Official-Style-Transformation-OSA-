# файл для обработки текстовых массивов
import csv
import os
  
def get_pair(file_name="corpus.tsv"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)

    source = []
    targeted = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            source.append(row.get('source_text', ''))
            targeted.append(row.get('target_text', ''))
    return source, targeted
