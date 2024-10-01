import os
import sys
from read_json import get_scores
import argparse

def main(args):
    dirPath = args.dirPath
    txtPath = args.txtPath
    separe_warning = args.separe_warning
    
    pass_list, fail_list, warning_list = get_folder_scores(dirPath, separe_warning)

    if txtPath:
        create_txt(dirPath, pass_list, fail_list, warning_list)
    
def get_folder_scores(path, separe_warning):
    folder_list =[folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    fail_list = []
    pass_list = []
    warning_list = []
    for folder in folder_list:
        folder_path = os.path.join(path, folder)
        json_files = [json for json in os.listdir(folder_path) if json.lower().endswith('.json')]
        if len(json_files) != 0:
            for json in json_files:
                json_score = get_scores(os.path.join(folder_path, json))
                try:
                    score = json_score['result_result'].values[0]
                    if score == 'FAIL':
                        fail_list.append(os.path.join(folder_path, json.replace('result.json', 'large.jpg')))
                    elif score == 'PASS':
                        pass_list.append(os.path.join(folder_path, json.replace('result.json', 'large.jpg')))
                    else:
                        if separe_warning:
                            warning_list.append(os.path.join(folder_path, json.replace('result.json', 'large.jpg')))
                        else:
                            pass_list.append(os.path.join(folder_path, json.replace('result.json', 'large.jpg')))
                               
                    # print(score)
                except:
                    # print(json_score)
                    continue
                
    return pass_list, fail_list, warning_list

def create_txt(dirPath, pass_list, fail_list, warning_list):
    lists_dict = {
        'pass_list': pass_list,
        'fail_list': fail_list,
        'warning_list': warning_list
    }
    
    for var_name, file_list in lists_dict.items():
            if len(file_list) != 0:
                with open(os.path.join(os.path.dirname(dirPath),f'{os.path.basename(dirPath)}_{var_name}.txt'), 'w') as file:
                    for idx, file_name in enumerate(file_list):
                        if idx < len(file_list) - 1:
                            file.write(f'{file_name}\n')
                        else:
                            file.write(f'{file_name}')
                            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirPath', type=str, required=True, help='Directory with (Moiré pattern) images.')
    
    parser.add_argument('--modelPath', type=str, required=False)
    parser.add_argument('--txtPath', action='store_true', default=False)
    parser.add_argument('--separe_warning', action='store_true', default=False)
    parser.add_argument('--model_compare', action='store_true', default=False)
    
    return parser.parse_args(argv)
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))