import pandas as pd 
import numpy as np
import re

def preprocess_csv(input_csv_file, output_csv_file):
    df = pd.read_csv(input_csv_file)
    # rename columns
    df.rename(
        columns = {
            'Input.label': 'label',
            'Input.text': 'text',
            'Answer.Q1Answer':  'human_label',
            'Answer.html_output': 'human_attention'
        },
        inplace=True
    )

    # convert yes or no labels to 1 and 0
    df['human_label'] = df['human_label'].map(dict(yes=1, no=0))
    # remove entries where human label disagrees with true label
    df = df[df.label == df.human_label]

    # convert html attention to binary vectors
    def attention_to_vec(attention):
        html_list = attention.split(' <')
        num_words = len(html_list) - 1
        attn_idxs = np.flatnonzero(np.core.defchararray.find(np.array(html_list),'class="active"')!=-1)
        vec = np.zeros(num_words)
        vec[attn_idxs] = 1
        return vec
    df['human_attention'] = df['human_attention'].map(attention_to_vec)

    df_result = df.groupby(['text']).agg({
                    # take first available label
                    'label': (lambda x: x.iat[0]),
                    # add attention vectors like numpy arrays
                    'human_attention': (lambda a: " ".join(map(str, sum([np.array(x)for x in a]).astype(int))))
                    }).reset_index()
    # save df result
    df_result.to_csv(output_csv_file, columns=['label', 'text', 'human_attention'], index=False)

def main():
    file_names = ['ham_part1(50words).csv', 'ham_part3.csv', 'ham_part4.csv', 'ham_part5.csv', 'ham_part6(100words).csv', 'ham_part7.csv', 'ham_part8(200words).csv']
    input_dir = 'raw_data'
    output_dir = 'processed_data'

    for file_name in file_names:
        preprocess_csv(f'{input_dir}/{file_name}', f'{output_dir}/{file_name}')
        print(f'processed {file_name}')


if __name__ == "__main__":
    main()