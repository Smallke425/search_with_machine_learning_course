import argparse
import csv
import fasttext
import pandas as pd

# fasttext directory
parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--word_file", default="/workspace/datasets/fasttext/top_words.txt", help="Word file")
general.add_argument("--model_file", default="/workspace/datasets/fasttext/title_model.bin", help="Model file")
general.add_argument("--output_file", default="/workspace/datasets/fasttext/synonyms.csv", help="the file to output to")


args = parser.parse_args()
word_file = args.word_file
model_file = args.model_file
output_file = args.output_file


def get_synonyms(model, word, threshold = 0.75):
    list_of_syns = [synonym for (score, synonym) in model.get_nearest_neighbors(word, k=2000) if score > threshold]
    return ','.join(list_of_syns)


if __name__ == '__main__':
    model = fasttext.load_model(model_file)
    words_df = pd.read_csv(word_file, header=None)
    words_df[0] = words_df[0].astype(str)
    words_df['synonyms'] = words_df[0].apply(lambda word: get_synonyms(model, word))
    words_df['output_format'] = words_df[0] + ',' + words_df['synonyms']
    words_df['output_format'].to_csv(output_file, sep=',', header=False, index=False, quoting=csv.QUOTE_NONE, quotechar = "", escapechar = " ")
