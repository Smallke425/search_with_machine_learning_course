import argparse
import multiprocessing
import glob
from tqdm import tqdm
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import re
from nltk.stem import SnowballStemmer
import pandas as pd

sb_stemmer = SnowballStemmer("english")

def transform_name(product_name):
    # replace / to space
    pm_slashes_replaced = re.sub('/', " ", product_name)
    # remove all non-alphanumeric characters other than underscore or space and lower everything
    pm_lower_alphanum = ''.join([c.lower() for c in pm_slashes_replaced if c.isalnum() or c == ' ' or c == '_'])
    # trim excess spaces
    pm_lower_alphanum = re.sub(' +', " ", pm_lower_alphanum)
    # stemming
    token_list = pm_lower_alphanum.split()
    pm_stem = [sb_stemmer.stem(t) for t in token_list]
    return ' '.join(pm_stem)

# Directory for product data
directory = r'/workspace/datasets/product_data/products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")
general.add_argument("--label", default="id", help="id is default and needed for downsteam use, but name is helpful for debugging")

# Consuming all of the product data, even excluding music and movies,
# takes a few minutes. We can speed that up by taking a representative
# random sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# IMPLEMENT: Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
min_products = args.min_products
sample_rate = args.sample_rate
names_as_labels = False
if args.label == 'name':
    names_as_labels = True


def _label_filename(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    labels = []
    for child in root:
        if random.random() > sample_rate:
            continue
        # Check to make sure category name is valid and not in music or movies
        if (child.find('name') is not None and child.find('name').text is not None and
            child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
            child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None and
            child.find('categoryPath')[0][0].text == 'cat00000' and
            child.find('categoryPath')[1][0].text != 'abcat0600000'):
              # Choose last element in categoryPath as the leaf categoryId or name
              if names_as_labels:
                  cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][1].text.replace(' ', '_')
              else:
                  cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text
              # Replace newline chars with spaces so fastText doesn't complain
              name = child.find('name').text.replace('\n', ' ')
              labels.append((cat, transform_name(name)))
    return labels

if __name__ == '__main__':
    files = glob.glob(f'{directory}/*.xml')
    df_file = os.path.splitext(output_file)[0]+'_df.csv'
    training_data_file = os.path.splitext(output_file)[0]+'.train'
    test_data_file = os.path.splitext(output_file)[0]+'.test'

    all_datasets_df = pd.DataFrame(columns=['category', 'name'])
    print("Writing results to %s" % output_file)
    with multiprocessing.Pool() as p:
        all_labels = tqdm(p.imap_unordered(_label_filename, files), total=len(files))

        with open(output_file, 'w') as output:
            for label_list in all_labels:
                for (cat, name) in label_list:
                    output.write(f'__label__{cat} {name}\n')
                    # get same data in a data frame object
                    all_datasets_df = pd.concat([all_datasets_df, pd.DataFrame({'category':f'__label__{cat}', 'name':name}, index=[0])], axis=0, ignore_index=True)
    
    # shuffle the data
    # output_file = '/workspace/datasets/fasttext/labeled_products_stemmed.txt'
    # df_file = '/workspace/datasets/fasttext/labeled_products_stemmed_df.csv'
    # all_datasets_df = pd.read_csv(df_file)
    all_datasets_df = all_datasets_df.sample(frac=1).reset_index(drop=True)
    
    # store all of it
    all_datasets_df.to_csv(df_file, index=False)

    # store train and test in format for fasttext
    if min_products > 0:
        all_datasets_df = all_datasets_df.groupby('category').filter(lambda x : len(x)>=min_products)
    all_datasets_df.head(10000).to_csv(training_data_file, sep=' ', index=False, header=False, quoting=csv.QUOTE_NONE, quotechar = "", escapechar = " ")
    all_datasets_df.tail(10000).to_csv(test_data_file, sep=' ', index=False, header=False, quoting=csv.QUOTE_NONE, quotechar = "", escapechar = " ")

    # store only titels in a separate file
    all_datasets_df

