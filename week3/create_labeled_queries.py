import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import re

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")
general.add_argument("--update_successively", default=True, help="Updating categories to parent categories one by one?")

args = parser.parse_args()
output_file_name = args.output
update_successively = args.update_successively

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])
parents_df = parents_df.set_index('category').sort_index()

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
def normalize(query):
    # replace / to space
    query_normalized = re.sub('/', " ", query)
    # treat all non-alphanumeric characters as space and lower everything else
    query_normalized = ''.join([c.lower() if c.isalnum() else ' ' for c in query_normalized])
    # trim excess spaces
    query_normalized = re.sub(' +', " ", query_normalized)
    query_normalized = query_normalized.strip()
    # stemming
    query_normalized = ' '.join([stemmer.stem(t) for t in query_normalized.split()])
    return query_normalized


df['query'] = df['query'].apply(normalize)

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
while min_count < min_queries:
    counts_df = df.groupby('category').size().reset_index(name='counts').sort_index()
    min_count = min(counts_df.counts)
    if update_successively == True:
        category_to_update = counts_df[counts_df.counts == min_count].category.iloc[0]
        parent_category = parents_df.loc[category_to_update].parent
        df.category.replace(category_to_update, parent_category, inplace=True)
    else:
        categories_to_update = counts_df[counts_df.counts == min_count].category
        update_dict = parents_df.loc[categories_to_update].to_dict()['parent']
        df.category.replace(update_dict, inplace=True)

# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
