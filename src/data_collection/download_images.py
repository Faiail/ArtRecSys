from neo4j import GraphDatabase
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import pandas as pd
import requests
import warnings
import argparse
import logging
import os
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
warnings.filterwarnings('ignore')
tqdm.pandas()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=False, default='images', help='root folder to store the images', type=str)
    parser.add_argument('--uri', required=False, default="bolt://localhost:7687", help='uri of the neo4j dbms', type=str)
    parser.add_argument('--user', required=False, default='neo4j', help='username of the target db', type=str)
    parser.add_argument('--pwd', required=False, default='neo4j', help='password of the target db', type=str)
    parser.add_argument('--db', required=False, default='recsys', help='name of the target db', type=str)

    return parser.parse_args()


def download_image(root, name, url):
    response = requests.get(url)
    if response.status_code != 200:
        response = requests.get(re.split('!', url)[0])
    Image.open(BytesIO(response.content)).convert('RGB').save(f'{root}/{name}')


def main():
    args = parse_args()
    driver = GraphDatabase.driver(uri=args.uri, auth=(args.user, args.pwd))
    with driver.session(database=args.db) as session:
        data = pd.DataFrame(session.run('match (a:Artwork) return a.name as name, a.image_url as url').data())

    if not os.path.exists(args.root):
        os.mkdir(os.path.abspath(args.root))

    data.progress_apply(lambda x: download_image(args.root, x['name'], x['url']), axis=1)


if __name__ == '__main__':
    main()

