import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
import requests
from utils import BASE_AUTH
import logging
import warnings
from artgraph_utils import update_graph

warnings.filterwarnings('ignore')

tqdm.pandas()
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def get_paintings_by_artist(artist_name):
    """

    :param artist_name:
    :return:
    """
    base_query = 'https://www.wikiart.org/en/App/Painting/PaintingsByArtist?artistUrl={artist}&json=2'
    return requests.get(base_query.format(artist=artist_name)).json()


def get_content_id(artist_name, artwork):
    """

    :param artist_name:
    :param artworks:
    :return:
    """
    title, name = artwork
    paintings = get_paintings_by_artist(artist_name)
    filtered = list(filter(lambda x: x['image'].split('/')[-1][:-10] == name or x['title'] == title, paintings))
    # special case: more than 1 retrieved-> filter again
    # priority to the image because
    assert(len(filtered) > 0), (artist_name, artwork)
    if len(filtered) > 1:
        filtered_out = list(filter(lambda x: x['image'].split('/')[-1][:-10] == name, filtered))
        if len(filtered_out) == 0:
            filtered = filtered[:1]
        else:
            filtered = filtered_out
    return filtered[0]['contentId']


def artist_in_artgraph(artist_name, driver, db):
    with driver.session(database=db) as session:
        ans = session.run(f'match (a:Artist{{name: "{artist_name}" }}) return count(distinct a) as num')
        return next(iter(ans))['num'] > 0


def get_artist_information(artist_name):
    # if in artgraph -> migrate, else get it from the web
    driver = GraphDatabase.driver(**BASE_AUTH)
    if artist_in_artgraph(artist_name, driver, 'neo4j'):
        with driver.session(database='neo4j') as session:
            artists_links = list(iter(session.run(f"""
            match p=(:Artist{{name: "{artist_name}" }})-->(n) where labels(n)[0] <> "Artwork"
            return relationships(p) as rels, nodes(p) as nodes
            """)))
            artists_links = list(map(lambda x: x['rels'][0], artists_links))
        return artists_links, 'artgraph'
    else:
        base_query = 'https://www.wikiart.org/en/{artist_name}?json=2'
        return requests.get(base_query.format(artist_name=artist_name)).json(), 'web'


def get_artwork_information(content_id):
    return {k: v for k, v in requests.get(f'https://www.wikiart.org/en/App/Painting/ImageJson/{content_id}').json()
            .items()if k != 'dictionaries'}


def stringfy_prop(props):
    return ', '.join([f'{x}: "{props[x]}"' if isinstance(props[x], str) else f"{x}: {props[x]}" for x in props.keys()
                      if props[x] is not None])


def save_artwork(raw, driver, db):
    metadata = get_artwork_information(raw['content_id'])
    artist_info, mode = get_artist_information(raw['artist_name'])
    # add artist
    if mode == 'artwork':
        update_graph(driver=driver, db=db, rels=artist_info)
    else:
        with driver.session(database=db) as session:
            session.run(f'merge(:Artist{{ {stringfy_prop(metadata)} }})')
    # add other info
    with driver.session(database=db) as session:
        # merge just the artwork
        session.run(f'''merge(:Artwork{{ code: "{raw.content_id}",
                                         title: "{metadata['title']}",
                                         year: "{metadata['yearAsString']},
                                         dimensions: '{metadata['height']} X {metadata['width']}',
                                         image_url: '{metadata['image']}',
                                         name: '{metadata['artist_url']}_{metadata['url']}.jpg'"}})''')
        # merge style and genre
        session.run(f'''match (a:Artwork{{code: "{raw.content_id}"}})
                        match (au:Artist{{name: "{raw.artist_name}"}})
                        merge (s:Style{{ name: "{metadata['style'].lower()}"}})
                        merge (g:Genre{{name: "{metadata['genre'].lower()}"}})
                        create (a)-[:createdBy]->(au)
                        create (a)-[:hasStyle]->(s)
                        create (a)-[:hasGenre]->(g)''')

        # merge media if there are
        medias = metadata['material']
        if medias is not None:
            query = f'''match (a:Artwork{{code: "{raw.ID}"}})'''
            query += '\n'.join([f'merge (t{i}: Tag{{name: "{tag}"}})\n create (a)-[:about]->(t{i})'\
                                for i, tag in enumerate(medias)])
            session.run(query)
        if metadata['serie'] is not None:
            session.run(f'''match (a:Artwork{{code: "{raw.content_id}"}})
                            merge (s:Serie{{ name: "{metadata['serie']}" }})
                            create (a)-[:partOf]->(s)''')
        if metadata['galleryName'] is not None:
            session.run(f'''match (a:Artwork{{code: "{raw.content_id}"}})
                                        merge (g:Gallery{{ name: "{metadata['galleryName']}" }})
                                        create (a)-[:locatedIn]->(g)''')
        if metadata['period'] is not None:
            session.run(f'''match (a:Artwork{{code: "{raw.content_id}"}})
                            merge (p:Period{{ name: {metadata['period']} }})
                            create (a)-[:hasPeriod]->(p)''')


def main():
    artwork_info = pd.read_csv('../notebooks/artwork_info_sources.csv', index_col=0)
    driver = GraphDatabase.driver(**BASE_AUTH)
    artwork_info.drop(artwork_info[(artwork_info.api_v1_artist == 0) &
                                   (artwork_info.name_in_artgraph == 0) &
                                   (artwork_info.api_v1_artist_1 == 0) &
                                   (artwork_info.api_v1_url == 0) &
                                   (artwork_info.api_v2 == 0)].index, inplace=True)

    # delete useless columns
    artwork_info.drop(['Category', 'Year'], axis=1, inplace=True)
    artwork_info_artist = artwork_info[artwork_info.api_v1_artist == 1]
    artwork_info_artist.drop(['name_in_artgraph', 'api_v1_artist',
                                'api_v1_artist_1', 'api_v1_url', 'api_v2'],
                               axis=1,
                               inplace=True)
    artwork_info_artist['artist_name'] = artwork_info_artist['Image URL'].apply(lambda x: x.split('/')[-2])
    artwork_info_artist['artist_name_1'] = artwork_info_artist['Artist']\
        .apply(lambda x: '-'.join(x.lower().split(' ')))
    # set content ids
    logging.info('Getting proper content ids...')
    artwork_info_artist['content_id'] = artwork_info_artist.progress_apply(lambda x: get_content_id(x.artist_name, x[
        ['Title', 'name']]), axis=1)

    logging.info("Saving new artworks and relations into db...")
    artwork_info_artist.progress_apply(lambda x: save_artwork(x, driver, 'recsys'), axis=1)


if __name__ == '__main__':
    main()