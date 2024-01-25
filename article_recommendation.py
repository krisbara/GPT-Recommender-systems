# imports
import pandas as pd
import openai
from typing import List
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, indices_of_nearest_neighbors_from_distances
from collections import Counter
import os

def print_recommendations_for_user(
    user_id: str,
    users_dict: dict,
) -> pd.DataFrame:
    user_embeddings = []
    user_articles_embeddings = users_dict[user_id][1]
    user_embeddings_np = np.array(user_articles_embeddings)
    user_embeddings = user_embeddings_np.tolist()

    content_dictionary = []
    for key in users_dict.keys():
        content_dictionary.append(users_dict[key][1])

    # Assume you have the following functions imported from embeddings_utils.py
    distances = distances_from_embeddings(user_embeddings, content_dictionary, distance_metric="cosine")
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)

    user_keys = list(users_dict.keys())  # Fixed a typo: users_dict instead of user_dict
    selected_user_keys = []
    selected_distances = []

    for i in indices_of_nearest_neighbors:
        if 0 <= i < len(user_keys) and distances[i] != 0:
            user_key = user_keys[i]
            selected_user_keys.append(user_key)
            selected_distances.append(distances[i])

    # Creating a DataFrame
    result_df = pd.DataFrame({
        'User': selected_user_keys,
        'distance': selected_distances
    })

    return result_df

#create list with interacted items for a particular user
def user_list(user_id, df):
    user_list = df.loc[df['User'] == user_id, 'ID'].tolist()
    if len(user_list) > 0:
        return user_list[0].split()
    else:
        return []

def print_article_recommendations(user_id, user_dict, users, news):
    recommendations = print_recommendations_for_user(user_id, user_dict)
    collab_recommender = pd.merge(recommendations, users, on='User', how='left')
    collab_recommender = collab_recommender.drop(['Interactions_emb'], axis=1)
    
    collab_recommender['ID'] = collab_recommender['ID'].str.split()
    collab_recommender = collab_recommender.explode('ID')

    article_distances = collab_recommender.groupby('ID').agg({'distance': np.sum}).reset_index()
    article_users = collab_recommender.groupby('ID')['User'].apply(list).reset_index()

    article_distances_users = pd.merge(article_distances, article_users, on='ID', how='left')
    article_distances_users['N_users'] = article_distances_users['User'].apply(len)
    article_distances_users['article_distance'] = article_distances_users['distance'] / article_distances_users['N_users']

    collab_final = pd.merge(news, article_distances_users[['ID', 'article_distance']], on='ID', how='left')
    collab_final = collab_final.drop(['Content_emb'], axis=1)
    
    collab_final['article_distance'].fillna(1, inplace=True)  # Replace NaN with 1
    collab_final = collab_final.sort_values(by='article_distance', ascending=True, na_position='last')
    
    # Delete the rows with articles read by the considered user 
    mask = collab_final['ID'].isin(user_list(user_id, users))

    # Invert the mask to keep the rows that are not in the list
    collab_rec = collab_final[~mask]
    collab_rec.to_csv('./collaborative_recommendations/' + user_id + '_collab.csv', index=False)
    # print(collab_rec)
    
    return collab_rec


if __name__ == '__main__':
    users = pd.read_csv("embeddings/users_emb_final.csv") #document with user interactions
    users.head()

    news = pd.read_csv("embeddings/news_emb_final.csv") #document with user interactions
    news = news.drop(['Category', 'SubCategory', 'Content'], axis=1)
    news.head()

    # Create a dictionary with user interactions
    user_dict = {}
    for index, row in users.iterrows():
        user = row['User']
        interactions = row['Interactions_emb']
        user_dict[user] = ('Content', interactions)

    for user, (content, embeddings_str) in user_dict.items():
        # Check for empty string before converting to float
        embeddings_list = [float(value) if value.strip() else 0.0 for value in embeddings_str.strip('[]').split(',')]
        user_dict[user] = (content, embeddings_list)
    
    users_list = users['User'].tolist()
    
    # import multiprocessing
    # from multiprocessing.pool import Pool
    # # Get the number of available CPU cores
    # num_processes = multiprocessing.cpu_count()
    # with Pool(processes = num_processes) as pool:
    #     pool.starmap(print_article_recommendations, [(user_id, user_dict, users, news) for user_id in users_list])
        
    for i in users_list:
        # Example usage
        recommendations = print_article_recommendations(i, user_dict, users, news)