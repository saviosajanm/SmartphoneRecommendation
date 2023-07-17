# Main libraries
import os
import pandas as pd
import numpy as np
import random
# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import pickle
# Libraries for recommendation systemshmm
from collections import defaultdict
from surprise import SVD
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

from google_trans_new import google_translator
from translate import Translator

import warnings


# Loading Data files
warnings.filterwarnings('ignore')
pd.set_option("display.max_rows", 50, "display.max_columns", 50)
pd.set_option('display.max_colwidth', None)
# create contants
RS=612

class SmartPhoneRecommendation:
  def __init__(self):

    review_1 = pd.read_csv('dataset/phone_user_review_file_1.csv', encoding='iso-8859-1')
    review_2 = pd.read_csv('dataset/phone_user_review_file_2.csv', encoding='iso-8859-1')
    review_3 = pd.read_csv('dataset/phone_user_review_file_3.csv', encoding='iso-8859-1')
    review_4 = pd.read_csv('dataset/phone_user_review_file_4.csv', encoding='iso-8859-1')
    review_5 = pd.read_csv('dataset/phone_user_review_file_5.csv', encoding='iso-8859-1')
    review_6 = pd.read_csv('dataset/phone_user_review_file_6.csv', encoding='iso-8859-1')

    # Merge the data into a single dataframe
    reviews = pd.concat([review_1,review_2,review_3,review_4,review_5,review_6], ignore_index=True)
    del review_1, review_2, review_3, review_4, review_5, review_6

    # As can be seen, same pattern is visible for the most comun types of phones.
    # Thus it is better to use phone name and model number rather than other details mentioned in 'product' column
    reviews['phone'] = reviews['phone_url'].str.split("/").apply(lambda col: col[2]).replace('-', ' ', regex=True)
    reviews['product'] = reviews['phone']

    # names like 'einer Kundin', 'einem Kunden','Anonymous' and 'unknown' can be interpreted in the same way i.e. an 'unknown customer'.
    # replace these names
    unknowns = ['Anonymous','einer Kundin','einem Kunden', 'unknown','Anonymous ']
    reviews['author'].replace(to_replace = unknowns,
                              value = 'Anonymous',
                              inplace=True)

    # Data cleaning, Imputation & rounding off
    relevant_features=['author','product','score']
    revs1 = reviews.copy()
    del reviews

    # Step1: Remove irrelevant features
    revs1 = revs1.loc[:,relevant_features]

    # Step2: Round-off score feature to nearest integer
    revs1['score'] = revs1['score'].round(0).astype('Int64')

    # Step3: Impute missing values in score feature with median
    revs1['score'] = revs1['score'].fillna(revs1['score'].median())

    # Step4: Remove samples with missing values in 'Product' and 'author' feature and also 'Anonymous' values
    revs1.dropna(inplace=True)
    revs1 = revs1[revs1["author"] != 'Anonymous']

    # Step5: remove duplicates, if any
    revs1 = revs1.drop_duplicates()

    # separate 1 million data samples
    revs_1m = revs1.sample(n=1000000, random_state=RS)

    # MOST RATED PRODUCTS
    self.MOST_RATED_PRODUCTS = revs_1m['product'].value_counts().head()

    # Top 50 products based on mean ratings
    self.top50_product = revs1['product'].value_counts()[0:50].rename('rating_count').to_frame()
    self.top50_product['mean_ratings'] = revs1[revs1['product'].isin(self.top50_product.index.tolist())].groupby(['product'])['score'].mean().astype('float64').round(1)

    # Select data with products having >50 ratings and users who have given > 50 ratings
    author50 = revs1['author'].value_counts()
    author50 = author50[author50>50].index.tolist() # list of authors with > 50 ratings

    product50 = revs1['product'].value_counts()
    product50 = product50[product50>50].index.tolist() # list of products with > 50 ratings

    self.revs_50 = revs1[(revs1['author'].isin(author50)) & (revs1['product'].isin(product50))]
    del author50, product50

    # Collaborative filtering model using SVD
    # Rearrange columns for SVD and prepare train and testsets
    revs50_ = Dataset.load_from_df(self.revs_50[['author','product','score']], Reader(rating_scale=(1, 10)))
    trainset, testset = train_test_split(revs50_, test_size=.25,random_state=RS)

    self.svd_pred, svd = self.svd_func(trainset,testset)
    self.svd_rmse = round(accuracy.rmse(self.svd_pred),3)

    # Collaborative filtering modle using KNNWithMeans_Item based
    self.knn_i_pred, knn_i = self.knn_item(trainset, testset)
    self.knn_i_rmse = round(accuracy.rmse(self.knn_i_pred),3)

    # Collaborative filtering model using KNNWithMeans_User based
    self.knn_u_pred, knn_u = self.knn_user(trainset, testset)
    self.knn_u_rmse = round(accuracy.rmse(self.knn_u_pred),3)


  # Choose the best model based on least RMSE value
  # (Helper function)
  def chooseModel(self):
    if(self.svd_rmse <= self.knn_i_rmse and self.svd_rmse <= self.knn_u_rmse):
      return self.svd_pred
    elif(self.knn_i_rmse <= self.svd_rmse and self.knn_i_rmse <= self.knn_u_rmse):
      return self.knn_i_pred
    else:
      return self.knn_u_pred

  # fit and predict using svd
  # (Helper function)
  def svd_func(self, train, test):
      if not os.path.exists("C:/Users/<USER>/SmartphoneRecommendation/models/svd/svd.sav"):
        svd = SVD(random_state=RS)
        svd.fit(train)
        pickle.dump(svd, open("C:/Users/<USER>/SmartphoneRecommendation/models/svd/svd.sav", 'wb'))
      svd = pickle.load(open("C:/Users/<USER>/SmartphoneRecommendation/models/svd/svd.sav", 'rb'))
      svd_pred = svd.test(test)
      return svd_pred, svd

  # fit and predict using knn
  # (Helper function)
  def knn_item(self, train, test):
      if not os.path.exists("C:/Users/<USER>/SmartphoneRecommendation/models/knni/knni.sav"):
        knn_i = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': False})
        knn_i.fit(train)
        pickle.dump(knn_i, open("C:/Users/<USER>/SmartphoneRecommendation/models/knni/knni.sav", 'wb'))
      knn_i = pickle.load(open("C:/Users/<USER>/SmartphoneRecommendation/models/knni/knni.sav", 'rb'))
      knn_i_pred = knn_i.test(test)
      return knn_i_pred, knn_i

  # fit and predict using knn
  # (Helper function)
  def knn_user(self, train, test):
      if not os.path.exists("C:/Users/<USER>/SmartphoneRecommendation/models/knnu/knnu.sav"):
        knn_u = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
        knn_u.fit(train)
        pickle.dump(knn_u, open("C:/Users/<USER>/SmartphoneRecommendation/models/knnu/knnu.sav", 'wb'))
      knn_u = pickle.load(open("C:/Users/<USER>/SmartphoneRecommendation/models/knnu/knnu.sav", 'rb'))
      knn_u_pred = knn_u.test(test)
      return knn_u_pred, knn_u


  def get5MostRatedProducts(self):
    return self.MOST_RATED_PRODUCTS

  def getTop50Products(self): # based on mean rating
    return self.top50_product.sort_values(by='mean_ratings',inplace=True)

  # Recommend top 5 mobile phones using popularity based model
  # (Using the data from the most popular phones amongst the most frequent users)
  def top5PopularityRecommendatation(self):
      ratings_mean_count = pd.DataFrame(self.revs_50.groupby('product')['score'].mean())
      ratings_mean_count['rating_counts'] = self.revs_50.groupby('product')['score'].count()
      ratings_mean_count = ratings_mean_count.sort_values(by=['score','rating_counts'], ascending=[False, False])
      print('Top 5 recommendations for the products are: \n')
      return ratings_mean_count.head()

  # Objective: To get top_n recommendation for each user
  def get_top_n(self, n=5):
      predictions = self.chooseModel()

      # First map the predictions to each user.
      top_n = defaultdict(list)
      for uid, iid, true_r, est, _ in predictions:
          top_n[uid].append((iid, est))

      # Then sort the predictions for each user and retrieve the n highest ones.
      for uid, user_ratings in top_n.items():
          user_ratings.sort(key=lambda x: x[1], reverse=True)
          top_n[uid] = user_ratings[:n]

      h, i, ll = random.sample(range(len(top_n)), 5), 0, []
      for k,v in top_n.items():
          if i in h:
              ll.append([k,v])
          i += 1
      return ll
