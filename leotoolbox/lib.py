# -*- coding: UTF-8 -*-
# Copyright (C) 2018 Jean Bizot <jean@styckr.io>
""" Main lib for edgar_toolbox Project
"""

from os.path import split
import pandas as pd
import datetime
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import csv
import datetime
import urllib.parse
import requests
from weather import search_city, BASE_URI

pd.set_option("display.width", 200)


def get_data():
    """get data from sklearn"""
    faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return faces


def data_projection(faces):
    pca = PCA(n_components=150)
    data_projected = pca.fit_transform(faces.data)
    return data_projected, pca


def data_reconstruction(data_projected, pca):
    data_reconstructed = pca.inverse_transform(data_projected)
    return data_reconstructed

def daily_forecast(woeid, year, month, day):
    url = urllib.parse.urljoin(BASE_URI, f"/api/location/{woeid}/{year}/{month}/{day}")
    return requests.get(url).json()

def monthly_forecast(woeid, year, month):
    forecasts = []
    date = datetime.date(year, month, 1)
    if month == 12:
        upper_bound = datetime.date(year + 1, 1, 1)
    else:
        upper_bound = datetime.date(year, month + 1, 1)
    while date < upper_bound:
        print(f"Fetching forecast for {date.strftime('%Y-%m-%d')}")
        forecasts += daily_forecast(woeid, date.year, date.month, date.day)
        date = date + datetime.timedelta(days=1)
    return forecasts

def write_csv(woeid, year, month, city, forecasts):
    filename = f"{year}_{'{:02d}'.format(month)}_{woeid}_{city.lower()}"
    with open(f"data/{filename}.csv", 'w') as output_file:
        keys = forecasts[0].keys()
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(forecasts)

def main():
    if len(sys.argv) > 2:
        city = search_city(sys.argv[1])
        if city:
            woeid = city['woeid']
            year = int(sys.argv[2])
            month = int(sys.argv[3])
            if 1 <= month <= 12:
                forecasts = monthly_forecast(woeid, year, month)
                if not forecasts:
                    print("Sorry, could not fetch any forecast")
                else:
                    write_csv(woeid, year, month, city['title'], forecasts)
            else:
                print("MONTH must be a number between 1 (Jan) and 12 (Dec)")
                sys.exit(1)
    else:
        print("Usage: python history.py CITY YEAR MONTH")
        sys.exit(1)

if __name__ == "__main__":
    # For introspections purpose to quickly get this functions on ipython
    import edgar_toolbox

    main()
    folder_source, _ = split(edgar_toolbox.__file__)
    faces = get_data()
    data_projected, pca = data_projection(faces)
    data_reconstructed = data_reconstruction(data_projected, pca)
    fig = plt.figure(figsize=(7, 10))
    for i in range(15):
        plt.subplot(5, 5, i + 1)
        plt.title(faces.target_names[faces.target[i]], size=12)
        plt.imshow(
            pca.inverse_transform(data_projected)[i].reshape((50, 37)), cmap=plt.cm.gray
        )
        plt.xticks(())
        plt.yticks(())
    print(" dataframe extracted")
