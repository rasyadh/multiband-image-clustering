import os, json

from flask import Flask, request, Response, jsonify, json
from flask import render_template, url_for, redirect, send_from_directory
from flask import make_response, abort, session
from werkzeug.utils import secure_filename

from project import app

from multiband.mband_img_cluster import MultibandImageCluster as MIC

# Web Profile
@app.route('/')
def index():
    return redirect(url_for('multiband'))

# Multiband Image Clustering
@app.route('/multiband/')
def multiband():
    cluster, cluster_img, result_img = [], [], []

    for i in range(2, 7):
        cluster.append(i)
        cluster_img.append(url_for('static', filename='multiband/cluster' + str(i) + '.jpg'))
        result_img.append(url_for('static', filename='multiband/result-multiband' + str(i) + '.jpg'))

    return render_template('multiband/index.html', title='Multiband Image Clustering', cluster=cluster, clusterimg=cluster_img, resultimg=result_img)

@app.route('/multiband/process/')
def process_mic():
    try:
        app = MIC('project/static/landsat7/')
        image = app.read_images()
        features = app.feature_space_transformation(image)
        feature_copy = features.copy()
        for i in range(2, 7):
            label = app.KMeans_clustering(features, cluster=i, iteration=100)
            app.image_creation(feature_copy, features, label, cluster=i)
    except RuntimeError:
        print('runtime error')

    return redirect(url_for('multiband'))
