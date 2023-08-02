import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

'''
Given N samples and constrained to n < N samples for in-context learning, select n samples from N samples
Fit a logistic regression model on the N samples, and choose the n that maximize entropy (highest uncertainty)
'''

def entropy_learn_sample(train_data_dict, sample_size):
    assert sample_size < len(train_data_dict), 'sample_size must be less than the number of samples'
    train_data = pd.DataFrame(train_data_dict).T
    label = 'Label'
    train_data.dropna(inplace=True)
    print(train_data)
    features = train_data.drop(label, axis=1)
    target = train_data[label]
    
    # Apply the StandardScaler only to the features
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)
    
    # Recombine the features and the target
    train_data = pd.concat([scaled_features, target], axis=1)
    train_data = train_data.sample(frac=1)  # shuffle
    # run logistic regression
    X = train_data.drop(label, axis=1)
    y = train_data[label]
    clf = LogisticRegression(random_state=0).fit(X, y)
    # get entropy
    train_data['pred_prob'] = clf.predict_proba(X)[:, 1]
    train_data['pred_prob'] = train_data['pred_prob'].apply(lambda x: min(x, 1 - x))
    train_data.sort_values(by='pred_prob', inplace=True, ascending=False)
    top_indices = train_data.index[:sample_size]
    res = {k: v for k, v in train_data_dict.items() if k in top_indices}
    return res

def active_learn_sample(train_data_dict, sample_size):
    assert sample_size < len(train_data_dict), 'sample_size must be less than the number of samples'
    train_data = pd.DataFrame(train_data_dict).T
    label = 'Label'
    train_data.dropna(inplace=True)
    features = train_data.drop(label, axis=1)
    target = train_data[label]
    
    # Apply the StandardScaler only to the features
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)
    
    # Recombine the features and the target
    train_data = pd.concat([scaled_features, target], axis=1)
    train_data = train_data.sample(frac=1)  # shuffle
    valid_data = train_data.copy()

    # get 4 random samples
    # Separate samples by class
    class_0_data = train_data[train_data[label] == 0]
    class_1_data = train_data[train_data[label] == 1]
    # Get 2 random samples from each class
    seed_indices = list(class_0_data.sample(n=1).index) + list(class_1_data.sample(n=1).index)
    train_data = train_data.loc[seed_indices]
    # drop seed indices from valid_data
    valid_data = valid_data.drop(seed_indices)


    def get_max_entropy(df):
        df['entropy'] = df['pred_prob'].apply(lambda x: min(x, 1 - x))
        df.sort_values(by='entropy', inplace=True, ascending=False)
        return df.index[0]
    
    while train_data.shape[0] < sample_size:
        # run logistic regression
        X_train = train_data.drop(label, axis=1, errors='ignore')
        y_train = train_data[label]
        clf = LogisticRegression(random_state=0).fit(X_train, y_train)

        X_test = valid_data.drop(label, axis=1, errors='ignore')
        y_test = valid_data[label]
        # get entropy
        valid_data['pred_prob'] = clf.predict_proba(X_test)[:, 1]
        max_entropy_index = get_max_entropy(valid_data)
        valid_data.drop(['pred_prob', 'entropy'], axis=1, inplace=True)
        # swap from valid_data to train_data
        train_data = train_data.append(valid_data.loc[max_entropy_index])
        valid_data = valid_data.drop(max_entropy_index)
        print('num training samples: ', train_data.shape[0], 'accuracy: ', clf.score(X_test, y_test))
        print()

    train_data_indices = train_data.index

    return {k: v for k, v in train_data_dict.items() if k in train_data_indices}


def cluster_learn_sample(train_data_dict, sample_size):
    assert sample_size < len(train_data_dict), 'sample_size must be less than the number of samples'
    train_data = pd.DataFrame(train_data_dict).T

    train_data.dropna(inplace=True)
    train_data = train_data.sample(frac=1)  # shuffle

    # Note: Here we are assuming that the 'Label' column is not considered
    # in the clustering process. If it should be, remove the next line.
    train_data_no_label = train_data.drop('Label', axis=1)

    # scale the features
    scaler = StandardScaler()
    train_data_scaled = pd.DataFrame(scaler.fit_transform(train_data_no_label), columns=train_data_no_label.columns, index=train_data_no_label.index)

    # run K-means clustering
    kmeans = KMeans(n_clusters=sample_size, random_state=0).fit(train_data_scaled)

    # get cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # find the closest data point from each cluster center
    # we will store indices of closest points here
    closest_points_indices = []
    for center in cluster_centers:
        distances = ((train_data_scaled - center)**2).sum(axis=1)
        closest_points_indices.append(distances.idxmin())

    # Filter the dictionary to return only those keys that are in closest_points_indices
    res = {k: v for k, v in train_data_dict.items() if k in closest_points_indices}

    return res
