def classify(features_train, labels_train):
    # import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB
    # create classifier
    # fit the classifier on the training features and labels
    # return the fit classifier

    # your code goes here!
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    # pred = clf.predict(features_train)
    return clf