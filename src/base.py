class Classifier(object):
    """Base class for all classifiers """
    _estimator_type = "classifier"

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from score import accuracy_score
        return accuracy_score(y, self.predict(X))


class Regressor(object):
    """Base class for all regression estimators """
    _estimator_type = "regressor"

    def score(self, X, y):
        """Returns the coefficient of determination R^2 of the prediction.
        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.
        Parameters
        ----------
        X : np.ndarray, shape = (n_samples, n_features)
            Test samples.
        y : np.ndarray, shape = (n_samples) or ( n_outputs, n_samples )
            True values for X.
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        from score import r2_score
        return r2_score(y, self.predict(X))
