from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import as_float_array
import numpy as np

class BiasedRandomForestClassifier(object):
    def __init__(
        self,
        p_rf_split=0.5,
        n_neighbors=5,
        # Nearest Neighbors Params
        knn_radius=1.0,
        knn_algorithm="auto",
        knn_leaf_size=30,
        knn_metric="minkowski",
        knn_p_minkowski=2,
        knn_metric_params=None,
        ## Random Forest Params
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        if p_rf_split < 0.0 or p_rf_split > 1.0:
            raise ("Error: p_rf_split out of proper bounds")
        self.p_rf_split = p_rf_split

        # K Nearest Neighbors
        self.kNN = NearestNeighbors(
            n_neighbors=n_neighbors,
            radius=knn_radius,
            algorithm=knn_algorithm,
            leaf_size=knn_leaf_size,
            metric=knn_metric,
            p=knn_p_minkowski,
            metric_params=knn_metric_params,
            n_jobs=n_jobs,
        )

        # Random Forest for full set
        self.N1 = int(n_estimators * p_rf_split)
        self.rf1 = RandomForestClassifier(
            n_estimators=self.N1,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

        # Random Forest for critical set
        self.N2 = n_estimators - self.N1
        self.rf2 = RandomForestClassifier(
            n_estimators=self.N2,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

    def fit(self, X, y, sample_weight=None):
        # Begin building critical set
        Xnp = as_float_array(X)
        ynp = as_float_array(y)

        self.kNN.fit(Xnp)

        target_classes = np.unique(ynp)
        if len(target_classes) != 2:
            raise ("Error: there must be two target classes only")

        index_first = np.where(ynp == target_classes[0])
        index_second = np.where(ynp == target_classes[1])

        if len(index_first) > len(index_second):
            Xmaj = Xnp[index_first]
            Xmin = Xnp[index_second]
        else:
            Xmaj = Xnp[index_second]
            Xmin = Xnp[index_first]

        crit_idx = np.unique(self.kNN.kneighbors(X=Xmin, return_distance=False).ravel())
        Xcrit = Xnp[crit_idx]
        ycrit = ynp[crit_idx]

        # Train the random forests
        self.rf1.fit(X=Xnp, y=ynp, sample_weight=sample_weight)
        self.rf2.fit(X=Xcrit, y=ycrit, sample_weight=sample_weight)

        # Params
        if np.array_equal(self.rf1.classes_, self.rf2.classes_):
            self.classes_ = self.rf1.classes_
        else:
            raise("Error: forests have different classes")

    def predict(self, X):
        y_prob = self.predict_proba(X)
        return np.where(
            y_prob[:, 0] < y_prob[:, 1], self.classes_[1], self.classes_[0]
        )

    def predict_proba(self, X):
        return self.p_rf_split * self.rf1.predict_proba(X) + (
            1 - self.p_rf_split
        ) * self.rf2.predict_proba(X)

