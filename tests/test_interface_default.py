from biasedrandomforest import BiasedRandomForestClassifier


def test_p_rf_split():
    clf = BiasedRandomForestClassifier()
    assert clf.p_rf_split == 0.5


def test_n1():
    clf = BiasedRandomForestClassifier()
    assert clf.N1 == 50


def test_n2():
    clf = BiasedRandomForestClassifier()
    assert clf.N2 == 50
