def test_real_fixture_shape(real_df):
    assert real_df.shape == (500, 6)
    assert set(real_df.columns) == {
        "age",
        "sex",
        "education",
        "hours_per_week",
        "income",
        "high_income",
    }


def test_syn_good_columns_match_real(real_df, syn_good_df):
    assert list(real_df.columns) == list(syn_good_df.columns)


def test_syn_leaky_overlaps_real(real_df, syn_leaky_df):
    real_set = set(map(tuple, real_df.values))
    overlap = sum(1 for row in syn_leaky_df.values if tuple(row) in real_set)
    assert overlap >= 60
