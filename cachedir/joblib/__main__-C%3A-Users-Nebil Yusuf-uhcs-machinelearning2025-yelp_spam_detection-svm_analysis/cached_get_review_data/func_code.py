# first line: 69
@memory.cache
def cached_get_review_data(engine_url, feature_type, batch_size=20000):
    engine = create_db_engine_from_url(engine_url)
    return get_review_data(engine, feature_type, batch_size)
