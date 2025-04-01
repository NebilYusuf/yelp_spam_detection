# first line: 59
@memory.cache
def extract_pos_sequences(review_text, batch_size):
    pos_sequences = []
    for i in tqdm(range(0, len(review_text), batch_size), desc="Extracting POS tags"):
        batch = review_text.iloc[i:i+batch_size]
        for doc in nlp.pipe(batch, disable=["ner", "parser"], batch_size=32, n_process=4):
            pos_sequence = " ".join([token.pos_ for token in doc])
            pos_sequences.append(pos_sequence)
    return pos_sequences
