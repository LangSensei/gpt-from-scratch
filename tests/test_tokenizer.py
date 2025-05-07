from tokenizer import Vocabulary, Tokenizer

def test_tokenizer():
    vocab = Vocabulary("the-verdict.txt")
    tokenizer = Tokenizer(vocab)

    ids = tokenizer.encode(""""The height of his glory"--that was what the women called it.""")
    decoded_text = tokenizer.decode(ids)
    assert decoded_text.__contains__("height")

    text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
    )
    ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(ids)
    assert decoded_text.__contains__("tea")