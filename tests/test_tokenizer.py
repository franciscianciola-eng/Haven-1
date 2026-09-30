import pytest

from haven.cortex.tokenizer import BASE, END, HAVEN, Tokenizer

TEXT = ["the cat sat on the mat. the cat was happy. " * 20, "a dog ran in the park and the dog was glad. " * 20]


def test_learns_pieces_and_round_trips():
    tok = Tokenizer()
    added = tok.learn(TEXT, 50)
    assert 0 < added <= 50 and len(tok) == BASE + added
    for text in ["the cat sat", "Ünïcode — works too 日本", "<|haven|>hello<|end|>"]:
        assert tok.decode(tok.encode(text), specials=True) == text
    assert len(tok.encode("the cat sat on the mat")) < len(b"the cat sat on the mat")
    assert tok.encode("<|haven|>x<|end|>")[0] == HAVEN and tok.encode("<|end|>") == [END]


def test_grows_without_renumbering(tmp_path):
    tok = Tokenizer()
    tok.learn(TEXT, 30)
    before = tok.encode("the cat sat")
    tok.learn(["wizards and dragons " * 50], 10)
    assert tok.encode("the cat sat") == before
    tok.save(tmp_path / "t.json")
    again = Tokenizer.load(tmp_path / "t.json")
    assert again.merges == tok.merges and again.encode("wizards") == tok.encode("wizards")


def test_decode_ignores_unknown_ids():
    assert Tokenizer().decode([104, 105, 10**9]) == "hi"


pytest.importorskip("torch")


def test_new_pieces_can_leave_what_it_knew_as_it_was():
    tok = Tokenizer()
    tok.learn(["the cat sat on the mat"] * 20, 20)
    known = ["the cat sat on the mat", "I'm hungry and the cat is here"]
    before = [tok.encode(t) for t in known]
    grown = Tokenizer(tok.merges)
    added = grown.learn(["a matter of catalogs and matters and cats and hats"] * 30, 40, keep=known)
    assert added > 0
    assert [grown.encode(t) for t in known] == before  # it reads what it knew in exactly the same pieces
    assert len(grown.encode("matters")) < len(tok.encode("matters"))  # and new words in fewer
