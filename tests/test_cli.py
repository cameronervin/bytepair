from bytepair.cli import main


def test_train_encode_decode(tmp_path, capsys):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the cat sat on the mat. " * 30)
    prefix = tmp_path / "m"
    main(["train", str(corpus), "--vocab-size", "280", "--out", str(prefix)])
    model = f"{prefix}.model"

    main(["encode", "the cat", "--model", model, "--ids"])
    ids = capsys.readouterr().out.strip().splitlines()[-1].split()

    main(["decode", *ids, "--model", model])
    assert capsys.readouterr().out.strip() == "the cat"
