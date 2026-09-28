import json

from haven.cli import main


def test_a_life_from_the_command_line(home, capsys):
    assert main(["live", "--ticks", "400", "--seed", "2"]) == 0
    out = capsys.readouterr().out
    assert "born" in out and "Lived 400 moments" in out
    assert main(["status"]) == 0
    assert "What it has concluded about itself" in capsys.readouterr().out
    assert main(["story"]) == 0
    assert "came into the world" in capsys.readouterr().out
    assert main(["check", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["indicators"]) == 14
    assert main(["reset", "--yes"]) == 0
    assert "Archived" in capsys.readouterr().out
    assert main(["status"]) == 1


def test_it_says_so_when_there_is_no_life_yet(home, capsys):
    assert main(["status"]) == 1
    assert "No Haven lives" in capsys.readouterr().out


def test_plain_haven_means_live():
    from haven.cli import with_default_command

    names = {"live", "status", "ask"}
    assert with_default_command([], names) == ["live"]
    assert with_default_command(["--cortex", "none"], names) == ["live", "--cortex", "none"]
    assert with_default_command(["--home", "x", "--speed", "3"], names) == ["--home", "x", "live", "--speed", "3"]
    assert with_default_command(["ask", "are you alive?"], names) == ["ask", "are you alive?"]
