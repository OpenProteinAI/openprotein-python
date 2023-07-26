def pytest_addoption(parser):

    # adapted from https://stackoverflow.com/a/33181491
    parser.addoption('--longrun', action='store_true', dest="longrun",
                 default=False, help="enable longrundecorated tests")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "longrun: mark tests that take a long time to run"
    )
    if not config.option.longrun:
        if config.option.markexpr != "":
            setattr(
                config.option,
                'markexpr',
                config.option.markexpr + ' and not longrun',
            )
        else:
            setattr(config.option, 'markexpr', 'not longrun')
