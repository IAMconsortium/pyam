import logging


def test_logger_namespacing(test_df, caplog):
    with caplog.at_level(logging.INFO, logger="pyam.core"):
        test_df.filter(model="junk")

    assert caplog.record_tuples == [
        (
            "pyam.core",  # namespacing
            logging.WARNING,  # level
            "Filtered IamDataFrame is empty!",  # message
        )
    ]


def test_adjusting_logger_level(test_df, caplog):
    def throw_warning():
        logging.warning("This is a root warning")

    with caplog.at_level(logging.INFO, logger="pyam.core"):
        test_df.filter(model="junk")
        throw_warning()

    assert caplog.record_tuples == [
        ("pyam.core", logging.WARNING, "Filtered IamDataFrame is empty!"),
        ("root", logging.WARNING, "This is a root warning"),
    ]

    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="pyam.core"):
        test_df.filter(model="junk")
        throw_warning()

    # only the root warning should come through now i.e. we can silence pyam
    # without silencing everything
    # TODO this test fails with pytest>=6.0.1, deactivated for now
    # assert caplog.record_tuples == [
    #     ("root", logging.WARNING, "This is a root warning"),
    # ]
