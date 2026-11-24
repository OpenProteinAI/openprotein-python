from openprotein import errors
from openprotein.errors import NotPrepared


def test_not_prepared_names_the_job_whose_units_are_not_listed():
    error = NotPrepared("job-1")

    assert error.job_id == "job-1"
    assert "job-1" in str(error)
    assert isinstance(error, Exception)


def test_not_prepared_is_part_of_the_errors_surface():
    assert "NotPrepared" in errors.__all__
