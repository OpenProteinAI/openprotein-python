import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "helper_names"),
    [
        (
            "generative.base",
            ("check_inputs", "query_ids", "takes_list", "with_param"),
        ),
        (
            "foundational.poet",
            ("check_inputs", "prompt_id", "query_ids", "takes_list", "with_param"),
        ),
        (
            "foundational.poet2",
            ("check_inputs", "prompt_id", "query_ids", "takes_list", "with_param"),
        ),
        ("estimators", ("as_dataset", "assay_id", "model_id")),
        ("model", ("assay_sequences",)),
        ("svd", ("as_dataset", "assay_id", "model_id")),
        ("gp", ("assay_id", "model_id")),
        ("umap", ("assay_sequences", "model_id")),
    ],
)
def test_input_helpers_are_not_public(module_name, helper_names):
    module = importlib.import_module(f"openprotein.models.{module_name}")
    for helper_name in helper_names:
        assert not hasattr(module, helper_name), helper_name


def test_deprecation_shims_export_no_helper_names_from_the_models_package():
    import openprotein.models

    assert not {"import_module", "Any", "warn_deprecated"} & set(
        openprotein.models.__all__
    )


def test_download_workers_is_not_bound_in_the_jobs_module():
    import openprotein.jobs.jobs

    assert "DOWNLOAD_WORKERS" not in vars(openprotein.jobs.jobs)


def test_prompt_content_read_is_not_bound_in_the_stored_prompts_module():
    import openprotein.data.prompts.prompts

    assert "get_prompt" not in vars(openprotein.data.prompts.prompts)
