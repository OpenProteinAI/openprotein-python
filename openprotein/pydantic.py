try:
    from pydantic.v1 import (
        BaseModel,
        Field,
        ConfigDict,
        validator,
        root_validator,
        parse_obj_as,
    )
    import pydantic.v1 as pydantic
except ImportError:
    from pydantic import (
        BaseModel,
        Field,
        ConfigDict,
        validator,
        root_validator,
        parse_obj_as,
    )
    import pydantic
