class InvalidParameterError(Exception):
    """InvalidParameterError"""
    def __init__(self, message="Invalid parameter"):
        self.message = message
        super().__init__(self.message)

class MissingParameterError(Exception):
    """MissingParameterError"""
    def __init__(self, message="Required parameter is missing"):
        self.message = message
        super().__init__(self.message)
 
class APIError(Exception):
    """APIError"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

