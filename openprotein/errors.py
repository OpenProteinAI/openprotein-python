# Errors for OpenProtein
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

class AuthError(Exception):
    """InvalidParameterError"""
    def __init__(self, message="Invalid authorization"):
        self.message = message
        super().__init__(self.message)
    
class InvalidJob(Exception):
    """InvalidParameterError"""
    def __init__(self, message="No such job"):
        self.message = message
        super().__init__(self.message)

class TimeoutException(Exception):
    """InvalidParameterError"""
    def __init__(self, message="Request timed out!"):
        self.message = message
        super().__init__(self.message)