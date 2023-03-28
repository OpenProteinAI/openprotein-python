from enum import Enum

class Backend(str, Enum):
    PROD = 'https://api.openprotein.ai/api/'
    DEV = 'https://backend-dev.openprotein.ai/api/'
    TEST = 'https://test.api.openprotein.ai/api/'

POLLING_INTERVAL = 2.5
POLLING_TIMEOUT = 600

BASE_PAGE_SIZE = 256
PROTS2PROT_PAGE_SIZE = 256