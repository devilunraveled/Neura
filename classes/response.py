class Response:
    time = -1.0
    lazy = False
    def_Success_Message = 'Success'
    def_Failure_Message = 'Failure'
    message = ""
    returnObject = {"value" : -1}

class Success(Response):
    def __init__(self, time = -1.0, message = "", lazy = False, returnObject = None):
        self.time = time
        if ( message == "" ):
            self.message = self.def_Success_Message
        self.lazy = lazy
        self.returnObject = returnObject

class Failure(Response):
    def __init__(self, time = -1.0, message = "", lazy = False, returnObject = None):
        self.time = time
        if ( message == "" ):
            self.message = self.def_Failure_Message
        self.lazy = lazy
        self.returnObject = returnObject
