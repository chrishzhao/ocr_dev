import abc

class InferenceAPI():

    @abc.abstractmethod
    def load_model(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def infer(self, image):
        raise NotImplementedError()
