
class OrderedDataset:
    def __init__(self, docs, order):
        self.docs = docs
        self.order = order 

    def docs_iter(self):
        for i in self.order:
            yield self.docs[i-1]

    def docs_count(self):
        return len(self.docs)
    





        
    