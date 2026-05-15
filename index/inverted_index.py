from .documents import tokenizer
import numpy as np

class Index:
    def __init__(self):
       self.index = {}
       self.dictionary = {}
    #    self.doc_ids = 0
       self.num_docs = 0

    def build_index(self, dataset, my_tokenizer, d_gap=False , max_num_docs=1001):
        if d_gap:
            return self.d_gap_index(dataset, my_tokenizer, max_num_docs)
        else:
            return self.simple_index(dataset, my_tokenizer, max_num_docs)

    def simple_index(self, dataset, my_tokenizer, max_num_docs=1001):
        count_docs = 0      # number of docs
        # count_tokens = 0    # number of tokens
        count_postings = 0  # number of postings
        docid = 0           # docid assigned sequentially during the parsing

        for i, doc in enumerate(dataset.docs_iter()):
            
            if i == max_num_docs:
                break

            docid = i+1 # assign docid

            tokens = my_tokenizer(doc.text)

            count_docs += 1
            # count_tokens += len(tokens)

            for el in tokens:
                if el in self.index:
                    last_docid, tf = self.index[el][-1]
                    if docid != last_docid:       # check if doc_id was already appended, multiple
                        self.index[el].append((docid,1))
                        count_postings += 1
                    else:
                        self.index[el][-1] = (docid, tf+1)
                else:
                    self.index[el] = [(docid, 1)]
                    count_postings += 1
                    
        self.num_docs = count_docs
        return self.index

    def d_gap_index(self, dataset, my_tokenizer, max_num_docs=1001):

        docid = 0           # docid assigned sequentially during the parsing
        count_docs = 0      # number of docs
        # count_tokens = 0    # number of tokens
        count_postings = 0  # number of postings
        previous_docid = {}

        for i, doc in enumerate(dataset.docs_iter()):
            
            if i == max_num_docs:
                break

            docid = i+1 # assign docid

            tokens = my_tokenizer(doc.text)

            count_docs += 1
            # count_tokens += len(tokens)

            for el in tokens:

                if el in self.index:
                    
                    last_gap, tf = self.index[el][-1]

                    prev_docid = previous_docid[el]
                    d_Gap = docid - prev_docid

                    if d_Gap > 0:       # check if doc_id was already appended, multiple occurrence of the same term in the dictionary     
                        self.index[el].append((d_Gap,1))
                        count_postings += 1
                        previous_docid[el] = docid

                    else:
                        self.index[el][-1] = (last_gap, tf+1)
                else:
                    previous_docid[el] = docid
                    self.index[el] = [(docid, 1)]
                    count_postings += 1

        return self.index
    
    def save_index(self, file_path):
        pass

    def load_index(self, file_path):
        pass

    def compress_index(self):
        pass

    def decompress_index(self):
        pass

    def get_terms(self):
        return self.index.keys()
    
    def get_posting_list(self, term):
        return self.index.get(term, [])

    def get_term_frequency(self, term, doc_id):
        posting_list = self.get_posting_list(term)
        for did, tf in posting_list:
            if did == doc_id:
                return np.log10(1+tf)
        return 0

    def get_inverse_document_frequency(self, term):
        df = len(self.get_posting_list(term))
        # if df == 0: #! non necessario -> se metto un termine c'è almeno su un documento
            # return 0
        assert df > 0, f"Document frequency for term '{term}' should be greater than 0."
        
        return np.log10(self.num_docs/df)
    
    def get_doc_ids(self):
        doc_ids = set()
        for posting_list in self.index.values():
            for doc_id, _tf in posting_list:
                doc_ids.add(doc_id)
        return sorted(doc_ids)
    
    # def get_approx_size(self):
    #     size = 0
    #     for term, posting_list in self.index.items():
    #         size += len(term) + 1  # term length + separator
    #         size += len(posting_list) * (4 + 4)  # doc_id (4 bytes) + tf (4 bytes)
    #     return size
    
    def get_approx_size(self, bytes_per_doc_id=4, bytes_per_tf=4, include_term_bytes=True, encoding="utf-8"):
        """
        Approximate serialized size of the inverted index in bytes.

        Assumptions:
        - each posting stores (doc_id, tf)
        - doc_id uses bytes_per_doc_id bytes
        - tf uses bytes_per_tf bytes
        - optionally include term bytes (UTF-8 by default)
        """
        posting_bytes = bytes_per_doc_id + bytes_per_tf
        total = 0

        for term, posting_list in self.index.items():
            if include_term_bytes:
                total += len(term.encode(encoding))
            total += len(posting_list) * posting_bytes

        return total