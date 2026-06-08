from index._compression import IndexCompressor
from index._storage import IndexStorage

import numpy as np

MAX_UINT32 = 2**32 - 1

class Index:
    def __init__(self):
        self.index = {}
        self.dictionary = {}
        
        self.num_docs = 0
        self.compression_method = None
        self.removed_tf = False

        # self.compression_function = None
        # self.decompression_function = None

        self.compressor = IndexCompressor(index_obj=self)
        self.storage = IndexStorage(index_obj=self, file_path=None)

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
    
    def remove_tf(self):
        """Remove term frequencies from the index, leaving only document IDs."""
        if self.removed_tf:
            raise ValueError("Term frequencies have already been removed from the index.")

        for term, posting_list in self.index.items():
            self.index[term] = [doc_id for doc_id, _tf in posting_list]
        
        self.removed_tf = True
        
    def save_index(self, file_path): #! temporary just save the postings not the terms 

        self.storage.set_file_path(file_path)
        self.storage.write_to_disk()
                

    def load_index(self, file_path):

        if self.index != None and len(self.index) > 0:
            raise ValueError("Inverted index is not empty. Clear the index before loading a new one.")

        self.storage.set_file_path(file_path)
        self.storage.read_from_disk()

    def clear_index(self):
        self.index = {}
        self.dictionary = {}
        self.num_docs = 0
        self.compression_method = None
        self.removed_tf = False

    def compress_index(self, method="gamma"):
        if self.index is None or len(self.index) == 0:
            raise ValueError("Inverted index is empty. Build the index before applying compression.")
        if self.compression_method != None:
            raise ValueError("Index is already compressed with method '{self.compression_method}'. Clear the index before applying new compression.")

        methods_map = {
            "gamma": self.compressor.gamma_encode,
            "delta": self.compressor.delta_encode,
            "varbyte": self.compressor.varbyte_encode
        }

        if method in methods_map:
            self.compression_method = method
            self.compressor.general_compressor(methods_map[method])
        else:
            raise ValueError(f"Compression method '{method}' is not supported. Allowed methods: {list(methods_map.keys())}")

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
    
    # def get_approx_size(self, bytes_per_doc_id=4, bytes_per_tf=4, include_term_bytes=True, encoding="utf-8"):
    #     """
    #     Approximate serialized size of the inverted index in bytes.

    #     Assumptions:
    #     - each posting stores (doc_id, tf)
    #     - doc_id uses bytes_per_doc_id bytes
    #     - tf uses bytes_per_tf bytes
    #     - optionally include term bytes (UTF-8 by default)
    #     """
    #     posting_bytes = bytes_per_doc_id + bytes_per_tf
    #     total = 0

    #     for term, posting_list in self.index.items():
    #         if include_term_bytes:
    #             total += len(term.encode(encoding))
    #         total += len(posting_list) * posting_bytes

    #     return total

    def get_approx_size(self, include_term_bytes=False, encoding="utf-8"):
        term_bytes = 0
        doc_id_bytes = 0
        tf_bytes = 0
        total = 0

        for term, posting_list in self.index.items():
            if include_term_bytes:
                term_size = len(term.encode(encoding))
                term_bytes += term_size
                total += term_size

            if self.compression_method is None:
                # uncompressed
                for posting in posting_list:
                    if self.removed_tf:
                        # posting is just a doc_id
                        doc_id_size = len(str(posting))
                        doc_id_bytes += doc_id_size
                        total += doc_id_size
                    else:
                        # posting is a tuple (doc_id, tf)
                        if isinstance(posting, tuple):
                            doc_id = posting[0]
                            tf = posting[1]
                            doc_id_size = len(str(doc_id))
                            tf_size = len(str(tf))
                            doc_id_bytes += doc_id_size
                            tf_bytes += tf_size
                            total += doc_id_size + tf_size
                        else:
                            doc_id_size = len(str(posting))
                            doc_id_bytes += doc_id_size
                            total += doc_id_size

            else:
                # compressed
                for posting in posting_list:
                    if self.removed_tf:
                        # posting is just a doc_id (string)
                        doc_id_size = len(posting)
                        doc_id_bytes += doc_id_size
                        total += doc_id_size
                    else:
                        # posting is a tuple of compressed strings
                        if isinstance(posting, tuple):
                            doc_id_size = len(posting[0])
                            tf_size = len(posting[1])
                            doc_id_bytes += doc_id_size
                            tf_bytes += tf_size
                            total += doc_id_size + tf_size
                        else:
                            doc_id_size = len(posting)
                            doc_id_bytes += doc_id_size
                            total += doc_id_size

        # Format sizes
        def format_size(bytes_val):
            kb = bytes_val / 1024
            mb = kb / 1024
            if mb >= 1:
                return f"{bytes_val:,} bytes ({kb:,.2f} KB / {mb:,.2f} MB)"
            elif kb >= 1:
                return f"{bytes_val:,} bytes ({kb:,.2f} KB)"
            else:
                return f"{bytes_val:,} bytes"

        # Build detailed report
        report = f"""
                === Index Size Breakdown ===
                Total Size: {format_size(total)}
                - Term Bytes: {format_size(term_bytes)} ({term_bytes/total*100:.2f}%)
                - Doc ID Bytes: {format_size(doc_id_bytes)} ({doc_id_bytes/total*100:.2f}%)
                - TF Bytes: {format_size(tf_bytes)} ({tf_bytes/total*100:.2f}%)
                """
        
        print(report)
        return total