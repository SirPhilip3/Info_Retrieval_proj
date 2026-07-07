from index._compression import IndexCompressor
from index._storage import IndexStorage

import numpy as np

MAX_UINT32 = 2**32 - 1

class Index:
    def __init__(self):
        self.index = {}
        self.dictionary = {}
        
        self.num_docs = 0
        self.compression_method = "None"
        self.removed_tf = False

        self.d_gap_index_ = False

        self.compressor = IndexCompressor(index_obj=self)
        self.storage = IndexStorage(index_obj=self, file_path=None)

    def build_index(self, dataset, my_tokenizer, d_gap=False, max_num_docs=1001):
        if d_gap:
            self.d_gap_index_ = True
            self.d_gap_index(dataset, my_tokenizer, max_num_docs)
        else:
            self.simple_index(dataset, my_tokenizer, max_num_docs)

    def simple_index(self, dataset, my_tokenizer, max_num_docs=1001):
        count_docs = 0      # number of docs
        count_postings = 0  # number of postings
        docid = 0           # docid assigned sequentially during the parsing

        total_docs = 0
        if max_num_docs < dataset.docs_count():
            total_docs = dataset.docs_count()
        else:
            total_docs = max_num_docs

        for i, doc in enumerate(dataset.docs_iter()):
            
            # print dynamic progress
            if i % 1000 == 0 and i > 0:
                percentage = (i / total_docs) * 100
                print(f"\rProcessing: {percentage:6.1f}% [{i}/{total_docs}] | Current docid: {i:<20}", end="", flush=True)

            if i == max_num_docs:
                break

            docid = i+1 # assign docid

            tokens = my_tokenizer(doc.text)

            count_docs += 1

            for el in tokens:
                if el in self.index:
                    last_docid, tf = self.index[el][-1]
                    if docid != last_docid:       # check if doc_id was already appended
                        self.index[el].append((docid,1))
                        count_postings += 1
                    else:
                        self.index[el][-1] = (docid, tf+1)
                else:
                    self.index[el] = [(docid, 1)]
                    count_postings += 1
                    
        self.num_docs = count_docs


    def d_gap_index(self, dataset, my_tokenizer, max_num_docs=1001):

        docid = 0           # docid assigned sequentially during the parsing
        count_docs = 0      # number of docs
        count_postings = 0  # number of postings
        
        previous_docid = {}
        

        total_docs = 0
        if max_num_docs < dataset.docs_count():
            total_docs = dataset.docs_count()
        else:
            total_docs = max_num_docs

        for i, doc in enumerate(dataset.docs_iter()):

            if i % 1000 == 0 and i > 0:
                percentage = (i / total_docs) * 100
                print(f"\rProcessing: {percentage:6.1f}% [{i}/{total_docs}] | Current docid: {i:<20}", end="", flush=True)
            
            if i == max_num_docs:
                break

            docid = i+1 # assign docid

            tokens = my_tokenizer(doc.text)

            count_docs += 1

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

    
    def remove_tf(self):
        """Remove term frequencies from the index, leaving only document IDs."""
        if self.removed_tf:
            raise ValueError("Term frequencies have already been removed from the index.")

        for term, posting_list in self.index.items():
            self.index[term] = [doc_id for doc_id, _tf in posting_list]
        
        self.removed_tf = True
        
    def save_index(self, file_path): 
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
        self.compression_method = "None"
        self.removed_tf = False

        self.d_gap_index_ = False

        self.compressor.clear_compressor()
        self.storage.clear_storage()

    def compress_index(self, method="gamma", k=8, block_size=128, extra_strategy="None"):
        if self.index is None or len(self.index) == 0:
            raise ValueError("Inverted index is empty. Build the index before applying compression.")
        if self.compression_method != "None":
            raise ValueError(f"Index is already compressed with method '{self.compression_method}'. Clear the index before applying new compression.")

        methods_map = { 
            "gamma": self.compressor.gamma_encode,
            "delta": self.compressor.delta_encode,
            "varbyte": self.compressor.varbyte_encode,
            "pfor_delta": self.compressor.pfor_delta_encode_bit #!test
        }

        if method in methods_map:
            self.compression_method = method

            if method == "pfor_delta":
                if self.removed_tf:
                    self.compressor.pfor_delta_k = k
                    self.compressor.pfor_delta_block_size = block_size
                    self.compressor.pfor_delta_extra_strategy = extra_strategy
                else:
                    raise ValueError("PForDelta compression requires term frequencies to be removed. Please call remove_tf() before applying PForDelta compression.")

            self.compressor.apply_index_transformation(methods_map[method])
        else:
            raise ValueError(f"Compression method '{method}' is not supported. Allowed methods: {list(methods_map.keys())}")
        
    def decompress_index(self):
        if self.index is None or len(self.index) == 0:
            raise ValueError("Inverted index is empty. Build the index before applying decompression.")
        if self.compression_method == "None":
            raise ValueError("Index is not compressed. Cannot apply decompression.")

        methods_map = {
            "gamma": self.compressor.gamma_decode,
            "delta": self.compressor.delta_decode,
            "varbyte": self.compressor.varbyte_decode,
            "pfor_delta": self.compressor.pfor_delta_decode_bit #! test
        }

        if self.compression_method in methods_map:
            self.compressor.apply_index_transformation(methods_map[self.compression_method], is_decompressing=True)
            self.compression_method = "None"  # Reset compression method after decompression
        else:
            raise ValueError(f"Decompression method '{self.compression_method}' is not supported. Allowed methods: {list(methods_map.keys())}")
        
    def to_dgaps(self):
        if self.d_gap_index_:
            raise ValueError("The index is a d-gap index. Cannot perform d-gap serialization.")
    
        self.d_gap_index_ = True

        for term, posting_list in self.index.items():
            current_doc_id = 0
            
            if self.removed_tf:
                for i, doc_id in enumerate(posting_list):
                    d_gap = doc_id - current_doc_id
                    self.index[term][i] = d_gap
                    current_doc_id = doc_id
            else:
                for i, (doc_id, tf) in enumerate(posting_list):
                    d_gap = doc_id - current_doc_id
                    self.index[term][i] = (d_gap, tf)
                    current_doc_id = doc_id

    def from_dgaps(self):
        if not self.d_gap_index_:
            raise ValueError("The index is not a d-gap index. Cannot perform d-gap deserialization.")
    
        self.d_gap_index_ = False

        for term, posting_list in self.index.items():
            current_doc_id = 0
            
            if self.removed_tf:
                for i, d_gap in enumerate(posting_list):
                    current_doc_id += d_gap
                    self.index[term][i] = current_doc_id
            else:
                for i, (d_gap, tf) in enumerate(posting_list):
                    current_doc_id += d_gap
                    self.index[term][i] = (current_doc_id, tf)

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

        assert df > 0, f"Document frequency for term '{term}' should be greater than 0."
        
        return np.log10(self.num_docs/df)
    
    def get_doc_ids(self):
        doc_ids = set()
        for posting_list in self.index.values():
            for doc_id, _tf in posting_list:
                doc_ids.add(doc_id)
        return sorted(doc_ids)
    
    def get_approx_size(self, include_term_bytes=False, verbose = False):
        if include_term_bytes:
            raise NotImplementedError("Term byte size calculation is not implemented yet.")

        term_bytes = 0
        doc_id_bits = 0
        tf_bits = 0

        for term, posting_list in self.index.items():
            if self.compression_method == "None":
                num_postings = len(posting_list)
                if self.removed_tf:
                    doc_id_bits += num_postings * 32
                else:
                    doc_id_bits += num_postings * 32
                    tf_bits += num_postings * 32
                    
            elif self.compression_method == "pfor_delta":
                for element in posting_list:
    
                    if isinstance(element, tuple):
                        doc_id_s, outliers, min_val = element
                        
                        
                        if hasattr(doc_id_s, 'len'):
                            doc_id_bits += doc_id_s.len()
                        
                        
                        if isinstance(outliers, list):
                            for outlier in outliers:
                                if hasattr(outlier, 'len'):
                                    doc_id_bits += outlier.len()
                                else:
                                    
                                    doc_id_bits += 32 
                        
                        doc_id_bits += 32

                    elif hasattr(element, 'len'):
                        doc_id_bits += element.len()


                    elif isinstance(element, list):
                        
                        doc_id_bits += len(element) * 32
                        
                    
                    elif isinstance(element, int):
                        doc_id_bits += 32
            else:
                for posting in posting_list:
                    if self.removed_tf:
                        doc_id_bits += len(posting)
                    else:
                        doc_id_bits += len(posting[0])
                        tf_bits += len(posting[1])

        total_bits = doc_id_bits + tf_bits

        result = {
            "term_bytes": term_bytes,
            "doc_id_bits": doc_id_bits,
            "tf_bits": tf_bits,
            "total_bits": total_bits,
            "total_bytes": total_bits / 8,
        }
        if verbose:
            print("\n=== Index Size Breakdown ===")
            print(f"Doc ID bits:  {doc_id_bits:,}")
            print(f"TF bits:      {tf_bits:,}")
            print(f"Total bits:   {total_bits:,}")
            print(f"Total bytes:  {total_bits / 8:,.2f}")

        return result
    

    def compute_docId_termId(self, dataset, tokenizer, max_num_docs=1001):
        terms = self.get_terms()
        term_index = {term: idx for idx, term in enumerate(terms)}

        doc_set = {}

        for i, doc in enumerate(dataset.docs_iter()):
            if i == max_num_docs:
                break

            doc_id = doc.doc_id
            doc_terms = tokenizer(doc.text)
            for term in doc_terms:
                if term in term_index: #useless maybe
                    term_id = term_index[term]
                    if doc_id not in doc_set:
                        doc_set[doc_id] = list()
                    doc_set[doc_id].append(term_id)

        
        return doc_set
            