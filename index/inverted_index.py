from .documents import tokenizer
import numpy as np

class Index:
    def __init__(self):
        self.index = {}
        self.dictionary = {}
        # self.doc_ids = 0
        self.num_docs = 0
        self.compression_method = None
        self.removed_tf = False

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
        for term, posting_list in self.index.items():
            self.index[term] = [doc_id for doc_id, _tf in posting_list]
        
        self.removed_tf = True

    def save_index(self, file_path):
        pass

    def load_index(self, file_path):
        # while reading it i should also read the compression method used, if any, and set self.compression_method accordingly
        pass

    def compress_index(self, method="gamma"):
        if self.index is None or len(self.index) == 0:
            raise ValueError("Inverted index is empty. Build the index before applying compression.")

        methods_map = {
            "gamma": self.__gamma_encode,
            "delta": self.__delta_encode,
            "varbyte": self.__varbyte_encode
        }

        if method in methods_map:
            self.compression_method = method
            self.__general_compressor(methods_map[method])
        else:
            raise ValueError(f"Compression method '{method}' is not supported. Allowed methods: {list(methods_map.keys())}")
    
    def __general_compressor(self, compression_func):
        for term, posting_list in self.index.items():
            compressed_postings = []
            for doc_id, tf in posting_list:
                # Gamma encode doc_id and tf
                compressed_doc_id = compression_func(doc_id)
                compressed_tf = compression_func(tf)
                compressed_postings.append((compressed_doc_id, compressed_tf))
            self.index[term] = compressed_postings

    def __transform_unary(self, number):
        return '1' * number + '0'

    def __gamma_encode(self, number):
        bin_num = bin(number)[2:]

        rem = bin_num[1:]  # remove leading '1'
        length = len(bin_num)
        una = self.__transform_unary(length - 1)

        # print(f"Gamma encoding number: {number}, Binary: {bin_num}, Unary: {una}, Remainder: {rem}")
        
        return una + rem

    def __delta_encode(self, number):
        bin_num = bin(number)[2:]

        rem = bin_num[1:]  # remove leading '1'
        length = len(bin_num)
        gamma_length = self.__gamma_encode(length)
        
        return gamma_length + rem

    def __varbyte_encode(self, number):
        bin_num = bin(number)[2:] # remove '0b' prefix

        # print(f"Encoding number: {number}, Binary representation: {bin_num}")

        cl = ((len(bin_num) + 6) // 7) * 7
        bin_num = bin_num.zfill(cl)  # Pad with leading zeros to make its length a multiple of 8
        spits = [bin_num[i:i+7] for i in range(0, len(bin_num), 7)]
        
        # print(f"Number: {number}, Binary: {bin_num}, Spits: {spits}")
        
        last_byte = '1' + spits[-1]
        other_bytes = ['0' + s for s in spits[:-1]]
        
        # print(f"Last byte: {last_byte}, Other bytes: {other_bytes}")
        
        res = ''.join(other_bytes + [last_byte])

        return res

    
    def __gamma_decode(self, encoded_str): #! use yield
        iterator = iter(encoded_str)
        com_string = ''
        for char in iterator:
            if char == '0':
                break
            else:
                com_string += char
        length = len(com_string) 
        com_string = ''
        for _ in range(length):
            com_string += next(iterator)

        com_string = '1' + com_string
        print(f"Gamma decoding string: {encoded_str}, Unary part: {'1'*length + '0'}, Remainder: {com_string[1:]}, Result: {int(com_string, 2)}")
        
        return com_string


    def __delta_decode(self, encoded_str):#! use yield
        iterator = iter(encoded_str)
        com_string = ''
        for char in iterator:
            if char == '0':
                break
            else:
                com_string += char
        lenght = len(com_string)
        com_string = ''
        for _ in range(lenght):
            com_string += next(iterator)
        
        length_ = '1' + com_string
        length__ = int(length_, 2)
        print(f"Delta decoding string: {encoded_str}, Gamma part: {'1'*lenght + '0'}, Length: {length__}, Remainder: {encoded_str[len(com_string)+lenght+1:]}")

        com_string = ''
        for _ in range(length__-1):
            com_string += next(iterator)

        com_string = '1' + com_string
        print(f"Delta decoding string: {encoded_str}, Gamma part: {'1'*lenght + '0'}, Length: {length__}, Remainder: {com_string[1:]}, Result: {int(com_string, 2)}")
        return com_string
    
    def __varbyte_decode(self, encoded_str): 
        com_string = ''
        for i in range(0, len(encoded_str), 8):
            byte = encoded_str[i:i+8]
            if byte[0] == '0':  # last byte
                com_string += byte[1:]
            else:
                com_string += byte[1:]
                print(f"Varbyte decoding string: {encoded_str}, Byte: {byte}, Result so far: {int(com_string, 2)}")
                yield com_string
                com_string = ''

        # return com_string


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