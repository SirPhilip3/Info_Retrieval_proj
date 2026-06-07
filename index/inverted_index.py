from .documents import tokenizer
import numpy as np
from bitarray import bitarray
from bitarray.util import ba2int, int2ba

import struct

COMPRESSION_CODES = {
    None: 0,
    "gamma": 1,
    "delta": 2,
    "varbyte": 3,
}

COMPRESSION_METHODS = {
    0: None,
    1: "gamma",
    2: "delta",
    3: "varbyte",
}

MAX_UINT32 = 2**32 - 1


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
        if self.removed_tf:
            raise ValueError("Term frequencies have already been removed from the index.")

        for term, posting_list in self.index.items():
            self.index[term] = [doc_id for doc_id, _tf in posting_list]
        
        self.removed_tf = True



    def build_header(self):
        compression_code = COMPRESSION_CODES[self.compression_method]
        header = struct.pack('<IB', compression_code, self.removed_tf)
        return header
        

    def save_index(self, file_path): #! temporary just save the postings not the terms 
        if self.compression_method != 'gamma':
            raise ValueError("Compression method is not set. Apply compression before saving the index. Only 'gamma' compression is currently supported for saving.")
        
        header = self.build_header()
        
        with open(file_path, 'wb') as f:
            f.write(header)
            if self.removed_tf:
                for term, posting_list in self.index.items():
                    l = len(posting_list)
                    
                    if l > MAX_UINT32:
                        raise ValueError(f"Posting list for term '{term}' exceeds maximum length of {MAX_UINT32}. Cannot save index.")
                    
                    f.write(struct.pack('<I', l))  #! write the length of the posting list maybe to optimize the loading phase

                    for doc_id in posting_list:
                        f.write(doc_id.tobytes())
                    # f.write(b'\x00')  # separator between posting lists            
            else :
                raise NotImplementedError("Saving index with term frequencies is not implemented yet.")
                

    def load_index(self, file_path):
        if self.index != None and len(self.index) > 0:
            raise ValueError("Inverted index is not empty. Clear the index before loading a new one.")

        with open(file_path, 'rb') as f:
            header = f.read(5)  # Read the first 5 bytes for the header
            compression_code, removed_tf_flag = struct.unpack('<IB', header)
            
            print(f"Loading index with compression code: {compression_code}, removed_tf_flag: {removed_tf_flag}")

            self.compression_method = COMPRESSION_METHODS[compression_code]
            self.removed_tf = bool(removed_tf_flag)

            if self.compression_method != 'gamma':
                raise NotImplementedError(f"Loading index with compression method '{self.compression_method}' is not implemented yet. Only 'gamma' compression is currently supported for loading.")
            # Now read the rest of the file for the index data
            index_data = f.read()
            self.decompress_index_from_file(index_data, 'gamma')  # Implement this method to parse and load the index data

        return self.index #!useless 
    
    def decompress_index_from_file(self, index_data, decompression_method):

        decomp_methods_map = { 
            "gamma": self.__gamma_decode,
            "delta": self.__delta_decode,
            "varbyte": self.__varbyte_decode
        }

        if decompression_method not in decomp_methods_map:
            raise ValueError(f"Decompression method '{decompression_method}' is not supported. Allowed methods: {list(decomp_methods_map.keys())}")
        
        decomp_func = decomp_methods_map[decompression_method]

        moch_term = 0 #! important change to real terms from file

        while len(index_data) > 0:
            pl_size = struct.unpack('<I', index_data[:4])[0] # [0] since unpack returns a tuple
            
            index_data = index_data[4:] # remove the bytes we just read
            
            print(f"Decompressing term: {moch_term}, Posting list size: {pl_size}, Remaining data length: {len(index_data)}")
            term = str(moch_term) #! important change to real terms from file
            self.index[term] = []

            for doc_id in decomp_func(bitarray(index_data[:pl_size])):
                self.index[term].append(doc_id)
            
            index_data = index_data[pl_size:] 

            moch_term += 1
        
        return self.index #! useless

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
            if self.removed_tf: #! can be optimzed 
                for doc_id in posting_list:
                    compressed_doc_id = compression_func(doc_id)
                    compressed_postings.append(compressed_doc_id)
            else:
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

        return bitarray(una + rem)

    def __delta_encode(self, number):
        bin_num = bin(number)[2:]

        rem = bin_num[1:]  # remove leading '1'
        length = len(bin_num)
        gamma_length = self.__gamma_encode(length)
        
        return bitarray(gamma_length + rem)

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

    def __gamma_decode(self, encoded_bites): #! use yield

        # if self.removed_tf:

        while len(encoded_bites) > 0:
            to_read = 0
            n = len(encoded_bites)
            # print(f"Gamma decoding bitarray: {encoded_bites.to01()}")
            print(f"remainig size to decode: {n} bits")

            for i in range(n):
                if encoded_bites[i] == 0:
                    to_read = i
                    break
            
            
            for _ in range(to_read+1):
                encoded_bites.pop(0)
            
            encoded_bites.insert(0, 1)

            sep = encoded_bites[:to_read+1]
            # print(f"Gamma decoding: Unary part: {'1'*to_read + '0'}, Remainder: {encoded_bites[to_read+1:].to01()}, Result: {ba2int(sep)}")

            yield ba2int(sep) 

        # encoded_bites.pop(0) #! remove the '0' that separates the unary part from the remainder
        # encoded_bites = encoded_bites[:to_read] #! the remainder is the next to_read bits



        # encoded_str = encoded_bites.to01() 
        # iterator = iter(encoded_str)
        # com_string = ''
        # for char in iterator:
        #     if char == '0':
        #         break
        #     else:
        #         com_string += char
        # length = len(com_string) 
        # com_string = ''
        # for _ in range(length):
        #     com_string += next(iterator)

        # com_string = '1' + com_string
        # print(f"Gamma decoding string: {encoded_str}, Unary part: {'1'*length + '0'}, Remainder: {com_string[1:]}, Result: {int(com_string, 2)}")
        
        # return com_string


    def __delta_decode(self, encoded_bytes): #! use yield
        encoded_str = encoded_bytes.to01()
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
    
    def __varbyte_decode(self, encoded_bytes): #! use yield
        encoded_str = encoded_bytes.to01()
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

    def pfor_delta_encode(self, d_gaps, k=4): #! compression given the full d-gap list returns the compressed d-gap list and the outliers
        min_ = min(d_gaps)

        outliers = []

        for i in range(len(d_gaps)):
            var = d_gaps[i] - min_
            if var >= 2**k-1:
                outliers.append(bin(var)[2:])
                d_gaps[i] = "1"*k
            else:
                d_gaps[i] = bin(var)[2:].zfill(k)
        
        # print(f"PForDelta encoding: D-gaps: {d_gaps}, Outliers: {outliers}")
        return d_gaps, outliers, min_

    def pfor_delta_decode(self, d_gap_array, outliers, min, k=4):
        decoded = []
        outlier_index = 0
        for d_gap in d_gap_array:
            if d_gap == "1"*k:
                decoded.append(int(outliers[outlier_index], 2) + min)
                outlier_index += 1
            else:
                decoded.append(int(d_gap, 2) + min)
        
        return decoded

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