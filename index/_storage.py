import struct
from bitarray import bitarray
from bitarray.util import int2ba

COMPRESSION_METHODS_ENC = {
    "None": 0,
    "gamma": 1,
    "delta": 2,
    "varbyte": 3,
    "pfor_delta": 4
}

COMPRESSION_METHODS_DEC = {
    0: "None",
    1: "gamma",
    2: "delta",
    3: "varbyte",
    4: "pfor_delta"
}

PFOR_DELTA_EXTRA_STRATEGY_ENC = { # strategy for the last block in PForDelta encoding and outliers handling
    0: "None",
    1: "gamma",
    2: "delta",
    3: "varbyte",
}

PFOR_DELTA_EXTRA_STRATEGY_DEC = {
    "None": 0,
    "gamma": 1,
    "delta": 2,
    "varbyte": 3,
}

MAX_UINT32 = 2**32 - 1

class IndexStorage: 
    def __init__(self, index_obj, file_path):
        self.index_obj = index_obj
        self.file_path = file_path
        # pass
    
    def clear_storage(self):
        self.file_path = None

    def set_file_path(self, file_path):
        self.file_path = file_path
    
    def build_header(self):
        compression_code = COMPRESSION_METHODS_ENC[self.index_obj.compression_method]
        header = struct.pack('<IB', compression_code, self.index_obj.removed_tf)
        return header
    
    def write_to_disk(self):
        header = self.build_header()
        
        with open(self.file_path, 'wb') as f:
            
            f.write(header)

            combined_bits = bitarray()

            for term, posting_list in self.index_obj.index.items():

                combined_bits.clear()  # Reset combined_bits for each term

                term_bytes = term.encode('utf-8')
                term_len = len(term_bytes)
                f.write(struct.pack('<I', term_len))
                f.write(struct.pack(f'<{term_len}s', term_bytes))
                
                if self.index_obj.removed_tf:
                    for doc_id in posting_list:

                        if isinstance(doc_id, int):
                            combined_bits.extend(int2ba(doc_id, length=32))  # Convert integer to bitarray of fixed length
                        else:
                            # print(f"Compressing term: {term}, doc_id: {doc_id}")
                            combined_bits.extend(doc_id) #! doesnt work with plain inegers
                    
                else: #! if we have term frequencies, we interleave doc_ids and tfs in the bitarray
                    for doc_id, tf in posting_list:
                        if isinstance(doc_id, int):
                            combined_bits.extend(int2ba(doc_id, length=32))
                            combined_bits.extend(int2ba(tf, length=32))  # Convert integer to bitarray of fixed length
                        else:   
                            # print(f"Compressing term: {term}, doc_id: {doc_id}, tf: {tf}")
                            combined_bits.extend(doc_id)
                            combined_bits.extend(tf)

                bit_length = len(combined_bits)

                if bit_length > MAX_UINT32:
                    raise ValueError(f"Posting list for term '{term}' is too large to be represented in 4 bytes. Bit length: {bit_length} bits.")

                f.write(struct.pack('<I', bit_length))

                # 2. Let bitarray handle the byte conversion and file write directly
                combined_bits.tofile(f)

    def read_from_disk(self):

        if self.index_obj.index != {}:
            raise ValueError("Index object is not empty. Cannot read from disk into a non-empty index.")

        with open(self.file_path, 'rb') as f:

            header = f.read(5)  
            compression_code, removed_tf_flag = struct.unpack('<IB', header)
            
            print(f"Loading index with compression code: {compression_code}, removed_tf_flag: {removed_tf_flag}")
            
            detected_method = COMPRESSION_METHODS_DEC[compression_code]
            # self.index_obj.compression_method = COMPRESSION_METHODS[compression_code]
            self.index_obj.compression_method = "None" #! bc at the end the index will have docids as integers
            self.index_obj.removed_tf = bool(removed_tf_flag)

            self.decompress_index_from_file(f, detected_method)  # Implement this method to parse and load the index data

            # self.raw_read(f)

    def raw_read(self, f): #!doesnt decompress keeps the compressed bitarrays in memory
        pass
    
    def decompress_index_from_file(self, index_data, decompression_method):

        decomp_methods_map = { 
            "None" : self.index_obj.compressor.no_decompression,
            "gamma": self.index_obj.compressor.gamma_decode,
            "delta": self.index_obj.compressor.delta_decode,
            "varbyte": self.index_obj.compressor.varbyte_decode
        }

        if decompression_method not in decomp_methods_map:
            raise ValueError(f"Decompression method '{decompression_method}' is not supported. Allowed methods: {list(decomp_methods_map.keys())}")
        
        decomp_func = decomp_methods_map[decompression_method]

        moch_term = 0 #! important change to real terms from file

        while True:
            term_len_bytes = index_data.read(4)
            if not term_len_bytes:
                break # Clean End of File reached

            term_len = struct.unpack('<I', term_len_bytes)[0]
            term_bytes = index_data.read(term_len)
            if not term_bytes:
                break # Clean End of File reached

            term = term_bytes.decode('utf-8') #! important change to real terms from file

            size_bytes = index_data.read(4)
            if not size_bytes:
                break # Clean End of File reached
                
            pl_size = struct.unpack('<I', size_bytes)[0] # [0] since unpack returns a tuple
            
            # print(f"Decompressing term: {moch_term}, Posting list size: {pl_size}, Reading: {index_data.tell()} bytes")
            # term = str(moch_term) #! important change to real terms from file
            self.index_obj.index[term] = []

            byte_length = (pl_size + 7) // 8
        
            # 1. Read the bits directly out of the file stream using bitarray
            bits = bitarray()
            bits.fromfile(index_data, byte_length)
            
            # 2. Slice off the trailing padding bits required by the filesystem
            bits = bits[:pl_size] 
            
            if self.index_obj.removed_tf:
                for doc_id in decomp_func(bits):
                    self.index_obj.index[term].append(doc_id)
            else:
                iterator = iter(decomp_func(bits)) #! maybe better to implement the decompression functions as generators that yield doc_id and tf pairs directly instead of returning a list, so we can avoid storing the whole decompressed posting list in memory at once

                # 2. Zip it with itself to group elements into pairs of 2
                for doc_id, tf in zip(iterator, iterator):
                    self.index_obj.index[term].append((doc_id, tf))
                
            moch_term += 1

        # return self.index #! useless
